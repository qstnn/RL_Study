"""Isaac Sim worker; run.py owns the isolated request and output directory."""

import argparse
import csv
import json
from pathlib import Path
import sys

from common import REPO, catalog, now, sha256, validate_checkpoint, write_json
import protocol


def runtime_check(env, cfg, run):
    import torch
    actual = (env.num_obs, env.num_privileged_obs, env.num_actions)
    if actual != (57, 236, 16):
        raise ValueError(f"Runtime dimensions changed: {actual}")
    base = env.unwrapped
    if list(base.observation_manager.active_terms["policy"]) != list(catalog.POLICY_TERMS):
        raise ValueError("Actor term order changed or privileged information leaked")
    robot = base.scene["robot"]
    action_names = []
    for name in ("joint_pos", "joint_vel"):
        term = base.action_manager.get_term(name)
        action_names.extend(robot.joint_names[i] for i in term._joint_ids)
    if action_names != catalog.CONTRACT["action_order"]:
        raise ValueError("Action joint order mismatch")
    obs, extras = env.get_observations()
    if not torch.isfinite(obs).all() or not torch.isfinite(extras["observations"]["critic"]).all():
        raise ValueError("Non-finite initial observations")
    if abs(base.step_dt - 0.02) > 1e-9 or abs(cfg.sim.dt - 0.001) > 1e-9:
        raise ValueError("Control/physics rate changed")
    write_json(run / "runtime_check.json", {
        "checked_at": now(), "dimensions": actual, "action_order": action_names,
        "policy_terms": base.observation_manager.active_terms["policy"],
        "physics_dt": cfg.sim.dt, "policy_dt": base.step_dt,
        "body_mass_kg": robot.root_physx_view.get_masses().sum(-1).cpu().tolist(),
        "status": "instantiated", "behavior_acceptance": "not_evaluated",
    })


def evaluate(env, runner, args, run, app):
    """Stop on termination before sampling the auto-reset state."""
    import math
    import torch
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.linear_leg_spec import LIMITS
    from isaaclab.utils.math import euler_xyz_from_quat

    validate_checkpoint(args["checkpoint"])
    runner.load(args["checkpoint"], load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    base, reports = env.unwrapped, []
    robot, sensor = base.scene["robot"], base.scene["contact_forces"]
    wheel_ids = sensor.find_bodies(list(catalog.WHEEL_BODIES), preserve_order=True)[0]
    body_ids = [i for i, name in enumerate(sensor.body_names) if name not in catalog.WHEEL_BODIES]
    joint_ids = robot.find_joints(list(catalog.LEG_NAMES + catalog.WHEEL_NAMES), preserve_order=True)[0]
    lower = torch.tensor([LIMITS[n][0] for n in catalog.LEG_NAMES] + [-37.0]*4, device=env.device)
    upper = torch.tensor([LIMITS[n][1] for n in catalog.LEG_NAMES] + [37.0]*4, device=env.device)
    seeds = args["eval_seeds"] if args["mode"] == "evaluate" else [args["seed"]]
    cases = list(protocol.CASES) if args["mode"] == "evaluate" else ["start_stop_03"]
    for seed in seeds:
        for case in cases:
            env.seed(seed)
            with torch.inference_mode():
                env.reset()
            command = base.command_manager.get_term("base_velocity")
            rows, reason, terminated, truncated, falls = [], None, 0, 0, 0
            initial = robot.data.root_state_w[0].detach().cpu().tolist()
            with torch.inference_mode():
                for step in range(protocol.STEPS):
                    if not app.is_running():
                        reason = "application_closed"
                        break
                    target = protocol.command_at(case, step * base.step_dt)
                    command.time_left.fill_(1e6)
                    command.is_standing_env.fill_(False)
                    command.is_heading_env.fill_(False)
                    command.vel_command_b[:] = torch.tensor(target, device=env.device)
                    obs, extras = env.get_observations()
                    if not torch.isfinite(obs).all() or not torch.isfinite(extras["observations"]["critic"]).all():
                        reason = "non_finite_observation"
                        break
                    actions = policy(obs)
                    if not torch.isfinite(actions).all():
                        reason = "non_finite_action"
                        break
                    _, _, dones, _ = env.step(actions)
                    if dones.any():
                        # Native step already reset these states; omit their new-spawn measurements.
                        terminated = int(base.termination_manager.terminated.sum())
                        truncated = int(base.termination_manager.time_outs.sum())
                        falls = int(base.termination_manager.get_term("bad_orientation_2").sum())
                        reason = "terminated" if terminated else "truncated"
                        break
                    data, forces = robot.data, sensor.data.net_forces_w[0]
                    targets = torch.cat([base.action_manager.get_term(n).processed_actions[0] for n in ("joint_pos", "joint_vel")])
                    clipped = ((targets <= lower+1e-5) | (targets >= upper-1e-5)).float()
                    torques, limits = data.applied_torque[0, joint_ids], data.joint_effort_limits[0, joint_ids]
                    tilt = torch.acos((-data.projected_gravity_b[0, 2]).clamp(-1., 1.)) * (180.0/math.pi)
                    row = {
                        "t": step * base.step_dt, "command_vx": target[0],
                        "vx": float(data.root_lin_vel_b[0, 0]), "vy": float(data.root_lin_vel_b[0, 1]),
                        "velocity_error": float((data.root_lin_vel_b[0, :2] - torch.tensor(target[:2], device=env.device)).norm()),
                        "speed": float(data.root_lin_vel_b[0, :2].norm()),
                        "yaw_rate_abs": float(data.root_ang_vel_b[0, 2].abs()), "tilt_deg": float(tilt),
                        "yaw_rad": float(euler_xyz_from_quat(data.root_quat_w)[2][0]),
                        "x": float(data.root_pos_w[0, 0]), "y": float(data.root_pos_w[0, 1]), "height": float(data.root_pos_w[0, 2]),
                        "four_wheel": float((forces[wheel_ids, 2] > 5.).all()),
                        "body_contact": float((forces[body_ids].norm(dim=-1) > 1.).any()),
                        "torque_abs_max": float(torques.abs().max()),
                        "torque_limit_fraction": float((torques.abs() >= limits*.99).float().mean()),
                        "target_clip_fraction": float(clipped.mean()),
                    }
                    for index, name in enumerate(catalog.LEG_NAMES + catalog.WHEEL_NAMES):
                        row["torque_"+name], row["target_"+name] = float(torques[index]), float(targets[index])
                        row["clipped_"+name] = float(clipped[index])
                    for index, name in enumerate(catalog.WHEEL_BODIES):
                        row["force_z_"+name] = float(forces[wheel_ids[index], 2])
                    if not all(math.isfinite(value) for value in row.values()):
                        reason = "non_finite_measurement"
                        break
                    rows.append(row)
            with (run / f"seed{seed}_{case}.csv").open("w", newline="") as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            report = protocol.summarize(rows, case, seed, reason)
            report.update(initial_root_state=initial, terminations=terminated, timeouts=truncated, fall_count=falls)
            reports.append(report)
            write_json(run / f"seed{seed}_{case}.json", report)
    write_json(run / "evaluation.json", {
        "protocol": protocol.VERSION, "checkpoint": args["checkpoint"],
        "checkpoint_sha256": sha256(args["checkpoint"]), "cases": reports, "seeds": seeds,
        "success_rate": sum(r["success"] for r in reports)/len(reports),
        "fall_rate": sum(r["fall_count"] for r in reports)/len(reports),
        "all_cases_complete": all(r["complete"] for r in reports),
        "final_acceptance": "not_evaluated", "ppo_updates": 0,
        "deterministic_screen": True, "random_initial_state_variation": False,
        "note": "Stage1 fixed reset; multiple seeds can yield identical trajectories; no robustness claim",
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    run = p.parse_args().run.resolve()
    args = json.loads((run / "request.json").read_text())
    sys.path.insert(0, str(REPO / "source/rl_training"))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args["headless"], device=args["device"]).app
    env, runner = None, None
    try:
        import gymnasium as gym
        import torch
        import rl_training.tasks  # noqa: F401
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        class RouteRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                return super().save(path, infos=dict(infos or {}, multiskill_contract=catalog.CONTRACT))

        catalog.register_tasks()
        cfg = load_cfg_from_registry(catalog.TASK, "env_cfg_entry_point")
        agent = load_cfg_from_registry(catalog.TASK, "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs = args["num_envs"]
        cfg.seed = agent.seed = args["seed"]
        cfg.sim.device = agent.device = args["device"]
        cfg.scene.robot.spawn.usd_dir = str(run / "asset_cache")
        cfg.scene.robot.spawn.usd_file_name = "lighthw_multiskill.usd"
        agent.max_iterations = args["max_iterations"]
        if args["mode"] in ("play", "evaluate"):
            cfg.observations.policy.enable_corruption = False
            cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
        dump_yaml(str(run / "params/env.yaml"), cfg)
        dump_yaml(str(run / "params/agent.yaml"), agent)
        env = RslRlVecEnvWrapper(gym.make(catalog.TASK, cfg=cfg), clip_actions=agent.clip_actions)
        runtime_check(env, cfg, run)
        if args["mode"] == "smoke":
            with torch.inference_mode():
                for step in range(100):
                    obs, reward, dones, extras = env.step(torch.zeros((env.num_envs, 16), device=env.device))
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all() or not torch.isfinite(extras["observations"]["critic"]).all():
                        raise RuntimeError("Non-finite smoke rollout")
                    if dones.any():
                        raise RuntimeError(f"Smoke rollout terminated at step {step}")
            write_json(run / "smoke.json", {"steps": 100, "num_envs": env.num_envs, "ppo_updates": 0,
                                           "runtime_passed": True, "behavior_acceptance": "not_evaluated"})
        else:
            runner = RouteRunner(env, agent.to_dict(), log_dir=str(run / "training") if args["mode"] == "train" else None,
                                 device=args["device"])
            if args["mode"] == "train":
                runner.learn(num_learning_iterations=args["max_iterations"], init_at_random_ep_len=True)
                checkpoints = sorted((run / "training").glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
                if not checkpoints:
                    raise RuntimeError("PPO returned without a checkpoint")
                validate_checkpoint(checkpoints[-1])
                write_json(run / "training_result.json", {"checkpoint": str(checkpoints[-1]),
                    "sha256": sha256(checkpoints[-1]), "requested_updates": args["max_iterations"],
                    "behavior_acceptance": "not_evaluated"})
            else:
                evaluate(env, runner, args, run, app)
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
