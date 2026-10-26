"""Check configured LightHW rewards, dimensions, zero-action dynamics and resets.

Run with Isaac Lab's Python launcher. The same checks intentionally fail on the
legacy StableRoll configuration; --task selects the configuration under test.
Artifacts are written only to --output_dir, normally inside route-local logs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def check_configured_terms(cfg):
    """Exercise the actual configured reward callables on upright/inverted states."""
    command = torch.zeros(4, 3)
    data = SimpleNamespace(
        projected_gravity_b=torch.tensor([[0., 0., -1.], [0., -.6, -.8], [0., -1., 0.], [0., 0., 1.]]),
        root_lin_vel_b=torch.zeros(4, 3),
        root_ang_vel_b=torch.zeros(4, 3),
    )
    fixture = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                              command_manager=SimpleNamespace(get_command=lambda name: command))
    checks = {}
    values = {}
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
        term = getattr(cfg.rewards, name)
        value = term.func(fixture, **term.params)
        values[name] = value.tolist()
        checks[name + "_rejects_inversion"] = bool(value[0] > 0 and value[3] == 0 and value[2] == 0)
        checks[name + "_finite_monotonic"] = bool(torch.isfinite(value).all() and (value[:-1] >= value[1:]).all())
    term = cfg.rewards.stand_flat_orientation_l2
    value = term.func(fixture, **term.params)
    values["stand_orientation"] = value.tolist()
    checks["stand_orientation_distinguishes_inversion"] = bool(value[0] == 0 and value[3] > value[2] > value[1] > 0)
    command[:, 0] = .4
    checks["stand_orientation_command_gate"] = bool((term.func(fixture, **term.params) == 0).all())
    term = cfg.terminations.bad_orientation_2
    checks["orientation_termination_enabled"] = term is not None
    if term is not None:
        checks["orientation_termination_values"] = term.func(fixture, **term.params).tolist() == [
            False, False, True, True
        ]
        checks["orientation_is_failure"] = not term.time_out
    term = cfg.terminations.illegal_contact
    checks["base_contact_termination_enabled"] = term is not None
    if term is not None:
        checks["base_only_contact_termination"] = term.params["sensor_cfg"].body_names == ["Base_link"]
        checks["base_contact_is_failure"] = not term.time_out
    checks["physical_step_1ms"] = math.isclose(cfg.sim.dt, .001)
    checks["control_50hz"] = math.isclose(cfg.sim.dt * cfg.decimation, .02)
    checks["render_at_control_rate"] = cfg.sim.render_interval == cfg.decimation
    checks["contact_at_physics_rate"] = math.isclose(cfg.scene.contact_forces.update_period, cfg.sim.dt)
    for name in ("height_scanner", "height_scanner_base"):
        checks[name + "_at_control_rate"] = math.isclose(getattr(cfg.scene, name).update_period, .02)
    return checks, values


def main():
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    checks, values = check_configured_terms(cfg)
    result = {"task": args.task, "configured_checks": checks, "fixture_values": values}
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[LightHWCheck] configured terms:", json.dumps(checks), flush=True)
    assert all(checks.values()), "Configured terms failed: " + ", ".join(k for k, v in checks.items() if not v)

    # Deterministic, flat, native-actuator plant check. Keep both observation
    # groups and the new failure terms, so dimensions and reset wiring are real.
    cfg.seed = 42
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.visual_material = None
    for name in cfg.events.to_dict():
        setattr(cfg.events, name, None)
    for name in cfg.curriculum.to_dict():
        setattr(cfg.curriculum, name, None)
    cfg.terminations.time_out = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        dims = {"policy": obs["policy"].shape[-1], "critic": obs["critic"].shape[-1],
                "action": env.action_manager.total_action_dim}
        result["dimensions"] = dims
        assert dims == {"policy": 60, "critic": 236, "action": 16}, dims
        robot = env.scene["robot"]
        leg_ids = [i for i, name in enumerate(robot.joint_names) if "wheel" not in name]
        wheel_ids, _ = env.scene["contact_forces"].find_bodies(".*_wheel_link")
        action = torch.zeros((1, dims["action"]), device=env.device)
        trace = []
        resets = 0
        for _ in range(args.steps):
            with torch.inference_mode():
                obs, reward, terminated, truncated, _ = env.step(action)
                resets += int((terminated | truncated).sum())
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
                assert torch.isfinite(reward).all()
                tilt = torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)) * 180 / math.pi
                wheel_force = env.scene["contact_forces"].data.net_forces_w[0, wheel_ids, 2]
                trace.append(torch.stack((robot.data.root_pos_w[0, 2], tilt,
                    robot.data.joint_vel[0, leg_ids].square().mean(),
                    robot.data.root_lin_vel_b[0, :2].norm(), (wheel_force > 1).all().float())).cpu())
        samples = torch.stack(trace)
        torch.save(samples, output / "zero_action_trace.pt")
        measured = samples[100:]
        assert len(measured) > 0, "Use more than 100 steps to exclude initial settling."
        result["zero_action"] = {
            "steps": args.steps, "control_dt": env.step_dt, "reset_count": resets,
            "height_mean": measured[:, 0].mean().item(),
            "tilt_p95_deg": measured[:, 1].quantile(.95).item(),
            "leg_speed_rms": measured[:, 2].mean().sqrt().item(),
            "xy_speed_mean": measured[:, 3].mean().item(),
            "four_wheel_contact_fraction": measured[:, 4].mean().item(),
        }
        m = result["zero_action"]
        plant_checks = {
            "no_resets": resets == 0, "height": abs(m["height_mean"] - .545) < .015,
            "tilt": m["tilt_p95_deg"] < 5, "leg_speed": m["leg_speed_rms"] < .1,
            "xy_speed": m["xy_speed_mean"] < .01, "wheel_contact": m["four_wheel_contact_fraction"] > .95,
        }
        result["plant_checks"] = plant_checks
        # Trigger the actual termination manager by placing an inverted robot
        # above the plane. This isolates orientation from the contact condition.
        with torch.inference_mode():
            pose = robot.data.root_pose_w.clone()
            pose[:, 2] = 1.5
            pose[:, 3:7] = torch.tensor([0., 1., 0., 0.], device=env.device)
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
            _, _, terminated, truncated, _ = env.step(action)
            result["inversion_causes_failure_reset"] = bool(terminated[0] and not truncated[0]
                and env.termination_manager.get_term("bad_orientation_2")[0])
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[LightHWCheck]", json.dumps(result, indent=2), flush=True)
        assert all(plant_checks.values()), "Zero-action plant failed: " + str(plant_checks)
        assert result["inversion_causes_failure_reset"], "Inversion did not produce a real failure reset."
        print("[LightHWCheck] PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Kit's fast shutdown can terminate Python before a pending exception
        # propagates. Let the failed process exit normally with its traceback.
        print("[LightHWCheck] FAIL", flush=True)
        raise
    else:
        simulation_app.close()
