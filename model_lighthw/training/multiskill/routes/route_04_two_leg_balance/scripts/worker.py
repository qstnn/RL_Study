"""R4 Isaac/PPO 子进程；每次独立USD缓存、物理读回、模型/Adam及1kHz冻结轨迹。"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from common import SOURCE, read, write, now, sha, manifest, checked_checkpoint
from two_leg_balance.spec import contract, runner_config, LEG_JOINTS, WHEELS, LOW, HIGH


def make_runner(run, c):
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    class Runner(OnPolicyRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.completed_updates = 0
            self.git_status_repos = []
            with torch.no_grad():
                self.alg.policy.actor[-1].weight.mul_(.01)
                self.alg.policy.actor[-1].bias.zero_()
                self.alg.policy.log_std[:12].fill_(np.log(.2))
                self.alg.policy.log_std[12:].fill_(np.log(.08))
            original_update = self.alg.update

            def update():
                before = {k: v.clone() for k, v in self.alg.policy.state_dict().items() if k.startswith("actor.")}
                losses = original_update()
                with torch.no_grad():
                    self.alg.policy.log_std.clamp_(np.log(.03), np.log(.6))
                    for value in self.alg.policy.state_dict().values():
                        if not torch.isfinite(value).all():
                            raise RuntimeError("R4 non-finite PPO parameters")
                    for state in self.alg.optimizer.state.values():
                        for value in state.values():
                            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                                raise RuntimeError("R4 non-finite Adam state")
                    storage = self.alg.storage
                    self.alg.policy.update_distribution(storage.observations.flatten(0, 1))
                    mu, sigma = self.alg.policy.action_mean, self.alg.policy.action_std
                    oldmu, olds = storage.mu.flatten(0, 1), storage.sigma.flatten(0, 1)
                    kl = (torch.log(sigma / olds) + (olds.square() + (oldmu - mu).square())
                          / (2 * sigma.square()) - .5).sum(-1).mean()
                    changed = any(not torch.equal(v, self.alg.policy.state_dict()[k]) for k, v in before.items())
                self.completed_updates += 1
                row = dict(completed_updates=self.completed_updates, actor_changed=changed,
                           learning_rate=self.alg.learning_rate, rollout_kl=float(kl),
                           losses={k: float(v) for k, v in losses.items()})
                with (run / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                return losses

            self.alg.update = update

        def save(self, path, infos=None):
            info = dict(contract=c, source_manifest=manifest(), completed_updates=self.completed_updates,
                        learning_rate=self.alg.learning_rate, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                        numpy_rng=np.random.get_state(), asset_manifest=read(run / "asset_manifest.json"), physics_state_saved=False)
            super().save(path, info)

    return Runner


def runtime(env, run):
    import torch
    import isaaclab.sim as sim
    from pxr import Usd
    b, d = env.unwrapped, env.unwrapped.robot.data
    assert [env.num_obs, env.num_privileged_obs, env.num_actions] == [305, 88, 16], (env.num_obs, env.num_privileged_obs, env.num_actions)
    assert [b.robot.joint_names[i] for i in b.joint_ids] == list(LEG_JOINTS + WHEELS)
    torch.testing.assert_close(b.mass, torch.full_like(b.mass, read(run / "geometry.json")["mass_kg"]), atol=.002, rtol=1e-4)
    import xml.etree.ElementTree as ET
    links = {link.get("name"): link for link in ET.parse(run / "assets/urdf/lighthw.urdf").findall("link")}
    actual_inertia = b.robot.root_physx_view.get_inertias()[0].cpu().reshape(-1, 3, 3)
    for i, name in enumerate(b.robot.body_names):
        inertial = links[name].find("inertial")
        mass = float(inertial.find("mass").get("value"))
        torch.testing.assert_close(b.masses[0, i].cpu(), torch.tensor(mass), atol=2e-4, rtol=1e-3)
        com = [float(x) for x in inertial.find("origin").get("xyz").split()]
        torch.testing.assert_close(d.body_com_pos_b[0, i].cpu(), torch.tensor(com), atol=2e-4, rtol=1e-3)
        iv = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        matrix = torch.tensor([[iv['ixx'], iv['ixy'], iv['ixz']], [iv['ixy'], iv['iyy'], iv['iyz']],
                               [iv['ixz'], iv['iyz'], iv['izz']]])
        torch.testing.assert_close(torch.linalg.eigvalsh(actual_inertia[i]), torch.linalg.eigvalsh(matrix), atol=2e-5, rtol=.005)
    expected_effort = [24. if "wheel" in n else 48. for n in b.robot.joint_names]
    expected_speed = [37.7 if "wheel" in n else 12.57 for n in b.robot.joint_names]
    torch.testing.assert_close(d.joint_effort_limits[0], d.joint_effort_limits.new_tensor(expected_effort))
    torch.testing.assert_close(d.joint_vel_limits[0], d.joint_vel_limits.new_tensor(expected_speed))
    assert (d.joint_pos_limits[0, b.leg_ids, 0] <= d.joint_pos_limits.new_tensor(LOW)).all()
    assert (d.joint_pos_limits[0, b.leg_ids, 1] >= d.joint_pos_limits.new_tensor(HIGH)).all()
    props = b.robot.root_physx_view.get_material_properties()
    torch.testing.assert_close(props[..., :2], torch.ones_like(props[..., :2]))
    self_collision, angular_caps, gyroscopic = [], [], []
    for prim in Usd.PrimRange(sim.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")):
        attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if attr and attr.HasAuthoredValueOpinion():
            self_collision.append(bool(attr.Get()))
        cap = prim.GetAttribute("physxRigidBody:maxAngularVelocity")
        if cap and cap.HasAuthoredValueOpinion():
            angular_caps.append(float(cap.Get()))
        gyro = prim.GetAttribute("physxRigidBody:enableGyroscopicForces")
        if gyro and gyro.HasAuthoredValueOpinion():
            gyroscopic.append(bool(gyro.Get()))
    assert self_collision and all(self_collision)
    assert len(angular_caps) == len(b.robot.body_names) and all(abs(x - 100000.) < .01 for x in angular_caps)
    assert len(gyroscopic) == len(b.robot.body_names) and all(gyroscopic)
    obs, extras = env.get_observations()
    again, _ = env.get_observations()
    torch.testing.assert_close(obs, again, atol=0, rtol=0)
    assert torch.isfinite(obs).all() and torch.isfinite(extras["observations"]["critic"]).all()
    owned = obs.clone()
    env.step(torch.zeros(env.num_envs, 16, device=env.device))
    torch.testing.assert_close(obs, owned, atol=0, rtol=0)
    # 对一个槽做局部reset，其他槽的时钟/历史必须保持；重置槽的五帧不能带入旧回合。
    other_clock = b.physics_steps[1:].clone()
    other_history = b.history[1:].clone()
    b._reset_idx(torch.tensor([0], device=b.device))
    b.scene.write_data_to_sim()
    b.sim.forward()
    env.get_observations()
    torch.testing.assert_close(b.physics_steps[1:], other_clock)
    torch.testing.assert_close(b.history[1:], other_history)
    torch.testing.assert_close(b.history[0], b.history[0, :1].expand(5, -1))
    env.reset()
    b.episode_count.zero_()
    b.last_reports = []
    write(run / "runtime_check.json", dict(
        dimensions=[305, 88, 16], mass_kg=float(b.mass[0]), effort_limits_Nm=expected_effort,
        velocity_limits_rad_s=expected_speed, robot_friction=props[0].cpu().tolist(),
        self_collisions=self_collision, rigid_angular_caps_deg_s=angular_caps, gyroscopic_forces=gyroscopic,
        urdf_body_mass_com_inertia_checked=len(links), observation_ownership=True, partial_reset_history=True,
        physics_dt=b.physics_dt, policy_dt=b.step_dt, behavior_accepted=False))


def evaluate(env, runner, run, request, app):
    import numpy as np
    import torch
    b = env.unwrapped
    policy = runner.get_inference_policy(device=env.device)
    result = []
    for seed in request["eval_seeds"]:
        env.seed(seed)
        with torch.inference_mode():
            obs, _ = env.reset()
            # 固定槽0前/槽1后；单模式时两个槽都是同模式，分别保存。
            b.mode[:] = torch.arange(env.num_envs, device=b.device) % 2 if b.cfg.mode == "both" else int(b.cfg.mode == "rear")
            b.history_reset[:] = True
            obs, _ = env.get_observations()
            active = torch.ones(env.num_envs, dtype=torch.bool, device=b.device)
            frames, policy_frames = [], []
            ended = {}

            def capture(base, tick):
                m = base.current_metrics
                row = {k: v.detach().clone() for k, v in m.items()}
                row.update(t=(base.physics_steps * base.physics_dt).clone(),
                           mode=base.mode.clone(), active=active.clone(), origin=base.state.origin.clone(),
                           target_gravity=base.command()[1].clone(), actions=base.actions.clone(),
                           leg_targets=base.joint_targets.clone(),
                           torque=base.robot.data.applied_torque[:, base.joint_ids].clone())
                frames.append(row)

            b.physics_observer = capture
            try:
                for _ in range(b.max_episode_length):
                    if not app.is_running():
                        raise RuntimeError("R4 playback closed")
                    started = time.monotonic()
                    action = policy(obs)
                    policy_frames.append(dict(observation=obs.cpu().numpy().copy(), action=action.cpu().numpy().copy()))
                    obs, _, dones, _ = env.step(action)
                    for report in b.last_reports:
                        i = report["env_id"]
                        if bool(dones[i]) and bool(active[i]):
                            ended[i] = dict(report)
                            active[i] = False
                    if not active.any():
                        break
                    if not request["headless"]:
                        time.sleep(max(0., b.step_dt - (time.monotonic() - started)))
            finally:
                b.physics_observer = None
            if len(ended) != env.num_envs:
                raise RuntimeError("R4 frozen evaluation ended without terminal reports")
            path = run / f"trace_seed{seed}.npz"
            arrays = {k: torch.stack([row[k] for row in frames]).cpu().numpy() for k in frames[0]}
            arrays.update({f"policy_{k}": np.stack([row[k] for row in policy_frames]) for k in policy_frames[0]})
            np.savez_compressed(path, **arrays)
            print(f"[R4 FROZEN] seed={seed} ended={len(ended)} trace={path.name}", flush=True)
            for i, report in ended.items():
                report.update(seed=seed, trace=path.name, trace_env=i)
                result.append(report)
    grouped = {}
    for mode in ("front", "rear"):
        rows = [r for r in result if r["mode"] == mode]
        if rows:
            grouped[mode] = dict(cases=len(rows), successes=sum(r["success"] and r["timeout"] for r in rows),
                                 best_hold_s=max(r["best_hold"] for r in rows),
                                 mean_best_hold_s=sum(r["best_hold"] for r in rows) / len(rows))
    report = dict(cases=result, by_mode=grouped, ppo_updates=0, behavior_accepted=False,
                  note="Fixed-seed screening only; inspect traces and expand reset/seed coverage before promotion.")
    write(run / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    app = env = runner = None
    try:
        request, c = read(run / "request.json"), read(run / "contract.json")
        if manifest() != read(run / "source_manifest.json"):
            raise ValueError("R4 sources changed since snapshot")
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=request["headless"], device=request["device"]).app
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from two_leg_balance.env import TwoLegBalanceCfg, TwoLegBalanceEnv
        from two_leg_balance.spec import TASK
        if TASK not in gym.registry:
            gym.register(TASK, entry_point=TwoLegBalanceEnv, disable_env_checker=True)
        cfg = TwoLegBalanceCfg()
        cfg.scene.num_envs = request["num_envs"]
        cfg.seed = request["seed"]
        cfg.sim.device = request["device"]
        cfg.hold_seconds = request["hold_seconds"]
        cfg.episode_length_s = 10. + cfg.hold_seconds
        cfg.mode = request["support_mode"]
        cfg.target_pitches = tuple(row["pitch_rad"] for row in read(run / "geometry.json")["candidates"])
        cfg.scene.robot.spawn.asset_path = str(run / "assets/urdf/lighthw.urdf")
        cfg.scene.robot.spawn.usd_dir = str(run / "asset_cache")
        cfg.scene.robot.spawn.usd_file_name = "r4.usd"
        agent = runner_config(request["seed"])
        agent["device"] = request["device"]
        dump_yaml(str(run / "params/env.yaml"), cfg)
        write(run / "params/agent.json", agent)
        import importlib.metadata as metadata
        write(run / "environment.json", dict(python=sys.version, executable=sys.executable,
              torch=torch.__version__, rsl_rl=metadata.version("rsl-rl-lib"),
              gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              env_source=str(Path(sys.modules["two_leg_balance.env"].__file__).resolve())))
        class Wrapper(RslRlVecEnvWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces["critic"])
        env = Wrapper(gym.make(TASK, cfg=cfg), clip_actions=1.)
        runtime(env, run)
        b = env.unwrapped

        def sink(rows):
            with (run / "episodes.jsonl").open("a") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

        b.episode_sink = sink
        if request["mode"] == "smoke":
            reports = []
            b.episode_sink = lambda rows: (sink(rows), reports.extend(rows))
            with torch.inference_mode():
                for _ in range(b.max_episode_length + 5):
                    env.step(torch.zeros(env.num_envs, 16, device=env.device))
            if not any(row["timeout"] for row in reports):
                raise RuntimeError("Neutral-action smoke did not reach a natural timeout; inspect physical setup")
            result = dict(mode="smoke", ppo_updates=0, ended_episodes=len(reports),
                          natural_timeouts=sum(row["timeout"] for row in reports), behavior_accepted=False)
        else:
            runner = make_runner(run, c)(env, agent, log_dir=str(run / "training"), device=env.device)
            runner.logger_type = "tensorboard"
            start = 0
            if request.get("checkpoint"):
                saved = checked_checkpoint(request["checkpoint"], c)
                info = runner.load(request["checkpoint"], load_optimizer=request["mode"] == "train")
                start = info["completed_updates"]
                runner.completed_updates = runner.current_learning_iteration = start
                runner.alg.learning_rate = info["learning_rate"]
                for name, value in saved["model_state_dict"].items():
                    torch.testing.assert_close(runner.alg.policy.state_dict()[name].cpu(), value, atol=0, rtol=0)
                if request["mode"] == "train":
                    # 恢复Adam后逐张量核对，避免只恢复Actor却声称连续训练。
                    restored = runner.alg.optimizer.state_dict()
                    for key, state in saved["optimizer_state_dict"]["state"].items():
                        for name, value in state.items():
                            if isinstance(value, torch.Tensor):
                                torch.testing.assert_close(restored["state"][key][name].cpu(), value.cpu(), atol=0, rtol=0)
                    torch.set_rng_state(info["torch_rng"].cpu())
                    torch.cuda.set_rng_state_all(info["cuda_rng"])
                    random.setstate(info["python_rng"])
                    np.random.set_state(info["numpy_rng"])
                write(run / "resume.json", dict(checkpoint=request["checkpoint"], sha256=sha(request["checkpoint"]),
                      start_updates=start, exact_model=True, exact_optimizer=request["mode"] == "train",
                      physics_state_restored=False))
            if request["mode"] == "train":
                runner.save(str(run / "initial.pt"))
                runner.learn(num_learning_iterations=request["updates"], init_at_random_ep_len=False)
                runner.save(str(run / "final.pt"))
                result = dict(completed_updates=runner.completed_updates, new_updates=request["updates"],
                              checkpoint_sha256=sha(run / "final.pt"), behavior_accepted=False)
            else:
                result = evaluate(env, runner, run, request, app)
        if request["mode"] in ("evaluate", "play"):
            from verify_trace import verify
            result["trace_verification"] = verify(run)
        if manifest() != read(run / "source_manifest.json"):
            raise RuntimeError("R4 sources changed during run; results require explicit review")
        write(run / "result.json", result)
    except BaseException as error:
        write(run / "worker_error.json", dict(at=now(), error=str(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
