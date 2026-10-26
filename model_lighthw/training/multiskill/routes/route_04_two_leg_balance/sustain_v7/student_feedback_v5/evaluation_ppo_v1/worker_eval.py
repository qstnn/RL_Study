"""Teacher-off PPO mean Actor on the original 1 kHz sustain_v7 environment."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import torch

from common_eval import read, write, now, sha, require


def write_progress(run: Path, value: dict):
    """Atomic, read-only-friendly status updated every 10 simulated seconds."""
    destination = run / "progress.json"
    temporary = run / "progress.json.tmp"
    value = {**value, "updated_at": now()}
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)


class BufferedPhysicsTrace:
    """Bound GPU trace memory while retaining every 1 ms physical sample."""

    def __init__(self, capacity: int = 20_000):
        self.capacity = capacity
        self.buffer = None
        self.chunks = None
        self.used = 0
        self.samples = 0

    @torch.no_grad()
    def append(self, row: dict[str, torch.Tensor]):
        if self.buffer is None:
            self.buffer = {key: torch.empty((self.capacity, *value.shape),
                             dtype=value.dtype, device=value.device) for key, value in row.items()}
            self.chunks = {key: [] for key in row}
        if set(row) != set(self.buffer):
            raise RuntimeError("1 kHz physics metric fields changed during evaluation")
        if self.used == self.capacity:
            self.flush()
        for key, value in row.items():
            if value.shape != self.buffer[key].shape[1:] or value.dtype != self.buffer[key].dtype:
                raise RuntimeError("1 kHz physics metric shape/dtype changed: " + key)
            self.buffer[key][self.used].copy_(value)
        self.used += 1
        self.samples += 1

    def flush(self):
        if not self.used:
            return
        for key, value in self.buffer.items():
            self.chunks[key].append(value[:self.used].cpu().numpy().copy())
        self.used = 0

    def arrays(self):
        self.flush()
        if not self.chunks:
            raise RuntimeError("No physical samples captured")
        result = {key: np.concatenate(parts, axis=0) for key, parts in self.chunks.items()}
        if any(len(value) != self.samples for value in result.values()):
            raise RuntimeError("Buffered physical trace length mismatch")
        return result


@torch.inference_mode()
def rollout(env, app, run: Path, actor, seed: int):
    from features_feedback import FeedbackFeatures

    base = env.unwrapped
    env.seed(seed)
    raw, _ = env.reset()
    builder = FeedbackFeatures(actor.prefix, 2, base.device)
    obs = builder(raw, torch.ones(2, dtype=torch.bool, device=base.device))
    active = torch.ones(2, dtype=torch.bool, device=base.device)
    base.last_reports = []
    trace = BufferedPhysicsTrace()
    controls, ended, terminals = [], {}, []
    write_progress(run, dict(seed=seed, control_step=0, simulation_time_s=0.,
        task_clock_s=[0., 0.], post_capture_s=[0., 0.],
        first_break_time_s=[-1., -1.], active=[True, True], status="running"))

    def capture(physical, tick):
        time = physical.physics_steps * physical.physics_dt
        row = dict(physical.current_metrics)
        row.update(t=time, task_t=physical.state.clock(time), mode=physical.mode,
            active=active, origin=physical.state.origin,
            target_gravity=physical.command()[1], phase=physical.command()[0],
            start_time=physical.state.start_time, actions=physical.actions,
            leg_targets=physical.joint_targets,
            torque=physical.robot.data.applied_torque[:, physical.joint_ids],
            joint_velocity=physical.robot.data.joint_vel[:, physical.joint_ids])
        for key in ("capture_time", "first_break_time", "break_reason", "return_time",
                    "stand_request", "post_capture_s", "held", "interrupted",
                    "observation_complete", "failed", "boundary"):
            row[key] = getattr(physical.state, key)
        trace.append(row)

    base.physics_observer = capture
    try:
        for step in range(base.max_episode_length):
            if not app.is_running():
                raise RuntimeError("Simulation closed before both autonomous reports")
            if obs.shape != (2, 325):
                raise RuntimeError("PPO Actor must receive exactly raw305 plus 20 causal features")
            action = actor(obs)  # Deterministic mean; checkpoint log_std never sampled.
            if action.shape != (2, 16) or not torch.isfinite(action).all():
                raise RuntimeError("PPO mean action shape/nonfinite failure")
            task = base.state.clock(base.physics_steps * base.physics_dt)
            control = {key: value.cpu().numpy().copy() for key, value in dict(
                observation=obs, raw_observation=raw,
                capture_memory=obs[:, 305:307], trigger_action=obs[:, 307:323],
                estimated_feedback=obs[:, 323:325], actor_action=action,
                teacher_action=torch.zeros_like(action), action=action,
                time=task, takeover=torch.zeros(2, dtype=torch.bool, device=base.device),
                active=active, physics_time=base.physics_steps * base.physics_dt,
                return_time=base.state.return_time,
                hold_origin=base.state.hold_origin).items()}
            next_raw, reward, done, extras = env.step(action)
            control["reward"] = reward.cpu().numpy().copy()
            control["reward_terms"] = torch.stack(list(base.reward_terms.values()), -1).cpu().numpy().copy()
            controls.append(control)
            for report in base.last_reports:
                slot = report["env_id"]
                if bool(active[slot]) and bool(done[slot]):
                    if (not bool(extras["terminal_observation_valid"][slot])
                            or not torch.isfinite(extras["terminal_critic_observation"][slot]).all()):
                        raise RuntimeError("Terminal pre-reset Critic observation invalid")
                    if (bool(extras["time_outs"][slot]) != bool(report["timeout"])
                            or bool(extras["physical_terminated"][slot]) != bool(report["failed"])):
                        raise RuntimeError("Terminal timeout/physical-failure flag differs from 1 kHz report")
                    ended[slot] = dict(report)
                    terminals.append(dict(slot=slot, timeout=bool(extras["time_outs"][slot]),
                        failed=bool(extras["physical_terminated"][slot]),
                        terminal_stand_request=float(extras["terminal_critic_observation"][slot, 60]),
                        terminal_held=float(extras["terminal_critic_observation"][slot, 85]),
                        reset_held=float(extras["observations"]["critic"][slot, 85])))
            active &= ~done.bool()
            if (step + 1) % 500 == 0 or not active.any():
                clock = base.state.clock(base.physics_steps * base.physics_dt)
                write_progress(run, dict(seed=seed, control_step=step + 1,
                    simulation_time_s=(step + 1) * .02,
                    task_clock_s=clock.cpu().tolist(),
                    post_capture_s=base.state.post_capture_s.cpu().tolist(),
                    first_break_time_s=base.state.first_break_time.cpu().tolist(),
                    active=active.cpu().tolist(),
                    status="seed_completed" if not active.any() else "running"))
            if not active.any():
                break
            raw = next_raw
            obs = builder(raw, done.bool())
    finally:
        base.physics_observer = None
    if len(ended) != 2:
        raise RuntimeError("Autonomous evaluation ended without two terminal reports")
    data = trace.arrays()
    data.update({"policy_" + key: np.stack([row[key] for row in controls]) for key in controls[0]})
    if len(data["t"]) != len(controls) * 20:
        raise RuntimeError("1 kHz physical trace and 50 Hz control frames are not aligned")
    trace_name = f"trace_autonomous_seed{seed}.npz"
    np.savez_compressed(run / trace_name, **data)
    cases = [dict(ended[slot], trace_env=slot, trace=trace_name, seed=seed,
                  blend_s=None, teacher=False, program="autonomous", windows=None,
                  deterministic_mean_actor=True) for slot in (0, 1)]
    write(run / f"terminal_autonomous_seed{seed}.json", terminals)
    write(run / "reward_names.json", list(base.reward_terms))
    print("[PPO-EVAL]", seed,
          [(c["duration_s"], c["best_hold"], c["post_capture_s"], c["reason"], c["success"])
           for c in cases], flush=True)
    return cases


def worker(run: Path):
    app = env = None
    run = Path(run).resolve()
    try:
        actor, metadata = require(run)
        request = read(run / "request.json")
        if request["teacher_off"] is not True or request["mean_action_only"] is not True:
            raise ValueError("PPO evaluation must be teacher-off deterministic mean")
        from worker_sustain import setup
        app, env = setup(run, request)
        runtime = read(run / "sustain_runtime.json")
        runtime.update(raw_env_observation_version="v7", raw_env_policy_dim=305,
            actor_input_dim=325, actor_uses_unaugmented_v7_observation=False,
            actor_uses_v7_observation=False, old_actor_uses_separate_legacy_adapter=False,
            causal_capture_memory=True, proprioceptive_feedback=True,
            teacher_constructed=False, deterministic_ppo_mean_actor=True,
            checkpoint_sha256=metadata["checkpoint_sha256"])
        write(run / "sustain_runtime.json", runtime)
        actor = actor.to(env.device).eval()
        cases = []
        for seed in request["seeds"]:
            cases.extend(rollout(env, app, run, actor, seed))
            write(run / "evaluation.json", dict(cases=cases,
                checkpoint_sha256=metadata["checkpoint_sha256"], mean_action_only=True))
        result = dict(kind="ppo_checkpoint_mean_autonomous_evaluation",
            cases=cases, autonomous=True, teacher_constructed=False,
            mean_action_only=True, observation_passes=sum(bool(case["success"]) for case in cases),
            checkpoint_sha256=metadata["checkpoint_sha256"], checkpoint_updates=metadata["checkpoint_updates"],
            source_bc_actor_sha256=metadata["source_bc_actor_sha256"],
            new_ppo_updates=0, ppo_training_run=metadata["training_run"],
            trace_hashes={path.name: sha(path) for path in run.glob("trace_autonomous_seed*.npz")},
            robot_behavior_accepted=False,
            note="Full PPO checkpoint mean Actor; independent physical verification is separate")
        write(run / "result.json", result)
    except BaseException as exc:
        write(run / "worker_error.json", dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()
