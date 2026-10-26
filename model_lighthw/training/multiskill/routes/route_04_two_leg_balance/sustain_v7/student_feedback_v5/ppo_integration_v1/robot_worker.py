"""Actual Isaac Lab R4a PPO worker; invoked only after CPU admission gates."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
STRUCTURED = V5 / "structured_v1"
SUSTAIN = V5.parent
ROUTE = SUSTAIN.parent
for directory in (STRUCTURED, V5 / "ppo_design", SUSTAIN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from common_feedback import load_actor, now, read, sha, write  # noqa: E402
from continuing_driver import ContinuingDriver, DriverConfig  # noqa: E402
from features_feedback import FeedbackFeatures  # noqa: E402
from gate import formal_review_gate, six_case_gate  # noqa: E402
from provenance import check  # noqa: E402


def _dump_line(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _rng_digest(value):
    import hashlib
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def _nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _nested_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, type(expected)) and len(actual) == len(expected) and all(
            _nested_equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


def verify_restored(driver, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    driver.load_checkpoint(checkpoint)
    if not _nested_equal(driver.policy.state_dict(), saved["model_state_dict"]):
        raise RuntimeError("Restored model, including log_std, differs")
    if not _nested_equal(driver.alg.optimizer.state_dict(), saved["optimizer_state_dict"]):
        raise RuntimeError("Restored Adam differs")
    if driver.alg.learning_rate != saved["learning_rate"]:
        raise RuntimeError("Restored learning rate differs")
    if any(getattr(driver, key) != value for key, value in saved["counters"].items()):
        raise RuntimeError("Restored PPO counters differ")
    rng = saved["rng"]
    if _rng_digest(__import__("random").getstate()) != _rng_digest(rng["python"]):
        raise RuntimeError("Python RNG differs")
    if _rng_digest(np.random.get_state()) != _rng_digest(rng["numpy"]):
        raise RuntimeError("NumPy RNG differs")
    if not torch.equal(torch.get_rng_state(), rng["torch_cpu"]):
        raise RuntimeError("CPU torch RNG differs")
    if rng["torch_cuda"] and not _nested_equal(torch.cuda.get_rng_state_all(), rng["torch_cuda"]):
        raise RuntimeError("CUDA RNG differs")
    return saved["counters"]


def migrate_formal_checkpoint(source, destination, request, target_config):
    """Explicitly migrate only PPO rollout length 512→32; retain every learned tensor."""
    migration = request["checkpoint_migration"]
    if request["mode"] != "formal-smoke" or migration is None:
        raise RuntimeError("Checkpoint migration is only valid in formal-config smoke")
    if sha(source) != migration["source_checkpoint_sha256"]:
        raise RuntimeError("Migration source SHA differs")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    old_config = dict(saved["config"])
    expected = dict(target_config.__dict__)
    if (old_config.get("num_steps_per_env") != 512 or expected.get("num_steps_per_env") != 32
            or {k: v for k, v in old_config.items() if k != "num_steps_per_env"}
            != {k: v for k, v in expected.items() if k != "num_steps_per_env"}
            or saved["num_envs"] != 16):
        raise RuntimeError("Migration attempted an unapproved PPO configuration change")
    if (migration["source_technical_limit_s"] != 20.
            or migration["destination_technical_limit_s"] != 360.
            or migration["source_steps_per_env"] != 512
            or migration["destination_steps_per_env"] != 32):
        raise RuntimeError("Migration request differs from reviewed 20 s→360 s, 512→32 plan")
    saved["config"] = expected
    saved["explicit_migration"] = dict(source_checkpoint=str(source),
        source_checkpoint_sha256=sha(source), source_config=old_config,
        destination_config=expected, unchanged_model_optimizer_rng_counters=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(saved, destination)
    roundtrip = torch.load(destination, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "rng", "counters"):
        if not _nested_equal(roundtrip[key], saved[key]):
            raise RuntimeError("Migration altered " + key)
    if roundtrip["learning_rate"] != saved["learning_rate"]:
        raise RuntimeError("Migration altered learning rate")
    return destination


def critic_adam_steps(driver):
    parameter = next(driver.policy.critic.parameters())
    state = driver.alg.optimizer.state.get(parameter, {})
    step = state.get("step", 0)
    return int(step.item() if isinstance(step, torch.Tensor) else step)


class AuditedEnv:
    """Observe real wrapper transitions without changing the reset/step contract."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.records = []
        self.reset_calls = 0
        self.initial_reads = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def reset(self):
        self.reset_calls += 1
        raise RuntimeError("PPO driver may not reset at an iteration boundary")

    def get_observations(self):
        self.initial_reads += 1
        return self.wrapped.get_observations()

    def step(self, action):
        raw, reward, done, info = self.wrapped.step(action)
        timeout = info["time_outs"].reshape(-1).bool()
        failed = info["physical_terminated"].reshape(-1).bool()
        valid = info["terminal_observation_valid"].reshape(-1).bool()
        if timeout.any() and not valid[timeout].all():
            raise RuntimeError("Timeout has no valid pre-reset Critic")
        self.records.append(dict(reward=reward.detach().reshape(-1).clone(),
            timeout=timeout.detach().clone(), failed=failed.detach().clone(),
            terminal=info["terminal_critic_observation"].detach().clone() if timeout.any() else None,
            reset=info["observations"]["critic"].detach().clone() if timeout.any() else None))
        return raw, reward, done, info

    @torch.no_grad()
    def inspect_bootstrap(self, driver, collection):
        if len(self.records) != driver.config.num_steps_per_env:
            raise RuntimeError("Missing audited real transitions")
        stored = driver.alg.storage.rewards[..., 0]
        base = torch.stack([row["reward"] for row in self.records])
        expected = base.clone()
        timeouts = failures = 0
        pre_reset_difference = 0.
        for k, row in enumerate(self.records):
            timeout, failed = row["timeout"], row["failed"]
            if (timeout & failed).any():
                raise RuntimeError("Failed terminal was also bootstrapped")
            n = int(timeout.sum())
            timeouts += n
            failures += int(failed.sum())
            if n:
                value = driver.policy.evaluate(row["terminal"][timeout]).reshape(-1)
                expected[k, timeout] += driver.config.gamma * value
                pre_reset_difference = max(pre_reset_difference,
                    float((row["terminal"][timeout] - row["reset"][timeout]).abs().max()))
        torch.testing.assert_close(stored, expected, atol=2e-5, rtol=2e-5)
        if timeouts != collection["timeouts"] or failures != collection["physical_terminations"]:
            raise RuntimeError("Stored terminal counters differ from real env")
        # If RSL applied its own timeout bonus too, storage rewards fail the equality above.
        self.records.clear()
        return dict(pre_reset_bootstrap_once_verified=True, timeout_count=timeouts,
                    physical_termination_count=failures,
                    terminal_vs_reset_critic_max_abs=pre_reset_difference,
                    audit_reset_calls=self.reset_calls, initial_observation_reads=self.initial_reads)


class PhysicsEvidence:
    """Accumulate actual per-1-ms behavior; no extra environment steps/evaluation."""

    def __init__(self, base, run, trace_ticks=0):
        self.base, self.run = base, Path(run)
        n, device = base.num_envs, base.device
        self.samples = torch.zeros(n, dtype=torch.int64, device=device)
        self.stable = torch.zeros_like(self.samples)
        self.capture = torch.zeros_like(self.samples)
        self.interrupted = torch.zeros_like(self.samples)
        self.max_body_force = torch.zeros(n, device=device)
        self.max_com_speed = torch.zeros(n, device=device)
        self.max_jitter_rms = torch.zeros(n, device=device)
        self.ended = 0
        self.ended_success = 0
        self.trace_index = 0
        self.trace = None
        if trace_ticks:
            # GPU buffering avoids synchronizing the simulator every 1 ms.
            fields = dict(physics_time_s=1, com_w=3, wheel_pos_w=12,
                origin_w_xy=2, hold_origin_w_xy=2, base_z_m=1,
                com_vel_m_s=3, line_error_m=1, gravity=3,
                target_gravity=3, ang_vel_rad_s=3, force_z_n=4, clearance_m=4,
                leg_position_rad=12, wheel_velocity_rad_s=4,
                body_force_n=1, torque_peak_nm=1, jitter_rms=4, executed_action=16,
                phase=1, mode=1, start_time_s=1, stand_request=1,
                hold_time_s=1, best_hold_s=1, post_capture_s=1,
                capture_time_s=1, first_break_time_s=1,
                held=1, interrupted=1, success=1, failed=1, boundary=1)
            self.trace = {name: torch.empty(trace_ticks, n, width, device=device)
                          for name, width in fields.items()}

    @torch.no_grad()
    def observe(self, base, tick):
        if base is not self.base or abs(base.physics_dt - .001) > 1e-12:
            raise RuntimeError("1 kHz physics observer contract changed")
        self.samples += 1
        state, metrics = base.state, base.current_metrics
        self.stable += (state.hold_time > 0).long()
        self.capture += state.held.long()
        self.interrupted += state.interrupted.long()
        self.max_body_force = torch.maximum(self.max_body_force, metrics["body_force"])
        self.max_com_speed = torch.maximum(self.max_com_speed, metrics["com_vel"].norm(dim=-1))
        self.max_jitter_rms = torch.maximum(self.max_jitter_rms,
            state.jitter.rms.amax(dim=-1))
        if self.trace is not None:
            if self.trace_index >= len(self.trace["physics_time_s"]):
                raise RuntimeError("Physics trace capacity exceeded")
            k = self.trace_index
            phase, target_gravity = base.command(base.physics_steps * base.physics_dt)
            row = dict(physics_time_s=(base.physics_steps * base.physics_dt)[:, None],
                com_w=metrics["com"], wheel_pos_w=metrics["wheel_pos"].flatten(1),
                origin_w_xy=state.origin, hold_origin_w_xy=state.hold_origin,
                base_z_m=metrics["base_z"][:, None],
                com_vel_m_s=metrics["com_vel"], line_error_m=metrics["line_error"][:, None],
                gravity=metrics["gravity"], target_gravity=target_gravity,
                ang_vel_rad_s=metrics["ang_vel"],
                force_z_n=metrics["force_z"], clearance_m=metrics["clearance"],
                leg_position_rad=metrics["leg_position"],
                wheel_velocity_rad_s=metrics["wheel_velocity"],
                body_force_n=metrics["body_force"][:, None],
                torque_peak_nm=metrics["torque_peak"][:, None],
                jitter_rms=state.jitter.rms,
                executed_action=base.actions,
                phase=phase[:, None], mode=base.mode[:, None],
                start_time_s=state.start_time[:, None],
                stand_request=state.stand_request[:, None],
                hold_time_s=state.hold_time[:, None], best_hold_s=state.best_hold[:, None],
                post_capture_s=state.post_capture_s[:, None], capture_time_s=state.capture_time[:, None],
                first_break_time_s=state.first_break_time[:, None],
                held=state.held[:, None], interrupted=state.interrupted[:, None],
                success=state.success[:, None], failed=state.failed[:, None],
                boundary=state.boundary[:, None])
            for name, value in row.items():
                if value.shape != self.trace[name][k].shape:
                    raise RuntimeError(f"Physics trace {name} shape {value.shape} changed")
                self.trace[name][k].copy_(value)
            self.trace_index += 1

    def episodes(self, reports):
        for report in reports:
            _dump_line(self.run / "ended_episodes.jsonl", report)
            self.ended += 1
            self.ended_success += int(report["success"])

    @torch.no_grad()
    def summary(self):
        base = self.base
        return dict(physics_dt_s=float(base.physics_dt), policy_dt_s=float(base.step_dt),
            physics_samples_per_slot=self.samples.cpu().tolist(),
            stable_samples_per_slot=self.stable.cpu().tolist(),
            capture_samples_per_slot=self.capture.cpu().tolist(),
            interrupted_samples_per_slot=self.interrupted.cpu().tolist(),
            max_body_force_n=self.max_body_force.cpu().tolist(),
            max_com_speed_m_s=self.max_com_speed.cpu().tolist(),
            max_jitter_rms=self.max_jitter_rms.cpu().tolist(),
            current_best_hold_s=base.state.best_hold.cpu().tolist(),
            current_post_capture_s=base.state.post_capture_s.cpu().tolist(),
            ended_episodes=self.ended, ended_success=self.ended_success,
            observation_complete_current=base.state.observation_complete.cpu().tolist(),
            standing_request_current=base.state.stand_request.cpu().tolist())

    def save_trace(self):
        if self.trace is None:
            return None
        if self.trace_index != len(self.trace["physics_time_s"]):
            raise RuntimeError("Incomplete 1 kHz physics trace")
        path = self.run / "physics_trace.npz"
        np.savez_compressed(path, **{name: value.cpu().numpy() for name, value in self.trace.items()})
        return path


def make_env(run, request):
    """Use original SustainBalanceEnv, fixed R4a physics/reward and a technical timeout."""
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from two_leg_balance_sustain_v7.env import SustainBalanceCfg, SustainBalanceEnv
    cfg = SustainBalanceCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = request["num_envs"], 42, "cuda:0"
    cfg.episode_length_s = request["technical_time_limit_s"]
    cfg.observation_hold_s = 10.
    cfg.stop_on_observation = False
    cfg.is_finite_horizon = False
    cfg.observation_noise = False
    cfg.reset_jitter = 0.
    cfg.target_pitches = tuple(c["pitch_rad"] for c in read(run / "geometry.json")["candidates"])
    cfg.scene.robot.spawn.asset_path = str(run / "assets/urdf/lighthw.urdf")
    cfg.scene.robot.spawn.usd_dir = str(run / "asset_cache")
    cfg.scene.robot.spawn.usd_file_name = "ppo_sustain.usd"

    class Wrapper(RslRlVecEnvWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_privileged_obs = gym.spaces.flatdim(
                self.unwrapped.single_observation_space.spaces["critic"])

    wrapped = Wrapper(SustainBalanceEnv(cfg=cfg), clip_actions=1.)
    spec = importlib.util.spec_from_file_location("r4_original_runtime", ROUTE / "split_v4/scripts/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime(wrapped, run)
    base = wrapped.unwrapped
    if [wrapped.num_obs, wrapped.num_privileged_obs, wrapped.num_actions] != [305, 88, 16]:
        raise RuntimeError("Original R4a 305/88/16 contract changed")
    if cfg.stop_on_observation or cfg.is_finite_horizon or cfg.observation_noise or cfg.reset_jitter:
        raise RuntimeError("Continuing fixed-condition configuration changed")
    expected_length = int(round(request["technical_time_limit_s"] / base.step_dt))
    if (abs(base.step_dt - .02) > 1e-12 or abs(base.physics_dt - .001) > 1e-12
            or base.max_episode_length != expected_length):
        raise RuntimeError("50 Hz/1 kHz/technical timeout contract changed")
    return app, wrapped, cfg


def run_robot(run):
    worker_started_at = now()
    run = Path(run).resolve()
    check(run)
    request = read(run / "request.json")
    admitted = six_case_gate(Path(request["six_case_verification"]), Path(request["actor"]))
    if admitted["actor_sha256"] != request["actor_sha256"]:
        raise RuntimeError("Actor changed after PPO request")
    if admitted["verification_sha256"] != request["six_case_verification_sha256"]:
        raise RuntimeError("Six-case verification changed after PPO request")
    mode = request["mode"]
    expected = {
        "smoke": (512, 20., 2),
        "resume-smoke": (512, 20., 1),
        "formal-smoke": (32, 360., 32),
        "train": (32, 360., None),
    }
    if (mode not in expected
            or (request["steps_per_env"], request["technical_time_limit_s"]) != expected[mode][:2]
            or (expected[mode][2] is not None and request["new_updates"] != expected[mode][2])):
        raise RuntimeError("PPO mode/rollout/technical timeout contract changed")
    if (request["num_envs"] not in (2, 16, 64)
            or mode in ("formal-smoke", "train") and request["num_envs"] != 16
            or request["new_updates"] <= 0
            or request["initial_std"] != .0005 or request["learning_rate"] != 3e-6
            or request["stop_on_observation"] is not False
            or request["is_finite_horizon"] is not False
            or request["no_observation_noise"] is not True
            or request["no_reset_randomization"] is not True
            or request["automatic_frozen_evaluation"] is not False):
        raise RuntimeError("PPO fixed-condition config changed")
    if mode in ("resume-smoke", "formal-smoke"):
        source = Path(request["source_smoke_run"])
        check(source)
        prior = read(source / "result.json")
        if (sha(source / "result.json") != request["source_smoke_result_sha256"]
                or read(source / "session.json")["status"] != "completed"
                or prior["mode"] != "smoke" or prior["actor_sha256"] != request["actor_sha256"]
                or prior["num_envs"] != request["num_envs"]
                or prior["final_checkpoint_sha256"] != request["resume_checkpoint_sha256"]):
            raise RuntimeError("Reviewed smoke source changed")
        if mode == "formal-smoke" and prior["technical_pass"] is not True:
            raise RuntimeError("Formal-config smoke requires passing timeout smoke")
    if mode == "train":
        reviewed = formal_review_gate(Path(request["review"]["review"]),
            request["actor_sha256"], check)
        if reviewed != request["review"]:
            raise RuntimeError("Formal independent review changed")
        prior = read(Path(reviewed["formal_smoke_run"]) / "result.json")
        if (request["num_envs"] != prior["num_envs"]
                or request["resume_checkpoint_sha256"] != prior["final_checkpoint_sha256"]):
            raise RuntimeError("Formal checkpoint differs from reviewed configuration")
    actor, _ = load_actor(request["actor"])
    if any(isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
           for module in actor.prefix.modules()):
        raise RuntimeError("Frozen prefix train/eval behavior requires separate validation")
    app = wrapped = None
    try:
        app, wrapped, cfg = make_env(run, request)
        base = wrapped.unwrapped
        initial_physics_steps = base.physics_steps.cpu().tolist()
        initial_episode_counts = base.episode_count.cpu().tolist()
        if any(initial_physics_steps) or any(initial_episode_counts):
            raise RuntimeError("PPO process did not begin in a fresh physical episode")
        trace_ticks = (request["new_updates"] * request["steps_per_env"] * cfg.decimation
                       if request["mode"] != "train" else 0)
        evidence = PhysicsEvidence(base, run, trace_ticks=trace_ticks)
        base.physics_observer = evidence.observe
        base.episode_sink = evidence.episodes
        env = AuditedEnv(wrapped)
        actor = actor.to(base.device)
        builder = FeedbackFeatures(actor.prefix, request["num_envs"], base.device)
        config = DriverConfig(actor_obs_dim=325, critic_obs_dim=88, num_actions=16,
            num_steps_per_env=request["steps_per_env"], initial_std=request["initial_std"],
            learning_rate=request["learning_rate"], num_learning_epochs=2, num_mini_batches=1)
        driver = ContinuingDriver(env, actor, builder, config,
            observation_contract="r4a_raw305_memory2_trigger16_fk2_v1",
            policy_contract=f"structured_feedback_bc_{request['actor_sha256']}", device=base.device)
        initial_updates = 0
        initial_control_steps = 0
        restored = False
        migration_verified = False
        if request["resume_checkpoint"]:
            checkpoint = Path(request["resume_checkpoint"])
            if sha(checkpoint) != request["resume_checkpoint_sha256"]:
                raise RuntimeError("Resume checkpoint SHA changed")
            if request["checkpoint_migration"]:
                checkpoint = migrate_formal_checkpoint(checkpoint,
                    run / "checkpoints" / "migrated_input.pt", request, config)
                migration_verified = True
            counters = verify_restored(driver, checkpoint)
            initial_updates = counters["completed_updates"]
            initial_control_steps = counters["control_steps"]
            restored = True
        initial_physics = evidence.samples.clone()
        initial_coefficients = driver.policy.actor.coefficients.detach().clone()
        initial_policy = {name: value.detach().clone() for name, value in driver.policy.state_dict().items()}
        initial_critic_adam_steps = critic_adam_steps(driver)
        last_checkpoint = None
        total_timeouts = total_failures = 0
        for local_update in range(1, request["new_updates"] + 1):
            if not app.is_running():
                raise RuntimeError("Simulator closed during PPO")
            collection = driver.collect_rollout()
            bootstrap = env.inspect_bootstrap(driver, collection)
            if bootstrap["audit_reset_calls"] != 0:
                raise RuntimeError("Driver called environment reset")
            stats = driver.update_from_rollout()
            total_timeouts += bootstrap["timeout_count"]
            total_failures += bootstrap["physical_termination_count"]
            expected_samples = local_update * request["steps_per_env"] * cfg.decimation
            actual_samples = evidence.samples - initial_physics
            if not torch.equal(actual_samples, torch.full_like(actual_samples, expected_samples)):
                raise RuntimeError("1 kHz samples differ from 50 Hz control steps")
            std = driver.policy.log_std.exp().detach()
            if not torch.isfinite(std).all() or float(std.min()) <= 0 or float(std.max()) > .01:
                raise RuntimeError("Exploration std escaped the explicit small range")
            if driver.completed_updates != initial_updates + local_update or driver.optimizer_steps <= 0:
                raise RuntimeError("Actual PPO/Adam update count did not advance")
            actual_critic_adam_steps = critic_adam_steps(driver)
            if actual_critic_adam_steps != initial_critic_adam_steps + local_update * 2:
                raise RuntimeError("Critic Adam step did not advance twice per PPO update")
            # The causal builder has one fresh-process initial call; restored
            # global control_steps intentionally include the previous process.
            if (driver.control_steps != initial_control_steps + local_update * request["steps_per_env"]
                    or driver.feature_build_calls != 1 + local_update * request["steps_per_env"]
                    or driver.initial_observation_reads != 1 or env.initial_reads != 1):
                raise RuntimeError("Feature memory advanced at the wrong control rate")
            row = dict(local_update=local_update, global_update=driver.completed_updates,
                expected_optimizer_steps=driver.optimizer_steps,
                actual_critic_adam_steps=actual_critic_adam_steps, collection=collection,
                process_control_steps=local_update * request["steps_per_env"],
                feature_build_calls=driver.feature_build_calls,
                bootstrap=bootstrap, losses=stats["losses"], std_min=float(std.min()),
                std_max=float(std.max()), learning_rate=driver.alg.learning_rate,
                actor_coefficients_changed=not torch.equal(initial_coefficients,
                    driver.policy.actor.coefficients.detach()),
                physics=evidence.summary(), no_reset_at_iteration=True,
                no_automatic_evaluation=True)
            _dump_line(run / "updates.jsonl", row)
            if (request["mode"] != "train" or local_update % 100 == 0
                    or local_update == request["new_updates"]):
                checkpoint = run / "checkpoints" / f"model_{driver.completed_updates:06d}.pt"
                driver.save_checkpoint(checkpoint)
                last_checkpoint = checkpoint
        if not driver.alg.optimizer.state:
            raise RuntimeError("Adam has no updated state")
        if last_checkpoint is None:
            raise RuntimeError("No final checkpoint")
        actor_changed = not torch.equal(initial_coefficients, driver.policy.actor.coefficients.detach())
        policy_changed = any(not torch.equal(value, initial_policy[name])
                             for name, value in driver.policy.state_dict().items())
        trace = evidence.save_trace()
        behavior_candidate = bool((base.state.success & ~base.state.interrupted
            & ~base.state.failed & ~base.state.boundary & base.state.stand_request
            & (base.state.post_capture_s >= 10. - 1e-6)).all())
        technical_pass = (driver.completed_updates == initial_updates + request["new_updates"]
            and driver.driver_reset_calls == 0 and env.reset_calls == 0
            and policy_changed and (request["mode"] != "smoke" or (total_timeouts > 0 and actor_changed))
            and (request["mode"] not in ("resume-smoke", "formal-smoke") or restored)
            and (request["mode"] != "formal-smoke" or migration_verified))
        result = dict(mode=request["mode"], num_envs=request["num_envs"],
            steps_per_env=request["steps_per_env"], technical_time_limit_s=request["technical_time_limit_s"],
            actor_sha256=request["actor_sha256"], new_updates=request["new_updates"],
            initial_updates=initial_updates, completed_updates=driver.completed_updates,
            expected_optimizer_steps=driver.optimizer_steps,
            actual_critic_adam_steps=critic_adam_steps(driver), control_steps=driver.control_steps,
            process_control_steps=request["new_updates"] * request["steps_per_env"],
            transition_samples=driver.transition_samples, timeout_count=total_timeouts,
            physical_termination_count=total_failures, bootstrap_events=driver.bootstrap_events,
            driver_reset_calls=driver.driver_reset_calls, audit_reset_calls=env.reset_calls,
            feature_build_calls=driver.feature_build_calls, actor_coefficients_changed=actor_changed,
            policy_changed=policy_changed, checkpoint_migration_verified=migration_verified,
            final_checkpoint=str(last_checkpoint), final_checkpoint_sha256=sha(last_checkpoint),
            physics_trace_sha256=sha(trace) if trace else None,
            physics_trace_samples=evidence.trace_index if trace else 0,
            restored_state_verified=restored, new_physics_episode=bool(request["resume_checkpoint"]),
            worker_pid=os.getpid(), worker_started_at=worker_started_at,
            initial_physics_steps=initial_physics_steps,
            initial_episode_counts=initial_episode_counts,
            standing_continuity_across_processes=False,
            technical_pass=bool(technical_pass), physics=evidence.summary(),
            behavior_candidate_pass=behavior_candidate, robot_behavior_accepted=False,
            automatic_frozen_evaluation=False,
            note="Technical PPO evidence only; review autonomous physical behavior separately")
        write(run / "result.json", result)
        if not technical_pass:
            raise RuntimeError("Robot PPO technical criteria failed; inspect result.json")
    finally:
        if wrapped is not None:
            wrapped.unwrapped.physics_observer = None
            wrapped.unwrapped.episode_sink = None
            wrapped.close()
        if app is not None:
            app.close()
