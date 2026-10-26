"""Independent physical and causal-action audit of PPO mean-Actor evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, lfilter

from common_eval import BodyEstimator, URDF, read, write, sha, require


HF_LIMITS = np.array([.02, .05, math.radians(.2), .5])
HF_NAMES = ("com_velocity", "angular_velocity", "leg_position", "wheel_velocity")


def _hf_rms(raw: np.ndarray):
    source = np.concatenate((raw["com_vel"], raw["ang_vel"],
                             raw["leg_position"], raw["wheel_velocity"]), -1).astype(np.float64)
    b, a = butter(2, 5., fs=1000., btype="highpass")
    high = lfilter(b, a, source - source[:1], axis=0)
    alpha = 1. - np.exp(-.001 / .1)
    energy = lfilter([alpha], [1., -(1. - alpha)], high * high, axis=0)
    return np.stack((np.sqrt(energy[:, :3].sum(-1)),
                     np.sqrt(energy[:, 3:6].sum(-1)),
                     np.sqrt(energy[:, 6:18].max(-1)),
                     np.sqrt(energy[:, 18:].max(-1))), -1)


def _post_capture_physics(raw: dict, capture_s: float):
    if capture_s < 0:
        return None
    after = raw["t"] > capture_s
    if not after.any():
        return None
    hf = _hf_rms(raw)
    if not np.allclose(hf, raw["smooth_rms"], atol=5e-5, rtol=2e-3):
        raise AssertionError("Independent 1 kHz HF RMS differs from environment")
    force, clearance = raw["force_z"][after], raw["clearance"][after]
    return dict(max_com_speed_m_s=float(np.linalg.norm(raw["com_vel"][after], axis=-1).max()),
        max_angular_speed_rad_s=float(np.linalg.norm(raw["ang_vel"][after], axis=-1).max()),
        max_support_line_error_m=float(raw["line_error"][after].max()),
        max_body_force_n=float(raw["body_force"][after].max()),
        min_rear_force_n=float(force[:, 2:].min()),
        max_front_abs_force_n=float(np.abs(force[:, :2]).max()),
        min_front_clearance_m=float(clearance[:, :2].min()),
        max_hf_ratio_each=dict(zip(HF_NAMES, (hf[after] / HF_LIMITS).max(axis=0).tolist())),
        hf_composite_gate_pass_after_capture=bool((hf[after] <= HF_LIMITS).all()))


def _feature_and_action_checks(data: dict, actor, estimator: BodyEstimator):
    from verify_memory import reconstruct_memory
    from two_leg_balance_smooth_v6.core import targets

    active = data["policy_active"].astype(bool)
    raw = data["policy_raw_observation"]
    obs = data["policy_observation"]
    if obs.shape[-1] != 325 or raw.shape[-1] != 305 or obs.shape[:-1] != raw.shape[:-1]:
        raise AssertionError("Expected raw305 plus causal20 for PPO Actor")
    np.testing.assert_array_equal(active, data["active"][::20])
    np.testing.assert_array_equal(obs[..., :305], raw)
    if not np.isin(raw[..., 60::61][active], [1.]).all():
        raise AssertionError("Standing request absent from active raw v7 frames")
    if len(raw) > 1:
        np.testing.assert_allclose(raw[1:, ..., -58:-55][active[1:]],
            data["gravity"][19::20][:-1][active[1:]], atol=1e-6, rtol=0)
    memory = reconstruct_memory(raw)
    np.testing.assert_array_equal(memory[..., 0][active], obs[..., 305][active])
    np.testing.assert_allclose(memory[active], obs[..., 305:307][active], atol=3e-7, rtol=0)

    flat = raw.reshape(-1, 305)
    estimated_parts = []
    for start in range(0, len(flat), 2048):
        values = estimator.estimate(flat[start:start + 2048])
        estimated_parts.append(np.stack((values["gap_m"], values["velocity_m_s"]), -1))
    estimate = np.concatenate(estimated_parts, 0).reshape(*raw.shape[:-1], 2)
    np.testing.assert_allclose(estimate[active], obs[..., 323:325][active], atol=2e-5, rtol=0)

    start_action = np.zeros((*raw.shape[:2], 16), np.float32)
    for slot in range(raw.shape[1]):
        latched_action = np.zeros(16, np.float32)
        was_latched = False
        for frame in range(len(raw)):
            latched = bool(memory[frame, slot, 0])
            if not latched:
                latched_action = np.zeros(16, np.float32)
            elif not was_latched:
                latched_action = raw[frame, slot, -27:-11].copy()
            start_action[frame, slot] = latched_action
            was_latched = latched
    np.testing.assert_allclose(start_action[active], obs[..., 307:323][active], atol=2e-6, rtol=0)

    # Independently reconstruct the affine capture head rather than calling
    # model_feedback.feedback_basis; compute the frozen prefix/model in chunks.
    f = raw[..., -61:]
    g = f[..., 3:6].astype(np.float64)
    omega = f[..., :3].astype(np.float64) / .2
    derivative = -np.cross(omega, g)
    pitch = np.arctan2(g[..., 0], -g[..., 2])
    rate = (-g[..., 2] * derivative[..., 0] + g[..., 0] * derivative[..., 2]) / np.maximum(
        g[..., 0] ** 2 + g[..., 2] ** 2, 1e-6)
    basis = np.stack((np.ones_like(pitch), estimate[..., 0], estimate[..., 1],
        pitch, rate, estimate[..., 0].clip(-.12, .12)), -1)
    capture_action = basis @ (actor.coefficients.detach().numpy() * actor.mask.numpy()).T
    actor_values = np.empty((*raw.shape[:2], 16), np.float32)
    prefix_values = np.empty_like(actor_values)
    with torch.inference_mode():
        flat_obs = obs.reshape(-1, 325)
        flat_prefix = np.concatenate((raw, memory), -1).reshape(-1, 307)
        for start in range(0, len(flat_obs), 2048):
            stop = min(start + 2048, len(flat_obs))
            actor_values.reshape(-1, 16)[start:stop] = actor(torch.from_numpy(
                flat_obs[start:stop])).numpy()
            prefix_values.reshape(-1, 16)[start:stop] = actor.prefix(torch.from_numpy(
                flat_prefix[start:stop])).numpy()
    blend = memory[..., 1:2]
    independent_action = np.where(memory[..., 0:1] > .5,
        start_action * (1. - blend) + capture_action * blend, prefix_values).clip(-1., 1.)
    error = float(np.abs(independent_action[active] - data["policy_actor_action"][active]).max())
    if error >= 5e-5:
        raise AssertionError(f"Independent PPO action formula error {error}")
    np.testing.assert_allclose(actor_values[active], data["policy_actor_action"][active], atol=2e-5, rtol=0)
    if data["policy_takeover"].any() or data["policy_teacher_action"].any():
        raise AssertionError("Teacher/takeover appeared in PPO autonomous evaluation")
    np.testing.assert_array_equal(data["policy_actor_action"][active], data["policy_action"][active])
    executed = data["policy_action"].clip(-1., 1.).copy()
    executed[data["policy_physics_time"] < .2 - 1e-6] = 0.
    np.testing.assert_array_equal(executed[active], data["actions"][::20][active])
    action_tensor = torch.from_numpy(data["actions"][::20])
    joint_target = targets(action_tensor.reshape(-1, 16)).reshape(*action_tensor.shape[:-1], 12).numpy()
    np.testing.assert_allclose(joint_target[active], data["leg_targets"][::20][active], atol=1e-6, rtol=0)
    return dict(independent_actor_max_error=error,
        independent_feature_max_error=float(np.abs(estimate[active] - obs[..., 323:325][active]).max()),
        teacher_frames=0, executed_action_error=0, active_policy_frames=int(active.sum()))


def verify(run: Path):
    from verify_sustain import analyse

    run = Path(run).resolve()
    actor, metadata = require(run)
    request, result = read(run / "request.json"), read(run / "result.json")
    if (result["kind"] != "ppo_checkpoint_mean_autonomous_evaluation"
            or result["checkpoint_sha256"] != metadata["checkpoint_sha256"]
            or result["checkpoint_updates"] != metadata["checkpoint_updates"]
            or not result["autonomous"] or result["teacher_constructed"]
            or not result["mean_action_only"]):
        raise AssertionError("Result does not describe this PPO mean checkpoint")
    runtime = read(run / "sustain_runtime.json")
    if (runtime["raw_env_observation_version"] != "v7" or runtime["raw_env_policy_dim"] != 305
            or runtime["actor_input_dim"] != 325 or runtime["teacher_constructed"]
            or runtime["old_actor_uses_separate_legacy_adapter"]
            or runtime["actor_uses_unaugmented_v7_observation"]
            or not runtime["stop_on_observation"] or not runtime["no_automatic_return"]
            or abs(runtime["observation_s"] - request["hold_seconds"]) > 1e-8
            or abs(runtime["time_limit_s"] - request["hold_seconds"] - 15.) > 1e-8):
        raise AssertionError("Runtime physical/no-return/observation contract differs")
    geometry = read(run / "geometry.json")
    estimator = BodyEstimator(URDF)
    cases, action_checks = [], []
    for path in sorted(run.glob("trace_autonomous_seed*.npz")):
        if sha(path) != result["trace_hashes"].get(path.name):
            raise AssertionError("Trace hash differs: " + path.name)
        with np.load(path) as archive:
            data = {key: archive[key] for key in archive.files}
        if not all(np.isfinite(value).all() for value in data.values()):
            raise AssertionError("Trace has nonfinite physical or control data")
        group = [case for case in result["cases"] if case["trace"] == path.name]
        if (len(group) != 2 or sorted(case["trace_env"] for case in group) != [0, 1]
                or len(data["t"]) != len(data["policy_observation"]) * 20):
            raise AssertionError("Two-slot 1 kHz trace/control count mismatch")
        terminal = read(run / f"terminal_autonomous_seed{group[0]['seed']}.json")
        if len(terminal) != 2 or sorted(item["slot"] for item in terminal) != [0, 1]:
            raise AssertionError("Two-slot terminal report missing")
        qualified = np.zeros(data["active"].shape, dtype=bool)
        for case in group:
            slot = case["trace_env"]
            ended = next(item for item in terminal if item["slot"] == slot)
            if (ended["timeout"] != bool(case["timeout"])
                    or ended["failed"] != bool(case["failed"])
                    or ended["terminal_stand_request"] != 1.):
                raise AssertionError("Terminal pre-reset stand/timeout/failure differs")
            active = data["active"][:, slot].astype(bool)
            raw = {key: value[active, slot] for key, value in data.items()
                   if not key.startswith("policy_")}
            row, stable = analyse(raw, case, geometry, request["hold_seconds"])
            if row["returned"] or case["teacher"] or case["seed"] not in request["seeds"]:
                raise AssertionError("Return/teacher/seed mismatch in autonomous evaluation")
            row.update(seed=case["seed"], program="autonomous",
                       checkpoint_sha256=metadata["checkpoint_sha256"],
                       post_capture_physics=_post_capture_physics(raw, row["capture_time_s"]))
            cases.append(row)
            qualified[active, slot] = stable
        checks = _feature_and_action_checks(data, actor, estimator)
        checks["trace"] = path.name
        action_checks.append(checks)
        active = data["policy_active"].astype(bool)
        reward_names = read(run / "reward_names.json")
        terms = data["policy_reward_terms"]
        np.testing.assert_allclose(terms.sum(-1)[active], data["policy_reward"][active], atol=2e-6, rtol=1e-5)
        balance = qualified.reshape(-1, 20, 2).sum(1) * .001 * 12.
        np.testing.assert_allclose(terms[..., reward_names.index("balance")][active],
            balance[active], atol=.02401, rtol=0)
        for name in ("hold_once", "success_once", "missed_hold", "drift",
                     "return_pose", "return_attitude", "recovery"):
            if (terms[..., reward_names.index(name)][active] != 0.).any():
                raise AssertionError("No-return reward unexpectedly active: " + name)
    if (len(cases) != 2 * len(request["seeds"])
            or set(result["trace_hashes"]) != {f"trace_autonomous_seed{seed}.npz" for seed in request["seeds"]}
            or sum(bool(case["observation_pass"]) for case in cases) != result["observation_passes"]):
        raise AssertionError("PPO evaluation case/trace/pass count mismatch")
    passes = sum(bool(case["observation_pass"]) for case in cases)
    report = dict(kind="independent_ppo_mean_actor_evaluation_review",
        checkpoint_sha256=metadata["checkpoint_sha256"],
        checkpoint_updates=metadata["checkpoint_updates"],
        training_result_sha256=metadata["training_result_sha256"],
        hold_seconds=request["hold_seconds"], seeds=request["seeds"],
        independent_physics_verified=True, independent_proprioception_verified=True,
        deterministic_mean_actor_verified=True, teacher_constructed=False,
        no_automatic_return_verified=True, cases=cases, passes=passes,
        case_count=len(cases), all_case_hold_gate=passes == len(cases),
        action_checks=action_checks, physics_samples=sum(case["physics_samples"] for case in cases),
        new_ppo_updates=0, robot_behavior_accepted=False,
        limitation="Fixed nominal starts and seeds; no perturbation or hardware robustness claim")
    write(run / "verification.json", report)
    return report
