"""Independent 1 kHz R4a PPO trace review; never drives the simulator.

This deliberately does not import the robot environment or its state machine.
All acceptance predicates are reconstructed from recorded physical quantities.
The five-field formal-smoke gate supports a separate evidence review; this
script does not itself write the formal PPO review decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter


DT = .001
OBSERVATION_S = 10.
SPEED_LIMIT = .12
ANGULAR_LIMIT = .25
ATTITUDE_LIMIT = math.radians(12.)
HF_LIMITS = np.array([.02, .05, math.radians(.2), .5])
HF_NAMES = ("com_velocity", "angular_velocity", "leg_position", "wheel_velocity")
WHEEL_RADIUS_CONSERVATIVE = .087501

# Every item below is an independently observed physical quantity or clock.
# Precomputed line_error, clearance, jitter and state flags are compared only
# after reconstruction; they are never used as inputs to a passing gate.
RAW_WIDTHS = {
    "physics_time_s": 1, "com_w": 3, "com_vel_m_s": 3,
    "wheel_pos_w": 12, "base_z_m": 1, "gravity": 3,
    "ang_vel_rad_s": 3, "force_z_n": 4, "body_force_n": 1,
    "leg_position_rad": 12, "wheel_velocity_rad_s": 4,
    "stand_request": 1, "mode": 1,
}
TELEMETRY_WIDTHS = {
    "line_error_m": 1, "clearance_m": 4, "jitter_rms": 4,
    "start_time_s": 1, "phase": 1,
    "hold_time_s": 1, "best_hold_s": 1, "post_capture_s": 1,
    "capture_time_s": 1, "first_break_time_s": 1,
    "held": 1, "interrupted": 1, "success": 1,
    "failed": 1, "boundary": 1, "target_gravity": 3,
}


class TraceContractError(ValueError):
    """The trace cannot support an independent, fail-closed review."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def trace_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {**RAW_WIDTHS, **TELEMETRY_WIDTHS}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise TraceContractError("Missing raw/telemetry fields: " + ", ".join(missing))
        data = {key: archive[key] for key in required}
    sizes = {value.shape[:2] for value in data.values()}
    if len(sizes) != 1:
        raise TraceContractError(f"Trace tick/slot shapes disagree: {sizes}")
    ticks, slots = next(iter(sizes))
    if ticks < 1 or slots < 1:
        raise TraceContractError("Empty physics trace")
    for key, width in required.items():
        if data[key].shape != (ticks, slots, width):
            raise TraceContractError(f"{key} has shape {data[key].shape}, expected {(ticks, slots, width)}")
        if not np.isfinite(data[key]).all():
            raise TraceContractError(f"{key} contains NaN/Inf")
    return data


def episode_slices(time: np.ndarray) -> list[slice]:
    """Split a single slot on actual 1 kHz episode-time rollback."""
    time = np.asarray(time, dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(time) < -.0005) + 1, len(time)]
    episodes = [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    for part in episodes:
        segment = time[part]
        if abs(float(segment[0]) - DT) > 2e-5:
            raise TraceContractError(f"Episode starts at {segment[0]} s, expected first 1 ms sample")
        if len(segment) > 1 and not np.all(np.abs(np.diff(segment) - DT) <= 4e-5):
            raise TraceContractError("1 kHz episode clock has a gap or duplicate sample")
    return episodes


def high_frequency_rms(raw: np.ndarray) -> np.ndarray:
    """SciPy reconstruction of the v6/v7 causal 5 Hz HP + 0.1 s RMS."""
    if raw.ndim != 2 or raw.shape[1] != 22:
        raise TraceContractError("HF input must be [tick,22]")
    b, a = butter(2, 5., fs=1000., btype="highpass")
    hp = lfilter(b, a, raw.astype(np.float64) - raw[0].astype(np.float64), axis=0)
    alpha = 1. - math.exp(-DT / .1)
    energy = lfilter([alpha], [1., -(1. - alpha)], hp * hp, axis=0)
    return np.stack((
        np.sqrt(energy[:, :3].sum(-1)),
        np.sqrt(energy[:, 3:6].sum(-1)),
        np.sqrt(energy[:, 6:18].max(-1)),
        np.sqrt(energy[:, 18:].max(-1)),
    ), axis=-1)


def _line_distance(com_xy: np.ndarray, rear_xy: np.ndarray) -> np.ndarray:
    a, b = rear_xy[:, 0], rear_xy[:, 1]
    line = b - a
    projection = ((com_xy - a) * line).sum(-1) / np.maximum((line * line).sum(-1), 1e-8)
    return np.linalg.norm(com_xy - (a + np.clip(projection, 0., 1.)[:, None] * line), axis=-1)


def _phase_and_target(t: float, start: float, pitch_rad: float):
    task = min(t, .999) if start < 0. else t - start + 1.
    phase = int(task >= 1.) + int(task >= 4.)
    x = min(max((task - 1.) / 3., 0.), 1.)
    amplitude = x * x * (3. - 2. * x)
    pitch = float(np.float32(pitch_rad)) * amplitude
    return phase, np.array((math.sin(pitch), 0., -math.cos(pitch)))


def _comparison(actual: np.ndarray, expected: np.ndarray, atol: float) -> dict:
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"max_abs": float(error.max()) if error.size else 0.,
            "mismatch_count": int(np.count_nonzero(error > atol)), "atol": atol}


def _checkpoint_equal(actual, expected) -> bool:
    """Exact CPU comparison for saved model, Adam, RNG and counters."""
    import torch
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _checkpoint_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, type(expected)) and len(actual) == len(expected) and all(
            _checkpoint_equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


def analyse_episode(data: dict[str, np.ndarray], slot: int, part: slice,
                    rear_pitch_rad: float, observation_s: float = OBSERVATION_S) -> dict:
    """Re-run the v7 entry/capture/first-break state machine on one physical episode."""
    if observation_s != 10.:
        raise TraceContractError("PPO review requires the original extra 10 s observation")
    d = {key: value[part, slot] for key, value in data.items()}
    t = d["physics_time_s"][:, 0].astype(np.float64)
    n = len(t)
    wheel = d["wheel_pos_w"].reshape(n, 4, 3).astype(np.float64)
    com = d["com_w"].astype(np.float64)
    force = d["force_z_n"].astype(np.float64)
    gravity = d["gravity"].astype(np.float64)
    com_vel = d["com_vel_m_s"].astype(np.float64)
    ang_vel = d["ang_vel_rad_s"].astype(np.float64)
    base_z = d["base_z_m"][:, 0].astype(np.float64)
    body_force = d["body_force_n"][:, 0].astype(np.float64)
    stand = d["stand_request"][:, 0] == 1.
    mode = d["mode"][:, 0]
    if not np.all(mode == 1.) or not stand.all():
        raise TraceContractError("PPO trace is not continuously commanded rear standing")

    clearance = wheel[:, :, 2] - WHEEL_RADIUS_CONSERVATIVE
    line_error = _line_distance(com[:, :2], wheel[:, 2:, :2])
    speed = np.linalg.norm(com_vel, axis=-1)
    angular_speed = np.linalg.norm(ang_vel, axis=-1)
    pair = (force[:, 2:] > 5.).all(-1) & (np.abs(force[:, :2]) < 2.).all(-1) & (clearance[:, :2].min(-1) >= .03)
    quiet_speed = speed <= SPEED_LIMIT
    quiet_angular = angular_speed <= ANGULAR_LIMIT
    four = (force > 5.).all(-1)
    upright = gravity[:, 2] < -math.cos(ATTITUDE_LIMIT)
    entry_pose = upright & (np.abs(base_z - .416) <= .07)
    recovered = four & quiet_speed & quiet_angular & entry_pose
    outside_instant = (np.abs(com[:, :2]) >= 198.).any(-1) | (np.abs(wheel[:, :, :2]) >= 198.).any((1, 2))
    contact = (t > .15) & (body_force > 5.)
    fall = (base_z < .18) | (gravity[:, 2] > -math.cos(math.radians(80.)))
    raw_hf = np.concatenate((com_vel, ang_vel, d["leg_position_rad"], d["wheel_velocity_rad_s"]), -1)
    hf = high_frequency_rms(raw_hf)
    hf_gates = hf <= HF_LIMITS
    smooth = hf_gates.all(-1)

    phase_before = np.zeros(n, np.int8)
    phase_after = np.zeros(n, np.int8)
    target_after = np.zeros((n, 3))
    angle = np.zeros(n)
    qualified = np.zeros(n, bool)
    entered_log = np.zeros(n, bool)
    failed_log = np.zeros(n, bool)
    boundary_log = np.zeros(n, bool)
    held_log = np.zeros(n, bool)
    interrupted_log = np.zeros(n, bool)
    success_log = np.zeros(n, bool)
    hold_log = np.zeros(n)
    best_log = np.zeros(n)
    post_log = np.zeros(n)
    start_log = np.full(n, -1.)
    capture_log = np.full(n, -1.)
    break_log = np.full(n, -1.)

    start = capture = first_break = -1.
    entry_streak = hold_streak = best_streak = post_samples = 0
    entered = held = interrupted = failed = boundary = False
    failure_reason = 0
    first_break_gates: list[str] = []
    for i in range(n):
        phase, target = _phase_and_target(float(t[i]), start, rear_pitch_rad)
        phase_before[i] = phase
        angle[i] = math.acos(float(np.clip(gravity[i] @ target, -1., 1.)))
        if outside_instant[i]:
            boundary = True
        code = 1 if contact[i] else 2 if fall[i] else 0
        if code and not failed:
            failure_reason = code
        failed |= bool(code)
        valid = not failed and not boundary
        if phase == 0 and valid and recovered[i]:
            entry_streak += 1
        else:
            entry_streak = 0
        entered |= entry_streak >= 500
        if start < 0. and entered and valid and t[i] >= 1.:
            start = float(t[i])
        if start < 0. and t[i] >= 3. - 1e-6 and valid:
            failed, valid, failure_reason = True, False, 5
        balance = (pair[i] and quiet_speed[i] and quiet_angular[i]
                   and line_error[i] <= .06 and angle[i] <= ATTITUDE_LIMIT)
        stable = bool(phase == 2 and balance and smooth[i] and entered and valid and stand[i])
        was_held = held
        hold_streak = hold_streak + 1 if stable else 0
        best_streak = max(best_streak, hold_streak)
        if hold_streak >= 2000 and not was_held:
            capture, held = float(t[i]), True
        if was_held and stand[i] and not stable and not interrupted:
            first_break, interrupted = float(t[i]), True
            gate = (pair[i], quiet_speed[i], quiet_angular[i], line_error[i] <= .06,
                    angle[i] <= ATTITUDE_LIMIT, *hf_gates[i], not failed, not boundary)
            names = ("rear_pair", "com_speed", "angular_speed", "support_line",
                     "target_posture", *(f"hf_{name}" for name in HF_NAMES),
                     "physical_failure", "world_boundary")
            first_break_gates = [name for name, good in zip(names, gate) if not good]
        if was_held and stable and not interrupted:
            post_samples += 1
        if (start >= 0. and t[i] - start >= 8. - 1e-6
                and not held and stand[i] and valid):
            failed, failure_reason = True, 6
        observation_complete = (post_samples * DT >= observation_s - 1e-8
                                and not interrupted and not failed and not boundary)
        success = bool(observation_complete and stand[i])
        qualified[i] = stable
        entered_log[i], failed_log[i], boundary_log[i] = entered, failed, boundary
        held_log[i], interrupted_log[i], success_log[i] = held, interrupted, success
        hold_log[i], best_log[i], post_log[i] = hold_streak * DT, best_streak * DT, post_samples * DT
        start_log[i], capture_log[i], break_log[i] = start, capture, first_break
        phase_after[i], target_after[i] = _phase_and_target(float(t[i]), start, rear_pitch_rad)

    time_checks = {
        name: _comparison(d[key][:, 0], value, 1e-4)
        for name, key, value in (
            ("start_time", "start_time_s", start_log),
            ("hold_time", "hold_time_s", hold_log),
            ("best_hold", "best_hold_s", best_log),
            ("post_capture", "post_capture_s", post_log),
            ("capture_time", "capture_time_s", capture_log),
            ("first_break_time", "first_break_time_s", break_log),
        )
    }
    bool_checks = {
        name: int(np.count_nonzero((d[key][:, 0] >= .5) != value))
        for name, key, value in (
            ("held", "held", held_log), ("interrupted", "interrupted", interrupted_log),
            ("success", "success", success_log), ("failed", "failed", failed_log),
            ("boundary", "boundary", boundary_log),
        )
    }
    line_check = _comparison(d["line_error_m"][:, 0], line_error, 1e-5)
    clear_check = _comparison(d["clearance_m"], clearance, 1e-6)
    hf_check = _comparison(d["jitter_rms"], hf, 5e-5)
    recorded_hf_gates = d["jitter_rms"] <= HF_LIMITS
    hf_component_disagreements = np.argwhere(recorded_hf_gates != hf_gates)
    hf_limit_disagreement = int(len(hf_component_disagreements))
    # The v7 decision uses the conjunction of all four HF limits.  Preserve
    # every channel-level discrepancy, but do not turn a numerically ambiguous
    # *inactive* channel into a false disagreement of the actual smooth gate.
    smooth_gate_disagreement = int(np.count_nonzero(recorded_hf_gates.all(-1) != smooth))
    hf_disagreement_details = [
        {
            "tick": int(index), "time_s": float(t[index]), "channel": HF_NAMES[component],
            "recorded_rms": float(d["jitter_rms"][index, component]),
            "independent_rms": float(hf[index, component]),
            "limit": float(HF_LIMITS[component]),
            "recorded_smooth_gate": bool(recorded_hf_gates[index].all()),
            "independent_smooth_gate": bool(smooth[index]),
            "phase": int(phase_after[index]), "capture_already_latched": bool(capture_log[index] >= 0.),
        }
        for index, component in hf_component_disagreements
    ]
    line_limit_disagreement = int(np.count_nonzero((d["line_error_m"][:, 0] <= .06) != (line_error <= .06)))
    clearance_limit_disagreement = int(np.count_nonzero((d["clearance_m"][:, :2].min(-1) >= .03)
                                                       != (clearance[:, :2].min(-1) >= .03)))
    phase_disagreement = int(np.count_nonzero(d["phase"][:, 0] != phase_after))
    target_check = _comparison(d["target_gravity"], target_after, 3e-6)
    reconstruction = (all(v["mismatch_count"] == 0 for v in time_checks.values())
                      and not any(bool_checks.values())
                      and line_check["mismatch_count"] == 0
                      and clear_check["mismatch_count"] == 0
                      and hf_check["mismatch_count"] == 0
                      and smooth_gate_disagreement == 0 and line_limit_disagreement == 0
                      and clearance_limit_disagreement == 0
                      and phase_disagreement == 0 and target_check["mismatch_count"] == 0)
    gate = {
        "captured_continuous_2s": capture >= 0.,
        "extra_continuous_10s": post_samples >= 10000,
        "no_first_interruption": not interrupted,
        "no_failure_or_boundary": not failed and not boundary,
        "qualified_at_final_tick": bool(qualified[-1]),
    }
    gate_pass = bool(all(gate.values()) and reconstruction)
    post_capture_mask = t > capture if capture >= 0. else np.zeros(n, dtype=bool)
    post_capture_physics = None
    if post_capture_mask.any():
        post_capture_physics = {
            "max_com_speed_m_s": float(speed[post_capture_mask].max()),
            "max_angular_speed_rad_s": float(angular_speed[post_capture_mask].max()),
            "max_support_line_error_m": float(line_error[post_capture_mask].max()),
            "max_target_attitude_rad": float(angle[post_capture_mask].max()),
            "min_rear_force_n": float(force[post_capture_mask, 2:].min()),
            "max_front_abs_force_n": float(np.abs(force[post_capture_mask, :2]).max()),
            "min_front_clearance_m": float(clearance[post_capture_mask, :2].min()),
            "max_hf_ratio_each": dict(zip(HF_NAMES,
                                           (hf[post_capture_mask] / HF_LIMITS).max(axis=0).tolist())),
        }
    return {
        "tick_start": part.start, "tick_end_exclusive": part.stop,
        "physics_samples": n, "end_time_s": float(t[-1]),
        "entry_start_time_s": start, "capture_time_s": capture,
        "first_break_time_s": first_break, "first_break_gates": first_break_gates,
        "post_capture_uninterrupted_s": post_samples * DT,
        "longest_qualified_s": best_streak * DT,
        "failure_reason": failure_reason, "failed": failed, "boundary": boundary,
        "interrupted": interrupted, "success_at_end": bool(success_log[-1]),
        "qualified_at_end": bool(qualified[-1]),
        "max_com_speed_m_s": float(speed.max()),
        "max_angular_speed_rad_s": float(angular_speed.max()),
        "max_body_force_n": float(body_force.max()),
        "max_support_line_error_m": float(line_error.max()),
        "max_target_attitude_rad": float(angle.max()),
        "max_hf_ratio_each": dict(zip(HF_NAMES, (hf / HF_LIMITS).max(axis=0).tolist())),
        "post_capture_physics": post_capture_physics,
        "gate_pass": gate_pass, "terminal_five_field_gate": gate,
        "recorded_final": {
            "held": bool(d["held"][-1, 0]), "interrupted": bool(d["interrupted"][-1, 0]),
            "success": bool(d["success"][-1, 0]), "failed": bool(d["failed"][-1, 0]),
            "boundary": bool(d["boundary"][-1, 0]),
            "post_capture_s": float(d["post_capture_s"][-1, 0]),
        },
        "reconstruction_verified": bool(reconstruction),
        "telemetry_comparison": {
            "times": time_checks, "boolean_mismatch_counts": bool_checks,
            "line_error": line_check, "clearance": clear_check, "high_frequency_rms": hf_check,
            "hf_limit_disagreement": hf_limit_disagreement,
            "hf_limit_disagreement_details": hf_disagreement_details,
            "smooth_gate_disagreement": smooth_gate_disagreement,
            "line_limit_disagreement": line_limit_disagreement,
            "clearance_limit_disagreement": clearance_limit_disagreement,
            "phase_disagreement": phase_disagreement, "target_gravity": target_check,
        },
    }


def audit_training_artifacts(run: Path, request: dict, result: dict) -> dict:
    """Cross-check PPO update, checkpoint lineage and timeout log; no physics gate substitution."""
    import torch

    rows_path = run / "updates.jsonl"
    if not rows_path.is_file():
        raise TraceContractError("Missing PPO updates.jsonl")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if len(rows) != request["new_updates"]:
        raise TraceContractError("PPO update rows differ from requested updates")
    actual_steps = [row["actual_critic_adam_steps"] for row in rows]
    if any(b - a != 2 for a, b in zip(actual_steps, actual_steps[1:])):
        raise TraceContractError("Critic Adam steps do not rise by two per update")
    if any(row["bootstrap"].get("pre_reset_bootstrap_once_verified") is not True for row in rows):
        raise TraceContractError("Pre-reset terminal Critic bootstrap audit is absent")
    if any(row["bootstrap"]["timeout_count"] != row["collection"]["timeouts"]
           or row["bootstrap"]["physical_termination_count"]
              != row["collection"]["physical_terminations"] for row in rows):
        raise TraceContractError("Collection and terminal bootstrap counts differ")
    if sum(row["bootstrap"]["timeout_count"] for row in rows) != result["timeout_count"]:
        raise TraceContractError("Timeout audit count differs from result")
    if sum(row["bootstrap"]["physical_termination_count"] for row in rows) != result["physical_termination_count"]:
        raise TraceContractError("Physical termination audit count differs from result")
    if actual_steps[-1] != result["actual_critic_adam_steps"]:
        raise TraceContractError("Final Critic Adam step differs from update log")
    checkpoint = Path(result["final_checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != result["final_checkpoint_sha256"]:
        raise TraceContractError("Final PPO checkpoint missing or changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    counters = saved["counters"]
    for key in ("completed_updates", "control_steps", "transition_samples", "optimizer_steps"):
        if counters[key] != result.get(key, result.get("expected_optimizer_steps") if key == "optimizer_steps" else None):
            raise TraceContractError(f"Final checkpoint {key} counter differs from result")
    if counters["completed_updates"] != result["initial_updates"] + request["new_updates"]:
        raise TraceContractError("Completed PPO updates differ from requested new updates")
    if result["feature_build_calls"] != request["new_updates"] * request["steps_per_env"] + 1:
        raise TraceContractError("Feedback memory was not built once per new policy input")
    actor_path = Path(request["actor"])
    if (not actor_path.is_file() or sha256(actor_path) != request["actor_sha256"]
            or request["actor_sha256"] != result["actor_sha256"]):
        raise TraceContractError("Designated BC Actor hash differs from request/result")
    source_actor = torch.load(actor_path, map_location="cpu", weights_only=False)
    initial = source_actor["actor_state_dict"]
    final = saved["model_state_dict"]
    frozen = {key: torch.equal(value, final["actor." + key])
              for key, value in initial.items() if key.startswith("prefix.")}
    if not frozen or not all(frozen.values()):
        raise TraceContractError("Frozen rising prefix differs from designated BC Actor")
    initial_coeff = initial["coefficients"]
    final_coeff = final["actor.coefficients"]
    source_checkpoint = request.get("resume_checkpoint")
    source_coeff_changed = bool(not torch.equal(initial_coeff, final_coeff))
    inherited_model_equal = None
    migration_exact = None
    initial_adam_step = 0
    coefficients_changed_this_run = source_coeff_changed
    if source_checkpoint:
        source_path = Path(source_checkpoint)
        if not source_path.is_file() or sha256(source_path) != request["resume_checkpoint_sha256"]:
            raise TraceContractError("Source PPO checkpoint missing or changed")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        # The worker verifies exact model/Adam/RNG/counters at the process's
        # initial restore; this checks the pinned source and retained prefix.
        inherited_model_equal = all(torch.equal(source["model_state_dict"][key], final[key])
                                    for key in final if key.startswith("actor.prefix."))
        if not inherited_model_equal:
            raise TraceContractError("PPO source/final prefix mismatch")
        if result.get("restored_state_verified") is not True:
            raise TraceContractError("Worker did not verify restored PPO state")
        if request.get("checkpoint_migration"):
            migrated_path = run / "checkpoints" / "migrated_input.pt"
            if not migrated_path.is_file() or result.get("checkpoint_migration_verified") is not True:
                raise TraceContractError("Formal checkpoint migration artifact/verification missing")
            migrated = torch.load(migrated_path, map_location="cpu", weights_only=False)
            source_config, migrated_config = source["config"], migrated["config"]
            only_rollout_changed = (source_config["num_steps_per_env"] == 512
                and migrated_config["num_steps_per_env"] == 32
                and {k: v for k, v in source_config.items() if k != "num_steps_per_env"}
                    == {k: v for k, v in migrated_config.items() if k != "num_steps_per_env"})
            unchanged = all(_checkpoint_equal(migrated[key], source[key])
                            for key in ("model_state_dict", "optimizer_state_dict", "rng", "counters"))
            migration_exact = bool(only_rollout_changed and unchanged
                and migrated["explicit_migration"]["source_checkpoint_sha256"] == sha256(source_path))
            if not migration_exact:
                raise TraceContractError("Formal checkpoint migration changed learned/RNG/counter state")
        source_coeff = source["model_state_dict"]["actor.coefficients"]
        coefficients_changed_this_run = bool(not torch.equal(source_coeff, final_coeff))
        initial_adam_step = source["counters"]["optimizer_steps"]
        if (result["initial_updates"] != source["counters"]["completed_updates"]
                or result["control_steps"] != source["counters"]["control_steps"]
                   + request["new_updates"] * request["steps_per_env"]):
            raise TraceContractError("PPO checkpoint inheritance counters differ")
    elif result["control_steps"] != request["new_updates"] * request["steps_per_env"]:
        raise TraceContractError("Fresh PPO control-step count differs from request")
    if actual_steps[0] != initial_adam_step + 2 or result["actual_critic_adam_steps"] != initial_adam_step + 2*len(rows):
        raise TraceContractError("Critic Adam step differs from checkpoint lineage")
    if bool(result["actor_coefficients_changed"]) != coefficients_changed_this_run:
        raise TraceContractError("Reported Actor coefficient change differs from checkpoint")
    optimizer_steps = [int(value["step"].item() if isinstance(value["step"], torch.Tensor)
                           else value["step"])
                       for value in saved["optimizer_state_dict"]["state"].values()
                       if "step" in value]
    if not optimizer_steps or max(optimizer_steps) != result["actual_critic_adam_steps"]:
        raise TraceContractError("Saved Adam state disagrees with recorded real steps")
    return {
        "update_rows": len(rows), "actual_critic_adam_steps_by_update": actual_steps,
        "timeout_count": result["timeout_count"],
        "pre_reset_bootstrap_once_all_updates": True,
        "timeout_bootstrap_exercised": result["timeout_count"] > 0,
        "frozen_prefix_tensor_count": len(frozen), "frozen_prefix_exact": True,
        "coefficients_changed_from_bc": source_coeff_changed,
        "coefficients_changed_this_run": coefficients_changed_this_run,
        "coefficients_max_abs_change_from_bc": float((final_coeff - initial_coeff).abs().max()),
        "actual_critic_adam_steps_initial": initial_adam_step,
        "checkpoint_optimizer_state_max_step": max(optimizer_steps),
        "inherited_prefix_exact": inherited_model_equal,
        "checkpoint_migration_exact": migration_exact,
        "checkpoint_sha256": sha256(checkpoint),
    }


def review_run(run: Path) -> dict:
    run = run.resolve()
    request, result = read_json(run / "request.json"), read_json(run / "result.json")
    if read_json(run / "session.json").get("status") != "completed":
        raise TraceContractError("PPO run is not completed")
    if (request["mode"] != result["mode"] or request["new_updates"] != result["new_updates"]
            or request["technical_time_limit_s"] != result["technical_time_limit_s"]
            or request["steps_per_env"] != result["steps_per_env"]):
        raise TraceContractError("PPO request/result configuration differs")
    if (request.get("no_observation_noise") is not True
            or request.get("no_reset_randomization") is not True
            or request.get("stop_on_observation") is not False
            or request.get("automatic_frozen_evaluation") is not False):
        raise TraceContractError("PPO fixed-condition review contract changed")
    trace_path = run / "physics_trace.npz"
    if not trace_path.is_file() or sha256(trace_path) != result.get("physics_trace_sha256"):
        raise TraceContractError("1 kHz physics trace missing or changed")
    data = trace_arrays(trace_path)
    ticks, slots = data["physics_time_s"].shape[:2]
    if slots != request["num_envs"] or slots != result["num_envs"]:
        raise TraceContractError("Trace slot count differs from request/result")
    expected_ticks = request["new_updates"] * request["steps_per_env"] * 20
    if ticks != expected_ticks or result.get("physics_trace_samples") != ticks:
        raise TraceContractError("Trace tick count differs from actual PPO collection budget")
    if result["physics"]["physics_samples_per_slot"] != [ticks] * slots:
        raise TraceContractError("1 kHz per-slot samples differ from result")
    ppo_manifest_path = run / "ppo_dependency_manifest.json"
    source_manifest_path = run / "source_manifest.json"
    for manifest_path in (ppo_manifest_path, source_manifest_path):
        manifest = read_json(manifest_path)
        for relative, digest in manifest.items():
            archived = run / "source_snapshot" / relative
            if not archived.is_file() or sha256(archived) != digest:
                raise TraceContractError(f"Archived source differs from {manifest_path.name}: {relative}")
    geometry = read_json(run / "geometry.json")
    rear_pitch = geometry["candidates"][1]["pitch_rad"]
    per_slot = []
    for slot in range(slots):
        episodes = [analyse_episode(data, slot, part, rear_pitch)
                    for part in episode_slices(data["physics_time_s"][:, slot, 0])]
        last = episodes[-1]
        per_slot.append({
            "slot": slot, "episode_count_in_trace": len(episodes),
            "episodes": episodes,
            "terminal_five_field_gate": last["terminal_five_field_gate"],
            "single_untouched_episode": len(episodes) == 1,
            "formal_smoke_slot_pass": bool(len(episodes) == 1 and last["gate_pass"]),
        })
    training = audit_training_artifacts(run, request, result)
    formal = request["mode"] == "formal-smoke"
    formal_gate = (formal and slots == 16 and all(row["formal_smoke_slot_pass"] for row in per_slot))
    behavior_cases = []
    for row in per_slot:
        episodes = row["episodes"]
        chosen = (len(episodes) - 1 if formal else
                  max(range(len(episodes)), key=lambda index: episodes[index]["post_capture_uninterrupted_s"]))
        episode = episodes[chosen]
        behavior_cases.append({
            "slot": row["slot"], "selected_episode_index": chosen,
            "episode_count_in_trace": len(episodes),
            "selected_episode_ends_at_run_end": chosen == len(episodes) - 1,
            "capture_time_s": episode["capture_time_s"],
            "post_capture_uninterrupted_s": episode["post_capture_uninterrupted_s"],
            "first_break_time_s": episode["first_break_time_s"],
            "success_at_episode_end": episode["success_at_end"],
            "reconstruction_verified": episode["reconstruction_verified"],
            "five_field_gate": episode["terminal_five_field_gate"],
            "five_field_pass": episode["gate_pass"],
            "formal_single_episode_pass": row["formal_smoke_slot_pass"] if formal else None,
            "post_capture_physics": episode["post_capture_physics"],
        })
    return {
        "schema": "r4a_ppo_independent_physics_review_v1",
        "run": str(run), "mode": request["mode"],
        "request_sha256": sha256(run / "request.json"),
        "result_sha256": sha256(run / "result.json"),
        "trace_sha256": sha256(trace_path),
        "analysis_source_sha256": sha256(Path(__file__)),
        "source_manifest_sha256": sha256(source_manifest_path),
        "ppo_dependency_manifest_sha256": sha256(ppo_manifest_path),
        "physics_dt_s": DT, "physics_ticks": ticks, "slots": slots,
        "thresholds": {
            "com_speed_m_s": SPEED_LIMIT, "angular_speed_rad_s": ANGULAR_LIMIT,
            "support_line_m": .06, "target_attitude_rad": ATTITUDE_LIMIT,
            "rear_contact_min_n": 5., "front_abs_force_max_n": 2.,
            "front_clearance_min_m": .03,
            "body_contact_max_n_after_0_15s": 5., "hf_limits": dict(zip(HF_NAMES, HF_LIMITS.tolist())),
            "capture_s": 2., "extra_observation_s": OBSERVATION_S,
        },
        "per_slot": per_slot,
        "behavior_cases": behavior_cases,
        "behavior_case_pass_count": sum(case["five_field_pass"] for case in behavior_cases),
        "training_artifacts": training,
        "all_episode_reconstructions_verified": all(
            episode["reconstruction_verified"] for row in per_slot for episode in row["episodes"]),
        "formal_smoke_16_slot_five_field_gate": bool(formal_gate),
        "formal_review_decision_emitted": False,
        "note": "Independent physical review evidence; formal review.json is separate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed PPO run containing physics_trace.npz")
    parser.add_argument("--output", type=Path, help="New JSON file; never overwrites an earlier review")
    args = parser.parse_args(argv)
    report = review_run(args.run)
    destination = args.output or Path(__file__).resolve().parent / "results" / f"{args.run.name}_physics_review.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(destination)
    print("all_episode_reconstructions_verified:", report["all_episode_reconstructions_verified"])
    print("formal_smoke_16_slot_five_field_gate:", report["formal_smoke_16_slot_five_field_gate"])


if __name__ == "__main__":
    main()
