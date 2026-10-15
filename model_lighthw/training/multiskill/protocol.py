"""Deterministic Stage1 screening; completion and final acceptance are separate."""

import math
import numpy as np

VERSION = "lighthw_multiskill_stage1_screen_v1"
CASES = {"zero": 0.0, "start_stop_03": 0.3, "start_stop_06": 0.6}
DURATION = 18.0
DT = 0.02
STEPS = round(DURATION / DT)
SEEDS = (43, 44, 45)
THRESHOLDS = {"velocity_mae_mps": 0.12, "tilt_p95_deg": 10.0,
              "hold_speed_mean_mps": 0.03, "hold_yaw_rate_mean_rps": 0.05,
              "hold_four_wheel_fraction": 0.95, "target_clip_fraction": 0.20}


def command_at(case, t):
    return (CASES[case] if 3.0 <= t < 11.0 else 0.0, 0.0, 0.0)


def summarize(rows, case, seed, reason=None):
    complete = len(rows) == STEPS and reason is None
    result = {"protocol": VERSION, "case": case, "seed": seed, "samples": len(rows),
              "complete": complete, "failure_reason": reason, "success": False,
              "basic_skill_screen_only": True, "final_acceptance": "not_evaluated",
              "payload_kg": 0.0, "towing_force_N": 0.0, "sample_hz": 50,
              "thresholds": THRESHOLDS, "metrics": None}
    if not rows:
        return result
    if not all(math.isfinite(float(v)) for row in rows for v in row.values()):
        result.update(complete=False, failure_reason="non_finite_measurement")
        return result
    moving = [r for r in rows if 5.0 <= r["t"] < 11.0]
    hold = [r for r in rows if 13.0 <= r["t"] <= 18.0]
    if not moving or not hold:
        return result
    mean = lambda rs, key: float(np.mean([r[key] for r in rs]))
    metrics = {
        "velocity_mae_mps": mean(moving, "velocity_error"),
        "tilt_p95_deg": float(np.percentile([r["tilt_deg"] for r in rows], 95)),
        "hold_speed_mean_mps": mean(hold, "speed"),
        "hold_yaw_rate_mean_rps": mean(hold, "yaw_rate_abs"),
        "hold_four_wheel_fraction": mean(hold, "four_wheel"),
        "target_clip_fraction": mean(rows, "target_clip_fraction"),
        "torque_abs_peak_Nm_50Hz": max(r["torque_abs_max"] for r in rows),
        "torque_limit_fraction": mean(rows, "torque_limit_fraction"),
        "hold_drift_max_m": max(math.hypot(r["x"]-hold[0]["x"], r["y"]-hold[0]["y"]) for r in hold),
        "body_contact_samples": sum(r["body_contact"] for r in rows),
        "hold_heading_change_max_deg": max(abs(math.atan2(math.sin(r["yaw_rad"]-hold[0]["yaw_rad"]),
            math.cos(r["yaw_rad"]-hold[0]["yaw_rad"]))) * 180/math.pi for r in hold),
    }
    result["metrics"] = metrics
    result["success"] = bool(complete and not metrics["body_contact_samples"] and all(
        metrics[key] >= limit if key == "hold_four_wheel_fraction" else metrics[key] <= limit
        for key, limit in THRESHOLDS.items()))
    return result
