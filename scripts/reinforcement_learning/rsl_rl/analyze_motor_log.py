"""Analyze motor monitor CSV torque consistency.

This script compares:
1. applied_torque vs clip(computed_torque, effort_limit)
2. computed_torque vs Kp * (q_target - q) + Kd * (qd_target - qd) + effort_target

The second comparison may differ by one actuator step for DelayedPDActuator logs because
the CSV records public target tensors, while the actuator may use delayed targets internally.
"""

import argparse
import csv
import math
import os


def _float(row, key, default=math.nan):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _clip(value, limit):
    return max(-limit, min(limit, value))


def _stats(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return {"n": 0, "mean_abs": math.nan, "max_abs": math.nan, "rmse": math.nan}
    mean_abs = sum(abs(value) for value in values) / len(values)
    max_abs = max(abs(value) for value in values)
    rmse = math.sqrt(sum(value * value for value in values) / len(values))
    return {"n": len(values), "mean_abs": mean_abs, "max_abs": max_abs, "rmse": rmse}


def _find_monitored_joints(headers):
    joints = []
    for header in headers:
        if not header.endswith("_torque") or header.endswith("_computed_torque"):
            continue
        name = header[: -len("_torque")]
        required = [
            f"{name}_pos",
            f"{name}_vel",
            f"{name}_computed_torque",
            f"{name}_pos_target",
            f"{name}_vel_target",
            f"{name}_effort_target",
        ]
        if all(item in headers for item in required):
            joints.append(name)
    return joints


def analyze(csv_path):
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames or []
        joints = _find_monitored_joints(headers)
        if not joints:
            raise RuntimeError(
                "No monitor joints with torque/computed_torque/target columns were found. "
                "Please rerun play_motor_monitor.py after the CSV target columns were added."
            )

        diff_applied_vs_clipped = {joint: [] for joint in joints}
        diff_computed_vs_formula = {joint: [] for joint in joints}
        saturation_count = {joint: 0 for joint in joints}
        total_count = {joint: 0 for joint in joints}

        for row in reader:
            for joint in joints:
                is_wheel = "wheel" in joint.lower()
                effort_limit = _float(row, "tune_wheel_effort" if is_wheel else "tune_joint_effort")
                kp = 0.0 if is_wheel else _float(row, "tune_joint_stiffness")
                kd = _float(row, "tune_wheel_damping" if is_wheel else "tune_joint_damping")

                applied = _float(row, f"{joint}_torque")
                computed = _float(row, f"{joint}_computed_torque")
                pos = _float(row, f"{joint}_pos")
                vel = _float(row, f"{joint}_vel")
                pos_target = _float(row, f"{joint}_pos_target")
                vel_target = _float(row, f"{joint}_vel_target")
                effort_target = _float(row, f"{joint}_effort_target", default=0.0)

                if any(math.isnan(value) for value in (effort_limit, applied, computed)):
                    continue

                clipped_computed = _clip(computed, effort_limit)
                diff_applied_vs_clipped[joint].append(applied - clipped_computed)
                total_count[joint] += 1
                if abs(computed) >= effort_limit - 1e-6:
                    saturation_count[joint] += 1

                if any(math.isnan(value) for value in (kp, kd, pos, vel, pos_target, vel_target, effort_target)):
                    continue
                formula_computed = kp * (pos_target - pos) + kd * (vel_target - vel) + effort_target
                diff_computed_vs_formula[joint].append(computed - formula_computed)

    print(f"CSV: {os.path.abspath(csv_path)}")
    print("Comparison A: applied_torque - clip(computed_torque, effort_limit)")
    print("Comparison B: computed_torque - formula_no_delay")
    print("")
    for joint in joints:
        stats_a = _stats(diff_applied_vs_clipped[joint])
        stats_b = _stats(diff_computed_vs_formula[joint])
        saturation_ratio = 0.0
        if total_count[joint] > 0:
            saturation_ratio = 100.0 * saturation_count[joint] / total_count[joint]
        print(
            f"{joint}: "
            f"A mean_abs={stats_a['mean_abs']:.6g}, A max_abs={stats_a['max_abs']:.6g}, "
            f"A rmse={stats_a['rmse']:.6g}; "
            f"B mean_abs={stats_b['mean_abs']:.6g}, B max_abs={stats_b['max_abs']:.6g}, "
            f"B rmse={stats_b['rmse']:.6g}; "
            f"saturation={saturation_ratio:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze motor_monitor/motor_log.csv torque consistency.")
    parser.add_argument("csv_path", help="Path to motor_monitor/motor_log.csv.")
    args = parser.parse_args()
    analyze(args.csv_path)


if __name__ == "__main__":
    main()
