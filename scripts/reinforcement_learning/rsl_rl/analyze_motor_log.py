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


def _value_stats(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return {"n": 0, "mean": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    return sum(values) / len(values)


def _rms(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    return math.sqrt(sum(value * value for value in values) / len(values))


def _fmt(value, digits=3):
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _split_episodes(rows):
    """Split a single-env monitor log whenever episode_step decreases."""
    episodes = []
    current = []
    previous_episode_step = None
    for row in rows:
        episode_step = _float(row, "episode_step")
        if (
            current
            and not math.isnan(episode_step)
            and previous_episode_step is not None
            and episode_step < previous_episode_step
        ):
            episodes.append(current)
            current = []
        current.append(row)
        if not math.isnan(episode_step):
            previous_episode_step = episode_step
    if current:
        episodes.append(current)
    return episodes


def _first_sustained_failure(rows, predicate, confirmation_steps):
    run_start = None
    run_length = 0
    for index, row in enumerate(rows):
        if predicate(row):
            if run_start is None:
                run_start = index
            run_length += 1
            if run_length >= confirmation_steps:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def _analyze_biped_stand(
    rows,
    *,
    reach_height,
    reach_verticality,
    fall_height,
    minimum_verticality,
    expected_episode_steps,
    failure_confirmation_steps,
):
    required_headers = {
        "episode_step",
        "root_x_env",
        "root_y_env",
        "root_z",
        "root_lin_vel_x",
        "root_lin_vel_y",
        "projected_gravity_x",
    }
    if not rows or not required_headers.issubset(rows[0]):
        return None

    episode_results = []
    episodes = _split_episodes(rows)
    for episode_index, episode in enumerate(episodes):
        stand_index = None
        for index, row in enumerate(episode):
            root_z = _float(row, "root_z")
            verticality = abs(_float(row, "projected_gravity_x"))
            if root_z >= reach_height and verticality >= reach_verticality:
                stand_index = index
                break

        max_episode_step = max(_float(row, "episode_step") for row in episode)
        horizon_completed = max_episode_step >= expected_episode_steps - 2
        closed_by_reset = episode_index < len(episodes) - 1
        result = {
            "index": episode_index + 1,
            "samples": len(episode),
            "horizon_completed": horizon_completed,
            "closed_by_reset": closed_by_reset,
            "stood": stand_index is not None,
            "stand_step": math.nan,
            "hold_steps": 0,
            "collapsed": False,
            "failure_kind": "",
            "continuous_horizon": False,
            "xy_speed_mean": math.nan,
            "xy_speed_rms": math.nan,
            "drift_max": math.nan,
            "verticality_mean": math.nan,
            "verticality_min": math.nan,
            "both_rear_5n": math.nan,
            "both_rear_12n": math.nan,
            "rear_joint_velocity_rms": math.nan,
            "rear_wheel_velocity_rms": math.nan,
            "rl_calf_pos_mean": math.nan,
            "rr_calf_pos_mean": math.nan,
        }
        if stand_index is None:
            episode_results.append(result)
            continue

        result["stand_step"] = _float(episode[stand_index], "episode_step")
        post_stand = episode[stand_index:]

        def _failed(row):
            root_z = _float(row, "root_z")
            verticality = abs(_float(row, "projected_gravity_x"))
            return root_z < fall_height or verticality < minimum_verticality

        first_failed_sample = next(
            (index for index, row in enumerate(post_stand) if _failed(row)),
            None,
        )
        collapse_offset = _first_sustained_failure(
            post_stand, _failed, failure_confirmation_steps
        )
        if collapse_offset is not None:
            result["failure_kind"] = "pose_collapse"
        elif closed_by_reset and not horizon_completed:
            # Terminating environments often reset on the first bad sample, so
            # the log cannot contain a sustained run. Any early reset after a
            # successful stand still fails the primary no-sitting/hold gate.
            collapse_offset = (
                first_failed_sample
                if first_failed_sample is not None
                else len(post_stand)
            )
            result["failure_kind"] = "early_reset"
        if collapse_offset is None:
            stable_rows = post_stand
            result["hold_steps"] = len(post_stand)
        else:
            stable_rows = post_stand[:collapse_offset]
            result["hold_steps"] = collapse_offset
            result["collapsed"] = True
        result["continuous_horizon"] = horizon_completed and not result["collapsed"]

        xy_speeds = [
            math.hypot(
                _float(row, "root_lin_vel_x"),
                _float(row, "root_lin_vel_y"),
            )
            for row in stable_rows
        ]
        result["xy_speed_mean"] = _mean(xy_speeds)
        result["xy_speed_rms"] = _rms(xy_speeds)
        anchor_x = _float(episode[stand_index], "root_x_env")
        anchor_y = _float(episode[stand_index], "root_y_env")
        result["drift_max"] = max(
            (
                math.hypot(
                    _float(row, "root_x_env") - anchor_x,
                    _float(row, "root_y_env") - anchor_y,
                )
                for row in stable_rows
            ),
            default=math.nan,
        )
        verticalities = [
            abs(_float(row, "projected_gravity_x")) for row in stable_rows
        ]
        result["verticality_mean"] = _mean(verticalities)
        result["verticality_min"] = min(verticalities, default=math.nan)

        if {"contact_rl_fz", "contact_rr_fz"}.issubset(episode[0]):
            result["both_rear_5n"] = _mean(
                [
                    float(
                        _float(row, "contact_rl_fz") >= 5.0
                        and _float(row, "contact_rr_fz") >= 5.0
                    )
                    for row in stable_rows
                ]
            )
            result["both_rear_12n"] = _mean(
                [
                    float(
                        _float(row, "contact_rl_fz") >= 12.0
                        and _float(row, "contact_rr_fz") >= 12.0
                    )
                    for row in stable_rows
                ]
            )

        rear_joint_velocity_headers = [
            f"R{side}_{joint}_joint_vel"
            for side in ("L", "R")
            for joint in ("hip", "thigh", "calf")
            if f"R{side}_{joint}_joint_vel" in episode[0]
        ]
        rear_wheel_velocity_headers = [
            name
            for name in ("RL_wheel_joint_vel", "RR_wheel_joint_vel")
            if name in episode[0]
        ]
        result["rear_joint_velocity_rms"] = _rms(
            [
                _float(row, name)
                for row in stable_rows
                for name in rear_joint_velocity_headers
            ]
        )
        result["rear_wheel_velocity_rms"] = _rms(
            [
                _float(row, name)
                for row in stable_rows
                for name in rear_wheel_velocity_headers
            ]
        )
        result["rl_calf_pos_mean"] = _mean(
            [_float(row, "RL_calf_joint_pos") for row in stable_rows]
        )
        result["rr_calf_pos_mean"] = _mean(
            [_float(row, "RR_calf_joint_pos") for row in stable_rows]
        )
        episode_results.append(result)

    return episode_results


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


def _find_contact_labels(headers):
    labels = []
    for header in headers:
        if not header.startswith("contact_") or not header.endswith("_fz"):
            continue
        label = header[len("contact_") : -len("_fz")]
        if f"contact_{label}_norm" not in headers:
            continue
        labels.append(label)
    return labels


def analyze(
    csv_path,
    *,
    stand_reach_height=0.56,
    stand_reach_verticality=0.88,
    stand_fall_height=0.48,
    stand_minimum_verticality=0.75,
    expected_episode_steps=300,
    failure_confirmation_steps=5,
):
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames or []
        joints = _find_monitored_joints(headers)
        contact_labels = _find_contact_labels(headers)
        if not joints and not contact_labels:
            raise RuntimeError(
                "No monitor joints or contact-force columns were found. "
                "Please rerun play_motor_monitor.py with the current logging script."
            )

        diff_applied_vs_clipped = {joint: [] for joint in joints}
        diff_computed_vs_formula = {joint: [] for joint in joints}
        saturation_count = {joint: 0 for joint in joints}
        total_count = {joint: 0 for joint in joints}
        contact_fz = {label: [] for label in contact_labels}
        contact_hist_max_fz = {label: [] for label in contact_labels}
        contact_unloaded_count = {label: 0 for label in contact_labels}
        contact_total_count = {label: 0 for label in contact_labels}
        contact_spread = []
        all_rows = []

        for row in reader:
            all_rows.append(row)
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

            for label in contact_labels:
                fz = _float(row, f"contact_{label}_fz")
                hist_max = _float(row, f"contact_{label}_hist_max_fz")
                if math.isnan(fz):
                    continue
                contact_fz[label].append(fz)
                contact_hist_max_fz[label].append(hist_max)
                contact_total_count[label] += 1
                if fz < 5.0:
                    contact_unloaded_count[label] += 1
            spread = _float(row, "contact_fz_spread")
            if not math.isnan(spread):
                contact_spread.append(spread)

    print(f"CSV: {os.path.abspath(csv_path)}")
    if joints:
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

    if contact_labels:
        print("")
        print("Contact force summary: current-frame |Fz|, unloaded means current Fz < 5N.")
        for label in contact_labels:
            fz_stats = _value_stats(contact_fz[label])
            hist_stats = _value_stats(contact_hist_max_fz[label])
            unloaded_ratio = 0.0
            if contact_total_count[label] > 0:
                unloaded_ratio = 100.0 * contact_unloaded_count[label] / contact_total_count[label]
            print(
                f"contact_{label}: "
                f"fz_mean={fz_stats['mean']:.3f}N, fz_min={fz_stats['min']:.3f}N, "
                f"fz_max={fz_stats['max']:.3f}N, unloaded={unloaded_ratio:.2f}%, "
                f"hist_max_fz_mean={hist_stats['mean']:.3f}N"
            )
        spread_stats = _value_stats(contact_spread)
        print(
            f"contact_spread: mean={spread_stats['mean']:.3f}, "
            f"min={spread_stats['min']:.3f}, max={spread_stats['max']:.3f}"
        )

    biped_results = _analyze_biped_stand(
        all_rows,
        reach_height=stand_reach_height,
        reach_verticality=stand_reach_verticality,
        fall_height=stand_fall_height,
        minimum_verticality=stand_minimum_verticality,
        expected_episode_steps=expected_episode_steps,
        failure_confirmation_steps=failure_confirmation_steps,
    )
    if biped_results is not None:
        stood_results = [result for result in biped_results if result["stood"]]
        print("")
        print("Biped stand episode summary (priority: no sitting, then stillness):")
        print(
            "Thresholds: "
            f"reach z>={stand_reach_height:.3f}m and verticality>={stand_reach_verticality:.3f}; "
            f"collapse z<{stand_fall_height:.3f}m or verticality<{stand_minimum_verticality:.3f} "
            f"for {failure_confirmation_steps} consecutive samples."
        )
        print(
            f"episodes={len(biped_results)}, stood={len(stood_results)}, "
            f"collapsed_after_stand={sum(result['collapsed'] for result in stood_results)}, "
            f"continuous_horizons={sum(result['continuous_horizon'] for result in stood_results)}, "
            f"completed_horizons={sum(result['horizon_completed'] for result in biped_results)}"
        )
        if stood_results:
            print(
                "Aggregate while continuously standing: "
                f"hold_steps_mean={_fmt(_mean([result['hold_steps'] for result in stood_results]), 1)}, "
                f"xy_speed_mean={_fmt(_mean([result['xy_speed_mean'] for result in stood_results]))}m/s, "
                f"xy_speed_rms={_fmt(_mean([result['xy_speed_rms'] for result in stood_results]))}m/s, "
                f"drift_max_mean={_fmt(_mean([result['drift_max'] for result in stood_results]))}m, "
                f"rear_joint_vel_rms={_fmt(_mean([result['rear_joint_velocity_rms'] for result in stood_results]))}rad/s, "
                f"rear_wheel_vel_rms={_fmt(_mean([result['rear_wheel_velocity_rms'] for result in stood_results]))}rad/s, "
                f"both_rear_12N={_fmt(100.0 * _mean([result['both_rear_12n'] for result in stood_results]), 1)}%"
            )
            print(
                "Rear calf position means: "
                f"RL={_fmt(_mean([result['rl_calf_pos_mean'] for result in stood_results]))}rad, "
                f"RR={_fmt(_mean([result['rr_calf_pos_mean'] for result in stood_results]))}rad"
            )
        for result in biped_results:
            state = "never_stood"
            if result["stood"]:
                if result["continuous_horizon"]:
                    state = "stood_to_horizon"
                elif result["collapsed"]:
                    state = result["failure_kind"]
                elif result["horizon_completed"]:
                    state = "stood_to_horizon"
                else:
                    state = "monitor_truncated"
            print(
                f"episode {result['index']}: state={state}, "
                f"stand_step={_fmt(result['stand_step'], 0)}, hold={result['hold_steps']}, "
                f"speed={_fmt(result['xy_speed_mean'])}m/s, "
                f"drift={_fmt(result['drift_max'])}m, "
                f"verticality_min={_fmt(result['verticality_min'])}, "
                f"both_rear_12N={_fmt(100.0 * result['both_rear_12n'], 1)}%"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze motor_monitor/motor_log.csv torque consistency.")
    parser.add_argument("csv_path", help="Path to motor_monitor/motor_log.csv.")
    parser.add_argument("--stand_reach_height", type=float, default=0.56)
    parser.add_argument("--stand_reach_verticality", type=float, default=0.88)
    parser.add_argument("--stand_fall_height", type=float, default=0.48)
    parser.add_argument("--stand_minimum_verticality", type=float, default=0.75)
    parser.add_argument(
        "--expected_episode_steps",
        type=int,
        default=1200,
        help="Expected full episode horizon for acceptance metrics.",
    )
    parser.add_argument("--failure_confirmation_steps", type=int, default=5)
    args = parser.parse_args()
    analyze(
        args.csv_path,
        stand_reach_height=args.stand_reach_height,
        stand_reach_verticality=args.stand_reach_verticality,
        stand_fall_height=args.stand_fall_height,
        stand_minimum_verticality=args.stand_minimum_verticality,
        expected_episode_steps=args.expected_episode_steps,
        failure_confirmation_steps=args.failure_confirmation_steps,
    )


if __name__ == "__main__":
    main()
