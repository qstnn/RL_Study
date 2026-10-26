"""CPU 离线评价：本体估计不读特权量，teacher COM/接触仅用于计分。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from body_estimator import BodyEstimator, TorchBodyEstimator, STAND, CANONICAL_URDF_SHA256


HERE = Path(__file__).resolve().parent
SUSTAIN = HERE.parents[1]
URDF = SUSTAIN.parents[4] / "lighthw_urdf/urdf/lighthw.urdf"
TRACES = {
    "teacher_30s_success": SUSTAIN / "runs/2026-10-22_sustain_30s_blend800/trace_seed43_blend800.npz",
    "memory_exposed_failure": SUSTAIN / "student_memory_v4/runs/2026-10-23_exposed_screen/trace_autonomous_seed43.npz",
    "memory_zero_failure": SUSTAIN / "student_memory_v4/runs/2026-10-23_zero_screen/trace_autonomous_seed43.npz",
}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(pred, true, mask, threshold):
    if not np.any(mask):
        return None
    p, t = pred[mask], true[mask]
    err = p - t
    significant = np.abs(t) >= threshold
    return dict(n=int(mask.sum()), mae=float(np.mean(np.abs(err))),
                p95_abs=float(np.quantile(np.abs(err), .95)), p99_abs=float(np.quantile(np.abs(err), .99)),
                max_abs=float(np.max(np.abs(err))), bias=float(np.mean(err)),
                abs_error_gt_0_12_n=int(np.sum(np.abs(err) > .12)),
                abs_error_gt_0_12_fraction=float(np.mean(np.abs(err) > .12)),
                significant_threshold=threshold, significant_n=int(significant.sum()),
                significant_wrong_sign_n=int(np.sum((np.sign(p) != np.sign(t)) & significant)),
                significant_wrong_sign_fraction=float(np.mean((np.sign(p) != np.sign(t))[significant])) if significant.any() else None)


def inspect_trace(name, path, estimator):
    with np.load(path) as d:
        raw = d["policy_raw_observation"] if "policy_raw_observation" in d else d["policy_observation"]
        if raw.shape[-1] not in (305, 307) or raw.shape[1] != 2:
            raise ValueError(f"wrong raw policy shape {path}: {raw.shape}")
        raw = raw[..., :305]
        k = np.arange(1, len(raw))
        physical = k * 20 - 1
        if physical[-1] >= len(d["gravity"]):
            raise ValueError("policy/frame alignment would exceed physics trace")
        active = d["policy_active"][k] & d["active"][physical]
        selected = raw[k][active]
        frame = selected[:, -61:]
        q = d["leg_position"][physical][active]
        qdot = d["joint_velocity"][physical][active]
        gravity = d["gravity"][physical][active]
        omega = d["ang_vel"][physical][active]
        alignment = dict(physics_index="policy_k*20-1; k>=1", n=int(len(selected)),
                         leg_q_max_abs_rad=float(np.max(np.abs(frame[:, 6:18] + STAND - q))),
                         joint_velocity_max_abs_rad_s=float(np.max(np.abs(frame[:, 18:34] / .05 - qdot))),
                         gravity_max_abs=float(np.max(np.abs(frame[:, 3:6] - gravity))),
                         angular_velocity_max_abs_rad_s=float(np.max(np.abs(frame[:, :3] / .2 - omega))))
        if alignment["leg_q_max_abs_rad"] > 1e-6 or alignment["joint_velocity_max_abs_rad_s"] > 1e-5 or alignment["gravity_max_abs"] > 1e-6:
            raise AssertionError(f"raw305/robot order or frame alignment differs: {name}: {alignment}")
        out = estimator.estimate(selected)
        wheel = d["wheel_pos"][physical][active]
        com = d["com"][physical][active]
        com_vel = d["com_vel"][physical][active]
        lateral = wheel[:, 3, :2] - wheel[:, 2, :2]
        forward = np.stack((lateral[:, 1], -lateral[:, 0]), axis=-1)
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
        true_gap = np.sum((com[:, :2] - wheel[:, 2:, :2].mean(1)) * forward, axis=-1)
        true_velocity = np.sum(com_vel[:, :2] * forward, axis=-1)
        t = d["t"][physical][active]
        forces = d["force_z"][physical][active]
        rear_loaded = np.all(forces[:, 2:] > 5., axis=-1)
        front_unloaded = np.all(np.abs(forces[:, :2]) < 2., axis=-1)
        masks = dict(all=np.ones(len(selected), dtype=bool), pre_3_2_s=t < 3.2,
                     transition_3_2_to_5_5_s=(t >= 3.2) & (t < 5.5), after_5_5_s=t >= 5.5,
                     both_rear_loaded=rear_loaded, rear_not_both_loaded=~rear_loaded,
                     rear_loaded_front_unloaded=rear_loaded & front_unloaded,
                     transition_rear_loaded=(t >= 3.2) & (t < 5.5) & rear_loaded)
        groups = {key: dict(gap=score(out["gap_m"], true_gap, mask, .03),
                            velocity=score(out["velocity_m_s"], true_velocity, mask, .10),
                            velocity_full_vertical=score(out["velocity_full_vertical_m_s"], true_velocity, mask, .10),
                            velocity_spin_only=score(out["velocity_spin_only_m_s"], true_velocity, mask, .10))
                  for key, mask in masks.items()}
        # 仅用于检验 URDF +Y 轮轴及观测轮速符号。teacher 速度不参与运行时标定。
        # per_rear = relative + rolling；从其 2 个分量间接比较错误反号。
        spin = out["rear_wheel_rate_rad_s"]
        both_fast = rear_loaded & (np.mean(np.abs(spin), axis=1) >= .5)
        # 已知轮轴在姿态变化下需乘以 (axis×up)·forward；从本体 FK 重建系数。
        _, _, _, _, wheel_axis, _ = estimator.geometry(frame[:, 6:18] + STAND)
        up = -frame[:, 3:6] / np.linalg.norm(frame[:, 3:6], axis=-1, keepdims=True)
        radial_up = up[:, None] - np.sum(up[:, None] * wheel_axis, axis=-1, keepdims=True) * wheel_axis
        radial_up /= np.linalg.norm(radial_up, axis=-1, keepdims=True)
        spin_rolling = .0875 * spin * np.sum(np.cross(wheel_axis, radial_up) * out["forward_body"][:, None], axis=-1)
        wrong_sign_velocity = (out["rear_velocity_m_s"] - 2. * spin_rolling).mean(1)
        axis_check = dict(joint_order=list(estimator.index), rear_wheel_indices=[14, 15], wheel_axis_urdf="+Y, RR then RL",
                          fast_loaded_n=int(both_fast.sum()),
                          correct_sign_mae_m_s=float(np.mean(np.abs(out["velocity_m_s"][both_fast] - true_velocity[both_fast]))) if both_fast.any() else None,
                          reversed_spin_sign_mae_m_s=float(np.mean(np.abs(wrong_sign_velocity[both_fast] - true_velocity[both_fast]))) if both_fast.any() else None)
        # PyTorch 接口与 NumPy 逐值对照且要求对本体观测导数有限。
        torch_estimator = TorchBodyEstimator(URDF)
        sample = selected[np.linspace(0, len(selected) - 1, min(32, len(selected)), dtype=int)]
        tensor = torch.tensor(sample, dtype=torch.float64, requires_grad=True)
        torch_out = torch_estimator(tensor)
        np_out = estimator.estimate(sample)
        torch_delta = {key: float(np.max(np.abs(torch_out[key].detach().numpy() - np_out[key])))
                       for key in ("gap_m", "velocity_m_s", "rear_velocity_m_s", "velocity_full_vertical_m_s", "velocity_spin_only_m_s")}
        (torch_out["gap_m"].square().sum() + torch_out["velocity_m_s"].square().sum()).backward()
        torch_check = dict(max_delta=torch_delta, finite_gradient=bool(torch.isfinite(tensor.grad).all()),
                           nonzero_gradient_entries=int(torch.count_nonzero(tensor.grad).item()))
        if max(torch_delta.values()) > 1e-9 or not torch_check["finite_gradient"] or torch_check["nonzero_gradient_entries"] == 0:
            raise AssertionError(f"Torch estimator mismatch or gradient failure: {name}: {torch_check}")
        return dict(path=str(path), trace_sha256=sha(path), active_policy_samples=int(len(selected)),
                    slots=2, physics_dt_s=.001, policy_dt_s=.02, alignment=alignment,
                    axis_and_sign=axis_check, torch=torch_check, groups=groups,
                    input_contract="raw305 last 61 frame only; teacher COM, COM velocity, contacts and root linear velocity for scoring only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/2026-10-23_offline.json")
    args = parser.parse_args()
    estimator = BodyEstimator(URDF)
    report = dict(generated_at=datetime.now().astimezone().isoformat(), urdf=str(URDF),
                  urdf_sha256=sha(URDF), expected_urdf_sha256=CANONICAL_URDF_SHA256,
                  estimator_sha256=sha(HERE / "body_estimator.py"), evaluator_sha256=sha(__file__),
                  radius_m=.0875, runtime_inputs="raw305 final frame: IMU gravity/angular rate, 12 leg angles, 16 joint velocities; no privileged input",
                  traces={name: inspect_trace(name, path, estimator) for name, path in TRACES.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, case in report["traces"].items():
        print(name, "n", case["active_policy_samples"], "transition_gap", case["groups"]["transition_3_2_to_5_5_s"]["gap"]["mae"],
              "transition_velocity", case["groups"]["transition_3_2_to_5_5_s"]["velocity"]["mae"],
              "velocity_wrong_sign", case["groups"]["transition_3_2_to_5_5_s"]["velocity"]["significant_wrong_sign_fraction"])
    print(args.output)


if __name__ == "__main__":
    main()
