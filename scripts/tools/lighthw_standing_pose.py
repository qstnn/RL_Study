#!/usr/bin/env python3
"""从 canonical URDF 计算 LightHW 平地站姿；无需 Isaac Sim/GPU。

约束：hip=0、机身水平、轮心在 thigh 轴正下方，前后腿镜像弯曲。
使用解析逆运动学求角度，再用独立的 URDF 齐次变换做正运动学复核。
轮半径来自本机 wheel collision STL 的 XZ 半径约 0.0875 m。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import runpy
import struct
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "model_lighthw/lighthw_urdf/urdf/lighthw.urdf"
POSE_CONFIG = REPO / (
    "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/"
    "config/wheeled/deeprobotics_lighthw/default_pose.py"
)
LEGS = ("FR", "FL", "RR", "RL")


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin(element):
    value = np.eye(4)
    if element is not None:
        value[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        value[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
    return value


def forward_kinematics(joints, angles):
    frames = {"Base_link": np.eye(4)}
    pending = list(joints.values())
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in frames]
        if not ready:
            raise ValueError("URDF is not a connected tree rooted at Base_link")
        for joint in ready:
            move = np.eye(4)
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                move[:3, :3] = rotation(axis, angles[joint.get("name")])
            frames[joint.find("child").get("link")] = (
                frames[joint.find("parent").get("link")] @ origin(joint.find("origin")) @ move
            )
            pending.remove(joint)
    return frames


def solve_leg(joints, leg, height, radius):
    hip, thigh, calf, wheel = [joints[f"{leg}_{name}_joint"] for name in ("hip", "thigh", "calf", "wheel")]
    for joint in (hip, thigh, calf, wheel):
        if not np.allclose(origin(joint.find("origin"))[:3, :3], np.eye(3)):
            raise ValueError("This analytic solver expects the canonical zero-rpy joint origins")
    for joint in (thigh, calf, wheel):
        if joint.find("axis").get("xyz") != "0 1 0":
            raise ValueError("This analytic solver expects +Y thigh/calf/wheel axes")
    calf_offset = origin(calf.find("origin"))[:3, 3]
    wheel_offset = origin(wheel.find("origin"))[:3, 3]
    if abs(calf_offset[0]) > 1e-9 or abs(wheel_offset[0]) > 1e-9:
        raise ValueError("This analytic solver expects zero sagittal X link offsets")
    l1, l2 = -calf_offset[2], -wheel_offset[2]
    shoulder_z = origin(hip.find("origin"))[2, 3] + origin(thigh.find("origin"))[2, 3]
    down = height + shoulder_z - radius
    cosine = (down * down - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if not -1 <= cosine <= 1 or down <= 0:
        raise ValueError(f"{leg}: height {height:.6f} m is unreachable at hip=0; maximum {radius+l1+l2-shoulder_z:.6f} m")
    knee = math.acos(cosine)
    # x = -l1*sin(thigh) - l2*sin(thigh+calf) = 0。
    upper = math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
    sign = 1 if leg.startswith("F") else -1
    return sign * upper, -sign * knee, float(radius + l1 + l2 - shoulder_z)


def binary_stl_vertices(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, help="Base_link origin height above flat ground, in metres.")
    parser.add_argument("--wheel_radius", type=float, default=0.0875)
    parser.add_argument("--verify_default_pose", action="store_true", help="Check the actual route defaults instead of just the solved angles.")
    parser.add_argument("--mesh_check", action="store_true", help="Also measure all local collision STL ground clearances.")
    args = parser.parse_args()
    defaults = runpy.run_path(str(POSE_CONFIG))
    height = defaults["LIGHTHW_DEFAULT_BASE_HEIGHT"] if args.height is None else args.height
    if not math.isfinite(height) or not math.isfinite(args.wheel_radius) or args.wheel_radius <= 0:
        parser.error("height must be finite and wheel_radius must be positive and finite")
    root = ET.parse(URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    angles = {name: 0.0 for name in joints}
    solutions = {}
    try:
        for leg in LEGS:
            thigh, calf, maximum = solve_leg(joints, leg, height, args.wheel_radius)
            angles[f"{leg}_thigh_joint"] = thigh
            angles[f"{leg}_calf_joint"] = calf
            solutions[leg] = {"thigh_rad": thigh, "calf_rad": calf, "thigh_deg": math.degrees(thigh), "calf_deg": math.degrees(calf), "max_height_m": maximum}
    except ValueError as exc:
        parser.error(str(exc))
    if args.verify_default_pose:
        for name in angles:
            values = [v for pattern, v in defaults["LIGHTHW_DEFAULT_JOINT_POS"].items() if re.fullmatch(pattern, name)]
            assert len(values) == 1, (name, values)
            angles[name] = values[0]
    frames = forward_kinematics(joints, angles)
    for name, joint in joints.items():
        if joint.get("type") == "continuous":
            continue
        limit = joint.find("limit")
        low, high = float(limit.get("lower")), float(limit.get("upper"))
        # 运行时 soft_joint_pos_limit_factor=0.9；同时复核默认角不触碰软限位。
        centre, half = (low + high) / 2, (high - low) * 0.9 / 2
        assert centre - half < angles[name] < centre + half, (name, angles[name], low, high)
    for leg in LEGS:
        centre = frames[f"{leg}_wheel_link"][:3, 3]
        solutions[leg]["verified_joint_angles_rad"] = [angles[f"{leg}_thigh_joint"], angles[f"{leg}_calf_joint"]]
        solutions[leg]["wheel_center_in_base_m"] = centre.tolist()
        solutions[leg]["fk_base_height_m"] = float(-centre[2] + args.wheel_radius)
        assert abs(solutions[leg]["fk_base_height_m"] - height) < 1e-6, solutions[leg]
        shoulder_x = frames[f"{leg}_thigh_link"][0, 3]
        assert abs(centre[0] - shoulder_x) < 1e-6, (leg, centre[0], shoulder_x)
    result = {"urdf": str(URDF), "target_base_height_m": height, "wheel_radius_m": args.wheel_radius, "verified_route_defaults": args.verify_default_pose, "legs": solutions}
    if args.verify_default_pose:
        clearance = defaults["LIGHTHW_DEFAULT_SPAWN_HEIGHT"] - height
        assert clearance >= 0.01
        result["spawn_wheel_clearance_m"] = clearance
    if args.mesh_check:
        clearances = {}
        for link in root.findall("link"):
            lows = []
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    raise ValueError("Mesh check expects collision meshes")
                vertices = binary_stl_vertices((URDF.parent / mesh.get("filename")).resolve())
                vertices = vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                transform = frames[link.get("name")] @ origin(collision.find("origin"))
                lows.append(float((vertices @ transform[:3, :3].T + transform[:3, 3])[:, 2].min()) + height)
            if lows:
                clearances[link.get("name")] = min(lows)
        assert min(clearances.values()) > -1e-5, clearances
        result["collision_mesh_ground_clearance_m"] = clearances
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
