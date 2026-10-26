"""URDF-only geometry and illustrative static loads; no PhysX/hardware feasibility claim."""

import json
import re
import xml.etree.ElementTree as ET

import numpy as np

from common import REPO, ROOT, URDF, asset_manifest, catalog, module_from_file, now, write_json


def audit():
    geometry = module_from_file("_multiskill_geometry", REPO / "scripts/tools/lighthw_standing_pose.py")
    defaults = module_from_file("_multiskill_pose", REPO / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_lighthw/default_pose.py")
    tree = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in tree.findall("joint")}
    angles = {name: next(v for pattern, v in defaults.LIGHTHW_DEFAULT_JOINT_POS.items() if re.fullmatch(pattern, name)) for name in joints}
    frames = geometry.forward_kinematics(joints, angles)
    masses, centers, mesh_bounds, radii = {}, {}, {}, {}
    for link in tree.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        masses[name] = float(inertial.find("mass").get("value"))
        centers[name] = (frames[name] @ geometry.origin(inertial.find("origin")))[:3, 3]
        points = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            vertices = geometry.binary_stl_vertices((URDF.parent / mesh.get("filename")).resolve())
            vertices = vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            transform = geometry.origin(collision.find("origin"))
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            if "wheel" in name:
                radii[name] = float(np.linalg.norm(vertices[:, [0,2]], axis=1).max())
            points.append(vertices @ frames[name][:3, :3].T + frames[name][:3, 3])
        points = np.concatenate(points)
        mesh_bounds[name] = [points.min(0).tolist(), points.max(0).tolist()]
    mass = sum(masses.values())
    com = sum(masses[n] * centers[n] for n in masses) / mass
    contacts = np.array([frames[n][:3, 3] for n in catalog.WHEEL_BODIES])
    radius = float(np.mean(list(radii.values())))
    contacts[:, 2] -= radius
    # Four nearly rectangular contacts; each case solves vertical-force force/moment balance.
    matrix = np.vstack((np.ones(4), contacts[:, 0], contacts[:, 1]))
    loads = []
    for payload in (0., 10., 15., 65.):
        payload_com = np.array([0., 0., .20])  # Declared example, NOT measured hardware mounting geometry.
        total_com = (mass*com + payload*payload_com) / (mass+payload)
        right = (mass+payload)*9.81*np.array([1., total_com[0], total_com[1]])
        forces = np.linalg.lstsq(matrix, right, rcond=None)[0]
        if not np.allclose(matrix @ forces, right, atol=1e-8):
            raise ValueError("Static force balance failed")
        torques = {}
        for name in catalog.LEG_NAMES:
            joint = joints[name]
            parent = joint.find("parent").get("link")
            transform = frames[parent] @ geometry.origin(joint.find("origin"))
            axis = transform[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            point = transform[:3, 3]
            leg = name[:2]
            children = {joint.find("child").get("link")}
            while True:
                expanded = children | {j.find("child").get("link") for j in joints.values() if j.find("parent").get("link") in children}
                if children == expanded:
                    break
                children = expanded
            moment = sum((np.cross(centers[n]-point, [0.,0.,-masses[n]*9.81]) for n in children), np.zeros(3))
            wheel_index = list(catalog.WHEEL_BODIES).index(leg+"_wheel_link")
            moment += np.cross(contacts[wheel_index]-point, [0.,0.,forces[wheel_index]])
            torques[name] = float(-axis @ moment)
        loads.append({"payload_kg": payload, "assumed_payload_com_in_base_m": payload_com.tolist(),
                      "total_com_in_base_m": total_com.tolist(), "vertical_wheel_forces_N": forces.tolist(),
                      "all_vertical_forces_nonnegative": bool((forces >= 0).all()),
                      "leg_static_torques_Nm": torques, "max_abs_leg_torque_Nm": max(abs(v) for v in torques.values())})
    return {"checked_at": now(), "evidence": "static_URDF_FK_and_ideal_vertical_contact_equilibrium",
            "assets": asset_manifest(), "default_joint_angles": angles, "base_com_m": com.tolist(),
            "wheel_contacts_in_base_m": contacts.tolist(), "wheel_collision_radii_m": radii,
            "default_collision_bbox_m": [np.min([b[0] for b in mesh_bounds.values()],0).tolist(), np.max([b[1] for b in mesh_bounds.values()],0).tolist()],
            "hip_zero_straight_leg_height_upper_bound_m": radius+.21+.25,
            "static_cases": loads, "hardware_feasible": None,
            "limitations": ["fixed original level pose; no collision-aware reachability sweep",
                "payload mount assumed at base xyz=(0,0,0.20)m", "minimum-norm vertical contact distribution, not unique or measured",
                "no friction/traction/dynamic/self-collision/structural/thermal/continuous actuator certification",
                "800mm access and acrobatics remain unresolved; nominal static torques do not certify65kg"]}


if __name__ == "__main__":
    result = audit()
    write_json(ROOT / "records/model_audit.json", result)
    print(json.dumps({k: result[k] for k in ("base_com_m", "wheel_contacts_in_base_m", "wheel_collision_radii_m", "default_collision_bbox_m", "hip_zero_straight_leg_height_upper_bound_m")}, indent=2))
    print("Illustrative static torque maxima:", [(r["payload_kg"], r["max_abs_leg_torque_Nm"]) for r in result["static_cases"]])
