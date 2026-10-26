"""规范URDF前/后双支撑候选：FK、全身COM、轮接触和理想静态关节力矩。"""
import importlib.util
import math
import xml.etree.ElementTree as ET
import numpy as np
from common import REPO, URDF, sha
from two_leg_balance.spec import LEG_JOINTS, WHEELS, WHEEL_BODIES, STAND, LOW, HIGH


def audit():
    if sha(URDF) != "781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9":
        raise ValueError("Canonical URDF changed; review and version the R4 contract before training")
    path = REPO / "scripts/tools/lighthw_standing_pose.py"
    spec = importlib.util.spec_from_file_location("_r4_geometry", path)
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    tree = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in tree.findall("joint")}
    angles = dict(zip(LEG_JOINTS, STAND))
    angles.update(dict.fromkeys(WHEELS, 0.))
    for i, name in enumerate(LEG_JOINTS):
        limits = joints[name].find("limit")
        if not float(limits.get("lower")) <= LOW[i] <= STAND[i] <= HIGH[i] <= float(limits.get("upper")):
            raise ValueError(f"R4 action interval is outside URDF: {name}")
    frames = geo.forward_kinematics(joints, angles)
    masses, centers, points = {}, {}, {}
    for link in tree.findall("link"):
        name = link.get("name")
        inertia = link.find("inertial")
        masses[name] = float(inertia.find("mass").get("value"))
        centers[name] = (frames[name] @ geo.origin(inertia.find("origin")))[:3, 3]
        vertices = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            v = geo.binary_stl_vertices((URDF.parent / mesh.get("filename")).resolve())
            v = v * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            transform = frames[name] @ geo.origin(collision.find("origin"))
            vertices.append(v @ transform[:3, :3].T + transform[:3, 3])
        points[name] = np.concatenate(vertices)
    mass = sum(masses.values())
    com = sum(masses[n] * centers[n] for n in masses) / mass
    wheels = np.array([frames[n][:3, 3] for n in WHEEL_BODIES])
    radius = .0875
    candidates = []
    for mode, support in (("front", [0, 1]), ("rear", [2, 3])):
        delta = com - wheels[support].mean(0)
        pitch = math.atan2(-delta[0], delta[2])
        rotation = geo.rotation([0., 1., 0.], pitch)
        wc = wheels @ rotation.T
        height = radius - wc[support, 2].mean()
        wc[:, 2] += height
        gc = {n: rotation @ c + [0., 0., height] for n, c in centers.items()}
        world_com = rotation @ com + [0., 0., height]
        contacts = wc.copy()
        contacts[:, 2] -= radius
        matrix = np.vstack((np.ones(2), contacts[support, 1]))
        forces = np.zeros(4)
        forces[support] = np.linalg.solve(matrix, mass * 9.81 * np.array([1., world_com[1]]))
        torques = {}
        for name in LEG_JOINTS:
            joint = joints[name]
            tf = frames[joint.find("parent").get("link")] @ geo.origin(joint.find("origin"))
            axis = rotation @ tf[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            origin = rotation @ tf[:3, 3] + [0., 0., height]
            children = {joint.find("child").get("link")}
            while True:
                expanded = children | {j.find("child").get("link") for j in joints.values()
                                       if j.find("parent").get("link") in children}
                if expanded == children:
                    break
                children = expanded
            moment = sum((np.cross(gc[n] - origin, [0., 0., -masses[n] * 9.81])
                          for n in children), np.zeros(3))
            index = list(WHEEL_BODIES).index(name[:2] + "_wheel_link")
            moment += np.cross(contacts[index] - origin, [0., 0., forces[index]])
            torques[name] = float(-axis @ moment)
        clearance = {n: float((v @ rotation.T)[:, 2].min() + height) for n, v in points.items()}
        nonwheel = min(v for n, v in clearance.items() if n not in WHEEL_BODIES)
        swing = [i for i in range(4) if i not in support]
        max_torque = max(abs(t) for t in torques.values())
        residual = np.cross(contacts - world_com, np.column_stack((np.zeros(4), np.zeros(4), forces))).sum(0)
        # 这里只评估一个原关节姿态的倾斜候选；零速PD需偏置，不能将FK姿态直接当可执行控制器。
        candidates.append(dict(
            mode=mode, pitch_rad=pitch, pitch_deg=math.degrees(pitch), root_height_m=height,
            leg_angles_rad=list(STAND), com_m=world_com.tolist(), wheel_centers_m=wc.tolist(),
            normal_forces_N=forces.tolist(), static_torques_Nm=torques,
            max_static_leg_torque_Nm=max_torque, nonwheel_ground_clearance_m=nonwheel,
            swing_clearance_m=contacts[swing, 2].tolist(), moment_residual_Nm=residual.tolist(),
            candidate_pass=bool(nonwheel > .02 and (forces[support] > 5).all()
                                and max_torque < 48. and np.linalg.norm(residual) < .05),
        ))
    return dict(urdf_sha256=sha(URDF), mass_kg=mass, candidates=candidates,
                evidence="CPU URDF FK + ideal vertical-force static equilibrium",
                limitations=["no dynamic transition or stability proof", "no self-collision mesh intersection test",
                             "no friction disturbance, motor thermal or hardware certification",
                             "pose is a reward target, never a root-state injection or teacher controller"])


if __name__ == "__main__":
    import json
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
