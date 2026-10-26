"""Solve an original-URDF front-support static pose at exactly +90 degrees."""
from pathlib import Path
import importlib.util
import itertools
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[6]
URDF = REPO / 'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
spec = importlib.util.spec_from_file_location('r4b_fk', REPO / 'scripts/tools/lighthw_standing_pose.py')
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)
LEGS = ('FR', 'FL', 'RR', 'RL')
JOINTS = tuple(f'{leg}_{part}_joint' for leg in LEGS for part in ('hip', 'thigh', 'calf'))
WHEELS = tuple(f'{leg}_wheel_link' for leg in LEGS)
STAND = np.array([0., .6, -1.5] * 2 + [0., -.6, 1.5] * 2)
LOW = np.array([-.35, -1.45, -2.4] * 2 + [-.35, -1.45, .12] * 2)
HIGH = np.array([.35, 1.45, -.12] * 2 + [.35, 1.45, 2.4] * 2)


def action_for_pose(q):
    delta = np.asarray(q) - STAND
    return delta / np.where(delta >= 0., HIGH - STAND, STAND - LOW)


class Geometry:
    def __init__(self):
        root = ET.parse(URDF).getroot()
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.mass, self.centers, self.points = {}, {}, {}
        for link in root.findall('link'):
            name = link.get('name'); inertia = link.find('inertial')
            self.mass[name] = float(inertia.find('mass').get('value'))
            self.centers[name] = fk.origin(inertia.find('origin'))[:3, 3]
            parts = []
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                vertices = fk.binary_stl_vertices((URDF.parent / mesh.get('filename')).resolve())
                vertices = vertices * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                tf = fk.origin(collision.find('origin'))
                parts.append(vertices @ tf[:3, :3].T + tf[:3, 3])
            vertices = np.concatenate(parts)
            # AABB corners conservatively bound mesh ground clearance in every pose.
            self.points[name] = np.array(list(itertools.product(*zip(vertices.min(0), vertices.max(0)))))
        self.rotation = fk.rotation([0., 1., 0.], math.pi / 2)
        self.total_mass = sum(self.mass.values())

    @staticmethod
    def angles(x):
        return np.array([0., x[0], x[1]] * 2 + [0., x[2], x[3]] * 2)

    def evaluate(self, x, torques=False):
        q = self.angles(x)
        angles = dict(zip(JOINTS, q)); angles.update({f'{leg}_wheel_joint': 0. for leg in LEGS})
        frames = fk.forward_kinematics(self.joints, angles)
        wheels = np.stack([frames[name][:3, 3] for name in WHEELS]) @ self.rotation.T
        height = .0875 - wheels[:2, 2].mean()
        wheels[:, 2] += height
        centers = {name: self.rotation @ (frames[name][:3, :3] @ c + frames[name][:3, 3])
                   + [0., 0., height] for name, c in self.centers.items()}
        com = sum(self.mass[name] * centers[name] for name in centers) / self.total_mass
        clearance = min(((p @ frames[name][:3, :3].T + frames[name][:3, 3]) @ self.rotation.T)[:, 2].min() + height
                        for name, p in self.points.items() if name not in WHEELS)
        gap = float(com[0] - wheels[:2, 0].mean())
        row = dict(leg_angles_rad=q.tolist(), pitch_deg=90., pitch_rad=math.pi / 2,
                   root_height_m=float(height), signed_support_gap_m=gap,
                   com_m=com.tolist(), wheel_centers_m=wheels.tolist(),
                   nonwheel_aabb_ground_clearance_m=float(clearance),
                   rear_wheel_clearance_m=(wheels[2:, 2] - .087501).tolist())
        if not torques:
            return row
        contacts = wheels.copy(); contacts[:, 2] -= .0875
        forces = np.zeros(4)
        forces[:2] = np.linalg.solve(np.vstack((np.ones(2), contacts[:2, 1])),
                                     self.total_mass * 9.81 * np.array([1., com[1]]))
        torque = []
        for name in JOINTS:
            j = self.joints[name]
            tf = frames[j.find('parent').get('link')] @ fk.origin(j.find('origin'))
            axis = self.rotation @ tf[:3, :3] @ np.fromstring(j.find('axis').get('xyz'), sep=' ')
            origin = self.rotation @ tf[:3, 3] + [0., 0., height]
            children = {j.find('child').get('link')}
            while True:
                expanded = children | {v.find('child').get('link') for v in self.joints.values()
                                       if v.find('parent').get('link') in children}
                if expanded == children: break
                children = expanded
            moment = sum((np.cross(centers[n] - origin, [0., 0., -self.mass[n] * 9.81])
                          for n in children), np.zeros(3))
            leg = LEGS.index(name[:2])
            moment += np.cross(contacts[leg] - origin, [0., 0., forces[leg]])
            torque.append(float(-axis @ moment))
        pd_target = q + np.array(torque) / 80.
        row.update(static_torques_Nm=torque, pd_targets_rad=pd_target.tolist(),
                   pd_action=action_for_pose(pd_target).tolist(), normal_forces_N=forces.tolist(),
                   max_static_leg_torque_Nm=max(map(abs, torque)))
        row['candidate_pass'] = bool(abs(gap) < .001 and clearance > .02 and height > .18
            and min(row['rear_wheel_clearance_m']) > .03 and min(forces[:2]) > 5.
            and max(map(abs, torque)) < 48. and ((pd_target >= LOW) & (pd_target <= HIGH)).all())
        return row


def solve():
    geometry = Geometry()
    neutral = np.array([.6, -1.5, -.6, 1.5])
    def objective(x):
        r = geometry.evaluate(x)
        return float(.02 * np.sum((x - neutral)**2) + 100 * r['signed_support_gap_m']**2)
    def constraint(x):
        r = geometry.evaluate(x)
        return np.array([r['nonwheel_aabb_ground_clearance_m'] - .025,
                         r['root_height_m'] - .20, min(r['rear_wheel_clearance_m']) - .05])
    candidates = []
    for start in ([1.3, -.25, -.6, 1.5], [-1.3, -.25, -.6, 1.5],
                  [.4, -2.2, -.6, 1.5], [-.8, -1., -.6, 1.5]):
        fit = minimize(objective, start, method='SLSQP',
            bounds=[(-1.4, 1.4), (-2.35, -.17), (-1.4, 1.4), (.17, 2.35)],
            constraints=[{'type': 'eq', 'fun': lambda x: geometry.evaluate(x)['signed_support_gap_m']},
                         {'type': 'ineq', 'fun': constraint}],
            options={'maxiter': 150, 'ftol': 1e-10})
        row = geometry.evaluate(fit.x, torques=True)
        row['optimizer_success'] = bool(fit.success)
        candidates.append(row)
    valid = [r for r in candidates if r['candidate_pass']]
    if not valid:
        raise ValueError('No feasible 90-degree static candidate; do not launch training')
    chosen = min(valid, key=lambda r: r['max_static_leg_torque_Nm'])
    return dict(target_pitch_deg=90., mass_kg=geometry.total_mass, selected=chosen,
                candidates=candidates, evidence='CPU ideal static front support only',
                dynamic_reachability_verified=False, self_collision_verified=False)


if __name__ == '__main__':
    import json
    print(json.dumps(solve(), indent=2))
