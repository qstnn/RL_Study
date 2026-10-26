"""R1 isolated posture screen and bounded no-PPO physical trials.

The static LP is only a candidate filter. Isaac Lab's native contact and the
existing distance/stop/hold evaluator remain the behavioral authority.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from model import Model, RADIUS
from pose_search import basis, force_lp
from research_common import BASELINE, LEGS, ROOT, ROUTE, launch
from settings import GROUND_FRICTION


REFERENCE_TRACE = ROUTE / 'unified_v5/results/2026-10-19_towing_recovery/control_screen_02/seed43_case00_policy_trace.npz'
CASES = ROOT / 'results/2026-10-22_no_ppo_1m_search/case_plus25_40kg_move.json'
TEACHER = ROOT / 'results/2026-10-22_no_ppo_1m_search/teacher_yaw10_balance001.json'
HITCH_LOCAL = np.array([-.38, 0., .07 - .416223])
MOTOR_LIMITS = np.r_[np.full(12, 96.), np.full(4, 48.)]
LEG_MARGIN_RAD = 1.e-5


def candidate_grid():
    """Include old best and low-body retest alongside new nose-up postures."""
    points = {(0.40, 0., 0.), (0.38, 0., 0.)}
    points.update(itertools.product((.39, .40, .41), (-6., -3.), (0., .02)))
    points.update(itertools.product((.39, .40, .41), (1., 2., 3.), (0., .02)))
    points.update({(.40, 0., .02), (.40, 0., .04)})
    return sorted(points)


def reference_wheels(model, trace=REFERENCE_TRACE):
    q = np.load(trace)['joint_pos'][100:140].mean(0).astype(float)
    state = model.state(np.zeros(3), np.array([1., 0., 0., 0.]), q)
    centers = np.array([state['trans'][leg + '_wheel_link'][:3, 3] for leg in LEGS])
    centers[:, 2] = RADIUS
    return q, centers


def solve_wheel_pose(model, reference_q, wheel_centers, height_m, pitch_deg, forward_shift_m):
    """Keep each wheel center fixed while changing body height, pitch and COM lane."""
    q = reference_q.copy()
    orientation = Rotation.from_rotvec([0., -np.deg2rad(pitch_deg), 0.])
    quat = np.roll(orientation.as_quat(), 1)
    position = np.array([forward_shift_m, 0., height_m])
    errors = []
    for k, leg in enumerate(LEGS):
        indices = np.arange(3*k, 3*k+3)
        def residual(values):
            trial = q.copy()
            trial[indices] = values
            state = model.state(position, quat, trial)
            return state['trans'][leg + '_wheel_link'][:3, 3] - wheel_centers[k]
        fit = least_squares(residual, q[indices],
            bounds=(model.limits[indices, 0] + LEG_MARGIN_RAD,
                    model.limits[indices, 1] - LEG_MARGIN_RAD),
            max_nfev=100, gtol=1.e-10, ftol=1.e-10, xtol=1.e-10)
        q[indices] = fit.x
        errors.append(float(np.linalg.norm(fit.fun)))
    margin = float(np.min(np.minimum(q[:12] - model.limits[:12, 0],
                                     model.limits[:12, 1] - q[:12])))
    return q, max(errors), margin


def evaluate_candidate(model, reference_q, wheel_centers, height_m, pitch_deg, forward_shift_m):
    ident = f'h{height_m:.2f}_p{pitch_deg:+g}_x{forward_shift_m:.2f}'
    entry = dict(id=ident, height_m=height_m, pitch_deg=pitch_deg,
                 forward_shift_m=forward_shift_m, feasible=False)
    q, error, margin = solve_wheel_pose(model, reference_q, wheel_centers,
                                        height_m, pitch_deg, forward_shift_m)
    entry.update(q=q.tolist(), ik_error_m=error, joint_limit_margin_rad=margin)
    if error > 1.e-4 or margin <= 0.:
        entry['reason'] = 'wheel_center_ik_or_joint_limit'
        return entry
    B = basis(25.)
    local_rotation = Rotation.from_rotvec([0., -np.deg2rad(pitch_deg), 0.]).as_matrix()
    world_rotation = B @ local_rotation
    quat = np.roll(Rotation.from_matrix(world_rotation).as_quat(), 1)
    position = B @ np.array([forward_shift_m, 0., height_m])
    state = model.state(position, quat, q)
    contact_matrix, _, _ = model.contacts(state, B[:, 2])
    mass_matrix, gravity = model.dynamics(state)
    mass_matrix[18:, 18:] += np.eye(4) * .00243216
    hitch = position + world_rotation @ HITCH_LOCAL
    clearance = float((hitch - B @ np.array([forward_shift_m, 0., 0.])) @ B[:, 2])
    com = B.T @ model.whole_com(state)
    entry.update(hitch_clearance_m=clearance, com_lane_m=com.tolist(), scenarios=[])
    for phase, acceleration, rolling in [('static', 0., False),
                                         ('start', .05, True),
                                         ('cruise', 0., True),
                                         ('brake', -.05, True)]:
        tension = 40 * 9.81 * (np.sin(np.deg2rad(25.)) +
                    (.02 * np.cos(np.deg2rad(25.)) if rolling else 0.)) + 42*acceleration
        external = state['jac']('Base_link', hitch)[:3].T @ (-tension * B[:, 0])
        acceleration_vector = np.zeros(22)
        acceleration_vector[:3] = acceleration * B[:, 0]
        for k in range(4):
            column = contact_matrix[3*k:3*k+3, 18+k]
            acceleration_vector[18+k] = -acceleration/(B[:, 0] @ column)
        bias = mass_matrix @ acceleration_vector + gravity - external
        solution = force_lp(contact_matrix, bias, B, GROUND_FRICTION)
        scenario = dict(slope_deg=25., phase=phase,
                        acceleration_mps2=acceleration, drawbar_signed_N=tension,
                        hitch_clearance_m=clearance, com_lane_m=com.tolist(),
                        feasible=bool(solution.success and clearance >= .01))
        if solution.success:
            forces = solution.x[:12].reshape(4, 3)
            normal = forces @ B[:, 2]
            torque = bias[6:] - contact_matrix.T[6:] @ solution.x[:12]
            scenario.update(normal_N=normal.tolist(),
                            minimum_normal_N=float(normal.min()),
                            front_normal_N=float(normal[:2].sum()),
                            rear_normal_N=float(normal[2:].sum()),
                            joint_torque_Nm=torque.tolist(),
                            max_torque_fraction=float(np.max(abs(torque)/MOTOR_LIMITS)))
        entry['scenarios'].append(scenario)
    entry['feasible'] = all(s['feasible'] for s in entry['scenarios'])
    if entry['feasible']:
        entry['worst_minimum_normal_N'] = min(s['minimum_normal_N'] for s in entry['scenarios'])
        entry['start_front_normal_N'] = entry['scenarios'][1]['front_normal_N']
    else:
        entry['reason'] = 'static_lp_or_hitch_clearance'
    return entry


def screen(output):
    model = Model()
    q, centers = reference_wheels(model)
    candidates = [evaluate_candidate(model, q, centers, *point) for point in candidate_grid()]
    selected = [candidate for candidate in candidates if candidate['feasible']]
    selected.sort(key=lambda c: (-c['worst_minimum_normal_N'],
                                 -c['start_front_normal_N'], c['id']))
    result = dict(stage='static_pose_screen', dynamic_success=False, ppo_updates=0,
                  ground_friction=GROUND_FRICTION, slope_deg=25., towed_kg=40.,
                  reference_trace=str(REFERENCE_TRACE),
                  reference_sha256=launch.sha256(REFERENCE_TRACE),
                  candidate_count=len(candidates), candidates=candidates,
                  selected=selected,
                  note='Ideal rigid-disc LP with original robot, motor limits and passive trailer. Physical success requires native-contact 1 m plus stop/hold validation.')
    launch.write_json(output, result)
    return result


def trial_command(pose_file, candidate_id, output, seconds, device):
    if not 0 < seconds <= 15.:
        raise ValueError('Only bounded physical diagnostics <=15 s are supported')
    poses = json.loads(Path(pose_file).read_text())
    matches = [i for i, p in enumerate(poses['selected']) if p['id'] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f'Pose {candidate_id} must occur exactly once among feasible selected poses')
    return [sys.executable, '-u', str(ROOT/'scripts/worker.py'),
            '--cases', str(CASES), '--output', str(output),
            '--checkpoint', str(BASELINE), '--seed', '43',
            '--device', device, '--hitch-height', '.07',
            '--ground-friction', '3', '--loaded-acceleration', '.05',
            '--pose-file', str(pose_file), '--pose-index', str(matches[0]),
            '--teacher-options', str(TEACHER), '--seconds', str(seconds),
            '--stop-on-first-failure']


def summarize_trial(directory):
    """Use native 1 kHz forces and cart root motion, not robot or rod travel."""
    directory = Path(directory)
    case = json.loads((directory/'evaluation.json').read_text())['cases'][0]
    physics = np.load(directory/'seed43_case00_physics_trace.npz')
    policy = np.load(directory/'seed43_case00_policy_trace.npz')
    phases = policy['phase'].reshape(-1)
    moving = np.flatnonzero(phases >= 1)
    if not len(moving):
        raise ValueError(f'No moving phase in {directory}')
    first = int(moving[0]) * 20  # 50 Hz policy / 1 kHz physics
    B = basis(25.)
    cart = physics['cart_pos_w']
    terminal = float(case['observed']['cart_progress_m'])
    # The evaluator owns the movement-start anchor. Reconstruct its lane
    # coordinate from the terminal reading, retaining its exact convention.
    progress = (cart-cart[-1]) @ B[:, 0] + terminal
    normals = physics['native_ground_normal_w'][first:] @ B[:, 2]
    contacts = physics['native_contact_count'][first:] > 0
    motor = json.loads((directory/'seed43_case00_motor.json').read_text())
    return dict(directory=str(directory), samples=int(case['samples']),
                duration_s=float(case['observed']['duration_s']),
                success=bool(case['success']),
                failure_reason=case['failure_reason'],
                teacher_failure=case.get('teacher_failure'),
                terminal_cart_root_progress_m=terminal,
                max_cart_root_progress_m=float(progress[first:].max()),
                min_hitch_clearance_m=float(case['observed']['minimum_hitch_clearance_m']),
                peak_tilt_deg=float(case['observed']['peak_tilt_deg']),
                four_wheel_native_contact_fraction=float(contacts.all(axis=1).mean()),
                per_wheel_native_contact_fraction=contacts.mean(axis=0).tolist(),
                mean_native_normal_N=normals.mean(axis=0).tolist(),
                peak_motor_fraction=max(float(j['peak_Nm'])/float(j['limit_Nm'])
                                        for j in motor['joints']),
                physical_sample_hz=1000, moving_start_s=first/1000.,
                ppo_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    static = sub.add_parser('screen')
    static.add_argument('--output', type=Path, required=True)
    physical = sub.add_parser('trial')
    physical.add_argument('--pose-file', type=Path, required=True)
    physical.add_argument('--candidate-id', required=True)
    physical.add_argument('--output', type=Path, required=True)
    physical.add_argument('--seconds', type=float, default=15.)
    physical.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    summary = sub.add_parser('summarize')
    summary.add_argument('--output', type=Path, required=True)
    summary.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    if args.mode == 'screen':
        result = screen(args.output)
        print('STATIC', len(result['selected']), '/', result['candidate_count'],
              'feasible; output', args.output, flush=True)
        for candidate in result['selected']:
            print(candidate['id'], 'front_N', round(candidate['start_front_normal_N'], 2),
                  'min_N', round(candidate['worst_minimum_normal_N'], 2),
                  'clear_m', round(candidate['hitch_clearance_m'], 4), flush=True)
        return
    if args.mode == 'summarize':
        rows = [summarize_trial(directory) for directory in args.directories]
        launch.write_json(args.output, dict(stage='bounded_posture_physics',
            all_passed=all(row['success'] for row in rows), ppo_updates=0,
            rows=rows))
        for row in rows:
            print(Path(row['directory']).name, row['duration_s'],
                  row['max_cart_root_progress_m'], row['failure_reason'], flush=True)
        return
    command = trial_command(args.pose_file, args.candidate_id,
                            args.output, args.seconds, args.device)
    print('PHYSICAL', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
