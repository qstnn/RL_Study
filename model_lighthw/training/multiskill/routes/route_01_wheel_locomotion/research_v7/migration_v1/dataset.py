"""只读取完整成功的普通GPU上下坡教师；没有BC/PPO/仿真入口。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .contract import CONTRACT, REPO, V7, SPEC_PATH, POSITION_SOURCE, masked_observations, sha256, validate_runtime
else:
    from contract import CONTRACT, REPO, V7, SPEC_PATH, POSITION_SOURCE, masked_observations, sha256, validate_runtime


class EvidenceError(ValueError):
    """缺证据或失败即关闭，不能产生成功数据。"""


def _json(path):
    return json.loads(Path(path).read_text())


def verify_snapshot(run, required=()):
    """核对运行时封存源；不要求历史档案等同当前研究控制器。"""
    combined = {}
    for filename in ('source_manifest.json', 'research_source_manifest.json'):
        manifest = _json(run/filename)
        if not manifest:
            raise EvidenceError('Empty source manifest: '+filename)
        for relative, expected in manifest.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise EvidenceError('Source manifest contains an unsafe path')
            archived = run/'source_snapshot'/path
            if not archived.is_file() or sha256(archived) != expected:
                raise EvidenceError('Archived execution source changed: '+relative)
            if relative in combined and combined[relative] != expected:
                raise EvidenceError('Source manifests disagree: '+relative)
            combined[relative] = expected
    missing = set(required)-combined.keys()
    if missing:
        raise EvidenceError('Required execution source absent from manifests: '+', '.join(sorted(missing)))
    return combined


def required_execution_sources():
    """当前教师入口实际执行的关键源；避免空/删减manifest冒充完整封存。"""
    files = [SPEC_PATH, POSITION_SOURCE, REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf']
    files += [SPEC_PATH.parent/name for name in ('env.py', 'env_cfg.py', 'cart.py', 'commands.py', 'mdp.py', 'robot.py', 'torque.py')]
    files += [V7/'scripts'/name for name in ('worker.py', 'loaded_cart_worker.py', 'ordinary_entry_controller.py',
        'rounded_cart.py', 'model.py', 'partial_support_control.py', 'wheel_load_control.py', 'real_cart_audit.py', 'rolling_load_audit.py')]
    return [str(path.relative_to(REPO)) for path in files]


def validate_physics(request, runtime, initial, solver, geometry):
    """只接受普通零速初态及实际GPU16/4；不将预承重/继续态改名通过。"""
    physics = request.get('execution_physics', {})
    if request.get('initial_state_mode') != 'ordinary' or initial.get('mode') != 'ordinary':
        raise EvidenceError('Only actual ordinary entry episodes are accepted')
    if not str(physics.get('device', '')).startswith('cuda') or initial.get('cuda_available') is not True or not str(initial.get('robot_tensor_device', '')).startswith('cuda'):
        raise EvidenceError('Only confirmed GPU teacher episodes are accepted')
    if request.get('seed') != 43:
        raise EvidenceError('Current V7 physical trace schema is seed43_case00; review schema before new seeds')
    if request.get('slope_deg') not in (25., -20.) or request.get('physical_cart_mass_kg') != 40:
        raise EvidenceError('Expected +25/-20 degree, 40kg heavy towing teacher')
    if (request.get('noise') is not False or not request.get('physical_trailer_connected')
            or request.get('actor_inference') is not False or request.get('ppo_updates') != 0):
        raise EvidenceError('Expected a clean executed teacher episode without PPO')
    expected = dict(hitch_height_m=.04, robot_hitch_offset_m=.2, ground_friction=3.,
                    rolling_coefficient=.02, base_com_shift_m=0., robot_mass_variant=False)
    if any(physics.get(key) != value for key, value in expected.items()) or request.get('ground_friction') != 3.:
        raise EvidenceError('Teacher physical assumptions do not match the accepted research line')
    if not request.get('rounded_cart_wheels') or geometry.get('variant') != 'spherical_tread_passive_wheels':
        raise EvidenceError('Expected the declared rounded passive cart asset')
    if geometry.get('total_mass_kg') != 40. or geometry.get('wheel_radius_m') != .14:
        raise EvidenceError('Rounded cart mass/radius differs')
    if [request.get('position_iterations'), request.get('velocity_iterations')] != [16, 4] or request.get('cart_solver_iterations') != [16, 4]:
        raise EvidenceError('Both robot and cart require the same GPU16/4 physical configuration')
    for suffix in ('/Robot/Base_link', '/Trailer/cart_base'):
        entries = [value for key, value in solver.items() if key.endswith(suffix)]
        if len(entries) != 1 or any(entries[0].get('physxArticulation:solver'+name+'IterationCount') != target
                                    for name, target in [('Position', 16), ('Velocity', 4)]):
            raise EvidenceError('Actual USD articulation solver differs: '+suffix)
    scene = solver.get('/physicsScene', {})
    if scene.get('physxScene:maxPositionIterationCount') != 16 or scene.get('physxScene:maxVelocityIterationCount') != 4:
        raise EvidenceError('Actual scene solver limits differ')
    angle = np.deg2rad(request['slope_deg'])
    normal = np.array([-np.sin(angle), 0., np.cos(angle)])
    default = np.array([0., .6, -1.5, 0., .6, -1.5, 0., -.6, 1.5, 0., -.6, 1.5, 0., 0., 0., 0.])
    root = np.asarray(initial.get('robot_root_state', []), dtype=float)
    cart = np.asarray(initial.get('cart_root_state', []), dtype=float)
    origin = np.asarray(initial.get('env_origin', []), dtype=float)
    initial_q = np.asarray(initial.get('joint_position_action_order', []), dtype=float)
    quaternion = np.array([np.cos(angle/2.), 0., -np.sin(angle/2.), 0.])
    if (root.shape != (13,) or cart.shape != (13,) or origin.shape != (3,)
            or initial_q.shape != (16,) or not all(np.isfinite(x).all() for x in (root, cart, origin, initial_q))
            or initial.get('episode_step') != 0 or initial.get('cart_mass_kg') != 40.
            or not np.allclose(initial_q, default, atol=1.e-6, rtol=0.)
            or not np.allclose(root[3:7], quaternion, atol=1.e-6, rtol=0.)
            or not np.allclose(cart[3:7], quaternion, atol=1.e-6, rtol=0.)
            or not np.allclose(root[7:], 0., atol=1.e-6, rtol=0.)
            or not np.allclose(cart[7:], 0., atol=1.e-6, rtol=0.)
            or abs((root[:3]-origin)@normal-.416223) > 1.e-5):
        raise EvidenceError('Ordinary reset content does not reproduce')
    validate_runtime(runtime, request['base_network_contract'])
    for key, expected_point in [('robot_hitch_local_m', [-.38, 0., -.176223]), ('cart_hitch_local_m', [.35, 0., -.26])]:
        if not np.allclose(runtime.get(key), expected_point, atol=1.e-6, rtol=0.):
            raise EvidenceError('Actual hitch geometry differs: '+key)
    result = dict(physics)
    result.pop('device')
    result.update(robot_solver=[16, 4], cart_solver=[16, 4], mass_kg=40., geometry=geometry,
                  robot_hitch_local_m=[-.38, 0., -.176223], cart_hitch_local_m=[.35, 0., -.26])
    return result


def validate_observation_trace(policy, physics, initial, tolerance=1.e-5):
    """由1kHz前态重建最新本体观测，并逐项核对五帧历史移位。"""
    x = np.asarray(policy['observation'])
    action = np.asarray(policy['executed_action'])
    if x.ndim != 2 or x.shape[1] != 300 or len(x) < 2 or action.shape != (len(x), 16):
        raise EvidenceError('Expected aligned deployed N x 300 observations and N x 16 executed labels')
    if not np.isfinite(x).all() or not np.isfinite(action).all():
        raise EvidenceError('Nonfinite observation/action label')
    if len(physics['joint_pos']) != len(x)*20:
        raise EvidenceError('Policy and 1kHz physical trace lengths differ')
    errors = {}
    for name, layout in CONTRACT['layout'].items():
        begin, end = layout['columns']
        history = x[:, begin:end].reshape(len(x), 5, layout['width'])
        errors['history_'+name] = float(abs(history[1:, :4]-history[:-1, 1:]).max())
    q = np.vstack((initial['joint_position_action_order'], physics['joint_pos'][19::20][:-1]))
    qd = np.vstack((np.zeros(16), physics['joint_velocity'][19::20][:-1]))
    quat = np.vstack((initial['robot_root_state'][3:7], physics['body_quat_w'][19::20][:-1]))
    omega = np.vstack((np.zeros(3), physics['body_omega_b'][19::20][:-1]))
    R = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    order = CONTRACT['observation_to_action_indices']
    relative = (q-np.asarray(initial['joint_position_action_order']))[:, order]
    relative[:, CONTRACT['wheel_position_channels_in_frame']] = 0.
    expected = dict(base_ang_vel=np.clip(omega, -100., 100.)*.5,
        projected_gravity=-R[:, 2, :], joint_pos=relative,
        joint_vel=np.clip(qd[:, order], -100., 100.)*.05,
        velocity_commands=np.column_stack((policy['command_vx'], np.zeros((len(x), 2)))),
        heading_error=np.arctan2(R[:, 1, 0], R[:, 0, 0])[:, None],
        task_context=np.tile([0., 1.], (len(x), 1)))
    for name, value in expected.items():
        begin, end = CONTRACT['layout'][name]['latest']
        errors[name] = float(abs(x[:, begin:end]-value).max())
    begin, end = CONTRACT['layout']['actions']['latest']
    errors['actual_previous_action'] = float(abs(x[1:, begin:end]-np.clip(action[:-1], -100., 100.)).max())
    if any(not np.isfinite(value) or value > tolerance for value in errors.values()):
        raise EvidenceError('Deployed observation contents do not reproduce: '+repr(errors))
    return errors


def _behavior_audit(run):
    scripts = str(V7/'scripts')
    sys.path.insert(0, scripts)
    try:
        from real_cart_audit import analyze
        return analyze(run, write_report=False)
    finally:
        sys.path.remove(scripts)


def stage_labels(policy, report, speed):
    phase = np.asarray(policy['phase'])
    if phase.ndim != 1 or not np.isin(phase, [0, 1, 2, 3]).all() or report.get('command_zero_time_s') is None:
        raise EvidenceError('Incomplete or unknown executed task phases')
    tm = np.arange(len(phase))*.02
    stage = np.full(len(phase), 'wait', dtype='<U12')
    if 'ordinary_entry_active' in policy:
        stage[(phase == 0)&(policy['ordinary_entry_active'] > 0)] = 'prepare'
    stage[(phase == 1)&(policy['command_vx'] < speed-1.e-4)] = 'start'
    stage[(phase == 1)&(policy['command_vx'] >= speed-1.e-4)] = 'move'
    stage[phase == 2] = 'brake'
    stage[(phase == 3)&(tm < report['command_zero_time_s']+2.)] = 'stop_settle'
    stage[(phase == 3)&(tm >= report['command_zero_time_s']+2.)] = 'stop_hold'
    if not np.isin(['start', 'brake', 'stop_settle', 'stop_hold'], np.unique(stage)).all():
        raise EvidenceError('Complete executed startup/brake/settle/hold stages are missing')
    return stage


def read_episode(directory):
    run = Path(directory).resolve(strict=True)
    request, runtime, initial = [_json(run/name) for name in ('request.json', 'runtime_check.json', 'initial_state_runtime.json')]
    profile = validate_physics(request, runtime, initial, _json(run/'solver_runtime_usd.json'), _json(run/'rounded_cart_geometry.json'))
    sources = verify_snapshot(run, required=required_execution_sources())
    urdf = 'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    if sources.get(urdf) != request['base_network_contract']['urdf_sha256']:
        raise EvidenceError('Robot asset does not match the deployed source contract')
    cart_path = run/'assets/cart.urdf'
    profile.update(robot_urdf_sha256=sources[urdf], cart_urdf_sha256=sha256(cart_path))
    checkpoint = Path(request['checkpoint'])
    if not checkpoint.is_file() or sha256(checkpoint) != request['checkpoint_sha256']:
        raise EvidenceError('Provenance checkpoint changed')
    before = {name: sha256(run/name) for name in ('request.json', 'runtime_check.json', 'initial_state_runtime.json',
        'source_manifest.json', 'research_source_manifest.json', 'solver_runtime_usd.json', 'rounded_cart_geometry.json', 'assets/cart.urdf',
        'seed43_case00.csv', 'seed43_case00_physics_trace.npz', 'seed43_case00_policy_trace.npz', 'seed43_case00_motor_trace.npz')}
    # Never trust a saved analysis JSON or a candidate's own success boolean.
    report = _behavior_audit(run)
    if not report.get('move_brake_hold_pass'):
        failed = [key for key, value in report.get('checks', {}).items() if not value]
        raise EvidenceError('Executed teacher failed complete move/brake/hold: '+', '.join(failed))
    with np.load(run/'seed43_case00_policy_trace.npz', allow_pickle=False) as data:
        policy = dict(data)
    with np.load(run/'seed43_case00_physics_trace.npz', allow_pickle=False) as data:
        physics = {key: data[key] for key in ('joint_pos', 'joint_velocity', 'body_quat_w', 'body_omega_b')}
    errors = validate_observation_trace(policy, physics, initial)
    cases = request['cases']
    if len(cases) != 1 or cases[0]['payload_kg'] != 0. or cases[0]['towed_kg'] != 40. or cases[0]['slope_deg'] != request['slope_deg']:
        raise EvidenceError('Requested physical case disagrees with recorded nominal task context')
    stage = stage_labels(policy, report, cases[0]['speed'])
    for name, expected in before.items():
        if sha256(run/name) != expected:
            raise EvidenceError('Source trace changed while reading: '+name)
    manifest = dict(directory=str(run), case=cases[0], seed=43, slot=0, physics_profile=profile,
        source_files=before, source_manifest_sha256=sha256(run/'source_manifest.json'),
        research_source_manifest_sha256=sha256(run/'research_source_manifest.json'),
        behavior_auditor_sources={name: sha256(V7/'scripts'/name) for name in ('real_cart_audit.py', 'rolling_load_audit.py')},
        command_timing=dict(actual_move_start_s=report['move_window_s'][0],
            command_zero_time_s=report['command_zero_time_s'],
            declared_move_start_s=request.get('move_start_s'),
            source='real_cart_audit recomputed from executed phase CSV; preparation motion excluded'),
        observation_reconstruction_max_errors=errors, complete_teacher_pass=True,
        samples=len(stage), successful_labels_are_executed=True, actor_success=False)
    return dict(observation=policy['observation'], action=policy['executed_action'], stage=stage, manifest=manifest)


def read_teacher_bundle(directories):
    """全提供回合必须通过；同配置上下坡缺一不可；不做随机帧拆分。"""
    runs = [Path(path).resolve(strict=True) for path in directories]
    if len(runs) != len(set(runs)):
        raise EvidenceError('Duplicate source episode')
    episodes = [read_episode(path) for path in runs]
    if {e['manifest']['case']['slope_deg'] for e in episodes} != {25., -20.}:
        raise EvidenceError('Both ordinary GPU uphill and downhill complete teachers are required')
    profiles = {json.dumps(e['manifest']['physics_profile'], sort_keys=True) for e in episodes}
    if len(profiles) != 1:
        raise EvidenceError('Do not combine different teacher physical configurations')
    groups = {}
    arrays = {key: [] for key in ('observation', 'action', 'stage', 'episode_id', 'group_id', 'case_key')}
    for index, episode in enumerate(episodes):
        case = episode['manifest']['case']
        key = f"slope{case['slope_deg']:g}_payload0_tow40_speed{case['speed']:g}"
        group = np.array([groups.setdefault((key, str(stage)), len(groups)) for stage in episode['stage']], dtype=np.int64)
        for name in ('observation', 'action', 'stage'):
            arrays[name].append(episode[name])
        arrays['episode_id'].append(np.full(len(group), index, dtype=np.int64))
        arrays['group_id'].append(group)
        arrays['case_key'].append(np.full(len(group), key))
    packed = {key: np.concatenate(value) for key, value in arrays.items()}
    packed['observation_masked'] = masked_observations(packed['observation'])
    packed['successful'] = np.ones(len(packed['action']), dtype=bool)
    packed['cache_kind'] = np.asarray('teacher_labels_for_offline_fit_only')
    manifest = dict(schema='r1_migration_v1_teacher_data', contract=CONTRACT,
        episodes=[e['manifest'] for e in episodes], groups=[dict(group_id=g, case_key=k, stage=s) for (k, s), g in groups.items()],
        samples=len(packed['action']), cache_kind='teacher_labels_for_offline_fit_only',
        split='not_assigned; split entire episode_id jointly across all stages, never neighboring frames',
        deployment_preparation_clock_matched=False,
        student_training_ready=False, unified_actor_accepted=False, formal_ppo_authorized=False,
        reason='Successful teacher labels only; this reader does not authorize a student/PPO or certify a selected deployment command clock')
    return packed, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh data-only directory; no directory on rejection')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    try:
        arrays, manifest = read_teacher_bundle(args.runs)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, 'Teacher collection remains closed: '+str(error)+'\n')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'teacher_data.npz', **arrays)
    manifest['data_sha256'] = sha256(args.output/'teacher_data.npz')
    manifest['reader_sources'] = {str(path.relative_to(REPO)): sha256(path) for path in
        (Path(__file__).resolve(), Path(__file__).with_name('contract.py').resolve(), SPEC_PATH, POSITION_SOURCE)}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(samples=manifest['samples'], formal_ppo_authorized=False, student_training_ready=False)))


if __name__ == '__main__':
    main()
