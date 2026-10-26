"""学生物理配置的纯Python复核；仅消费实际读回字典，不导入/启动Isaac。"""
from pathlib import Path
import numpy as np

from .contract import CONTRACT, validate_runtime, sha256
from .dataset import EvidenceError


def preparation_step_offset(preparation_seconds, policy_dt=.02):
    """原命令内置3s准备；只计算从reset开始的纯时钟偏移，不读取状态。"""
    value = float(preparation_seconds)
    if not np.isfinite(value) or value < 3. or not np.isfinite(policy_dt) or policy_dt <= 0.:
        raise EvidenceError('Preparation must be finite and at least the original 3 seconds')
    steps = round(value/policy_dt)
    if abs(steps*policy_dt-value) > 1.e-8:
        raise EvidenceError('Preparation must lie exactly on the original control-step clock')
    return steps-round(3./policy_dt)


def check_source_preparation(manifest, preparation_seconds, case_slope=None):
    """真实起点匹配所选任务；无case时保留全bundle检查，不读取运行时ready。"""
    preparation_step_offset(preparation_seconds)
    episodes = manifest['teacher_bundle']['episodes']
    if {episode['case']['slope_deg'] for episode in episodes} != {25., -20.}:
        raise EvidenceError('Preparation alignment requires both complete teacher directions')
    selected = list(range(len(episodes)))
    if case_slope is not None:
        value = float(case_slope)
        selected = [index for index,episode in enumerate(episodes) if episode['case']['slope_deg'] == value]
        if not np.isfinite(value) or not selected:
            raise EvidenceError('No complete source direction for selected preparation case')
    starts = [episodes[index].get('command_timing', {}).get('actual_move_start_s') for index in selected]
    if (any(value is None for value in starts) or not np.isfinite(starts).all()
            or not np.allclose(starts, preparation_seconds, atol=1.e-8, rtol=0.)):
        raise EvidenceError('Teacher actual preparation clocks do not all match selected student preparation: '+repr(starts))
    return dict(protocol='fixed_clock_from_reset', preparation_seconds=float(preparation_seconds),
        teacher_actual_move_starts_s=starts, teacher_clock_matched=True,
        alignment_scope='all_bundle_episodes' if case_slope is None else 'selected_task_direction',
        selected_case_slope_deg=None if case_slope is None else float(case_slope),
        matched_source_episode_ids=selected, fixed_before_reset=True, runtime_ready_or_contact_gate=False,
        student_clock_runtime_verified=False)


def check_pre_action_preparation(elapsed_s, phase, command_vx, preparation_seconds):
    """每次Actor推理前只检查命令时钟；不使用姿态/接触/教师ready，也不修改动作。"""
    if not np.isfinite([elapsed_s, phase, command_vx, preparation_seconds]).all() or phase not in (0,1,2,3):
        raise EvidenceError('Invalid pre-action command clock evidence')
    if elapsed_s < preparation_seconds-1.e-8:
        if phase != 0 or command_vx != 0.:
            raise EvidenceError('Student command moved before the selected fixed preparation clock')
    elif phase == 0:
        raise EvidenceError('Student command did not start at the selected fixed preparation clock')
    if abs(elapsed_s-preparation_seconds) <= 1.e-8 and (phase != 1 or command_vx <= 0.):
        raise EvidenceError('Student first movement phase/command differs from the selected fixed clock')


def select_case(manifest, slope_deg):
    """从已验证教师声明取速度/制动阈值，不能把1m验收错当1.2m命令。"""
    episodes = manifest['teacher_bundle']['episodes']
    selected = [item for item in episodes if item['case']['slope_deg'] == slope_deg]
    if not selected:
        raise EvidenceError('No complete source episode for the requested student direction')
    cases = [item['case'] for item in selected]
    if any(case != cases[0] for case in cases[1:]):
        raise EvidenceError('Source episodes disagree on the requested command case')
    if cases[0]['payload_kg'] != 0. or cases[0]['towed_kg'] != 40.:
        raise EvidenceError('Initial student worker only implements the two heavy-towing anchors')
    profile = selected[0]['physics_profile']
    if any(item['physics_profile'] != profile for item in episodes):
        raise EvidenceError('Student sources contain different physics profiles')
    return dict(cases[0]), dict(profile)


def check_runtime(runtime, initial, solver, materials, assets, profile, case):
    """实际runtime/普通reset/求解器/材质/资产均匹配，仍不是行为通过。"""
    validate_runtime(runtime, CONTRACT)
    if (runtime.get('num_envs') != 1 or runtime.get('slopes_deg') != [case['slope_deg']]
            or runtime.get('payloads_kg') != [0.] or runtime.get('towed_masses_kg') != [40.]
            or runtime.get('physical_active_cart_masses_kg') != [40.]):
        raise EvidenceError('Actual student condition differs from the declared heavy-towing anchor')
    if (runtime.get('loaded_acceleration_mps2') != profile['loaded_acceleration_mps2']
            or runtime.get('command_profile') != 'slew'
            or runtime.get('hitch_height_initial_m') != profile['hitch_height_m']):
        raise EvidenceError('Actual student command/hitch configuration differs from the source profile')
    if any(runtime.get(key) is not True for key in ('physical_com_inertia_verified',
            'unloaded_inertia_urdf_verified', 'nonbase_inertia_unchanged')):
        raise EvidenceError('Actual student mass/COM/inertia verification is missing')
    if initial.get('robot_tensor_device') != 'cuda:0' or initial.get('simulation_device') != 'cuda:0':
        raise EvidenceError('Student physical evaluation requires actual CUDA tensors and simulation')
    if profile['robot_solver'] != [16, 4] or profile['cart_solver'] != [16, 4]:
        raise EvidenceError('Unreviewed source solver profile')
    for suffix in ('/Robot/Base_link', '/Trailer/cart_base'):
        entries = [value for name, value in solver.items() if name.endswith(suffix)]
        if len(entries) != 1 or any(entries[0].get('physxArticulation:solver'+key+'IterationCount') != value
                                  for key, value in [('Position', 16), ('Velocity', 4)]):
            raise EvidenceError('Actual student articulation solver mismatch: '+suffix)
    scene = solver.get('/physicsScene', {})
    if any(scene.get('physxScene:max'+key+'IterationCount') != value for key, value in [('Position', 16), ('Velocity', 4)]):
        raise EvidenceError('Actual student scene solver mismatch')
    if runtime['ground_friction'] != profile['ground_friction'] or runtime['rolling_coefficient'] != profile['rolling_coefficient']:
        raise EvidenceError('Student friction/rolling coefficient mismatch')
    for key in ('robot_hitch_local_m', 'cart_hitch_local_m'):
        if not np.allclose(runtime[key], profile[key], rtol=0., atol=1.e-6):
            raise EvidenceError('Actual student hitch mismatch: '+key)
    for name in ('robot', 'trailer'):
        values = np.asarray(materials['assets'][name], dtype=float)
        if values.shape[-1] != 2 or not np.isfinite(values).all() or not np.allclose(values, profile['ground_friction'], atol=1.e-6, rtol=0.):
            raise EvidenceError('Actual asset friction mismatch: '+name)
    if not materials.get('terrain') or any(value != dict(static=profile['ground_friction'], dynamic=profile['ground_friction'], combine='min') for value in materials['terrain'].values()):
        raise EvidenceError('Actual terrain material/friction combine mismatch')
    for name, key in [('robot', 'robot_urdf_sha256'), ('cart', 'cart_urdf_sha256')]:
        if sha256(Path(assets[name])) != profile[key]:
            raise EvidenceError('Actual asset SHA mismatch: '+name)
    slope = np.deg2rad(case['slope_deg'])
    normal = np.array([-np.sin(slope), 0., np.cos(slope)])
    quaternion = np.array([np.cos(slope/2.), 0., -np.sin(slope/2.), 0.])
    root = np.asarray(initial['robot_root_state']); cart = np.asarray(initial['cart_root_state'])
    q = np.asarray(initial['joint_position_action_order']); qd = np.asarray(initial['joint_velocity_action_order'])
    default = np.array([0., .6, -1.5, 0., .6, -1.5, 0., -.6, 1.5, 0., -.6, 1.5, 0., 0., 0., 0.])
    if (root.shape != (13,) or cart.shape != (13,) or q.shape != (16,) or qd.shape != (16,)
            or not all(np.isfinite(value).all() for value in (root, cart, q, qd))
            or not np.allclose(q, default, atol=1.e-6, rtol=0.) or not np.allclose(qd, 0., atol=1.e-7)
            or not np.allclose(root[3:7], quaternion, atol=1.e-6, rtol=0.)
            or not np.allclose(cart[3:7], quaternion, atol=1.e-6, rtol=0.)
            or not np.allclose(root[7:], 0., atol=1.e-7) or not np.allclose(cart[7:], 0., atol=1.e-7)
            or not np.allclose(root[:3], np.asarray(initial['env_origin'])+.416223*normal, atol=1.e-5, rtol=0.)
            or initial['episode_step'] != 0 or abs(initial['cart_mass_kg']-40.) > 1.e-4):
        raise EvidenceError('Student ordinary reset differs from verified default zero-speed initial state')
    expected_cart = np.asarray(initial['env_origin'])+.30*normal-1.73*np.array([np.cos(slope), 0., np.sin(slope)])
    if not np.allclose(cart[:3], expected_cart, atol=1.e-5, rtol=0.):
        raise EvidenceError('Student cart position includes an unreviewed preload/reset override')
    if not np.allclose(initial['cart_joint_velocity'], 0., atol=1.e-7) or initial['command_vx'] != 0.:
        raise EvidenceError('Student must start with zero cart wheel speed and zero command')
    return dict(runtime_profile_verified=True, ordinary_reset_verified=True,
        student_behavior_accepted=False, formal_ppo_authorized=False)


def check_action_mapping(actor_raw, manager_raw, leg_target, wheel_target, default_legs, leg_clip, wrapper_clip=None):
    """核对Actor原始输出→原wrapper限幅→原动作manager映射；拒绝任何控制器增量。"""
    raw, applied, legs, wheels, default, limits = map(np.asarray,
        (actor_raw, manager_raw, leg_target, wheel_target, default_legs, leg_clip))
    n = len(raw)
    if (raw.shape != (n,16) or applied.shape != raw.shape or legs.shape != (n,12)
            or wheels.shape != (n,4) or default.shape != (n,12) or limits.shape != (n,12,2)
            or not all(np.isfinite(value).all() for value in (raw,applied,legs,wheels,default,limits))):
        raise EvidenceError('Malformed student action/mapping evidence')
    expected = raw if wrapper_clip is None else np.clip(raw, -wrapper_clip, wrapper_clip)
    target = expected[:,:12]*np.array([.08,.14,.14]*4, dtype=raw.dtype)+default
    target = np.clip(target, limits[:,:,0], limits[:,:,1])
    wheel = np.clip(expected[:,12:]*5., -50., 50.)
    if not np.array_equal(applied, expected):
        raise EvidenceError('Student manager action differs from Actor plus declared original wrapper clip')
    if not np.allclose(legs, target, atol=1.e-6, rtol=0.) or not np.allclose(wheels, wheel, atol=1.e-6, rtol=0.):
        raise EvidenceError('Student action manager mapping differs from the original deployed contract')
    return dict(actor_to_manager_max_error=float(abs(applied-expected).max()),
        leg_mapping_max_error=float(abs(legs-target).max()), wheel_mapping_max_error=float(abs(wheels-wheel).max()),
        action_override=False)
