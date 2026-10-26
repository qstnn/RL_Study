"""原12项空载/背载回归；复用原V5验收，不继承重拖准备时钟。"""
import importlib.util
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contract import CONTRACT, V7, sha256, validate_runtime
from .dataset import EvidenceError

PROTOCOL_SOURCE = V7.parent/'unified_v5/scripts/protocol.py'
_spec = importlib.util.spec_from_file_location('_r1_retention_original_protocol', PROTOCOL_SOURCE)
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)

DIAGNOSTIC_PHYSICS_VARIANTS = ('mu4_solver16_4','mu4_solver4_1')


def execution_profile(source_profile,diagnostic_physics_override=None):
    """Keep source provenance intact; allow only the two explicitly named contrasts."""
    if (source_profile.get('ground_friction') != 3. or source_profile.get('robot_solver') != [16,4]
            or source_profile.get('cart_solver') != [16,4]):
        raise EvidenceError('Retention source must remain the reviewed mu3 / 16-4 profile')
    actual = deepcopy(source_profile)
    if diagnostic_physics_override is None:
        return actual,None
    if diagnostic_physics_override not in DIAGNOSTIC_PHYSICS_VARIANTS:
        raise EvidenceError('Unknown diagnostic physics override; only named mu4 contrasts are permitted')
    solver = [16,4] if diagnostic_physics_override == 'mu4_solver16_4' else [4,1]
    actual.update(ground_friction=4.,robot_solver=solver.copy(),cart_solver=solver.copy())
    return actual,dict(variant=diagnostic_physics_override,ground_friction=4.,
        robot_solver=solver.copy(),cart_solver=solver.copy(),scene_solver=solver.copy(),
        current_mu3_retention_admission_eligible=False)


def cases():
    return [dict(case) for case in original.cases(1.) if case['towed_kg'] == 0.]


def select_case(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 12:
        raise EvidenceError('Retention case index must be an integer in [0,11]')
    return cases()[index]


def configure_case(cfg, case, profile):
    if case not in cases():
        raise EvidenceError('Retention case is outside the original 12 non-towing boundaries')
    cfg.fixed_slope_deg = case['slope_deg']
    cfg.fixed_payload_kg = case['payload_kg']
    cfg.fixed_towed_kg = 0.
    cfg.fixed_speed_mps = case['speed']
    cfg.command_profile = 'slew'
    cfg.unloaded_acceleration = 1.
    cfg.loaded_acceleration = .3
    cfg.eval_distance_m = 1.
    cfg.episode_length_s = original.max_steps(case)*original.DT+5.
    cfg.ground_friction = profile['ground_friction']
    cfg.rolling_coefficient = profile['rolling_coefficient']
    cfg.hitch_height_m = profile['hitch_height_m']
    cfg.observations.policy.enable_corruption = False


def check_pre_action(observation, elapsed, phase, command, case):
    obs = np.asarray(observation)
    if obs.shape != (1,300) or not np.isfinite(obs).all():
        raise EvidenceError('Retention must execute finite deployed 300-dimensional observations')
    context = np.array([case['payload_kg']/15., 0.])
    if not np.allclose(obs[0,290:300].reshape(5,2), context, atol=1.e-7, rtol=0.):
        raise EvidenceError('Retention policy nominal payload/towing history differs from actual case')
    if not np.isfinite([elapsed, phase, command]).all() or phase not in (0,1,2,3):
        raise EvidenceError('Invalid retention command-clock evidence')
    if case['speed'] == 0. or elapsed < 3.-1.e-8:
        if phase != 0 or command != 0.:
            raise EvidenceError('Retention static/initial-zero protocol changed')
    elif phase == 0:
        raise EvidenceError('Retention movement failed to begin on the original 3-second clock')


def check_runtime(runtime, initial, solver, materials, assets, profile, case, physical,*,
                  diagnostic_physics_override=None,source_profile=None):
    """核验读回数据；不将通过接线等同于实际行为通过。"""
    validate_runtime(runtime, CONTRACT)
    expected,diagnostic = execution_profile(profile if source_profile is None else source_profile,
                                            diagnostic_physics_override)
    if profile != expected:
        raise EvidenceError('Executed profile differs from the explicit source/diagnostic profile')
    mu = profile['ground_friction']; position,velocity = profile['robot_solver']
    if (runtime.get('num_envs') != 1 or runtime.get('slopes_deg') != [case['slope_deg']]
            or runtime.get('payloads_kg') != [case['payload_kg']]
            or runtime.get('towed_masses_kg') != [0.]
            or runtime.get('physical_active_cart_masses_kg') != []):
        raise EvidenceError('Actual retention case/towing assignment differs')
    if (runtime.get('unloaded_acceleration_mps2') != 1.
            or runtime.get('loaded_acceleration_mps2') != .3
            or runtime.get('command_profile') != 'slew'
            or not np.isclose(initial.get('requested_vx',np.nan),case['speed'],atol=1.e-7,rtol=0.)
            or initial.get('goal_distance_m') != 1.):
        raise EvidenceError('Original retention command speed/acceleration/distance changed')
    if any(runtime.get(key) is not True for key in ('physical_com_inertia_verified',
            'unloaded_inertia_urdf_verified', 'nonbase_inertia_unchanged')):
        raise EvidenceError('Actual payload mass/COM/inertia readback verification is missing')
    if initial.get('robot_tensor_device') != 'cuda:0' or initial.get('simulation_device') != 'cuda:0':
        raise EvidenceError('Retention requires actual GPU simulation and robot tensors')
    for suffix in ('/Robot/Base_link', '/Trailer/cart_base'):
        entries = [value for name,value in solver.items() if name.endswith(suffix)]
        if len(entries) != 1 or any(entries[0].get('physxArticulation:solver'+key+'IterationCount') != value
                for key,value in [('Position',position),('Velocity',velocity)]):
            raise EvidenceError('Actual retention articulation solver mismatch: '+suffix)
    if any(solver.get('/physicsScene',{}).get('physxScene:max'+key+'IterationCount') != value
            for key,value in [('Position',position),('Velocity',velocity)]):
        raise EvidenceError('Actual retention scene solver mismatch')
    if (runtime['ground_friction'] != mu
            or runtime['rolling_coefficient'] != profile['rolling_coefficient']):
        raise EvidenceError('Retention actual friction/rolling coefficient differs from its explicit profile')
    for name in ('robot','trailer'):
        material = np.asarray(materials['assets'][name], dtype=float)
        if material.shape[-1] != 2 or not np.isfinite(material).all() or not np.allclose(material,mu,atol=1.e-6,rtol=0.):
            raise EvidenceError('Actual retention asset friction differs: '+name)
    if not materials['terrain'] or any(value != dict(static=mu,dynamic=mu,combine='min') for value in materials['terrain'].values()):
        raise EvidenceError('Actual retention ground material differs')
    for name,key in [('robot','robot_urdf_sha256'),('cart','cart_urdf_sha256')]:
        if sha256(assets[name]) != profile[key]:
            raise EvidenceError('Actual retention asset source differs: '+name)
    for key in ('robot_hitch_local_m','cart_hitch_local_m'):
        if not np.allclose(runtime[key],profile[key],atol=1.e-6,rtol=0.):
            raise EvidenceError('Retention asset hitch differs: '+key)
    slope = np.deg2rad(case['slope_deg'])
    normal = np.array([-np.sin(slope),0.,np.cos(slope)])
    quat = np.array([np.cos(slope/2.),0.,-np.sin(slope/2.),0.])
    root,cart,q,qd = [np.asarray(initial[key]) for key in ('robot_root_state','cart_root_state',
        'joint_position_action_order','joint_velocity_action_order')]
    default = np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])
    if (root.shape != (13,) or cart.shape != (13,) or q.shape != (16,) or qd.shape != (16,)
            or not all(np.isfinite(value).all() for value in (root,cart,q,qd))
            or not np.allclose(q,default,atol=1.e-6,rtol=0.) or not np.allclose(qd,0.,atol=1.e-7)
            or not np.allclose(root[3:7],quat,atol=1.e-6,rtol=0.)
            or not np.allclose(root[7:],0.,atol=1.e-7) or not np.allclose(cart[7:],0.,atol=1.e-7)
            or not np.allclose(root[:3],np.asarray(initial['env_origin'])+.416223*normal,atol=1.e-5,rtol=0.)
            or not np.allclose(cart[1:3],[500.,1.],atol=1.e-5,rtol=0.)
            or initial['episode_step'] != 0 or initial['command_vx'] != 0.
            or not np.allclose(initial['cart_joint_velocity'],0.,atol=1.e-7)):
        raise EvidenceError('Retention ordinary reset/disconnected parked cart differs')
    masses = np.asarray(physical['masses_kg']); com = np.asarray(physical['coms'])
    inertia = np.asarray(physical['inertias']).reshape(-1,3,3)
    original_mass = np.asarray(physical['unloaded_masses_kg'])
    original_inertia = np.asarray(physical['unloaded_inertias']).reshape(-1,3,3)
    names = physical['body_names']; base_id = names.index('Base_link')
    declared = {link.get('name'):float(link.find('inertial/mass').get('value'))
        for link in ET.parse(assets['robot']).getroot().findall('link') if link.find('inertial/mass') is not None}
    if not np.allclose(original_mass,[declared[name] for name in names],atol=1.e-5,rtol=1.e-5):
        raise EvidenceError('Unloaded link mass readback differs from original robot URDF')
    expected_mass = original_mass.copy(); expected_mass[base_id] += case['payload_kg']
    expected = runtime['payload_examples'][0]
    unchanged = [i for i in range(len(names)) if i != base_id]
    if (not np.allclose(masses,expected_mass,atol=1.e-4,rtol=1.e-5)
            or not np.allclose(com[base_id,:3],expected['base_com_m'],atol=1.e-5,rtol=0.)
            or not np.allclose(inertia[base_id],expected['inertia_body_frame'],atol=1.e-5,rtol=1.e-4)
            or not np.allclose(inertia[unchanged],original_inertia[unchanged],atol=1.e-6,rtol=1.e-4)):
        raise EvidenceError('Actual retention mass/COM/inertia does not match original rigid payload model')
    return dict(runtime_profile_verified=True,ordinary_reset_verified=True,
        diagnostic_physics_override=diagnostic,
        physical_robot_mass_kg=float(masses.sum()),payload_mass_com_inertia_readback_verified=True,
        disconnected_cart_reset_verified=True,autonomous_behavior_accepted=False,formal_ppo_authorized=False)


def check_no_towing(physics, rows):
    maxima = {}
    for key in ('real_cart_robot_force_w_N','real_cart_robot_torque_w_Nm',
                'real_cart_cart_forces_w_N','real_cart_cart_torques_w_Nm'):
        value = np.asarray(physics[key])
        if not value.size or not np.isfinite(value).all():
            raise EvidenceError('Missing actual no-towing force evidence: '+key)
        maxima[key] = float(np.abs(value).max())
        if maxima[key] != 0.:
            raise EvidenceError('Unexpected external force/torque in non-towing case: '+key)
    if not rows or any(row['towed_kg'] != 0. or row['drawbar_force_N'] != 0. for row in rows):
        raise EvidenceError('Actual non-towing nominal context/drawbar force differs')
    return dict(no_towing_force_verified=True,actual_peak_absolute=maxima)
