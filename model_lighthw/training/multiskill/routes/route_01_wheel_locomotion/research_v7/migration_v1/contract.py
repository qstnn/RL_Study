"""部署观测/动作合同；由当前V7实际V6源核对，不引入教师真值。"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent
REPO = next(p for p in ROOT.parents if (p/'source/rl_training').is_dir())
SOURCE = REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw'
SPEC_PATH = SOURCE/'wheel_unified_v6/spec.py'
POSITION_SOURCE = REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_lighthw/reference_observations.py'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_contract():
    """只读无仿真源，核对历史排列/相对关节位置语义与实际action顺序。"""
    spec = _load(SPEC_PATH, '_r1_migration_v6_contract')
    position = _load(POSITION_SOURCE, '_r1_migration_position_contract')
    declared = spec.CONTRACT
    terms = list(spec.POLICY_TERMS.items())
    expected = [('base_ang_vel', 3), ('projected_gravity', 3), ('velocity_commands', 3),
                ('joint_pos', 16), ('joint_vel', 16), ('actions', 16), ('heading_error', 1), ('task_context', 2)]
    if terms != expected or spec.HISTORY != 5:
        raise ValueError('V7 deployed observation term order/history changed; review a new migration contract')
    layout = {}
    offset = 0
    for name, width in terms:
        end = offset+width*spec.HISTORY
        layout[name] = dict(width=width, history=spec.HISTORY, columns=[offset, end], latest=[end-width, end])
        offset = end
    masked = layout['actions']['columns']
    if offset != 300 or declared['policy_dim'] != 300 or declared['action_dim'] != 16 or declared['ignored_policy_columns'] != masked:
        raise ValueError('V6 input dimensions/action-history mask do not match the deployed layout')
    joints = list(position.OBSERVATION_JOINT_NAMES)
    wheels = list(position.WHEEL_JOINT_NAMES)
    actions = [name for name in joints if name not in wheels]+wheels
    if actions != declared['action_order']:
        raise ValueError('Observation named-joint layout and actuator order disagree')
    selected = {key: deepcopy(declared[key]) for key in ('leg_scale', 'leg_mapping', 'leg_pd',
        'wheel_scale', 'wheel_clip', 'wheel_joint_velocity_limit_rad_s', 'wheel_damping', 'physics_dt', 'policy_dt')}
    if selected != dict(leg_scale={'hip': .08, 'thigh': .14, 'calf': .14},
        leg_mapping='clip(default + scale * raw, URDF_lower, URDF_upper)', leg_pd=[80., 3.],
        wheel_scale=5., wheel_clip=[-50., 50.], wheel_joint_velocity_limit_rad_s=60., wheel_damping=2.,
        physics_dt=.001, policy_dt=.02):
        raise ValueError('Original deployed actuator contract changed')
    return dict(version='r1_migration_v1_deployed_data_contract', policy_dim=300, action_dim=16,
        teacher_critic_dim=declared['critic_dim'], student_critic_contract='not_implemented',
        hidden_dims=[512, 256, 128], activation='elu', policy_terms=dict(terms), layout=layout,
        history_layout=declared['history_layout'], ignored_policy_columns=masked,
        observation_joint_order=joints, action_order=actions,
        observation_to_action_indices=[actions.index(name) for name in joints],
        wheel_position_channels_in_frame=[joints.index(name) for name in wheels],
        observation_scale=dict(base_ang_vel=.5, projected_gravity=1., velocity_commands=1.,
            joint_pos=1., joint_vel=.05, actions=1., heading_error=1., task_context=1.),
        position_semantics='relative to default named angles; four wheel-position channels are zero',
        task_context=['nominal payload_kg/15', 'nominal towed_kg/40'],
        actor_truth_inputs=False, observation_noise=False, physical_randomization=False,
        actor_updates=0, formal_ppo_authorized=False, **selected)


CONTRACT = checked_contract()


def validate_runtime(runtime, network_contract):
    """shape之外同时核对来源的terms、动作顺序、scale、原生执行器与无噪声。"""
    c = CONTRACT
    for key in ('policy_dim', 'action_dim', 'policy_terms', 'action_order', 'ignored_policy_columns',
                'history_layout', 'leg_scale', 'leg_pd', 'wheel_scale', 'wheel_clip'):
        if network_contract.get(key) != c[key]:
            raise ValueError('Recorded deployed contract mismatch: '+key)
    if runtime.get('dimensions') != [300, c['teacher_critic_dim'], 16]:
        raise ValueError('Recorded teacher runtime dimensions differ')
    if runtime.get('policy_terms') != list(c['policy_terms']) or runtime.get('policy_history_frames') != 5:
        raise ValueError('Recorded runtime observation history/order differs')
    if runtime.get('action_order') != c['action_order'] or runtime.get('policy_enable_corruption') is not False:
        raise ValueError('Recorded action order or observation noise differs')
    for key, expected in [('leg_kp', [80.]*12), ('leg_kd', [3.]*12),
        ('software_effort_limits_Nm', [96.]*12+[48.]*4), ('wheel_target_clip_rad_s', [[-50., 50.]]*4),
        ('physics_dt', .001), ('policy_dt', .02), ('wheel_damping', 2.)]:
        if np.asarray(runtime.get(key)).shape != np.asarray(expected).shape or not np.allclose(runtime[key], expected, atol=1.e-7, rtol=0.):
            raise ValueError('Recorded physical interface mismatch: '+key)
    for name in ('base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel'):
        if runtime.get('noise_terms', {}).get(name, {}).get('scale') != c['observation_scale'][name]:
            raise ValueError('Recorded observation scale differs: '+name)


def masked_observations(observation):
    """复制并屏蔽原80列动作历史；原始观测保留在数据包供重建核验。"""
    result = np.asarray(observation).copy()
    if result.ndim != 2 or result.shape[1] != 300 or not np.isfinite(result).all():
        raise ValueError('Expected finite N x 300 deployed observations')
    begin, end = CONTRACT['ignored_policy_columns']
    result[:, begin:end] = 0.
    return result


def install_actor_mask(actor, optimizer=None):
    """未来拟合复用：输入权重/梯度/已有Adam矩都屏蔽，不执行优化器更新。"""
    import torch
    first = actor[0].weight
    if tuple(first.shape) != (512, 300):
        raise ValueError('Expected the existing 300 -> 512 Actor input layer')
    begin, end = CONTRACT['ignored_policy_columns']
    with torch.no_grad():
        first[:, begin:end] = 0.
        if optimizer is not None:
            for value in optimizer.state.get(first, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == first.shape:
                    value[:, begin:end] = 0.
    mask = torch.ones_like(first)
    mask[:, begin:end] = 0.
    previous = getattr(actor, '_r1_migration_input_mask', None)
    if previous is not None:
        previous.remove()
    actor._r1_migration_input_mask = first.register_hook(lambda gradient: gradient*mask)
    return actor._r1_migration_input_mask


def validate_actor_mask(actor, optimizer=None):
    import torch
    first = actor[0].weight
    begin, end = CONTRACT['ignored_policy_columns']
    if tuple(first.shape) != (512, 300) or torch.count_nonzero(first[:, begin:end]):
        raise ValueError('Actor has unmasked action-history inputs')
    if optimizer is not None:
        for value in optimizer.state.get(first, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == first.shape and torch.count_nonzero(value[:, begin:end]):
                raise ValueError('Adam state has unmasked action-history inputs')
