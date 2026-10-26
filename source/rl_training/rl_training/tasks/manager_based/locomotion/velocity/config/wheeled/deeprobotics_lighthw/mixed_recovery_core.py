"""Route53：固定任务配额与纯Tensor命令采样；不读取实速，不产生关节动作。"""
from copy import deepcopy
import hashlib
import json
import torch

VERSION = 'lighthw_mixed_recovery_sampling_v1'
ROLE_NAMES = ('forward', 'yaw', 'lateral', 'combined', 'stairs')
COMMAND_CONTRACT = dict(
    version=VERSION, role_fractions=[.30, .20, .20, .10, .20], role_names=list(ROLE_NAMES),
    zero_command_probability=.25, resampling_seconds=8., heading_command=False,
    forward=dict(range=[.2, 3.], anchor1_probability=.2, anchor3_probability=.2),
    yaw=dict(abs_range=[.2, 1.], equal_directions=True),
    lateral=dict(abs_range=[.2, 1.], equal_directions=True, original_clamped_uniform=True),
    combined=dict(vx=[.2, 1.], abs_vy=[.2, .5], abs_wz=[.2, .5], independent_signs=True),
    stairs=dict(vx=[.3, 1.], vy=0., wz=0., spawn_levels=[0, 2, 4, 6],
                positive_stair_column='first original M20 positive-stairs column', from_bottom=True),
    terrain='original seeded M20 map; sample flat and continuous positive stairs only',
    flat_spawn_column='central original flat column',
    terrain_progression=False, all_16_actions_from_actor=True, behavior_cloning=False,
    wheel_feedforward=False, old_trajectories_used_for_training=False,
)


def roles_for_ids(ids, num_envs):
    """固定环境配额，避免短run偶然完全没有某技能；不消耗RNG。"""
    decile = ids * 10 // num_envs
    return torch.where(decile < 3, 0, torch.where(decile < 5, 1,
           torch.where(decile < 7, 2, torch.where(decile < 8, 3, 4)))).long()


def commands_from_uniform(roles, random):
    """输入[N]角色和[N,7]独立U[0,1)；返回机身命令和独立停车mask。

    25%零命令按所有角色同样采样。侧移保持旧clamp幅值规则；前进覆盖
    0.2～3.0且显式含1.0/3.0锚点；组合两个转向符号独立，不强制同号。
    """
    if random.shape != (len(roles), 7) or not torch.isfinite(random).all():
        raise ValueError('Expected finite[N,7] uniform samples')
    if (random < 0).any() or (random > 1).any() or (roles < 0).any() or (roles > 4).any():
        raise ValueError('Sample or role outside contract')
    stop = random[:, 0] < .25
    sign_y = torch.where(random[:, 3] >= .5, 1., -1.)
    sign_z = torch.where(random[:, 4] >= .5, 1., -1.)
    forward = .2 + 2.8 * random[:, 1]
    forward = torch.where(random[:, 2] < .2, 1., torch.where(random[:, 2] < .4, 3., forward))
    cmd = torch.zeros((len(roles), 3), device=random.device, dtype=random.dtype)
    cmd[:, 0] = torch.where(roles == 0, forward, torch.where(roles == 3, .2 + .8 * random[:, 1],
                    torch.where(roles == 4, .3 + .7 * random[:, 1], 0.)))
    signed = random[:, 3] * 2 - 1
    lateral = sign_y * signed.abs().clamp_min(.2)
    cmd[:, 1] = torch.where(roles == 2, lateral, torch.where(roles == 3, sign_y * (.2 + .3 * random[:, 5]), 0.))
    cmd[:, 2] = torch.where(roles == 1, sign_z * (.2 + .8 * random[:, 2]),
                    torch.where(roles == 3, sign_z * (.2 + .3 * random[:, 6]), 0.))
    cmd[stop] = 0.
    return cmd, stop


def migrate_state(state, target_mapping):
    """仅迁移命令分布签名；原观测、物理映射、课程计数不变。"""
    result = deepcopy(state); expected = deepcopy(state['action_mapping'])
    expected['command_distribution'] = deepcopy(COMMAND_CONTRACT)
    if expected != target_mapping:
        raise ValueError('Unexpected physical action or observation change')
    result['action_mapping'] = expected
    result['signature'] = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    return result
