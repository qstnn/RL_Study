"""LightHW完整M20地形与3m/s长训的纯数据契约；不导入仿真器。"""
from copy import deepcopy

TERRAIN_VERSION = 'lighthw_m20_full_rough_speed3_v1'
SPEED_CAP = 3.0
# 保留M20六种类型；75%按M20原比例分配，额外25%平地。
SURFACES = {
    'flat': dict(proportion=.25, function='flat_terrain'),
    'pyramid_stairs': dict(proportion=.15, function='pyramid_stairs_terrain', step_height_range=(.05, .23), step_width=.3, platform_width=3., border_width=1., holes=False),
    'pyramid_stairs_inv': dict(proportion=.15, function='inverted_pyramid_stairs_terrain', step_height_range=(.05, .23), step_width=.3, platform_width=3., border_width=1., holes=False),
    'boxes': dict(proportion=.15, function='random_grid_terrain', grid_width=.45, grid_height_range=(.025, .20), platform_width=2.),
    # Isaac Lab random_uniform不使用difficulty参数；本轮用户选择完整M20幅度，不把row0描述为低噪声。
    'random_rough': dict(proportion=.15, function='random_uniform_terrain', noise_range=(.01, .16), noise_step=.01, border_width=.25),
    'hf_pyramid_slope': dict(proportion=.075, function='pyramid_sloped_terrain', slope_range=(0., .4), platform_width=2., border_width=.25, inverted=False),
    'hf_pyramid_slope_inv': dict(proportion=.075, function='pyramid_sloped_terrain', slope_range=(0., .4), platform_width=2., border_width=.25, inverted=True),
}
GENERATOR = dict(seed=42, size=(8., 8.), border_width=20., num_rows=10, num_cols=20,
                 horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75,
                 curriculum=True, difficulty_range=(0., 1.), use_cache=False)
NATIVE_CURRICULUM_FUNCTION = 'isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums:terrain_levels_vel'
CURRICULUM_FUNCTION = 'rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.m20_rough_transfer_env_cfg:terrain_levels_initial_guard'


def contract():
    return dict(version=TERRAIN_VERSION, generator=deepcopy(GENERATOR), surfaces=deepcopy(SURFACES),
                max_init_terrain_level=0, speed_cap=SPEED_CAP, flat_fraction=.25,
                native_curriculum=NATIVE_CURRICULUM_FUNCTION,
                effective_curriculum=CURRICULUM_FUNCTION,initial_reset_guard=True,
                random_rough_difficulty='full M20 noise1..16cm at every row; native generator ignores difficulty')


def columns_for_surface(name):
    """严格复现Isaac Lab课程地形按列起点加0.001选择类型的规则，避免单环境默认为平地。"""
    if name not in SURFACES:
        raise ValueError('Unknown terrain: '+str(name))
    total=sum(s['proportion'] for s in SURFACES.values())
    lower=0.
    for key, spec in SURFACES.items():
        upper=lower+spec['proportion']/total
        if key==name:
            return [i for i in range(GENERATOR['num_cols']) if lower <= i/GENERATOR['num_cols']+.001 < upper]
        lower=upper
    raise AssertionError(name)
