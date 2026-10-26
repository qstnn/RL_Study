"""LightHW完整M20地形与3m/s长训的纯数据契约；不导入仿真器。"""
from copy import deepcopy

TERRAIN_VERSION = 'lighthw_m20_safe_traversal_curriculum_v1'
SPEED_CAP = 3.0
# 保留M20六种类型；75%按M20原比例分配，额外25%平地。
SURFACES = {
    'flat': dict(proportion=.25, function='flat_terrain'),
    'pyramid_stairs': dict(proportion=.15, function='pyramid_stairs_terrain', step_height_range=(.05, .23), step_width=.3, platform_width=3., border_width=1., holes=False),
    'pyramid_stairs_inv': dict(proportion=.15, function='inverted_pyramid_stairs_terrain', step_height_range=(.05, .23), step_width=.3, platform_width=3., border_width=1., holes=False),
    'boxes': dict(proportion=.15, function='random_grid_terrain', grid_width=.45, grid_height_range=(.025, .20), platform_width=2.),
    # 仅课程改变：noise_range保留M20端点，由新函数根据difficulty生成实际幅度。
    'random_rough': dict(proportion=.15, function='progressive_random_terrain', noise_range=(.01, .16), noise_step=.01, border_width=.25),
    'hf_pyramid_slope': dict(proportion=.075, function='pyramid_sloped_terrain', slope_range=(0., .4), platform_width=2., border_width=.25, inverted=False),
    'hf_pyramid_slope_inv': dict(proportion=.075, function='pyramid_sloped_terrain', slope_range=(0., .4), platform_width=2., border_width=.25, inverted=True),
}
GENERATOR = dict(seed=42, size=(8., 8.), border_width=20., num_rows=10, num_cols=20,
                 horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75,
                 curriculum=True, difficulty_range=(0., 1.), use_cache=False)
CURRICULUM_FUNCTION = 'rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.safe_terrain_env_cfg:SafeTraversalCurriculum'
RANDOM_FUNCTION = 'rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.safe_terrain_env_cfg:progressive_random_terrain'
INITIAL_NOISE_MAX = .02
PROMOTION_TILT_DEG = 45.0
CONTACT_LIMIT_N = 1.0


def random_noise_range(difficulty, low=.01, high=.16, step=.01):
    """上界从2cm线性增加至16cm，向下量化到原1cm采样网格，禁止离散采样越过课程上界。"""
    import math
    if not 0. <= difficulty <= 1.:raise ValueError('Difficulty must be in[0,1]')
    upper=INITIAL_NOISE_MAX+(high-INITIAL_NOISE_MAX)*difficulty
    upper=low+math.floor((upper-low)/step+1e-10)*step
    return (low,min(high,upper))


def contract():
    return dict(version=TERRAIN_VERSION,generator=deepcopy(GENERATOR),surfaces=deepcopy(SURFACES),
        max_init_terrain_level=0,speed_cap=SPEED_CAP,flat_fraction=.25,
        effective_curriculum=CURRICULUM_FUNCTION,initial_reset_guard=True,
        random_rough_difficulty='upper2..16cm linear in difficulty, rounded down to original1cm grid; lower1cm unchanged',
        initial_noise_max=INITIAL_NOISE_MAX,promotion='one complete same-column8m tile, no sampled nonwheel>1N/tilt>45deg/nonfinite over episode; no unsafe termination',
        sample_hz=50,spawn_partial_tile_not_counted=True,changes_applied_at_episode_reset=True,
        demotion='unsafe episode or no full traversal after integrated positive command>=12m; insufficient moving exposure stays; top-row stall uses4m forward displacement',
        flat_levels='fixed0; identical flat geometry at all rows',maximum_level='clamp9, no automatic random drop',
        moving_low_load='not a curriculum failure; wheel support never used',training_termination_unchanged=True)


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
