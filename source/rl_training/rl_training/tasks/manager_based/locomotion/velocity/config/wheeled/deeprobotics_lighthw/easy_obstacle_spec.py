"""Route36 障碍高度课程；纯数据，可在不启动 Isaac Sim 时审查。

只重排台阶/方块的高度随 difficulty 的变化，不改变地形行列、宽度、
随机采样次数或升级判据。方块参数是正负高度幅值，不是相邻方块落差。
"""
import math

VERSION = 'lighthw_easy_obstacle_heights_v1'
MODULE = 'rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.easy_obstacle_terrains'
# 对应 difficulty=0, .1, ... 1；每行仍按原生成器在本行区间内抽样。
STAIR_HEIGHTS_M = (.02, .03, .04, .05, .065, .08, .10, .125, .155, .19, .23)
BOX_AMPLITUDES_M = (.01, .015, .02, .03, .045, .065, .09, .115, .145, .175, .20)
SURFACE_CHANGES = {
    'pyramid_stairs': dict(function=MODULE + ':easy_pyramid_stairs',
                          step_height_range=(.02, .23), height_knots=STAIR_HEIGHTS_M),
    'pyramid_stairs_inv': dict(function=MODULE + ':easy_inverted_pyramid_stairs',
                              step_height_range=(.02, .23), height_knots=STAIR_HEIGHTS_M),
    'boxes': dict(function=MODULE + ':easy_random_grid',
                  grid_height_range=(.01, .20), height_knots=BOX_AMPLITUDES_M),
}


def obstacle_height(difficulty, knots):
    """按原 difficulty 分段线性插值，返回米；非法配置失败而不静默限幅。"""
    if not math.isfinite(difficulty) or not 0. <= difficulty <= 1.:
        raise ValueError('Obstacle difficulty must be finite and in [0, 1]')
    if (len(knots) != 11 or any(not math.isfinite(h) or h <= 0 for h in knots)
            or any(a >= b for a, b in zip(knots, knots[1:]))):
        raise ValueError('Exactly 11 positive, strictly increasing height knots are required')
    position = difficulty * 10
    index = min(int(position), 9)
    fraction = position - index
    return knots[index] * (1 - fraction) + knots[index + 1] * fraction


def geometry_contract():
    return dict(version=VERSION, difficulty_knots=[i / 10 for i in range(11)],
        stair_heights_m=list(STAIR_HEIGHTS_M), box_amplitudes_m=list(BOX_AMPLITUDES_M),
        row0_stair_height_m=[.02, .03], row0_box_amplitude_m=[.01, .015],
        final_stair_endpoint_m=.23, final_box_amplitude_endpoint_m=.20,
        box_interpretation='grid top sampled in [-amplitude,+amplitude]; adjacent height difference can approach2*amplitude',
        interpolation='piecewise linear; original within-row difficulty RNG unchanged',
        other_terrain_geometry_unchanged=True, promotion_and_demotion_unchanged=True,
        endpoint_note='Endpoints are at difficulty1; original random sampling within last row usually stays below1')
