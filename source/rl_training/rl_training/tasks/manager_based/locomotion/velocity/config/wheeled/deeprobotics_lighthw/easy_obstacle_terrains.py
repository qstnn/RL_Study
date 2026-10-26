"""只改变障碍高度，复用本机 Isaac Lab 原生网格生成及随机数序列。"""
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains import (
    pyramid_stairs_terrain, inverted_pyramid_stairs_terrain, random_grid_terrain,
)
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPyramidStairsTerrainCfg, MeshInvertedPyramidStairsTerrainCfg, MeshRandomGridTerrainCfg,
)
from .easy_obstacle_spec import STAIR_HEIGHTS_M, BOX_AMPLITUDES_M, obstacle_height


def _effective_config(difficulty, cfg, range_name):
    """复制后固定此次网格高度，不覆盖保存配置、旧任务或后续地形。"""
    if tuple(getattr(cfg, range_name)) != (cfg.height_knots[0], cfg.height_knots[-1]):
        raise ValueError('Saved terrain range and height knots disagree')
    height = obstacle_height(difficulty, cfg.height_knots)
    effective = cfg.copy()
    setattr(effective, range_name, (height, height))
    return effective


def easy_pyramid_stairs(difficulty, cfg):
    return pyramid_stairs_terrain(difficulty, _effective_config(difficulty, cfg, 'step_height_range'))


def easy_inverted_pyramid_stairs(difficulty, cfg):
    return inverted_pyramid_stairs_terrain(difficulty, _effective_config(difficulty, cfg, 'step_height_range'))


def easy_random_grid(difficulty, cfg):
    return random_grid_terrain(difficulty, _effective_config(difficulty, cfg, 'grid_height_range'))


@configclass
class EasyPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    function = easy_pyramid_stairs
    height_knots: tuple[float, ...] = STAIR_HEIGHTS_M


@configclass
class EasyInvertedPyramidStairsTerrainCfg(MeshInvertedPyramidStairsTerrainCfg):
    function = easy_inverted_pyramid_stairs
    height_knots: tuple[float, ...] = STAIR_HEIGHTS_M


@configclass
class EasyRandomGridTerrainCfg(MeshRandomGridTerrainCfg):
    function = easy_random_grid
    height_knots: tuple[float, ...] = BOX_AMPLITUDES_M
