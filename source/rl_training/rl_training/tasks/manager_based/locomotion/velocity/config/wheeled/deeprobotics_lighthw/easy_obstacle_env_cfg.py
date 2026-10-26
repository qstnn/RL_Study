"""Route36 独立低障碍课程：动作/命令/PPO/奖励/PD及安全课程规则沿用 route35。"""
from isaaclab.utils import configclass
from .terrain_speed_env_cfg import LightHWTerrainSpeedEnvCfg
from .easy_obstacle_terrains import (
    EasyPyramidStairsTerrainCfg, EasyInvertedPyramidStairsTerrainCfg, EasyRandomGridTerrainCfg,
)
from .easy_obstacle_spec import SURFACE_CHANGES


@configclass
class LightHWEasyObstacleEnvCfg(LightHWTerrainSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        surfaces = self.scene.terrain.terrain_generator.sub_terrains
        classes = dict(pyramid_stairs=EasyPyramidStairsTerrainCfg,
                       pyramid_stairs_inv=EasyInvertedPyramidStairsTerrainCfg,
                       boxes=EasyRandomGridTerrainCfg)
        for name, cls in classes.items():
            values = {key: value for key, value in vars(surfaces[name]).items() if key != 'function'}
            values.update({key: value for key, value in SURFACE_CHANGES[name].items() if key != 'function'})
            surfaces[name] = cls(**values)
        print('[LightHW route36] policy=60, critic=236, action=16; independent obstacle height curriculum; '
              'row0 stairs2..3cm, box amplitude1..1.5cm; endpoints23/20cm; '
              'flat0..3, rough0..1; all16 PPO; source999 actions/rewards/PD/noise unchanged', flush=True)
