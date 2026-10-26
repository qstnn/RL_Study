"""Route32：完整M20 rough与平地高速混训，旧任务不变。"""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel
from ..deeprobotics_m20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from .moving_acceleration_env_cfg import LightHWMovingAccelerationEnvCfg
from .speed3_env_cfg import set_speed_cap
from .rough_long_spec import GENERATOR, SURFACES, SPEED_CAP


def m20_reference_terrain():
    """读取M20最终实例化的地形，包括M20对boxes和random_rough的覆盖。"""
    return DeeproboticsM20RoughEnvCfg().scene.terrain.terrain_generator.copy()


def make_transfer_terrain():
    terrain=m20_reference_terrain()
    expected=set(SURFACES)-{'flat'}
    if set(terrain.sub_terrains)!=expected:
        raise ValueError('M20 terrain families changed; review the transfer contract')
    original=terrain.sub_terrains
    terrain.sub_terrains={'flat': MeshPlaneTerrainCfg(proportion=.25), **original}
    for key,value in GENERATOR.items():
        setattr(terrain,key,value)
    for name,spec in SURFACES.items():
        surface=terrain.sub_terrains[name]
        if surface.function.__name__!=spec['function']:
            raise ValueError('M20 terrain generator changed: '+name)
        for key,value in spec.items():
            if key!='function':setattr(surface,key,value)
        # TerrainGenerator也会写入这些值；提前设置使启动前/后有效配置一致。
        surface.size=terrain.size
        for key in ('horizontal_scale','vertical_scale','slope_threshold'):
            if hasattr(surface,key):setattr(surface,key,getattr(terrain,key))
    return terrain


# 原生距离课程和首次reset保护不变，无额外速度/支撑升级门槛。
from .m20_rough_transfer_env_cfg import terrain_levels_initial_guard


@configclass
class LightHWM20FullRoughEnvCfg(LightHWMovingAccelerationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        set_speed_cap(self,SPEED_CAP)
        self.scene.terrain.terrain_type='generator'
        self.scene.terrain.terrain_generator=make_transfer_terrain()
        self.scene.terrain.max_init_terrain_level=0
        # M20同一距离升降级逻辑，使用Isaac Lab原生版本，避免项目副本改全局gait奖励。
        self.curriculum.terrain_levels=CurriculumTermCfg(func=terrain_levels_initial_guard)
        print('[LightHW Rough transfer] policy=60, critic=236, action=16; source=model4799 full restore; '
              'vx=0..3.0, vy=wz=0;25%zero;25%flat+75%full M20 terrain;row0 start; '
              'all16 PPO; source posture/actions/PD/rewards/noise preserved',flush=True)
