"""Route31：从高速PPO迁入M20地形家族的低速适应任务，旧平地任务不变。"""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel
from ..deeprobotics_m20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from .moving_acceleration_env_cfg import LightHWMovingAccelerationEnvCfg
from .speed3_env_cfg import set_speed_cap
from .rough_transfer_spec import GENERATOR, SURFACES, SPEED_CAP


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


def terrain_levels_initial_guard(env,env_ids):
    """首个reset保留level0；已有物理步之后使用原生距离课程，无奖励副作用。

    初次reset时资产尚未经过有效episode，位置差不能视为行走成绩。
    返回当前平均地形等级，只用于原生课程日志。
    """
    if env.common_step_counter==0:
        terrain=env.scene.terrain
        robot=env.scene['robot']
        distance=torch.norm(robot.data.root_pos_w[env_ids,:2]-env.scene.env_origins[env_ids,:2],dim=1)
        env._rough_initial_reset_audit=dict(skipped_before_first_physics_step=True,
            levels_min=int(terrain.terrain_levels.min()),levels_max=int(terrain.terrain_levels.max()),
            native_would_move_up_count=int((distance>terrain.cfg.terrain_generator.size[0]/2).sum()),
            pre_episode_max_distance_m=float(distance.max()))
        return terrain.terrain_levels.float().mean()
    return terrain_levels_vel(env,env_ids)


@configclass
class LightHWM20RoughTransferEnvCfg(LightHWMovingAccelerationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        set_speed_cap(self,SPEED_CAP)
        self.scene.terrain.terrain_type='generator'
        self.scene.terrain.terrain_generator=make_transfer_terrain()
        self.scene.terrain.max_init_terrain_level=0
        # M20同一距离升降级逻辑，使用Isaac Lab原生版本，避免项目副本改全局gait奖励。
        self.curriculum.terrain_levels=CurriculumTermCfg(func=terrain_levels_initial_guard)
        print('[LightHW Rough transfer] policy=60, critic=236, action=16; source=model3799 full restore; '
              'vx=0..1.0, vy=wz=0;25%zero;25%flat+75%gentle M20 terrain;row0 start; '
              'all16 PPO; source posture/actions/PD/rewards/noise preserved',flush=True)
