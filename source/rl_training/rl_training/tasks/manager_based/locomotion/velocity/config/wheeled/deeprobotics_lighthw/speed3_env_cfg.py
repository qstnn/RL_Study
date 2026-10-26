"""Route22/23：独立速度分支；原奖励/16维动作/观测契约不变。"""
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfPyramidSlopedTerrainCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel
from .ppo_reference_env_cfg import LightHWPPOReferenceEnvCfg
from .contact_gate_rewards import apply_variant


def set_speed_cap(cfg, speed):
    """同步命令与单stage课程上限；不改变采样概率、腿包络或动作缩放。"""
    cfg.actions.joint_pos.stage_max_speeds = (float(speed),)
    cfg.commands.base_velocity.stage_max_speeds = (float(speed),)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, float(speed))


@configclass
class LightHWFlatSpeed3EnvCfg(LightHWPPOReferenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_variant(self, 'gated')
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        set_speed_cap(self, 3.0)
        print('[Route22 effective] flat vx=0..3.0; 25% zero, moving 50% cap anchors; '
              '60/236/16; scale5/clip37; unchanged legs/PD/rewards; full PPO restore required')


@configclass
class LightHWRoughSpeedTransferEnvCfg(LightHWFlatSpeed3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 第一轮只升地形难度；速度固定低上限0.8，不把3.0与高难度地形同时开启。
        set_speed_cap(self, .8)
        self.scene.terrain.terrain_type = 'generator'
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42, size=(20., 20.), border_width=20., num_rows=5, num_cols=8,
            horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75,
            curriculum=True, difficulty_range=(0., 1.), use_cache=False,
            sub_terrains={
                'flat': MeshPlaneTerrainCfg(proportion=.25),
                'gentle_up': HfPyramidSlopedTerrainCfg(proportion=.375, slope_range=(0., .08),
                    platform_width=2., border_width=.25),
                'gentle_down': HfPyramidSlopedTerrainCfg(proportion=.375, slope_range=(0., .08),
                    platform_width=2., border_width=.25, inverted=True),
            })
        # 使用Isaac Lab原生地形课程，避免项目另一个函数的全局gait奖励副作用。
        self.curriculum.terrain_levels = CurriculumTermCfg(func=terrain_levels_vel)
        print('[Route23 effective] independently gated rough transfer; vx=0..0.8; initial terrain row0; '
              '25% plane/75% gentle slopes; same named60/236/16 and PPO actions')
