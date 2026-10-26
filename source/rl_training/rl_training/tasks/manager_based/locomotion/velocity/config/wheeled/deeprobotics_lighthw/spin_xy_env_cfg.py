"""独立任务：yaw 权重固定 2，只对原地转向的 XY 核宽度做匹配对照。"""
from isaaclab.utils import configclass
from .yaw_weight_env_cfg import LightHWYawDoubleEnvCfg
from .spin_xy_rewards import track_lin_vel_xy_spin_exp


@configclass
class LightHWSpinXYControlEnvCfg(LightHWYawDoubleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        reward = self.rewards.track_lin_vel_xy_exp
        reward.func = track_lin_vel_xy_spin_exp
        reward.params.update(spin_std=1.5, linear_command_threshold=0.05, yaw_command_threshold=0.05)


@configclass
class LightHWSpinXYNarrowEnvCfg(LightHWSpinXYControlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.params['spin_std'] = 0.5
