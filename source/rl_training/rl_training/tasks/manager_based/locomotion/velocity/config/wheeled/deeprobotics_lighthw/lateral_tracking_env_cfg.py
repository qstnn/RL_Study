"""Route46：保留route44整体训练任务，仅改变侧移命令的y跟踪宽度。"""
from isaaclab.utils import configclass
from .stair_completion_env_cfg import LightHWStairCompletionEnvCfg
from .lateral_tracking_rewards import track_lin_vel_xy_lateral_exp


@configclass
class LightHWLateralTrackingEnvCfg(LightHWStairCompletionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        reward = self.rewards.track_lin_vel_xy_exp
        reward.func = track_lin_vel_xy_lateral_exp
        reward.params.update(lateral_std=0.5, lateral_command_threshold=0.05)
        print('[LightHW route46] policy=60 critic=236 action=16; all16 PPO; '
              '|command vy|>0.05: xstd=1.5 ystd=0.5; weight5; other settings unchanged', flush=True)
