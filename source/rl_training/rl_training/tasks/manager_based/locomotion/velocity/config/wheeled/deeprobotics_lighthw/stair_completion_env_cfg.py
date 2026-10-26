"""Route44：固定源yaw2/999配置，只对比新整段进展奖励0/1。"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
from .spin_xy_env_cfg import LightHWSpinXYControlEnvCfg
from .stair_completion_rewards import StairCompletionReward

@configclass
class LightHWCompletionControlEnvCfg(LightHWSpinXYControlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_completion_event=RewardTermCfg(func=StairCompletionReward,weight=0.)
        print('[LightHW route44] policy=60 critic=236 action=16; all16 PPO; source50% retention and spinstd1.5; no clearance bonus',flush=True)

@configclass
class LightHWStairCompletionEnvCfg(LightHWCompletionControlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_completion_event.weight=1.
