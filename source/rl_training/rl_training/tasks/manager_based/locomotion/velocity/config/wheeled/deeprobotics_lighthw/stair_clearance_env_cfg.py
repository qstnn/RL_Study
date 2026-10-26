"""独立净空奖金实验；两组仅新增奖励的 weight=0/1 不同。"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
from .retention_fraction_env_cfg import LightHWRetention75EnvCfg
from .stair_clearance_rewards import StairClearanceReward


@configclass
class LightHWClearanceControlEnvCfg(LightHWRetention75EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_clearance_event = RewardTermCfg(func=StairClearanceReward, weight=0.0)


@configclass
class LightHWStairClearanceEnvCfg(LightHWClearanceControlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_clearance_event.weight = 1.0
