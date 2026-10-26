"""独立转向奖励对照：只改变偏航速度跟踪权重，不改变运动控制或课程。"""
from isaaclab.utils import configclass
from .omni_rough_env_cfg import LightHWOmniRoughEnvCfg


@configclass
class LightHWYawControlEnvCfg(LightHWOmniRoughEnvCfg):
    """同源续训对照，保留原偏航权重1。"""
    pass


@configclass
class LightHWYawDoubleEnvCfg(LightHWOmniRoughEnvCfg):
    """实验组：所有状态的原偏航跟踪项权重1→2，std和所有门控原样保留。"""
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_z_exp.weight = 2.0
