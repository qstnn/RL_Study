"""Route24独立宽度实验：仅改变速度指数奖励std；所有16维仍由PPO输出。"""
from isaaclab.utils import configclass
from .speed3_env_cfg import LightHWFlatSpeed3EnvCfg

@configclass
class LightHWSpeedWidth15EnvCfg(LightHWFlatSpeed3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 不改变动作探索噪声；这是奖励核宽度，零命令也使用同一速度奖励函数。
        self.rewards.track_lin_vel_xy_exp.params['std']=1.5
        print('[Route24 effective] speed reward std=1.5 weight5; vx0..3; scale5/clip37; original PPO noise/legs/PD/25%zero;60/236/16')
