"""R5A 独立 PPO；不加载 R1/R6 策略。"""
from isaaclab.utils import configclass
from ..ppo_cfg import WheelStage1RunnerCfg


@configclass
class JumpRunnerCfg(WheelStage1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='lighthw_r5a_highjump_v3'
        self.run_name='J1_highjump_v3'
        self.clip_actions=1.
        self.policy.init_noise_std=.05
        self.algorithm.learning_rate=1e-5
        self.algorithm.schedule="fixed"
        self.algorithm.entropy_coef=.001
        self.num_steps_per_env=24
        self.save_interval=50
