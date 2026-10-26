"""R5A 独立 PPO；不加载 R1/R6 策略。"""
from isaaclab.utils import configclass
from ..ppo_cfg import WheelStage1RunnerCfg


@configclass
class JumpRunnerCfg(WheelStage1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='lighthw_r5a_jump_landing_v2'
        self.run_name='J1_recovery_v2'
        self.clip_actions=1.
        self.policy.init_noise_std=.05
        self.algorithm.learning_rate=1e-4
        self.algorithm.entropy_coef=.001
        self.num_steps_per_env=24
        self.save_interval=100
