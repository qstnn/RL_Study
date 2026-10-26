"""统一R1仍为一个标准MLP ActorCritic和单个PPO优化器。"""
from isaaclab.utils import configclass
from ..ppo_cfg import WheelStage1RunnerCfg


@configclass
class UnifiedRunnerCfg(WheelStage1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name="lighthw_r1_unified_v4"
        self.run_name="route_frame_slew_hold_motor_sizing"
        self.max_iterations=200
        self.algorithm.learning_rate=1.e-5
        self.algorithm.schedule="fixed"
        self.save_interval=50
        self.policy.init_noise_std=.05
