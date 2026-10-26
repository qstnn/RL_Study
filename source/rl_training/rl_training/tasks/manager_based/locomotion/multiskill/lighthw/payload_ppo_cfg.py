"""R6独立PPO；从零训练或恢复同契约R6 checkpoint。"""
from isaaclab.utils import configclass
from .ppo_cfg import WheelStage1RunnerCfg


@configclass
class HeavyPayloadRunnerCfg(WheelStage1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_multiskill_heavy_payload_r6"
        self.run_name = "fixed_payload"
        self.policy.init_noise_std = .15
        self.algorithm.learning_rate = 3e-4
        self.save_interval = 100
