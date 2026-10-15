"""Route01 owns its PPO settings; inherits only the standard Rough runner schema."""

from isaaclab.utils import configclass
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.agents.rsl_rl_ppo_cfg import (
    DeeproboticsLightHWRoughPPORunnerCfg,
)


@configclass
class WheelStage1RunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_multiskill_wheel_stage1"
        self.run_name = "fresh"
        self.max_iterations = 1000
        self.save_interval = 100
        self.clip_actions = None
        self.policy.init_noise_std = 0.25
        self.algorithm.learning_rate = 3e-4
        self.algorithm.entropy_coef = 0.001
