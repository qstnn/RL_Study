"""V6：单 Actor、全工况混合；Actor/Critic 固定独立学习率。"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from ..ppo_cfg import WheelStage1RunnerCfg


@configclass
class V6PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # Runner 仅接受 class_name=PPO；V6 worker 临时绑定本路线 V6PPO。
    class_name: str = "PPO"
    critic_learning_rate: float = 1.e-4
    kl_stop_threshold: float = .01
    critic_warmup_updates: int = 20
    # raw 单位，前12腿/后4轮；条件探索在零指令/非拖载时乘1/8。
    initial_std_by_action: tuple = (.04,) * 12 + (.08,) * 4
    min_std_by_action: tuple = (.04,) * 12 + (.08,) * 4
    max_std_by_action: tuple = (.04,) * 12 + (.08,) * 4
    reference_loss_coef: float = .1
    reference_batch_size: int = 256
    # 成功缓存保留软约束；本轮不使用参考误差硬拒绝。
    reference_max_group_mse: float = 0.
    reference_action_scale: tuple = (.2,) * 16
    actor_input_mask: tuple = (205, 285)


@configclass
class UnifiedRunnerCfg(WheelStage1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_r1_unified_v6"
        self.run_name = "rl_towing_motion_exploration"
        self.max_iterations = 1000
        self.num_steps_per_env = 64
        self.save_interval = 25
        self.policy.init_noise_std = .005
        self.policy.noise_std_type = "log"
        self.algorithm = V6PpoAlgorithmCfg(
            value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
            entropy_coef=.001, num_learning_epochs=3, num_mini_batches=4,
            learning_rate=3.e-5, schedule="fixed", gamma=.995, lam=.95,
            desired_kl=.01, max_grad_norm=1.)
