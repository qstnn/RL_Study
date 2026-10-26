"""V5：单 Actor、全工况混合；Actor/Critic 固定独立学习率。"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from ..ppo_cfg import WheelStage1RunnerCfg


@configclass
class V5PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # Runner 仅接受 class_name=PPO；V5 worker 临时绑定本路线 V5PPO。
    class_name: str = "PPO"
    critic_learning_rate: float = 1.e-4
    kl_stop_threshold: float = .01
    critic_warmup_updates: int = 50
    # raw 单位，前12腿/后4轮；候选须通过冻结扰动验证才可准入。
    initial_std_by_action: tuple = (.005,) * 12 + (.01,) * 4
    min_std_by_action: tuple = (.001,) * 12 + (.002,) * 4
    max_std_by_action: tuple = (.005,) * 12 + (.01,) * 4
    reference_loss_coef: float = .1
    reference_batch_size: int = 256
    # 每个工况/阶段独立保持，不能由全局均值掩盖稀有重拖样本。
    reference_max_group_mse: float = .02
    reference_action_scale: tuple = (.005,) * 12 + (.01,) * 4
    actor_input_mask: tuple = (205, 285)


@configclass
class UnifiedRunnerCfg(WheelStage1RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_r1_unified_v5"
        self.run_name = "protocol_critic_state_kl_guard"
        self.max_iterations = 1000
        self.num_steps_per_env = 64
        self.save_interval = 25
        self.policy.init_noise_std = .005
        self.policy.noise_std_type = "log"
        self.algorithm = V5PpoAlgorithmCfg(
            value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
            entropy_coef=.001, num_learning_epochs=3, num_mini_batches=4,
            learning_rate=1.e-5, schedule="fixed", gamma=.995, lam=.95,
            desired_kl=.01, max_grad_norm=1.)
