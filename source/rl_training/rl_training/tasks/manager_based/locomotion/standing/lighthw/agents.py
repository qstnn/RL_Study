"""独立站立 PPO 起点；训练入口默认从零初始化。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LightHWStandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "lighthw_stand_flat"
    run_name = "stand_flat_v1"
    logger = "tensorboard"
    empirical_normalization = False
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15, noise_std_type="log",
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.001, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )
