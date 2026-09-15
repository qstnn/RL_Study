# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsLightHWRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    # Keep this parameter-only static-hold pass in its own TensorBoard tree.
    experiment_name = "deeprobotics_lighthw_static_hold_hs_v1"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.40,
        noise_std_type="log",
        actor_hidden_dims=[384, 256, 128],
        critic_hidden_dims=[384, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsLightHWFlatPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 4000
        self.experiment_name = "deeprobotics_lighthw_flat"


@configclass
class DeeproboticsLightHWBalancedHighSpeedPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    """Runner for the balanced 3.5 m/s plus zero-command route."""

    max_iterations = 12000
    experiment_name = "deeprobotics_lighthw_balanced_highspeed"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 12000
        self.experiment_name = "deeprobotics_lighthw_balanced_highspeed"
        self.algorithm.learning_rate = 1.5e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.008
        # This is effective for actor-only transfer and avoids the excessive
        # wheel exploration seen in the failed static route.
        self.policy.init_noise_std = 0.25


@configclass
class DeeproboticsLightHWBlindRoughPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    """PPO runner for the no-height-observation rough LightHW task."""

    experiment_name = "deeprobotics_lighthw_blind_rough"
    run_name = "lighthw_rough_blind"
    fresh_start_only = True
