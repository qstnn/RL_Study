# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LightHWRollStopPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """独立从零PPO；只允许本路线完整checkpoint续训。"""

    lighthw_roll_stop = True
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "lighthw_ppo_roll_stop"
    run_name = "lighthw_roll_stop_scratch_v1"
    empirical_normalization = False
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(class_name="LightHWRollStopActorCritic",
        init_noise_std=0.12, noise_std_type="log", actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True,
        clip_param=0.2, entropy_coef=0.001, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3e-4, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0)

    def __post_init__(self):
        self.policy.grouped_noise_config = {
            "leg_std_init": 0.12, "leg_std_min": 0.08, "leg_std_max": 0.15,
            "wheel_std_init": 0.50, "wheel_std_min": 0.12, "wheel_std_max": 0.60,
            "stop_wheel_multiplier": 0.25,
            "hip_limit": 0.04, "leg_limit": 0.10, "wheel_scale": 5.0,
        }


@configclass
class DeeproboticsLightHWRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    # Use an explicit M20-derived name for the LightHW comparison route.
    experiment_name = "lighthw_rough_fromM20"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
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
    """Runner for the balanced 3.2 m/s plus zero-command route."""

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
class DeeproboticsLightHWFlatStraightHighSpeedPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    """Fresh route runner for flat straight-line speed transfer."""

    max_iterations = 12000
    experiment_name = "deeprobotics_lighthw_flat_straight_highspeed"
    run_name = "lighthw_flat_straight_highspeed_from_low_v1"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 12000
        self.experiment_name = "deeprobotics_lighthw_flat_straight_highspeed"
        self.run_name = "lighthw_flat_straight_highspeed_from_low_v1"
        # Warm-start loads the source policy noise.  These values control only
        # the new optimizer and keep the transfer from changing too abruptly.
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.008


@configclass
class DeeproboticsLightHWFlatStraightProgressivePPORunnerCfg(
    DeeproboticsLightHWFlatStraightHighSpeedPPORunnerCfg
):
    """Fresh progressive-speed runner initialized from the model_2498 policy."""

    max_iterations = 12000
    experiment_name = "deeprobotics_lighthw_flat_straight_progressive"
    run_name = "lighthw_flat_straight_progressive_v2"
    # Warm-start keeps the trained actor/critic but deliberately restores
    # enough wheel exploration for the newly exposed 37 rad/s action range.
    warm_start_noise_std = (
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.12,
        0.12,
        0.12,
        0.12,
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 12000
        self.experiment_name = "deeprobotics_lighthw_flat_straight_progressive"
        self.run_name = "lighthw_flat_straight_progressive_v2"
        self.algorithm.learning_rate = 1.0e-4
        # The previous adaptive schedule spent 96.5% of this route at its
        # 1e-5 floor. A fixed rate makes the staged curriculum observable.
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.008


@configclass
class DeeproboticsLightHWFlatStraightRecoveryPPORunnerCfg(
    DeeproboticsLightHWFlatStraightHighSpeedPPORunnerCfg
):
    """Conservative PPO runner for the fixed +/-1 m/s recovery route."""

    max_iterations = 8000
    experiment_name = "deeprobotics_lighthw_flat_straight_recovery"
    run_name = "lighthw_flat_straight_recovery_v2"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "deeprobotics_lighthw_flat_straight_recovery"
        self.run_name = "lighthw_flat_straight_recovery_v2"
        # Start below route 06's fixed 1e-4 rate and restore KL adaptation.
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.001


@configclass
class DeeproboticsLightHWFlatStraightPosturePPORunnerCfg(
    DeeproboticsLightHWFlatStraightRecoveryPPORunnerCfg
):
    """Low-speed warm-start runner for the isolated posture-correction route."""

    max_iterations = 6000
    experiment_name = "deeprobotics_lighthw_flat_straight_posture"
    run_name = "lighthw_flat_straight_posture_v1"

    # The route changes the command distribution and adds posture terms.  A
    # small initial rate preserves the transferred velocity policy while the
    # reset optimizer learns those new terms.
    warm_start_noise_std = (
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.08,
        0.05,
        0.05,
        0.05,
        0.05,
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.experiment_name = "deeprobotics_lighthw_flat_straight_posture"
        self.run_name = "lighthw_flat_straight_posture_v1"
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.006
        self.algorithm.entropy_coef = 0.0005


@configclass
class DeeproboticsLightHWBlindRoughPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    """PPO runner for the no-height-observation rough LightHW task."""

    experiment_name = "deeprobotics_lighthw_blind_rough"
    run_name = "lighthw_rough_blind"
    fresh_start_only = True


@configclass
class DeeproboticsLightHWStableRollPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    """Fresh PPO runner for the stable low-speed rolling route."""

    experiment_name = "deeprobotics_lighthw_stable_roll"
    run_name = "lighthw_stable_roll_v1"
    fresh_start_only = True

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_lighthw_stable_roll"
        self.run_name = "lighthw_stable_roll_v1"
        # Preserve the M20-compatible network and optimizer layout while
        # reducing the initial wheel exploration that destabilized the prior run.
        self.policy.init_noise_std = 0.50
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 5.0e-4


@configclass
class DeeproboticsLightHWStableRollV3PPORunnerCfg(DeeproboticsLightHWStableRollPPORunnerCfg):
    """Isolated fresh-start runner for the corrected 1 ms physics baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_lighthw_stable_roll_v3"
        self.run_name = "lighthw_stable_roll_height0545_phys1ms_v3"


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsPPORunnerCfg(
    DeeproboticsLightHWRoughPPORunnerCfg
):
    """Fresh PPO runner for the 1 ms flat straight-line baseline.

    No checkpoint is loaded by the supplied route command.  ``fresh_start_only``
    also rejects an accidental resume or warm-start from the saturated route-08
    policy until the new task passes its first fixed-command playback.
    """

    max_iterations = 6000
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics"
    run_name = "lighthw_flat_straight_stable_physics_v1"
    fresh_start_only = True
    log_all_episode_metric_keys = True

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics"
        self.run_name = "lighthw_flat_straight_stable_physics_v1"
        self.fresh_start_only = True
        self.log_all_episode_metric_keys = True
        # Fresh actor noise is lower than the generic M20 default so the first
        # 1 ms baseline does not immediately command large wheel/hip excursions.
        self.policy.init_noise_std = 0.50
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.002


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsContinuePPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsPPORunnerCfg
):
    """Conservative low-speed continuation initialized from ``model_999``.

    This is a separate task/run entry point so the original fresh-start route
    remains protected from accidental checkpoint loading.  The warm-start
    command loads the complete actor, critic and learned per-action noise,
    resets only the PPO optimizer state and starts a new 1,500-update run.
    """

    max_iterations = 1500
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_continue"
    run_name = "lighthw_flat_straight_stable_physics_continue_v1"
    fresh_start_only = False
    log_all_episode_metric_keys = True

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_continue"
        self.run_name = "lighthw_flat_straight_stable_physics_continue_v1"
        self.fresh_start_only = False
        self.log_all_episode_metric_keys = True
        # Keep the learned action-noise vector from model_999; only lower the
        # optimizer step for the fine-tuning stage after the fresh baseline.
        self.algorithm.learning_rate = 2.5e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.001


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsPrecisionPPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsContinuePPORunnerCfg
):
    """Low-speed precision fine-tuning initialized from ``model_1499``.

    The learned actor, critic, and per-action exploration standard deviations
    are loaded by ``--warm_start_checkpoint``.  Only the PPO optimizer state
    and iteration counter are reset for this isolated reward-shaping run.
    """

    max_iterations = 1000
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_precision"
    run_name = "lighthw_flat_straight_stable_physics_precision_v1"
    fresh_start_only = False
    log_all_episode_metric_keys = True

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_precision"
        self.run_name = "lighthw_flat_straight_stable_physics_precision_v1"
        self.fresh_start_only = False
        self.log_all_episode_metric_keys = True
        # Keep the continuation optimizer settings and the source model's
        # learned action noise; this run isolates only reward precision.
        self.algorithm.learning_rate = 2.5e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.001


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsContactPPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsPrecisionPPORunnerCfg
):
    """500-update reward-removal experiment from continuation ``model_1499``."""

    max_iterations = 500
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_contact"
    run_name = "lighthw_flat_straight_stable_physics_contact_v1"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_contact"
        self.run_name = "lighthw_flat_straight_stable_physics_contact_v1"
        # 继承 precision 的全部 PPO 参数；不设置 warm_start_noise_std，
        # 保留来源模型学得的分动作探索噪声，以相同起点对照奖励移除效果。


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandGatePPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsContactPPORunnerCfg
):
    """500-update standing-gate experiment initialized from Contact model_499."""

    max_iterations = 500
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_gate"
    run_name = "lighthw_flat_straight_stable_physics_stand_gate_v1"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_gate"
        self.run_name = "lighthw_flat_straight_stable_physics_stand_gate_v1"
        # 完整继承 Contact 的 PPO 设置与来源模型学得的动作 std；
        # 优化器和局部轮次由 --warm_start_checkpoint 入口重置。


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandGatePPORunnerCfg
):
    """500-update zero-command holding trial initialized from StandGate."""

    max_iterations = 500
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_hold"
    run_name = "lighthw_flat_straight_stable_physics_stand_hold_v1"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_hold"
        self.run_name = "lighthw_flat_straight_stable_physics_stand_hold_v1"
        # All PPO settings and learned per-action noise are inherited. The
        # warm-start command resets only the optimizer and local iteration.


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionPPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPPORunnerCfg
):
    """500-update tracking-precision trial initialized from StandHold."""

    max_iterations = 500
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_hold_precision"
    run_name = "lighthw_flat_straight_stable_physics_stand_hold_precision_v1"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_hold_precision"
        self.run_name = "lighthw_flat_straight_stable_physics_stand_hold_precision_v1"
        # Keep the StandHold PPO settings and learned action std. The warm
        # start resets only optimizer state and the local iteration counter.


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionVx06PPORunnerCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionPPORunnerCfg
):
    """500-update +/-0.6 m/s trial initialized from StandHoldPrecision model_300."""

    max_iterations = 500
    experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_hold_precision_vx06"
    run_name = "lighthw_flat_straight_stable_physics_stand_hold_precision_vx06_v1"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "deeprobotics_lighthw_flat_straight_stable_physics_stand_hold_precision_vx06"
        self.run_name = "lighthw_flat_straight_stable_physics_stand_hold_precision_vx06_v1"
        # Keep the inherited PPO settings and learned per-action exploration
        # std. The warm-start entry resets only optimizer and local iteration.


@configclass
class DeeproboticsLightHWPPOActionCurriculumPPORunnerCfg(DeeproboticsLightHWRoughPPORunnerCfg):
    """Fresh PPO actor with checkpointed, performance-gated action progression.

    Only checkpoints carrying this route's curriculum state may be resumed.
    The route runner initializes per-action noise after resolving joint order;
    the generic scalar is kept valid for the installed RSL-RL constructor.
    """

    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "lighthw_ppo_action_curriculum"
    run_name = "lighthw_ppo_action_curriculum_v1"
    fresh_start_only = False
    lighthw_ppo_action_curriculum = True
    log_all_episode_metric_keys = True

    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 24
        self.max_iterations = 4000
        self.save_interval = 100
        self.experiment_name = "lighthw_ppo_action_curriculum"
        self.run_name = "lighthw_ppo_action_curriculum_v1"
        self.fresh_start_only = False
        self.lighthw_ppo_action_curriculum = True
        self.log_all_episode_metric_keys = True
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.noise_std_type = "log"
        self.policy.init_noise_std = 0.30
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.001

@configclass
class DeeproboticsLightHWPPOForwardSpeedPPORunnerCfg(DeeproboticsLightHWPPOActionCurriculumPPORunnerCfg):
    """PPO continuation with an explicit route-10 transfer or same-route resume."""

    experiment_name = "lighthw_ppo_forward_speed"
    run_name = "lighthw_ppo_forward_speed_v1"
    max_iterations = 3000
    lighthw_forward_speed = True
    # 只控制显示频率；所有原生 TensorBoard 标量仍每次 PPO 更新写入。
    lighthw_compact_console = True
    lighthw_console_interval = 10

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_ppo_forward_speed"
        self.run_name = "lighthw_ppo_forward_speed_v1"
        self.max_iterations = 3000
        self.lighthw_forward_speed = True


@configclass
class DeeproboticsLightHWPPOSpeedBridgePPORunnerCfg(DeeproboticsLightHWPPOForwardSpeedPPORunnerCfg):
    """从 route 11 的 0.8 策略继续 PPO；继承 Adam/LR，只细化速度课程。"""

    experiment_name = "lighthw_ppo_speed_bridge"
    run_name = "lighthw_speed_bridge_from1500_v1"
    max_iterations = 1200
    lighthw_speed_bridge = True

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_ppo_speed_bridge"
        self.run_name = "lighthw_speed_bridge_from1500_v1"
        self.max_iterations = 1200
        self.lighthw_speed_bridge = True


@configclass
class DeeproboticsLightHWGroupedNoisePPORunnerCfg(DeeproboticsLightHWPPOForwardSpeedPPORunnerCfg):
    """独立轮/腿探索短对照；全部16动作仍由原生PPO控制。"""

    experiment_name = "lighthw_ppo_grouped_noise"
    run_name = "lighthw_grouped_noise_from900_v1"
    max_iterations = 400
    lighthw_grouped_noise = True
    lighthw_speed_bridge = False

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_ppo_grouped_noise"
        self.run_name = "lighthw_grouped_noise_from900_v1"
        self.max_iterations = 400
        self.lighthw_grouped_noise = True
        self.lighthw_speed_bridge = False
        self.empirical_normalization = False
        self.policy.class_name = "LightHWGroupedNoiseActorCritic"
        # 命令观测为9:12、scale1；runner启动时核对真实manager布局。
        # wheel std为raw，乘动作scale5后得到rad/s；停车使用同一参数的0.4倍。
        self.policy.grouped_noise_config = {
            "policy_obs_dim": 60, "critic_obs_dim": 236, "actions": 16,
            "command_start": 9, "command_size": 3, "command_scale": (1.0, 1.0, 1.0),
            "stop_threshold": 0.05,
            "leg_std_init": 0.12, "leg_std_min": 0.08, "leg_std_max": 0.15,
            "wheel_std_init": 0.25, "wheel_std_min": 0.20, "wheel_std_max": 0.35,
            "stop_wheel_multiplier": 0.4,
            "hip_limit": 0.04, "leg_limit": 0.10, "wheel_scale": 5.0,
        }


@configclass
class DeeproboticsLightHWLegDesaturationPPORunnerCfg(DeeproboticsLightHWGroupedNoisePPORunnerCfg):
    """继承route13的PPO与分组噪声；独立保存仅新增腿raw奖励的400次对照。"""

    experiment_name = "lighthw_ppo_leg_desaturation"
    run_name = "lighthw_leg_desaturation_from900_v1"
    max_iterations = 400

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_ppo_leg_desaturation"
        self.run_name = "lighthw_leg_desaturation_from900_v1"
        self.max_iterations = 400


@configclass
class LightHWPPOReferenceRunnerCfg(DeeproboticsLightHWLegDesaturationPPORunnerCfg):
    """独立从零参考；PPO/噪声数值沿用route14，首轮1000次仅作观察预算。"""
    lighthw_reference = True
    lighthw_reference_run_role = "formal"
    lighthw_reference_short_check = False
    lighthw_observation_version = "lighthw_named_joint_positions_v1"

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "lighthw_ppo_observation_reference"
        self.run_name = "lighthw_reference_obs_v1"
        self.max_iterations = 1000
        self.save_interval = 100
        self.seed = 42
        self.resume = False
