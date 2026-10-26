"""R4拆分v4独立契约：所有阈值是首轮工程筛查标准，不是已验收性能。"""
VERSION = "lighthw_r4_split_v4"
REWARD_PROFILES = ("baseline_v4", "unload_diag_v1", "soft_transfer_v2")
TASKS = {"rear": "MultiSkill-LightHW-R4a-RearBalance-v4",
         "front": "MultiSkill-LightHW-R4b-FrontBalance-v4"}
SKILLS = {"R4a": "rear", "R4b": "front"}
LEGS = ("FR", "FL", "RR", "RL")
LEG_JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
WHEELS = tuple(f"{leg}_wheel_joint" for leg in LEGS)
WHEEL_BODIES = tuple(n.replace("_joint", "_link") for n in WHEELS)
STAND = (0., .6, -1.5, 0., .6, -1.5, 0., -.6, 1.5, 0., -.6, 1.5)
# 采用各前/后腿均可达的保守共同角区间；保留膝弯曲方向和硬限位余量。
LOW = (-.35, -1.45, -2.40) * 2 + (-.35, -1.45, .12) * 2
HIGH = (.35, 1.45, -.12) * 2 + (.35, 1.45, 2.40) * 2
# 连续项按秒积分；阶段完成/失败项一次性结算。先学会真正承载和卸载，再学安静保持。
BASE_WEIGHTS = dict(
    attitude=4., com_line=8., support=8., clearance=6., capture=8., pair=4.,
    swing_load=-2., swing_asymmetry=0., quiet=2., entry_quiet=1., entry_pose=.5,
    return_pose=1., return_attitude=2., balance=12., recovery=6.,
    hold_once=20., success_once=40., missed_hold=-8.,
    action_rate=-.025, torque=-.03, joint_speed=-.03, wheel_speed=-.015,
    saturation=-.05, drift=-1., failure=-20., time=-.1,
)

# 诊断档案只调整任务引导项；物理、严格保持和失败门槛保持不变。
DIAGNOSTIC_WEIGHTS = dict(
    BASE_WEIGHTS,
    attitude=3., com_line=5., support=10., clearance=8., capture=10.,
    swing_load=-4., swing_asymmetry=-2.,
)
# 新诊断保持D1的权重，只调整过渡门控、捕获稳定性及保持漂移的函数形状。
SOFT_TRANSFER_WEIGHTS = dict(DIAGNOSTIC_WEIGHTS)
TRANSFER_GATE_FLOOR = .25
HOLD_DRIFT_WIDTH_M = .10
WEIGHTS = BASE_WEIGHTS


def weights_for(profile="baseline_v4"):
    if profile == "baseline_v4":
        return BASE_WEIGHTS
    if profile == "unload_diag_v1":
        return DIAGNOSTIC_WEIGHTS
    if profile == "soft_transfer_v2":
        return SOFT_TRANSFER_WEIGHTS
    raise ValueError(f"unknown R4 reward profile: {profile}")


def contract(hold_seconds=2., mode="rear", reward_profile="baseline_v4"):
    if mode not in TASKS:
        raise ValueError("split v4 only supports rear (R4a) or front (R4b); no joint training")
    weights = weights_for(reward_profile)
    result = dict(
        robot_mass_kg=22.0, urdf_sha256="781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9", robot_model_version="lighthw_uniform_22kg_v1",
        version=VERSION, task=TASKS[mode], skill="R4a" if mode == "rear" else "R4b",
        reward_profile=reward_profile,
        hold_seconds=float(hold_seconds), mode=mode, independent_policy=True,
        observation_noise=False, reset_jitter_rad=0., domain_randomization=False, external_pushes=False,
        init_at_random_ep_len=False, automatic_frozen_evaluation=False,
        review_policy="only after user reports training completion and requests review",
        physics_dt=.001, policy_dt=.02, episode_seconds=12. + hold_seconds, task_seconds_after_entry=9. + hold_seconds,
        entry_min_start_s=1., entry_deadline_s=3., entry_noise_scale=.03, quiet_noise_phases=[0, 4], quiet_noise_scale=.03, hold_noise_scale=.25,
        phase_clock="event start after real 0.5s stable entry; transfer 3s, hold+1s, return 3s, recovery 2s",
        attitude_reward="normalized progress toward full final target, transfer/hold only",
        com_progress_width_m=.45, capture_com_width_m=.12, capture_attitude_width_rad=.35,
        support_reward="selected pair bearing >=65% total bodyweight gate AND swing unloading",
        recovery_reward_requires_held=True, missed_hold_penalty_once=8.,
        hold_exploration="0.25x std; 0.03x when observable target error<=12deg and angular speed<=0.25rad/s",
        hold_noise_target_tilt_deg=12., hold_noise_ang_speed_rad_s=.25,
        frame_dim=61, history=5, policy_dim=305, critic_dim=88, action_dim=16,
        action_order=list(LEG_JOINTS + WHEELS), low=list(LOW), high=list(HIGH), stand=list(STAND),
        action_mapping="piecewise linear about STAND to LOW/HIGH; four wheel velocity targets +/-15rad/s",
        leg_pd=[80., 3.], wheel_damping=1., leg_effort_Nm=48., wheel_effort_Nm=24.,
        leg_velocity_rad_s=12.57, wheel_velocity_rad_s=37.7, wheel_target_rad_s=15.,
        friction=1., gravity=[0., 0., -9.81], restitution=0., self_collisions=True,
        solver_iterations=[8, 2], rigid_max_angular_velocity_deg_s=100000.,
        gyroscopic_forces=True, payload_kg=0., towing_kg=0.,
        actor="IMU angular velocity/gravity, joint encoders, previous action, mode, target gravity, task time; 5 frames",
        privileged="COM/velocity, contact forces, clearance, base height, drift, hold/recovery timers",
        support_force_N=5., airborne_force_N=2., min_clearance_m=.03,
        max_line_error_m=.06, max_com_speed_m_s=.12, max_ang_speed_rad_s=.25,
        max_target_tilt_deg=12., max_hold_drift_m=.10, recovery_seconds=1.,
        max_recovery_drift_m=.10, max_episode_displacement_m=1.5,
        entry_seconds=.5, illegal_contact_N=5., max_tilt_deg=80.,
        weights=weights, optimizer="LR starts 1e-4; full-rollout KL>.03 restores model and Adam, halves LR to floor 1e-6", automatic_promotion=False, deployment_ready=False,
        initialization="fresh; never search for R1/R5/legacy checkpoints",
    )
    if reward_profile == "soft_transfer_v2":
        result.update(
            transfer_gate_floor=TRANSFER_GATE_FLOOR,
            support_reward="selected pair bearing * continuous total-load quality * worst-wheel unloading",
            support_total_load_gate="continuous clamp(total_wheel_load / (0.65 * bodyweight), 0, 1)",
            transfer_guidance="attitude and COM only: bearing * (0.25 + 0.75 * worst_wheel_unloading)",
            capture_stability="capture/pair multiply exp(-(angular_speed/0.35)^2-(com_speed/0.18)^2); capture/pair/hold quiet multiply exp(-(hold_drift/0.10)^2) in hold phase",
            hold_drift_cost="min((hold_drift / 0.10m)^2, 4); other phase costs unchanged",
        )
    return result


def runner_config(seed=42):
    return dict(
        seed=seed, device="cuda:0", num_steps_per_env=32, max_iterations=1000,
        save_interval=100, empirical_normalization=False, clip_actions=1., logger="tensorboard",
        policy=dict(class_name="PhaseActorCritic", init_noise_std=.2, noise_std_type="log",
                    actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation="elu"),
        algorithm=dict(class_name="PPO", num_learning_epochs=5, num_mini_batches=4,
                       clip_param=.2, gamma=.995, lam=.95, value_loss_coef=1.,
                       entropy_coef=.003, learning_rate=1e-4, max_grad_norm=1.,
                       use_clipped_value_loss=True, schedule="fixed", desired_kl=.01),
    )
