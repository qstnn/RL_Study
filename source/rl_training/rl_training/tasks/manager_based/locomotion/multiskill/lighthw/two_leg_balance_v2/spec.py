"""R4 v2 独立契约：所有阈值是首轮工程筛查标准，不是已验收性能。"""
VERSION = "lighthw_r4_two_leg_balance_v2"
TASK = "MultiSkill-LightHW-TwoLegBalance-v2"
LEGS = ("FR", "FL", "RR", "RL")
LEG_JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
WHEELS = tuple(f"{leg}_wheel_joint" for leg in LEGS)
WHEEL_BODIES = tuple(n.replace("_joint", "_link") for n in WHEELS)
STAND = (0., .6, -1.5, 0., .6, -1.5, 0., -.6, 1.5, 0., -.6, 1.5)
# 采用各前/后腿均可达的保守共同角区间；保留膝弯曲方向和硬限位余量。
LOW = (-.35, -1.45, -2.40) * 2 + (-.35, -1.45, .12) * 2
HIGH = (.35, 1.45, -.12) * 2 + (.35, 1.45, 2.40) * 2
WEIGHTS = dict(
    attitude=4., support=3., clearance=2., com_line=4., quiet=2., return_pose=1., entry_quiet=2.,
    balance=6., recovery=4., hold_once=10., success_once=20.,
    action_rate=-.025, torque=-.03, joint_speed=-.03, wheel_speed=-.015,
    saturation=-.05, drift=-1., failure=-8., time=-.1,
)


def contract(hold_seconds=2., mode="both"):
    return dict(
        robot_mass_kg=22.0, urdf_sha256="781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9", robot_model_version="lighthw_uniform_22kg_v1",
        version=VERSION, task=TASK, hold_seconds=float(hold_seconds), mode=mode,
        physics_dt=.001, policy_dt=.02, episode_seconds=12. + hold_seconds, task_seconds_after_entry=9. + hold_seconds,
        entry_min_start_s=1., entry_deadline_s=3., entry_noise_scale=.03, quiet_noise_phases=[0, 2, 4], quiet_noise_scale=.03,
        phase_clock="event start after real 0.5s stable entry; transfer 3s, hold+1s, return 3s, recovery 2s",
        attitude_width_rad=.9, com_width_m=.35, support_reward="selected load AND swing unloading; four-phase weight .25",
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
        weights=WEIGHTS, optimizer="LR starts 1e-4; full-rollout KL>.03 restores model and Adam, halves LR to floor 1e-6", automatic_promotion=False, deployment_ready=False,
        initialization="fresh; never search for R1/R5/legacy checkpoints",
    )


def runner_config(seed=42):
    return dict(
        seed=seed, device="cuda:0", num_steps_per_env=32, max_iterations=1000,
        save_interval=100, empirical_normalization=False, clip_actions=1., logger="tensorboard",
        policy=dict(class_name="PhaseActorCritic", init_noise_std=.2, noise_std_type="log",
                    actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation="elu"),
        algorithm=dict(class_name="PPO", num_learning_epochs=5, num_mini_batches=4,
                       clip_param=.2, gamma=.99, lam=.95, value_loss_coef=1.,
                       entropy_coef=.003, learning_rate=1e-4, max_grad_norm=1.,
                       use_clipped_value_loss=True, schedule="fixed", desired_kl=.01),
    )
