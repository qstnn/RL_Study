"""统一 R1 的可序列化契约；无仿真导入，保留 Stage1 原契约。"""
from copy import deepcopy

PACKAGE = "rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8"
TASK = "MultiSkill-LightHW-Wheel-MotorSizingV8-v0"
VERSION = "lighthw_r1_direct_v8"
HISTORY = 5
BASE_TERMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
              "joint_pos": 16, "joint_vel": 16, "actions": 16}
POLICY_TERMS = dict(BASE_TERMS, heading_error=1, task_context=2)
ANGLES = (-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.)
SLOPE_MIX = (0., 0., 0., 0., -5., 5., -10., 10., -15., 15., -20., 20., -25., 25.)
PAYLOADS = (0., 5., 10., 15.)
TOWED_MASSES = (0., 20., 40.)
CONTRACT = {
    "version": VERSION, "task": TASK, "policy_dim": 300, "critic_dim": 268, "action_dim": 16,
    "requirements_revision": "2026-10-16_no_towing_at_minus25",
    "excluded_conditions": [{"slope_deg": -25., "towed_kg_gt": 0.}],
    "training_condition_count": 64, "evaluation_case_count": 18,
    "minus25_requirements": "unloaded >=3m/s; payload15 downhill without speed threshold; no trailer",
    "history_frames": HISTORY, "history_layout": "term-major, oldest-to-newest; last frame is current",
    "policy_terms": POLICY_TERMS, "policy_base_linear_velocity": False, "policy_height_scan": False,
    "heading_observation": "IMU heading relative to reset heading, radians",
    "critic_extra": ["surface_normal_b_x", "surface_normal_b_y", "surface_normal_b_z",
                     "payload_kg/15", "heading_error_rad", "towed_mass_kg/40"],
    "reward_profile": "v5_complete_cycle_smooth_loaded_progress",
    "critic_task_state_extra": ["hold_route_displacement/.05", "hold_lateral_displacement/.05", "hold_elapsed/7", "requested_vx/3.1", "phase0", "phase1", "phase2", "phase3", "phase_elapsed/30", "remaining_distance/goal"],
    "slope_posture": "uphill relative pitch [-min(.4*slope,10deg),0]; downhill +/-min(.2*abs(slope),5deg); roll constrained",
    "default_leg_pose_penalty": "flat zero-command only; disabled on slopes >=2deg",
    "hitch_clearance_margin_m": .05, "hitch_failure_clearance_m": .005,
    "physics_dt": .001, "policy_dt": .02, "episode_s": 65.,
    "action_order": [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                     for joint in ("hip", "thigh", "calf")] + [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")],
    "leg_mapping": "clip(default + scale * raw, URDF_lower, URDF_upper)",
    "leg_scale": {"hip": .08, "thigh": .14, "calf": .14}, "leg_pd": [80., 3.],
    "wheel_scale": 5., "wheel_clip": [-50., 50.], "wheel_joint_velocity_limit_rad_s":60., "wheel_damping": 2.,
    "actuator_delay_steps": [0, 0], "empirical_normalization": False,
    "payload_kg": list(PAYLOADS), "payload_center_m": [0., 0., .20],
    "payload_size_m": [.30, .25, .20], "payload_model": "rigid_composite_mass_com_principal_inertia",
    "angles_deg": list(ANGLES), "slope_mix": list(SLOPE_MIX),
    "terrain": "eleven long constant-gradient lanes; signed tan(degrees)",
    "unloaded_speed_mps": [.2, 3.1], "unloaded_acceptance_speed_mps": 3.0, "loaded_command_mps": [.2,.6], "loaded_minimum_speed_requirement": None, "zero_probability": .25, "standing_command_zero_on_reset": True,
    "payload_towing_mutually_exclusive": True, "condition_class_weights": {"unloaded":.4,"payload_only":.3,"towing_only":.3},
    "unloaded_high_speed_anchor_probability_given_moving": .4,
    "target_condition_fraction": .20, "command_period_s": "full_task_cycle",
    "command_slew_mps2": {"unloaded":1.0,"loaded":.3}, "training_reset_zero_s":3.0,
    "standing_after_applied_command_zero":True, "hold_drift_tolerance_m":.05,
    "tracking_frame":"fixed route tangent and lateral, independent of body heading",
    "positive_tracking_safety_gate":"smooth posture/heading/height/hitch; negative costs remain active",
    "initialization":"Actor and compatible Critic; fresh Adam and calibrated per-action noise",
    "towed_mass_kg": list(TOWED_MASSES), "towing_model": "passive_four_wheel_cart_compliant_drawbar",
    "default_rolling_coefficient": .02, "default_ground_friction": 4.0,
    "default_leg_torque_limit_Nm":96., "default_wheel_torque_limit_Nm":48., "default_hitch_height_m":.07,
    "torque_measurement_hz":1000, "torque_measurement_frame":"joint output shaft",
    "effort_reward_denominator":"explicit actuator effort_limit in actual joint order",
    "loaded_evaluation":"travel 1m, slew to zero, settle 2s, hold 5s; 60s budget, no moving speed threshold",
    "unloaded_evaluation":"zero3s, target3.1 until11s, slew to zero, settle2s, hold5s; 30s budget",
    "drawbar_rest_m": 1.0, "drawbar_stiffness_Npm": 2000., "drawbar_damping_Nspm": 150.,
    "training": "all conditions from update zero; one shared ActorCritic and one PPO optimizer",
    "urdf_sha256": "781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9",
    "robot_mass_kg": 22.0, "robot_model_version": "lighthw_uniform_22kg_v1",
}
CONTRACT.update(policy_known_task_context=["nominal payload_kg/15", "nominal trailer_kg/40"],
    wheel_body_angular_cap_deg_s=4000., other_body_angular_cap_deg_s=1000.,
    reset_height_m=.416223, hitch_reference="actual reset clearance on nominal surface",
    initialization="known V4/V5/V8 Actor and compatible Critic transfer; zero-extended task columns; new Adam and calibrated noise",
    behavior_scope="Training readiness is separate from final 16-case task acceptance",
    reward_weights_changed={"surface_height":-200., "stand_drift":-2., "towing_support_margin":-3.},
    towing_support_target_fraction=.5,
    ppo_learning_rate=1.e-5, critic_learning_rate=1.e-4, ppo_schedule="fixed_with_measured_kl_stop",
    initial_noise_std_by_action=[.005]*12+[.01]*4, observation_noise_scale=.1,
    ppo_kl_stop_threshold=.01, ppo_kl_guard="full-rollout post-step backtracking 1..1/256; reject Actor and Adam if none acceptable; Critic continues",
    ppo_optimizer_state_version=3,
    ppo_rollout_steps=64, gamma=.995, gae_lambda=.95, critic_warmup_updates=50,
    periodic_evaluation="baseline and every100; named-case regression gate and best eligible checkpoint",
    loaded_progress_reference_mps=.2, loaded_progress_weight=6.,
    reference_cache_schema="case_stage_success_v1",
    reference_static_stages=["initial_settle", "static_mid", "static_hold"],
    reference_moving_stages=["wait", "move", "brake", "stop_settle", "stop_hold"],
    reference_loss_coef=.1, reference_sampling="uniform_case_phase_group_then_uniform_frame",
    reference_max_group_mse=.02, reference_action_scale=[.005]*12+[.01]*4,
    min_noise_std_by_action=[.001]*12+[.002]*4, max_noise_std_by_action=[.005]*12+[.01]*4,
    ignored_policy_columns=[205,285], ignored_policy_column_semantics="all five previous-action frames; weights and Adam moments remain zero")

CONTRACT.update(
    reward_profile="v8_rl_towing_signed_robot_cart_progress",
    exploration="same 300->16 mean Actor; commanded towing motion multiplies std up to8; all probabilities use conditional std",
    initial_noise_std_by_action=[.04]*12+[.08]*4,
    min_noise_std_by_action=[.04]*12+[.08]*4, max_noise_std_by_action=[.04]*12+[.08]*4,
    ppo_learning_rate=3.e-5, critic_warmup_updates=20,
    reference_loss_coef=.1, reference_max_group_mse=0., reference_action_scale=[.2]*16,
    towing_support_target_fraction=.25,
    towing_height_free_band_m=[.36,.44],
    condition_class_weights={"unloaded":.25,"payload_only":.25,"towing_only":.5},
    towing_progress_weight=8., towing_slip_weight=-.05, towing_failure_extra_weight=-400.,
    distance_reached_bonus=10., teacher_required=False,
    training_readiness="software/runtime integration; heavy towing behavior is the PPO learning objective",
    observation_noise_scale=.1,
)



# V8为独立直接RL，不依赖V7教师门；部署Actor/物理接口保持，Critic增加拖车真值。
CONTRACT.update(
    reward_profile="v8_joint_net_progress_brake_complete_hold",
    critic_extra_v8=["slope_rad/(pi/4)","cart_vx", "cart_vy", "cart_progress",
                     "robot_progress", "joint_progress", "drawbar_force/300", "drawbar_extension",
                     "hitch_clearance/.07", "wheel_N_FR/30", "wheel_N_FL/30",
                     "wheel_N_RR/30", "wheel_N_RL/30", "hold_support_fraction",
                     "cart_hold_drift/.05", "cart_tilt_rad"],
    exploration="towing command >=.05: raw std .12/.24; zero and other tasks .005/.01; one conditional Gaussian",
    initial_noise_std_by_action=[.12]*12+[.24]*4,
    min_noise_std_by_action=[.12]*12+[.24]*4,
    max_noise_std_by_action=[.12]*12+[.24]*4,
    ppo_learning_rate=1.e-5, ppo_gamma=.998, ppo_lam=.97, rollout_steps=128,
    reference_loss_coef=.2, reference_max_group_mse=0., reference_action_scale=[.1]*12+[.2]*4,
    target_condition_fraction=.30,
    hard_tow_anchors=[{"slope_deg":25.,"towed_kg":40.},{"slope_deg":-20.,"towed_kg":40.}],
    hard_tow_standing_probability=.1, loaded_command_mps=[.3,.3],
    command_sampling="deterministic per-slot cyclic sequence; unloaded3.1, loaded.3, standing0",
    policy_enable_corruption=False, physical_randomization=False, initial_state_randomization=False,
    training_stage="fixed_conditions_first; PPO Gaussian action exploration retained",
    training_readiness="physics/interface, sampling, finite accepted PPO updates and resume; task success is an outcome",
    distance_trigger="min(robot,cart) net displacement >=1m for towing; robot displacement for payload",
    towing_progress_weight=12., towing_progress_units="joint net displacement m/s, no positive-only accumulation",
    towing_brake_weight=-4., complete_hold_bonus=20., teacher_required=False,
    default_ground_friction=3., ppo_rollout_steps=128, gamma=.998, gae_lambda=.97,
    ppo_optimizer_state_version=4, behavior_scope="18 boundary screen then 128 cases and independent disturbances",
    evaluation_inference="row by default; training batch/noise requires separate validation",
)


def register_task():
    import gymnasium as gym
    kwargs = {"env_cfg_entry_point": f"{PACKAGE}.env_cfg:UnifiedEnvCfg",
              "rsl_rl_cfg_entry_point": f"{PACKAGE}.ppo_cfg:UnifiedRunnerCfg"}
    entry = f"{PACKAGE}.env:UnifiedEnv"
    if TASK in gym.registry:
        if gym.spec(TASK).entry_point != entry or gym.spec(TASK).kwargs != kwargs:
            raise ValueError(f"Conflicting registration: {TASK}")
        return
    gym.register(TASK, entry_point=entry, disable_env_checker=True, kwargs=kwargs)


def latest_source_columns():
    """Stage1 的57列映射到每个历史项的最新帧；新heading及旧历史初始权重为零。"""
    columns, offset = [], 0
    for width in BASE_TERMS.values():
        columns.extend(range(offset + (HISTORY-1)*width, offset + HISTORY*width))
        offset += HISTORY*width
    return columns


def checkpoint(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (value.get("infos") or {}).get("r1_unified_contract") != CONTRACT:
        raise ValueError("Expected the current V8 contract; use compatible V4/V5/V8 network initialization for changed training settings")
    for name, tensor in value["model_state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite model tensor: {name}")
    if value["model_state_dict"]["actor.0.weight"].shape[1] != CONTRACT["policy_dim"]:
        raise ValueError("Unified Actor dimension mismatch")
    return value


def initialization_checkpoint(path):
    """已知V4/V5/V8策略只作网络初始化；不接受任意57/60维旧Actor。"""
    import torch
    value=torch.load(path,map_location="cpu",weights_only=False)
    contract=(value.get("infos") or {}).get("r1_unified_contract",{})
    if contract.get("version") not in (VERSION,"lighthw_r1_unified_v6","lighthw_r1_unified_v5","lighthw_r1_unified_v4"):
        raise ValueError("Initialization requires a known 300-dimensional R1 V4/V5/V8 checkpoint")
    state=value["model_state_dict"]
    if state["actor.0.weight"].shape!=(512,300) or state["critic.0.weight"].shape not in ((512,242),(512,252),(512,268)):
        raise ValueError("R1 initialization dimensions do not match")
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Non-finite initialization checkpoint")
    if torch.count_nonzero(state["actor.0.weight"][:,205:285]):
        raise ValueError("V8 initialization must not restore the unstable action-history shortcut")
    return value


def transfer_policy(source,destination):
    """迁移同接口Actor和Critic；V4新增10列先清零，保持旧Critic函数，再通过预热学习新任务状态。

    新Adam和按关节标定的探索噪声保留；只接受声明过的观测布局，不按维度猜测任意旧路线。
    """
    migrated=deepcopy(destination)
    for name,tensor in migrated.items():
        if not name.startswith(("actor.","critic.")):
            continue
        previous=source["model_state_dict"][name].to(device=tensor.device,dtype=tensor.dtype)
        if name=="critic.0.weight" and previous.shape[0]==512 and previous.shape[1] in (242,252) and tensor.shape==(512,268):
            tensor.zero_();tensor[:,:previous.shape[1]].copy_(previous)
        elif previous.shape==tensor.shape:
            tensor.copy_(previous)
        else:
            raise ValueError(f"Network transfer shape mismatch: {name}: {previous.shape} -> {tensor.shape}")
    return migrated


def install_policy_input_mask(policy):
    """固定屏蔽上一动作80列，防止PPO重新引入离线迁移中已排除的自反馈捷径。

    其他220列继续训练，Critic不受影响；导出仍为标准300→16的单MLP。
    新Adam和合法恢复的这部分动量均为零，因此零梯度使其严格保持为零。
    """
    import torch
    first=policy.actor[0]
    if first.weight.shape!=(512,300):raise ValueError("Unexpected Actor input layer")
    mask=torch.ones_like(first.weight);mask[:,205:285]=0.
    with torch.no_grad():first.weight[:,205:285]=0.
    policy._r1_input_mask_hook=first.weight.register_hook(lambda gradient:gradient*mask)


def validate_policy_input_mask(policy,optimizer=None):
    import torch
    first=policy.actor[0].weight
    if torch.count_nonzero(first[:,205:285]):raise RuntimeError("PPO reintroduced masked action history")
    if optimizer is not None:
        state=optimizer.state.get(first,{})
        for name in ("exp_avg","exp_avg_sq"):
            if name in state and torch.count_nonzero(state[name][:,205:285]):
                raise RuntimeError("Masked inputs have nonzero Adam history: "+name)
