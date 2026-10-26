"""统一 R1 的可序列化契约；无仿真导入，保留 Stage1 原契约。"""
from copy import deepcopy

PACKAGE = "rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified"
TASK = "MultiSkill-LightHW-Wheel-Unified-v0"
VERSION = "lighthw_r1_unified_v1"
HISTORY = 5
BASE_TERMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
              "joint_pos": 16, "joint_vel": 16, "actions": 16}
POLICY_TERMS = dict(BASE_TERMS, heading_error=1)
ANGLES = (-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.)
SLOPE_MIX = (0., 0., 0., 0., -5., 5., -10., 10., -15., 15., -20., 20., -25., 25.)
PAYLOADS = (0., 5., 10., 15.)
TOWED_MASSES = (0., 20., 40.)
CONTRACT = {
    "version": VERSION, "task": TASK, "policy_dim": 290, "critic_dim": 242, "action_dim": 16,
    "history_frames": HISTORY, "history_layout": "term-major, oldest-to-newest; last frame is current",
    "policy_terms": POLICY_TERMS, "policy_base_linear_velocity": False, "policy_height_scan": False,
    "heading_observation": "IMU heading relative to reset heading, radians",
    "critic_extra": ["surface_normal_b_x", "surface_normal_b_y", "surface_normal_b_z",
                     "payload_kg/15", "heading_error_rad", "towed_mass_kg/40"],
    "physics_dt": .001, "policy_dt": .02, "episode_s": 20.,
    "action_order": [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                     for joint in ("hip", "thigh", "calf")] + [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")],
    "leg_mapping": "clip(default + scale * raw, URDF_lower, URDF_upper)",
    "leg_scale": {"hip": .08, "thigh": .14, "calf": .14}, "leg_pd": [80., 3.],
    "wheel_scale": 5., "wheel_clip": [-37., 37.], "wheel_damping": 3.,
    "actuator_delay_steps": [0, 0], "empirical_normalization": False,
    "payload_kg": list(PAYLOADS), "payload_center_m": [0., 0., .20],
    "payload_size_m": [.30, .25, .20], "payload_model": "rigid_composite_mass_com_principal_inertia",
    "angles_deg": list(ANGLES), "slope_mix": list(SLOPE_MIX),
    "terrain": "eleven long constant-gradient lanes; signed tan(degrees)",
    "speed_mps": [.2, 3.1], "acceptance_speed_mps": 3.0, "zero_probability": .25,
    "high_speed_anchor_probability_given_moving": .4, "full_target_high_speed_probability": .7,
    "target_condition_fraction": .20, "command_period_s": [6., 8.],
    "towed_mass_kg": list(TOWED_MASSES), "towing_model": "passive_four_wheel_cart_compliant_drawbar",
    "default_rolling_coefficient": .02, "default_ground_friction": 1.3,
    "drawbar_rest_m": 1.0, "drawbar_stiffness_Npm": 2000., "drawbar_damping_Nspm": 150.,
    "training": "all conditions from update zero; one shared ActorCritic and one PPO optimizer",
    "urdf_sha256": "781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9",
    "robot_mass_kg": 22.0, "robot_model_version": "lighthw_uniform_22kg_v1",
}


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
        raise ValueError("Expected unified R1 checkpoint; use --warm_start for original Stage1")
    for name, tensor in value["model_state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite model tensor: {name}")
    if value["model_state_dict"]["actor.0.weight"].shape[1] != CONTRACT["policy_dim"]:
        raise ValueError("Unified Actor dimension mismatch")
    return value


def transfer_stage1(source, destination):
    """仅迁移网络参数；新观测列为零，优化器重建，不能称完整续训。"""
    import torch
    old = source["model_state_dict"]
    migrated = deepcopy(destination)
    for name in migrated:
        source_tensor = old[name].to(device=migrated[name].device, dtype=migrated[name].dtype)
        if name in ("actor.0.weight", "critic.0.weight"):
            migrated[name].zero_()
            if name == "actor.0.weight":
                if old[name].shape[1] != 57:
                    raise ValueError("Warm start requires original 57-dimensional Stage1 Actor")
                migrated[name][:, latest_source_columns()] = source_tensor
            else:
                if old[name].shape[1] != 236:
                    raise ValueError("Warm start requires original 236-dimensional Critic")
                migrated[name][:, :236] = source_tensor
        else:
            if migrated[name].shape != old[name].shape:
                raise ValueError(f"Incompatible warm-start tensor: {name}")
            migrated[name].copy_(source_tensor)
        if not torch.isfinite(migrated[name]).all():
            raise ValueError(f"Invalid warm-start tensor: {name}")
    return migrated
