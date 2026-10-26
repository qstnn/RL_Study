"""R6 固定载荷分级持载契约；不导入仿真。"""
from copy import deepcopy

TASK = "MultiSkill-LightHW-HeavyPayload-v0"
VERSION = "lighthw_multiskill_heavy_payload_v1"
DIRECTORY = "route_06_heavy_payload"
PACKAGE = "rl_training.tasks.manager_based.locomotion.multiskill.lighthw"
CONTRACT = {
    "version": VERSION, "task": TASK, "policy_dim": 58, "critic_dim": 237, "action_dim": 16,
    "payload_model": "fixed_rigid_composite_base_inertia_v1",
    "payload_center_m": [0.0, 0.0, 0.20], "payload_box_size_m": [0.30, 0.25, 0.20],
    "payload_mass_range_kg": [0.0, 65.0], "payload_observation": "configured_mass_kg / 65; last channel",
    "policy_base_linear_velocity": False, "policy_height_scan": False,
    "physics_dt": 0.001, "policy_dt": 0.02, "leg_pd": [80.0, 3.0],
    "wheel_damping": 1.0, "leg_scale": {"hip": 0.08, "thigh_calf": 0.14},
    "wheel_scale": 5.0, "wheel_clip": [-37.0, 37.0], "empirical_normalization": False,
    "urdf_sha256": "781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9",
    "robot_mass_kg": 22.0, "robot_model_version": "lighthw_uniform_22kg_v1",
    "command": [0.0, 0.0, 0.0], "terrain": "plane", "episode_s": 20.0,
}
STAGES_KG = (0, 5, 10, 15, 25, 40, 65)
SCREEN = {"tilt_p95_deg": 5.0, "tilt_max_deg": 15.0, "speed_p95_m_s": 0.05,
          "yaw_rate_p95_rad_s": 0.05, "height_error_p95_m": 0.04,
          "four_wheel_fraction": 0.95, "torque_saturation_fraction": 0.05,
          "hold_displacement_max_m": 0.05}


def register_task():
    import gymnasium as gym
    kwargs = {"env_cfg_entry_point": f"{PACKAGE}.payload_env_cfg:HeavyPayloadEnvCfg",
              "rsl_rl_cfg_entry_point": f"{PACKAGE}.payload_ppo_cfg:HeavyPayloadRunnerCfg"}
    if TASK in gym.registry:
        spec = gym.spec(TASK)
        if spec.entry_point != "isaaclab.envs:ManagerBasedRLEnv" or spec.kwargs != kwargs:
            raise ValueError(f"Conflicting task registration: {TASK}")
        return
    gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs=kwargs)


def validate_mass(value):
    import math
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 65.0:
        raise ValueError("R6 payload must be finite and between 0 and 65 kg")
    return value


def checkpoint(path):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("infos") or {}).get("r6_contract") != CONTRACT:
        raise ValueError("Expected an R6 checkpoint; R1/legacy policies are incompatible")
    for name, tensor in saved["model_state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    return saved
