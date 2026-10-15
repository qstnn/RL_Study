"""Simulation-free route catalog and versioned actor contract."""

from copy import deepcopy

PACKAGE = "rl_training.tasks.manager_based.locomotion.multiskill.lighthw"
VERSION = "lighthw_multiskill_wheel_stage1_v1"
TASK = "MultiSkill-LightHW-Wheel-Stage1-v0"
ROUTES = {
    "00": {"directory": "route_00_baseline_reference", "status": "reference", "task_id": None},
    "01": {"directory": "route_01_wheel_locomotion", "status": "implemented_runtime_pending", "task_id": TASK},
    "02a": {"directory": "route_02_legged_stairs/running", "status": "planned", "task_id": None},
    "02b": {"directory": "route_02_legged_stairs/stairs", "status": "planned", "task_id": None},
    "03": {"directory": "route_03_high_obstacle", "status": "feasibility_pending", "task_id": None},
    "04": {"directory": "route_04_two_leg_balance", "status": "planned", "task_id": None},
    "05a": {"directory": "route_05_acrobatics/backflip", "status": "feasibility_pending", "task_id": None},
    "05b": {"directory": "route_05_acrobatics/sideflip", "status": "feasibility_pending", "task_id": None},
    "06": {"directory": "route_06_heavy_payload", "status": "implemented_runtime_pending", "task_id": "MultiSkill-LightHW-HeavyPayload-v0"},
}
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                    for joint in ("hip", "thigh", "calf", "wheel"))
LEG_NAMES = tuple(n for n in JOINT_NAMES if "wheel" not in n)
WHEEL_NAMES = tuple(n for n in JOINT_NAMES if "wheel" in n)
WHEEL_BODIES = tuple(n.replace("_joint", "_link") for n in WHEEL_NAMES)
POLICY_TERMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
                "joint_pos": 16, "joint_vel": 16, "actions": 16}
CONTRACT = {
    "version": VERSION, "task": TASK, "stage": 1,
    "urdf_sha256": "5abb7c07c0ad401f7940018fee739ecc2fb4e0d3e8fc83274e75d1886db19fbb",
    "leg_mapping": "clip(default + scale * raw, URDF_lower, URDF_upper)",
    "default_leg_angles": [0.0, 0.6, -1.5, 0.0, 0.6, -1.5, 0.0, -0.6, 1.5, 0.0, -0.6, 1.5],
    "leg_pd": [80.0, 3.0], "wheel_damping": 1.0, "actuator_delay_steps": [0, 0],
    "policy_scales": {"base_ang_vel": 0.5, "projected_gravity": 1.0, "velocity_commands": 1.0,
                      "joint_pos": 1.0, "joint_vel": 0.05, "actions": 1.0},
    "policy_dim": 57, "critic_dim": 236, "action_dim": 16,
    "policy_terms": POLICY_TERMS,
    "action_order": list(LEG_NAMES + WHEEL_NAMES), "joint_observation_order": list(JOINT_NAMES),
    "wheel_position_channels": [12, 16, 20, 24], "wheel_velocity_channels": [28, 32, 36, 40],
    "leg_scale": {"hip": 0.08, "thigh": 0.14, "calf": 0.14},
    "leg_clip": "canonical_URDF_limits", "wheel_scale": 5.0, "wheel_clip": [-37.0, 37.0],
    "physics_dt": 0.001, "policy_dt": 0.02, "empirical_normalization": False,
    "actor_base_linear_velocity": False, "actor_height_scan": False,
    "payload_kg": 0.0, "towing_force_N": 0.0, "terrain": "plane",
}


def implemented_route(key):
    route = deepcopy(ROUTES[key])
    if route["task_id"] is None:
        raise ValueError(f"Route {key} is {route['status']}; no implemented training task")
    return route


def register_tasks():
    """Register only implemented tasks, without importing Isaac Sim configs."""
    import gymnasium as gym

    kwargs = {"env_cfg_entry_point": f"{PACKAGE}.wheel_env_cfg:WheelStage1EnvCfg",
              "rsl_rl_cfg_entry_point": f"{PACKAGE}.ppo_cfg:WheelStage1RunnerCfg"}
    if TASK in gym.registry:
        spec = gym.spec(TASK)
        if spec.entry_point != "isaaclab.envs:ManagerBasedRLEnv" or spec.kwargs != kwargs:
            raise ValueError(f"Conflicting task registration: {TASK}")
        return
    gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                 disable_env_checker=True, kwargs=kwargs)
