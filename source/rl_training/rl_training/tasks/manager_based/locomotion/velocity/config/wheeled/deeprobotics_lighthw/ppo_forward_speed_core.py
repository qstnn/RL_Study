"""Pure-Python gates and explicit route 10 migration checks for forward PPO."""

from __future__ import annotations

import hashlib
import json
import math

from .ppo_curriculum_core import LegCurriculumSchedule


PROFILE_ID = "lighthw_forward_speed_v1"
FORWARD_SPEEDS = (0.2, 0.4, 0.8, 1.2, 1.8, 2.4, 3.2)
GATE_PARAMETERS = {
    "settle_seconds": 2.0,
    "fast_command_fraction": 0.8,
    "standing_command_threshold": 0.05,
    "moving_speed_error_min": 0.06,
    "moving_speed_error_fraction": 0.12,
    "moving_yaw_rate": 0.08,
    "moving_lateral_min": 0.04,
    "moving_lateral_fraction": 0.05,
    "moving_tilt_degrees": 10.0,
    "moving_tilt_bad_fraction": 0.02,
    "moving_four_wheel_support": 0.90,
    "stop_speed_error": 0.025,
    "stop_yaw_rate": 0.035,
    "stop_lateral_speed": 0.025,
    "stop_tilt_degrees": 5.0,
    "stop_tilt_bad_fraction": 0.01,
    "stop_four_wheel_support": 0.98,
    "body_contact_fraction": 0.001,
    "wheel_support_force_n": 5.0,
    "body_contact_force_n": 1.0,
    "failed_episode_fraction": 0.01,
    "min_completed_episodes": 8,
    "min_samples": 100,
    "samples_per_env": 10,
}


def forward_quality_gate(metrics: dict[str, float], *, max_speed: float, min_samples: int = 100,
                         parameters: dict | None = None) -> bool:
    """独立检查正向、当前上限附近和停车；不要求后退或名义机身高度。

    数据来自含探索噪声的 50 Hz 训练窗口。高速子集不可被低速样本平均掉；
    通过只允许扩大下一阶段命令，不替代确定性回放和物理步接触验收。
    """
    gate = GATE_PARAMETERS if parameters is None else parameters
    if not math.isfinite(max_speed) or max_speed <= 0:
        return False
    required = ["completed_episodes", "failed_episode_fraction", "nonfinite_samples"]
    keys = ("samples", "speed_error", "four_wheel_support", "tilt_bad_fraction",
            "lateral_speed", "yaw_rate", "body_contact_fraction")
    required.extend(f"{group}/{key}" for group in ("forward", "fast", "stop") for key in keys)
    if any(key not in metrics or not math.isfinite(metrics[key]) or metrics[key] < 0 for key in required):
        return False
    if (metrics["nonfinite_samples"] != 0
            or metrics["completed_episodes"] < gate["min_completed_episodes"]
            or metrics["failed_episode_fraction"] > gate["failed_episode_fraction"]):
        return False
    for group in ("forward", "fast", "stop"):
        moving = group != "stop"
        speed_error = (max(gate["moving_speed_error_min"], gate["moving_speed_error_fraction"] * max_speed)
                       if moving else gate["stop_speed_error"])
        lateral = (max(gate["moving_lateral_min"], gate["moving_lateral_fraction"] * max_speed)
                   if moving else gate["stop_lateral_speed"])
        prefix = "moving" if moving else "stop"
        if (metrics[f"{group}/samples"] < min_samples
                or metrics[f"{group}/speed_error"] > speed_error
                or metrics[f"{group}/four_wheel_support"] < gate[f"{prefix}_four_wheel_support"]
                or metrics[f"{group}/four_wheel_support"] > 1.0
                or metrics[f"{group}/tilt_bad_fraction"] > gate[f"{prefix}_tilt_bad_fraction"]
                or metrics[f"{group}/lateral_speed"] > lateral
                or metrics[f"{group}/yaw_rate"] > gate[f"{prefix}_yaw_rate"]
                or metrics[f"{group}/body_contact_fraction"] > gate["body_contact_fraction"]):
            return False
    return True


def validate_route10_transfer_state(state: dict, target_mapping: dict) -> dict:
    """仅批准完整 route 10 阶段 2 metadata 的显式迁移，返回可记录的来源。

    网络 60/236/16 和张量有限性由加载网络的 runner 另行检查。这里校验签名、
    原课程合法计数、当前腿映射、原姿态和关节顺序。轮速仅允许 10→37 rad/s
    的裁剪范围扩展，scale 必须保持 5；迁移不代表可继续使用旧 PPO 优化器。
    """
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Route 10 transfer requires complete version-1 curriculum metadata")
    source = state.get("action_mapping")
    if not isinstance(source, dict):
        raise ValueError("Route 10 transfer requires an action mapping")
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if state.get("signature") != digest:
        raise ValueError("Route 10 transfer action mapping signature is invalid")
    original = LegCurriculumSchedule()
    expected = {
        "mapping": "default_plus_tanh_limits_v1",
        "hip_limits": list(original.hip_limits), "leg_limits": list(original.leg_limits),
        "max_speeds": list(original.max_speeds), "min_stage_updates": original.min_stage_updates,
        "evaluation_interval": original.evaluation_interval, "required_passes": original.required_passes,
    }
    if "profile_id" in source or any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("Transfer source is not the original route 10 curriculum")
    try:
        original.load_state_dict(state["schedule"], reset_window=False)
    except (KeyError, TypeError) as exc:
        raise ValueError("Route 10 transfer has incomplete curriculum counters") from exc
    if (original.stage != 2 or state.get("hip_limit_rad") != 0.04 or state.get("leg_limit_rad") != 0.10
            or state.get("max_speed_m_s") != 0.20):
        raise ValueError("Only route 10 stage 2 with hip 0.04 / thigh-calf 0.10 rad can transfer")
    steps = state.get("control_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < original.total_updates:
        raise ValueError("Route 10 transfer has invalid control step count")
    for key in ("mapping", "leg_joints", "wheel_joints", "default_positions", "absolute_leg_clip",
                "physics_dt", "control_dt"):
        # JSON gives tuple/list clips the same canonical representation after torch/JSON round trips.
        if key not in source or json.dumps(source[key], sort_keys=True) != json.dumps(target_mapping.get(key), sort_keys=True):
            raise ValueError(f"Route 10 transfer changes the physical mapping: {key}")
    if len(source["leg_joints"]) != 12 or len(source["wheel_joints"]) != 4:
        raise ValueError("Route 10 transfer requires 12 leg and 4 wheel actions")
    if target_mapping.get("profile_id") != PROFILE_ID:
        raise ValueError("Transfer destination must be the isolated forward-speed profile")
    if source.get("wheel_scale") != 5.0 or target_mapping.get("wheel_scale") != 5.0:
        raise ValueError("Route 10 transfer must preserve wheel action scale=5")
    if (json.dumps(source.get("wheel_clip"), sort_keys=True) != json.dumps({".*_wheel_joint": [-10.0, 10.0]}, sort_keys=True)
            or json.dumps(target_mapping.get("wheel_clip"), sort_keys=True) != json.dumps({".*_wheel_joint": [-37.0, 37.0]}, sort_keys=True)):
        raise ValueError("Only the explicit wheel clip extension from +/-10 to +/-37 rad/s is allowed")
    if (target_mapping.get("hip_limits") != [0.04] * len(FORWARD_SPEEDS)
            or target_mapping.get("leg_limits") != [0.10] * len(FORWARD_SPEEDS)):
        raise ValueError("Route 10 transfer must preserve stage-2 leg limits throughout the forward curriculum")
    return {
        "source_profile": "route_10_ppo_action_curriculum", "source_signature": digest,
        "source_stage": original.stage, "source_total_updates": original.total_updates,
        "source_control_steps": steps, "transfer_type": "actor_critic_weights_only",
    }
