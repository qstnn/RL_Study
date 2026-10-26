"""Pure-Python validation for the isolated route-13 PPO exploration experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from .ppo_curriculum_core import LegCurriculumSchedule
from .ppo_forward_speed_core import GATE_PARAMETERS
from .ppo_speed_bridge_core import BRIDGE_SPEEDS, PROFILE_ID as BRIDGE_PROFILE_ID


PROFILE_ID = "lighthw_grouped_noise_v1"
GROUPED_NOISE_SPEEDS = (1.0,)


def _validate_source_provenance(provenance: dict) -> None:
    """核对原route 12保存的来源链；记录须完整，不能静默丢失更早迁移来源。"""
    if not isinstance(provenance, dict):
        raise ValueError("Route 12 transfer requires its original route-11 provenance")
    expected = {
        "source_profile": "lighthw_forward_speed_v1", "source_stage": 2,
        "source_max_speed_m_s": 0.8,
        "transfer_type": "actor_critic_optimizer_with_new_curriculum",
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Route 12 transfer provenance is not the approved route-11 stage-2 transfer")
    prior = provenance.get("source_transfer_source")
    if (not isinstance(prior, dict) or prior.get("source_profile") != "route_10_ppo_action_curriculum"
            or prior.get("source_stage") != 2 or prior.get("transfer_type") != "actor_critic_weights_only"):
        raise ValueError("Route 12 transfer requires the preserved route-10 provenance")
    for record in (provenance, prior):
        signature = record.get("source_signature")
        if (not isinstance(signature, str) or len(signature) != 64
                or any(character not in "0123456789abcdef" for character in signature)):
            raise ValueError("Route 12 transfer provenance has an invalid source signature")
        updates, steps = record.get("source_total_updates"), record.get("source_control_steps")
        if (isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0
                or isinstance(steps, bool) or not isinstance(steps, int) or steps != updates * 24):
            raise ValueError("Route 12 transfer provenance has inconsistent PPO/control-step counters")


def _validate_complete_quality_window(quality: dict, schedule: LegCurriculumSchedule) -> None:
    """允许速度/支撑未达标，但只接收当前1.0阶段真实完整且有限的评估窗口。

    这是候选迁移的完整性检查，不是行为验收：不能因为需要改善欠速，就要求
    源策略先通过速度门槛。原有所有质量字段保留，新的PPO窗口重新收集。
    """
    required = {"completed_episodes", "failed_episode_fraction", "nonfinite_samples",
                "evaluated_stage", "evaluated_max_speed_m_s", "window_passed"}
    group_keys = ("samples", "speed_error", "four_wheel_support", "tilt_bad_fraction", "base_height",
                  "lateral_speed", "yaw_rate", "body_contact_fraction", "command_vx", "actual_vx", "tilt_deg")
    required.update(f"{group}/{key}" for group in ("forward", "fast", "stop") for key in group_keys)
    if not isinstance(quality, dict) or not required.issubset(quality):
        raise ValueError("Route 12 transfer requires a complete evaluated 1.0 m/s quality window")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in quality.values()):
        raise ValueError("Route 12 transfer quality metrics must be finite numeric values")
    if (quality["evaluated_stage"] != 1 or quality["evaluated_max_speed_m_s"] != 1.0
            or quality["window_passed"] not in (0.0, 1.0)
            or schedule.stage_updates - schedule.window_updates < schedule.evaluation_interval):
        raise ValueError("Route 12 transfer quality window belongs to another stage or is incomplete")
    if (quality["completed_episodes"] <= 0 or quality["nonfinite_samples"] != 0
            or not 0.0 <= quality["failed_episode_fraction"] <= 1.0):
        raise ValueError("Route 12 transfer quality window has invalid episode or nonfinite-sample evidence")
    for group in ("forward", "fast", "stop"):
        if quality[f"{group}/samples"] < GATE_PARAMETERS["min_samples"]:
            raise ValueError("Route 12 transfer quality window is missing moving, fast or stop samples")
        for key in ("four_wheel_support", "tilt_bad_fraction", "body_contact_fraction"):
            if not 0.0 <= quality[f"{group}/{key}"] <= 1.0:
                raise ValueError("Route 12 transfer quality fractions must lie in [0, 1]")
        for key in ("speed_error", "lateral_speed", "yaw_rate", "tilt_deg"):
            if quality[f"{group}/{key}"] < 0:
                raise ValueError("Route 12 transfer quality errors and absolute measurements cannot be negative")


def validate_route12_transfer_state(state: dict, target_mapping: dict) -> dict:
    """只迁入route 12的1.0阶段；不修改姿态、动作、门槛或命令分布。

    唯一动作metadata变化是独立profile和固定1.0阶段表。分组噪声语义由runner
    单独保存和校验；模型/normalizer/Adam的完整性亦由runner预检。
    """
    if not isinstance(state, dict) or isinstance(state.get("version"), bool) or state.get("version") != 1:
        raise ValueError("Route 12 transfer requires complete version-1 curriculum metadata")
    source = state.get("action_mapping")
    if not isinstance(source, dict) or not isinstance(target_mapping, dict):
        raise ValueError("Route 12 transfer requires complete source and target action mappings")
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if state.get("signature") != digest:
        raise ValueError("Route 12 transfer action mapping signature is invalid")
    destination_expected = {
        "profile_id": PROFILE_ID, "hip_limits": [0.04], "leg_limits": [0.10],
        "max_speeds": list(GROUPED_NOISE_SPEEDS),
        "min_stage_updates": 200, "evaluation_interval": 100, "required_passes": 2,
        "gate_parameters": GATE_PARAMETERS,
        "observation_action_dimensions": {"policy": 60, "critic": 236, "action": 16},
        "command_distribution": {"forward_only": True, "zero_command_probability": 0.25,
                                 "anchor_probability": 0.5, "minimum_abs_speed": 0.08},
    }
    for key, value in destination_expected.items():
        if json.dumps(target_mapping.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Route 13 transfer destination changed the approved configuration: {key}")
    expected_source = deepcopy(target_mapping)
    expected_source.update(profile_id=BRIDGE_PROFILE_ID, hip_limits=[0.04] * len(BRIDGE_SPEEDS),
                           leg_limits=[0.10] * len(BRIDGE_SPEEDS), max_speeds=list(BRIDGE_SPEEDS))
    if json.dumps(source, sort_keys=True) != json.dumps(expected_source, sort_keys=True):
        raise ValueError("Route 12 transfer changes the profile, physical mapping, gates or command distribution")
    if len(source.get("leg_joints", [])) != 12 or len(source.get("wheel_joints", [])) != 4:
        raise ValueError("Route 12 transfer requires 12 leg and 4 wheel actions")
    original = LegCurriculumSchedule(
        hip_limits=(0.04,) * len(BRIDGE_SPEEDS), leg_limits=(0.10,) * len(BRIDGE_SPEEDS),
        max_speeds=BRIDGE_SPEEDS, min_stage_updates=200, evaluation_interval=100, required_passes=2,
    )
    try:
        original.load_state_dict(state["schedule"], reset_window=False)
    except (KeyError, TypeError) as exc:
        raise ValueError("Route 12 transfer has incomplete curriculum counters") from exc
    if (original.stage != 1 or state.get("hip_limit_rad") != 0.04
            or state.get("leg_limit_rad") != 0.10 or state.get("max_speed_m_s") != 1.0):
        raise ValueError("Only route 12 stage 1 at 1.0 m/s with unchanged leg limits can transfer")
    steps = state.get("control_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != original.total_updates * 24:
        raise ValueError("Route 12 transfer control steps must equal completed PPO updates times 24")
    _validate_complete_quality_window(state.get("last_quality_metrics"), original)
    _validate_source_provenance(state.get("transfer_source"))
    return {
        "source_profile": BRIDGE_PROFILE_ID, "source_signature": digest,
        "source_stage": original.stage, "source_total_updates": original.total_updates,
        "source_control_steps": steps, "source_max_speed_m_s": 1.0,
        "source_transfer_source": deepcopy(state["transfer_source"]),
        "source_quality_metrics": deepcopy(state["last_quality_metrics"]),
        "transfer_type": "actor_critic_optimizer_with_grouped_noise",
    }
