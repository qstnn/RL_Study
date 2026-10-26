"""Pure-Python validation for the isolated 0.8 -> 1.0 -> 1.2 m/s PPO route."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .ppo_curriculum_core import LegCurriculumSchedule
from .ppo_forward_speed_core import (
    FORWARD_SPEEDS, GATE_PARAMETERS, PROFILE_ID as FORWARD_PROFILE_ID, forward_quality_gate,
)


PROFILE_ID = "lighthw_speed_bridge_v1"
BRIDGE_SPEEDS = (0.8, 1.0, 1.2)


def validate_route11_transfer_state(state: dict, target_mapping: dict) -> dict:
    """只接受已通过一个完整0.8窗口的route 11阶段2，并核对全部动作语义。

    唯一允许变化的是profile及阶段表，腿包络、命令分布、门槛、观测维度、
    姿态和物理步长必须相同。保留来源链，供runner在完整网络/优化器检查后
    记录迁移；此函数不修改输入或环境。新课程不继承旧窗口与升速计数。
    """
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError("Route 11 transfer requires complete version-1 curriculum metadata")
    source = state.get("action_mapping")
    if not isinstance(source, dict):
        raise ValueError("Route 11 transfer requires a complete action mapping")
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if state.get("signature") != digest:
        raise ValueError("Route 11 transfer action mapping signature is invalid")

    destination_expected = {
        "profile_id": PROFILE_ID,
        "hip_limits": [0.04] * len(BRIDGE_SPEEDS),
        "leg_limits": [0.10] * len(BRIDGE_SPEEDS),
        "max_speeds": list(BRIDGE_SPEEDS),
        "min_stage_updates": 200, "evaluation_interval": 100, "required_passes": 2,
        "gate_parameters": GATE_PARAMETERS,
        "observation_action_dimensions": {"policy": 60, "critic": 236, "action": 16},
        "command_distribution": {"forward_only": True, "zero_command_probability": 0.25,
                                 "anchor_probability": 0.5, "minimum_abs_speed": 0.08},
    }
    for key, value in destination_expected.items():
        if json.dumps(target_mapping.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Route 12 transfer destination changed the approved configuration: {key}")

    # Canonical JSON treats tuples/lists identically after checkpoint/YAML round trips.
    expected_source = deepcopy(target_mapping)
    expected_source.update(profile_id=FORWARD_PROFILE_ID, hip_limits=[0.04] * len(FORWARD_SPEEDS),
                           leg_limits=[0.10] * len(FORWARD_SPEEDS), max_speeds=list(FORWARD_SPEEDS))
    if json.dumps(source, sort_keys=True) != json.dumps(expected_source, sort_keys=True):
        raise ValueError("Route 11 transfer changes the profile, physical mapping, gates or command distribution")
    if len(source.get("leg_joints", [])) != 12 or len(source.get("wheel_joints", [])) != 4:
        raise ValueError("Route 11 transfer requires 12 leg and 4 wheel actions")

    original = LegCurriculumSchedule(
        hip_limits=(0.04,) * len(FORWARD_SPEEDS), leg_limits=(0.10,) * len(FORWARD_SPEEDS),
        max_speeds=FORWARD_SPEEDS, min_stage_updates=200, evaluation_interval=100, required_passes=2,
    )
    try:
        original.load_state_dict(state["schedule"], reset_window=False)
    except (KeyError, TypeError) as exc:
        raise ValueError("Route 11 transfer has incomplete curriculum counters") from exc
    if (original.stage != 2 or state.get("hip_limit_rad") != 0.04
            or state.get("leg_limit_rad") != 0.10 or state.get("max_speed_m_s") != 0.8):
        raise ValueError("Only route 11 stage 2 at 0.8 m/s with unchanged leg limits can transfer")
    steps = state.get("control_steps")
    # 原任务固定每次PPO更新收集24控制步；不接受来自截断/不一致计数的快照。
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != original.total_updates * 24:
        raise ValueError("Route 11 transfer control steps must equal completed PPO updates times 24")
    quality = state.get("last_quality_metrics")
    if (not isinstance(quality, dict) or quality.get("window_passed") != 1.0
            or original.consecutive_passes < 1
            or not forward_quality_gate(quality, max_speed=0.8, parameters=GATE_PARAMETERS)):
        raise ValueError("Route 11 transfer requires a complete passing 0.8 m/s quality window")
    if (quality.get("evaluated_stage", 2) != 2
            or quality.get("evaluated_max_speed_m_s", 0.8) != 0.8):
        raise ValueError("Route 11 transfer quality window belongs to another speed stage")
    return {
        "source_profile": FORWARD_PROFILE_ID, "source_signature": digest,
        "source_stage": original.stage, "source_total_updates": original.total_updates,
        "source_control_steps": steps, "source_max_speed_m_s": 0.8,
        "source_transfer_source": deepcopy(state.get("transfer_source")),
        "transfer_type": "actor_critic_optimizer_with_new_curriculum",
    }
