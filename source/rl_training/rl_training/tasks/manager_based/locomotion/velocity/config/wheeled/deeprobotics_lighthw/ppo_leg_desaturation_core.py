"""Pure-Python guards for the isolated thigh/calf raw-action PPO ablation."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

from .ppo_grouped_noise_core import (
    PROFILE_ID as GROUPED_NOISE_PROFILE_ID,
    validate_route12_transfer_state as validate_grouped_noise_route12_transfer,
)


PROFILE_ID = "lighthw_leg_desaturation_v1"
DESATURATION_REWARD_NAME = "thigh_calf_raw_saturation_l1"
DESATURATION_FUNCTION = (
    "rl_training.tasks.manager_based.locomotion.velocity.config.wheeled."
    "deeprobotics_lighthw.adapted_rewards:lighthw_joint_action_saturation_l1"
)
DESATURATION_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("thigh", "calf")
)
DESATURATION_RAW_LIMIT = math.atanh(0.95)
DESATURATION_WEIGHT = -0.01


def desaturation_reward_fingerprint() -> dict:
    """返回本轮唯一新增奖励的完整定义；每次复制，避免调用方改写常量。"""
    return {
        "term_name": DESATURATION_REWARD_NAME,
        "function": DESATURATION_FUNCTION,
        "action_term_name": "joint_pos",
        "joint_names": list(DESATURATION_JOINT_NAMES),
        "raw_limit": DESATURATION_RAW_LIMIT,
        "weight": DESATURATION_WEIGHT,
    }


def validate_desaturation_reward_cfg(term_cfg, expected_function) -> dict:
    """核对实际奖励对象后返回指纹，不能用预设常量掩盖实际奖励配置漂移。

    检查发生在导出/迁移/恢复/回放的动作签名生成时。八个关节按动作名称
    选择，全部命令都激活；拒绝额外参数、错函数、错关节或更改权重/阈值。
    """
    expected = desaturation_reward_fingerprint()
    if term_cfg is None or getattr(term_cfg, "func", None) is not expected_function:
        raise ValueError("Leg-desaturation reward must use the approved raw-action L1 function")
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict) or set(params) != {"action_term_name", "joint_names", "raw_limit"}:
        raise ValueError("Leg-desaturation reward requires the complete, ungated raw-action parameters")
    names = params["joint_names"]
    if (params["action_term_name"] != "joint_pos" or not isinstance(names, (tuple, list))
            or list(names) != expected["joint_names"]):
        raise ValueError("Leg-desaturation reward must select exactly the eight thigh/calf action names")
    for key, actual in (("weight", getattr(term_cfg, "weight", None)), ("raw_limit", params["raw_limit"])):
        if (isinstance(actual, bool) or not isinstance(actual, Real) or not math.isfinite(actual)
                or actual != expected[key]):
            raise ValueError(f"Leg-desaturation reward changed the approved {key}")
    return expected


def validate_route12_transfer_state(state: dict, target_mapping: dict) -> dict:
    """保留原route12来源检查，只允许新profile与唯一新增奖励定义的差异。

    不接收route13模型作为本轮起点，以保持model900、分组探索重置以及
    原生PPO优化器迁移条件相同。旧来源链和完整质量窗由原验证器保留。
    """
    if not isinstance(target_mapping, dict) or target_mapping.get("profile_id") != PROFILE_ID:
        raise ValueError("Leg-desaturation transfer requires its independent destination profile")
    fingerprint = desaturation_reward_fingerprint()
    if target_mapping.get("leg_desaturation_reward") != fingerprint:
        raise ValueError("Leg-desaturation transfer reward fingerprint does not match this experiment")
    grouped_target = deepcopy(target_mapping)
    grouped_target["profile_id"] = GROUPED_NOISE_PROFILE_ID
    grouped_target.pop("leg_desaturation_reward")
    source = validate_grouped_noise_route12_transfer(state, grouped_target)
    source["transfer_type"] = "actor_critic_optimizer_with_grouped_noise_and_leg_desaturation"
    source["target_leg_desaturation_reward"] = fingerprint
    return source
