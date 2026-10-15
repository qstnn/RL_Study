"""LightHW 参考任务 v1 的观测内容契约；旧任务继续使用旧共享函数。

用关节名称分别解析物理索引和观测通道，不把全局索引用于重排后的张量。
本模块不导入仿真，便于对真实计算函数进行内容测试。
"""
from __future__ import annotations

from functools import lru_cache

OBSERVATION_VERSION = "lighthw_named_joint_positions_v1"
REFERENCE_PROFILE = "lighthw_ppo_observation_reference_v1"
REFERENCE_TASK = "Flat-Deeprobotics-LightHW-PPOReference-v0"
OBSERVATION_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf", "wheel")
)
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL"))


@lru_cache(maxsize=16)
def named_channel_indices(physical_names, observation_names, wheel_names):
    """返回物理取数索引及局部轮角通道；拒绝重复名、缺失名和歧义布局。"""
    if len(set(physical_names)) != len(physical_names):
        raise ValueError("Duplicate physical joint names")
    if tuple(observation_names) != OBSERVATION_JOINT_NAMES or tuple(wheel_names) != WHEEL_JOINT_NAMES:
        raise ValueError("LightHW observation v1 requires the declared FR/FL/RR/RL named layout")
    lookup = {name: index for index, name in enumerate(physical_names)}
    missing = set(observation_names) - lookup.keys()
    if missing:
        raise ValueError(f"Missing LightHW observation joints: {sorted(missing)}")
    selected = [lookup[name] for name in observation_names]
    wheel_channels = [i for i, name in enumerate(observation_names) if name in wheel_names]
    return selected, wheel_channels


def joint_positions_v1(env, asset_cfg, wheel_joint_names=WHEEL_JOINT_NAMES):
    """返回16个相对位置通道；12腿按名称取值，4轮位置为零。

    对所有命令生效，只替换新任务的joint_pos项，不屏蔽joint_vel。
    ObservationManager仍按原配置添加噪声/缩放；旧任务语义保持原样。
    """
    asset = env.scene[asset_cfg.name]
    selected, wheel_channels = named_channel_indices(
        tuple(asset.joint_names), tuple(asset_cfg.joint_names), tuple(wheel_joint_names)
    )
    result = asset.data.joint_pos[:, selected] - asset.data.default_joint_pos[:, selected]
    result[:, wheel_channels] = 0.0
    return result


def observation_contract():
    return {
        "version": OBSERVATION_VERSION,
        "policy_dim": 60, "critic_dim": 236, "action_dim": 16,
        "position_function": f"{joint_positions_v1.__module__}:{joint_positions_v1.__name__}",
        "position_channels": list(OBSERVATION_JOINT_NAMES),
        "masked_wheel_position_channels": [15, 19, 23, 27],
        "wheel_velocity_channels": [31, 35, 39, 43],
        "wheel_velocity_scale": 0.05,
        "critic_height_scan_channels": [60, 236],
        "empirical_normalization": False,
    }


def validate_observation_cfg(cfg):
    """校验训练/回放共用定义；不能只依赖60/236的维度相等。"""
    if cfg.lighthw_observation_version != OBSERVATION_VERSION:
        raise ValueError("Wrong LightHW observation version")
    for group in (cfg.observations.policy, cfg.observations.critic):
        if group.joint_pos.func is not joint_positions_v1:
            raise ValueError("Reference task must use named joint positions v1")
        if tuple(group.joint_pos.params['asset_cfg'].joint_names) != OBSERVATION_JOINT_NAMES:
            raise ValueError("Wrong reference position channel order")
        if tuple(group.joint_pos.params['wheel_joint_names']) != WHEEL_JOINT_NAMES:
            raise ValueError("Wrong reference wheel position mask")
        if tuple(group.joint_vel.params['asset_cfg'].joint_names) != OBSERVATION_JOINT_NAMES:
            raise ValueError("Wheel/leg velocity feedback must preserve its named order")
        if group.joint_vel.scale != 0.05 or group.joint_pos.scale != 1.0:
            raise ValueError("Reference position/velocity observation scale drift")
    return observation_contract()
