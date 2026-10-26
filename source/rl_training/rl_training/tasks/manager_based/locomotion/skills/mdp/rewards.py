# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# 奖励函数约定
#
# 每个函数返回形状为 ``(num_envs,)`` 的逐环境原始值，RewardManager 再乘配置中的
# weight 和仿真步长。函数名带 ``_l2`` 的通常返回非负误差，需要负权重；带
# ``_exp`` 的通常返回 0~1 得分，需要正权重。这里不要写死实验权重，权重应放在
# skill_env_cfg.py，才能为不同任务阶段复用同一数学定义。
# =============================================================================


def _robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _skill_gate(env: ManagerBasedRLEnv, command_name: str | None, active_skill_id: int | None) -> torch.Tensor:
    if command_name is None or active_skill_id is None:
        return torch.ones(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    return command[:, active_skill_id]


def _target_pos_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return env.scene.env_origins + command[:, -4:-1]


def _local_x_window_gate(
    env: ManagerBasedRLEnv,
    min_local_x: float | None,
    max_local_x: float | None,
    sharpness: float,
) -> torch.Tensor:
    root_x_env = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    gate = torch.ones(env.num_envs, device=env.device)
    if min_local_x is not None:
        gate *= torch.sigmoid(sharpness * (root_x_env - min_local_x))
    if max_local_x is not None:
        gate *= torch.sigmoid(sharpness * (max_local_x - root_x_env))
    return gate


def _bodies_over_edge_gate(
    env: ManagerBasedRLEnv,
    platform_front_x: float,
    min_height: float,
    asset_cfg: SceneEntityCfg,
    x_sharpness: float = 10.0,
    z_sharpness: float = 10.0,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_x = torch.min(
        body_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1),
        dim=1,
    ).values
    body_z = torch.min(body_pos[..., 2], dim=1).values
    return torch.sigmoid(x_sharpness * (body_x - platform_front_x)) * torch.sigmoid(
        z_sharpness * (body_z - min_height)
    )


def _root_height_gate(
    env: ManagerBasedRLEnv,
    min_height: float | None,
    sharpness: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if min_height is None:
        return torch.ones(env.num_envs, device=env.device)
    asset = _robot(env, asset_cfg)
    return torch.sigmoid(sharpness * (asset.data.root_pos_w[:, 2] - min_height))


def _aggregate_scores(scores: torch.Tensor, aggregation: str) -> torch.Tensor:
    if aggregation == "minimum":
        return torch.min(scores, dim=1).values
    if aggregation == "maximum":
        return torch.max(scores, dim=1).values
    if aggregation == "mean":
        return torch.mean(scores, dim=1)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _climb_stage(env: ManagerBasedRLEnv, stage_cfg: dict) -> torch.Tensor:
    """Return a monotonic climb stage: approach, front support, body raised, platform hold."""
    if not hasattr(env, "_lighthw_climb_stage"):
        stage_cfg["front_asset_cfg"].resolve(env.scene)
        stage_cfg["front_sensor_cfg"].resolve(env.scene)
        if stage_cfg.get("front_brace_asset_cfg") is not None:
            stage_cfg["front_brace_asset_cfg"].resolve(env.scene)
        if stage_cfg.get("front_brace_sensor_cfg") is not None:
            stage_cfg["front_brace_sensor_cfg"].resolve(env.scene)
        stage_cfg["rear_asset_cfg"].resolve(env.scene)
        stage_cfg["rear_sensor_cfg"].resolve(env.scene)
        stage_cfg["all_wheel_asset_cfg"].resolve(env.scene)
        env._lighthw_climb_stage_cfg = stage_cfg
        env._lighthw_climb_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._lighthw_front_support_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._lighthw_platform_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._lighthw_climb_stage_last_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    stage_cfg = env._lighthw_climb_stage_cfg

    current_step = env.episode_length_buf
    update_mask = current_step != env._lighthw_climb_stage_last_step
    if torch.any(update_mask):
        reset_mask = update_mask & (current_step <= 1)
        platform_reset_mask = getattr(
            env,
            "_lighthw_platform_reset_mask",
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
        env._lighthw_climb_stage[reset_mask] = torch.where(
            platform_reset_mask[reset_mask],
            torch.full_like(env._lighthw_climb_stage[reset_mask], 3),
            torch.zeros_like(env._lighthw_climb_stage[reset_mask]),
        )
        env._lighthw_front_support_steps[reset_mask] = 0
        env._lighthw_platform_hold_steps[reset_mask] = torch.where(
            platform_reset_mask[reset_mask],
            torch.full_like(
                env._lighthw_platform_hold_steps[reset_mask], stage_cfg["hold_support_steps"]
            ),
            torch.zeros_like(env._lighthw_platform_hold_steps[reset_mask]),
        )

        front_asset = _robot(env, stage_cfg["front_asset_cfg"])
        front_pos = front_asset.data.body_pos_w[:, stage_cfg["front_asset_cfg"].body_ids]
        contact_sensor: ContactSensor = env.scene.sensors[stage_cfg["front_sensor_cfg"].name]
        forces = contact_sensor.data.net_forces_w_history[:, :, stage_cfg["front_sensor_cfg"].body_ids]
        front_contact = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0
        front_x = front_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
        wheel_supported = torch.any(
            (front_x >= stage_cfg["platform_front_x"])
            & (front_pos[..., 2] >= stage_cfg["platform_height"] - stage_cfg["wheel_height_margin"])
            & front_contact,
            dim=1,
        )

        brace_supported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        front_brace_asset_cfg = stage_cfg.get("front_brace_asset_cfg")
        front_brace_sensor_cfg = stage_cfg.get("front_brace_sensor_cfg")
        if front_brace_asset_cfg is not None and front_brace_sensor_cfg is not None:
            brace_asset = _robot(env, front_brace_asset_cfg)
            brace_pos = brace_asset.data.body_pos_w[:, front_brace_asset_cfg.body_ids]
            brace_sensor: ContactSensor = env.scene.sensors[front_brace_sensor_cfg.name]
            brace_forces = brace_sensor.data.net_forces_w_history[
                :, :, front_brace_sensor_cfg.body_ids
            ]
            brace_contact = torch.max(torch.norm(brace_forces, dim=-1), dim=1)[0] > 1.0
            brace_x = brace_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
            brace_supported = torch.any(
                (brace_x >= stage_cfg.get("front_brace_x", stage_cfg["platform_front_x"] - 0.08))
                & (brace_pos[..., 2] >= stage_cfg.get("front_brace_min_height", stage_cfg["platform_height"] - 0.20))
                & brace_contact,
                dim=1,
            )

        rear_asset = _robot(env, stage_cfg["rear_asset_cfg"])
        rear_pos = rear_asset.data.body_pos_w[:, stage_cfg["rear_asset_cfg"].body_ids]
        rear_sensor: ContactSensor = env.scene.sensors[stage_cfg["rear_sensor_cfg"].name]
        rear_forces = rear_sensor.data.net_forces_w_history[:, :, stage_cfg["rear_sensor_cfg"].body_ids]
        rear_contact = torch.max(torch.norm(rear_forces, dim=-1), dim=1)[0] > 1.0
        rear_x = rear_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
        rear_supported = torch.all(
            rear_contact
            & (rear_pos[..., 2] <= stage_cfg["rear_ground_max_height"])
            & (rear_x <= stage_cfg["platform_front_x"] + stage_cfg["rear_edge_margin"]),
            dim=1,
        )
        front_placed = wheel_supported | brace_supported

        count_up = env._lighthw_front_support_steps + 1
        env._lighthw_front_support_steps = torch.where(
            update_mask & front_placed,
            count_up,
            torch.where(update_mask, torch.zeros_like(count_up), env._lighthw_front_support_steps),
        )
        was_front_stage = env._lighthw_climb_stage >= 1
        front_stage_ready = env._lighthw_front_support_steps >= stage_cfg["front_support_steps"]
        env._lighthw_climb_stage = torch.where(
            update_mask & front_stage_ready,
            torch.maximum(env._lighthw_climb_stage, torch.ones_like(env._lighthw_climb_stage)),
            env._lighthw_climb_stage,
        )

        base_height = env.scene["robot"].data.root_pos_w[:, 2]
        body_stage_ready = (
            was_front_stage
            & front_placed
            & rear_supported
            & (base_height >= stage_cfg["body_stage_height"])
        )
        env._lighthw_climb_stage = torch.where(
            update_mask & body_stage_ready,
            torch.full_like(env._lighthw_climb_stage, 2),
            env._lighthw_climb_stage,
        )

        all_wheel_asset = _robot(env, stage_cfg["all_wheel_asset_cfg"])
        all_wheel_pos = all_wheel_asset.data.body_pos_w[:, stage_cfg["all_wheel_asset_cfg"].body_ids]
        all_wheel_x = torch.min(
            all_wheel_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1), dim=1
        ).values
        all_wheel_z = torch.min(all_wheel_pos[..., 2], dim=1).values
        upright = -env.scene["robot"].data.projected_gravity_b[:, 2] > stage_cfg["hold_upright"]
        transition_ready = (
            (env._lighthw_climb_stage >= 2)
            & (all_wheel_x >= stage_cfg["hold_front_x"])
            & (all_wheel_z >= stage_cfg["platform_height"] - stage_cfg["wheel_height_margin"])
            & upright
        )
        hold_count_up = env._lighthw_platform_hold_steps + 1
        env._lighthw_platform_hold_steps = torch.where(
            update_mask & transition_ready,
            hold_count_up,
            torch.where(update_mask, torch.zeros_like(hold_count_up), env._lighthw_platform_hold_steps),
        )
        platform_hold_ready = env._lighthw_platform_hold_steps >= stage_cfg["hold_support_steps"]
        env._lighthw_climb_stage = torch.where(
            update_mask & platform_hold_ready,
            torch.full_like(env._lighthw_climb_stage, 3),
            env._lighthw_climb_stage,
        )
        env._lighthw_climb_stage_last_step[update_mask] = current_step[update_mask]

    return env._lighthw_climb_stage


def _climb_stage_gate(
    env: ManagerBasedRLEnv,
    stage_cfg: dict | None,
    required_stage: int | None,
    maximum_stage: int | None,
) -> torch.Tensor:
    if stage_cfg is None:
        return torch.ones(env.num_envs, device=env.device)
    stage = _climb_stage(env, stage_cfg)
    gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if required_stage is not None:
        gate &= stage >= required_stage
    if maximum_stage is not None:
        gate &= stage <= maximum_stage
    return gate.float()


def climb_stage_reward(env: ManagerBasedRLEnv, stage_cfg: dict) -> torch.Tensor:
    """Expose climb-stage progress to rewards and TensorBoard."""
    return _climb_stage(env, stage_cfg).float() / 3.0


def root_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    reward = torch.exp(-error / std**2) * _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return reward


def target_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_skill_id: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    target = _target_pos_w(env, command_name)
    error = torch.sum(torch.square(asset.data.root_pos_w[:, :2] - target[:, :2]), dim=1)
    return torch.exp(-error / std**2) * _skill_gate(env, command_name, active_skill_id)


def target_heading_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_skill_id: int | None = None,
) -> torch.Tensor:
    """Reward the robot yaw heading pointing toward the commanded target."""
    asset = _robot(env, asset_cfg)
    target = _target_pos_w(env, command_name)
    delta_xy = target[:, :2] - asset.data.root_pos_w[:, :2]
    target_heading = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
    heading_error = torch.atan2(
        torch.sin(asset.data.heading_w - target_heading),
        torch.cos(asset.data.heading_w - target_heading),
    )
    return torch.exp(-torch.square(heading_error) / std**2) * _skill_gate(
        env, command_name, active_skill_id
    )


def minimum_body_height_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
    min_local_x: float | None = None,
    max_local_x: float | None = None,
    gate_sharpness: float = 8.0,
    heading_std: float | None = None,
    activation_asset_cfg: SceneEntityCfg | None = None,
    activation_front_x: float = 0.0,
    activation_min_height: float = 0.0,
    activation_sharpness: float = 10.0,
    aggregation: str = "minimum",
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Dense progress for lifting every selected body from a start to a target height."""
    asset = _robot(env, asset_cfg)
    body_height = _aggregate_scores(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2], aggregation
    )
    progress = torch.clamp(
        (body_height - start_height) / max(target_height - start_height, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    progress *= _local_x_window_gate(env, min_local_x, max_local_x, gate_sharpness)
    if heading_std is not None and command_name is not None:
        progress *= target_heading_exp(env, command_name, heading_std, asset_cfg, active_skill_id)
    else:
        progress *= _skill_gate(env, command_name, active_skill_id)
    if activation_asset_cfg is not None:
        progress *= _bodies_over_edge_gate(
            env,
            activation_front_x,
            activation_min_height,
            activation_asset_cfg,
            activation_sharpness,
            activation_sharpness,
        )
    progress *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return progress


def minimum_body_forward_progress(
    env: ManagerBasedRLEnv,
    start_x: float,
    target_x: float,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
    activation_asset_cfg: SceneEntityCfg | None = None,
    activation_front_x: float = 0.0,
    activation_min_height: float = 0.0,
    activation_sharpness: float = 10.0,
    activation_root_height: float | None = None,
    root_height_sharpness: float = 12.0,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Dense local-x progress for advancing every selected body onto the platform."""
    asset = _robot(env, asset_cfg)
    body_x = torch.min(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 0] - env.scene.env_origins[:, 0].unsqueeze(-1),
        dim=1,
    ).values
    progress = torch.clamp((body_x - start_x) / max(target_x - start_x, 1.0e-6), min=0.0, max=1.0)
    progress *= _skill_gate(env, command_name, active_skill_id)
    if activation_asset_cfg is not None:
        progress *= _bodies_over_edge_gate(
            env,
            activation_front_x,
            activation_min_height,
            activation_asset_cfg,
            activation_sharpness,
            activation_sharpness,
        )
    progress *= _root_height_gate(env, activation_root_height, root_height_sharpness)
    progress *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return progress


def body_pair_axis_difference_l2(
    env: ManagerBasedRLEnv,
    axis: int,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    """Penalize a selected body pair being offset along a world-frame axis."""
    asset = _robot(env, asset_cfg)
    positions = asset.data.body_pos_w[:, asset_cfg.body_ids, axis]
    return torch.square(positions[:, 0] - positions[:, 1]) * _skill_gate(
        env, command_name, active_skill_id
    )


def knee_outward_progress(
    env: ManagerBasedRLEnv,
    target_offset: float,
    front_hip_cfg: SceneEntityCfg,
    front_knee_cfg: SceneEntityCfg,
    rear_hip_cfg: SceneEntityCfg,
    rear_knee_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward front knees ahead of their hips and rear knees behind their hips."""
    asset = _robot(env, asset_cfg)

    def body_x_in_root(cfg: SceneEntityCfg) -> torch.Tensor:
        body_pos_w = asset.data.body_pos_w[:, cfg.body_ids]
        root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1)
        root_pos = asset.data.root_pos_w.unsqueeze(1)
        return math_utils.quat_apply_inverse(root_quat, body_pos_w - root_pos)[..., 0]

    front_offset = body_x_in_root(front_knee_cfg) - body_x_in_root(front_hip_cfg)
    rear_offset = body_x_in_root(rear_hip_cfg) - body_x_in_root(rear_knee_cfg)
    all_offsets = torch.cat((front_offset, rear_offset), dim=1)
    minimum_offset = torch.min(all_offsets, dim=1).values
    return torch.clamp(minimum_offset / target_offset, min=0.0, max=1.0)


def leg_chevron_pose(
    env: ManagerBasedRLEnv,
    target_knee_offset: float,
    target_vertical_span: float,
    wheel_alignment_std: float,
    knee_midpoint_std: float,
    vertical_span_std: float,
    hip_cfg: SceneEntityCfg,
    knee_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward front legs shaped like ``>`` and rear legs shaped like ``<`` in the sagittal plane."""
    asset = _robot(env, asset_cfg)

    def positions_in_root(cfg: SceneEntityCfg) -> torch.Tensor:
        pos_w = asset.data.body_pos_w[:, cfg.body_ids]
        root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, pos_w.shape[1], -1)
        return math_utils.quat_apply_inverse(root_quat, pos_w - asset.data.root_pos_w.unsqueeze(1))

    hip = positions_in_root(hip_cfg)
    knee = positions_in_root(knee_cfg)
    wheel = positions_in_root(wheel_cfg)

    # Body order is front-left, front-right, rear-left, rear-right.
    outward_sign = torch.tensor((1.0, 1.0, -1.0, -1.0), device=env.device)
    leg_midpoint_x = 0.5 * (hip[..., 0] + wheel[..., 0])
    knee_offset = outward_sign * (knee[..., 0] - leg_midpoint_x)
    outward_score = torch.clamp(knee_offset / target_knee_offset, min=0.0, max=1.0)

    wheel_alignment = torch.exp(-torch.square(wheel[..., 0] - hip[..., 0]) / wheel_alignment_std**2)
    knee_midpoint_z = 0.5 * (hip[..., 2] + wheel[..., 2])
    midpoint_score = torch.exp(-torch.square(knee[..., 2] - knee_midpoint_z) / knee_midpoint_std**2)
    vertical_span = hip[..., 2] - wheel[..., 2]
    span_score = torch.exp(-torch.square(vertical_span - target_vertical_span) / vertical_span_std**2)

    per_leg_score = (
        0.35 * outward_score
        + 0.25 * wheel_alignment
        + 0.20 * midpoint_score
        + 0.20 * span_score
    )
    # Mean gives every leg a gradient; the minimum prevents one bad leg from being hidden.
    reward = 0.6 * torch.mean(per_leg_score, dim=1) + 0.4 * torch.min(per_leg_score, dim=1).values
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return reward


def platform_wheel_contact(
    env: ManagerBasedRLEnv,
    platform_front_x: float,
    platform_height: float,
    height_std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
    aggregation: str = "minimum",
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward every selected wheel contacting the horizontal platform surface."""
    asset = _robot(env, asset_cfg)
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_x = body_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    on_platform_x = torch.sigmoid(16.0 * (body_x - platform_front_x))
    height_score = torch.exp(-torch.square(body_pos[..., 2] - platform_height) / height_std**2)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact = (torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0).float()
    reward = _aggregate_scores(on_platform_x * height_score * contact, aggregation)
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return reward


def platform_wheel_landing(
    env: ManagerBasedRLEnv,
    platform_front_x: float,
    platform_height: float,
    height_std: float,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
    x_sharpness: float = 12.0,
    aggregation: str = "minimum",
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward both selected wheels moving past the edge and settling onto the platform height."""
    asset = _robot(env, asset_cfg)
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_x = body_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    x_score = torch.sigmoid(x_sharpness * (body_x - platform_front_x))
    height_score = torch.exp(-torch.square(body_pos[..., 2] - platform_height) / height_std**2)
    per_wheel = x_score * height_score
    reward = _aggregate_scores(per_wheel, aggregation)
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return reward


def platform_contact_traction(
    env: ManagerBasedRLEnv,
    max_forward_speed: float,
    platform_front_x: float,
    platform_height: float,
    height_std: float,
    sensor_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
    aggregation: str = "minimum",
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward body motion only while both front wheels press on the platform."""
    contact_score = platform_wheel_contact(
        env,
        platform_front_x,
        platform_height,
        height_std,
        sensor_cfg,
        wheel_asset_cfg,
        command_name=command_name,
        active_skill_id=active_skill_id,
        aggregation=aggregation,
        climb_stage_cfg=climb_stage_cfg,
        required_climb_stage=required_climb_stage,
        maximum_climb_stage=maximum_climb_stage,
    )
    asset = _robot(env, root_asset_cfg)
    forward_speed = torch.clamp(asset.data.root_lin_vel_w[:, 0] / max_forward_speed, min=0.0, max=1.0)
    return contact_score * forward_speed


def quasi_static_step_support(
    env: ManagerBasedRLEnv,
    platform_front_x: float,
    platform_height: float,
    rear_ground_max_height: float,
    rear_edge_margin: float,
    deactivate_root_x: float,
    start_height: float,
    target_height: float,
    vertical_speed_std: float,
    angular_speed_std: float,
    front_asset_cfg: SceneEntityCfg,
    front_sensor_cfg: SceneEntityCfg,
    rear_asset_cfg: SceneEntityCfg,
    rear_sensor_cfg: SceneEntityCfg,
    front_knee_asset_cfg: SceneEntityCfg | None = None,
    front_knee_sensor_cfg: SceneEntityCfg | None = None,
    knee_min_height: float | None = None,
    knee_edge_margin: float = 0.08,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward the pictured climb: a front wheel or front knee supported and both rear wheels grounded."""
    front = _robot(env, front_asset_cfg)
    front_pos = front.data.body_pos_w[:, front_asset_cfg.body_ids]
    front_x = front_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    front_sensor: ContactSensor = env.scene.sensors[front_sensor_cfg.name]
    front_forces = front_sensor.data.net_forces_w_history[:, :, front_sensor_cfg.body_ids]
    front_contact = (
        torch.max(torch.norm(front_forces, dim=-1), dim=1)[0] > 1.0
    ).float()
    wheel_front_score = torch.max(
        torch.sigmoid(16.0 * (front_x - platform_front_x))
        * torch.exp(-torch.square(front_pos[..., 2] - platform_height) / 0.12**2)
        * front_contact,
        dim=1,
    ).values
    front_score = wheel_front_score

    if front_knee_asset_cfg is not None and front_knee_sensor_cfg is not None:
        knee = _robot(env, front_knee_asset_cfg)
        knee_pos = knee.data.body_pos_w[:, front_knee_asset_cfg.body_ids]
        knee_x = knee_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
        knee_sensor: ContactSensor = env.scene.sensors[front_knee_sensor_cfg.name]
        knee_forces = knee_sensor.data.net_forces_w_history[:, :, front_knee_sensor_cfg.body_ids]
        knee_contact = (
            torch.max(torch.norm(knee_forces, dim=-1), dim=1)[0] > 1.0
        ).float()
        knee_height_min = (
            platform_height - 0.20 if knee_min_height is None else knee_min_height
        )
        knee_score = torch.max(
            torch.sigmoid(16.0 * (knee_x - (platform_front_x - knee_edge_margin)))
            * torch.sigmoid(12.0 * (knee_pos[..., 2] - knee_height_min))
            * knee_contact,
            dim=1,
        ).values
        front_score = torch.maximum(front_score, knee_score)

    rear = _robot(env, rear_asset_cfg)
    rear_pos = rear.data.body_pos_w[:, rear_asset_cfg.body_ids]
    rear_x = rear_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    rear_sensor: ContactSensor = env.scene.sensors[rear_sensor_cfg.name]
    rear_forces = rear_sensor.data.net_forces_w_history[:, :, rear_sensor_cfg.body_ids]
    rear_contact = (
        torch.max(torch.norm(rear_forces, dim=-1), dim=1)[0] > 1.0
    ).float()
    rear_ground_score = torch.sigmoid(20.0 * (rear_ground_max_height - rear_pos[..., 2]))
    rear_edge_score = torch.sigmoid(
        16.0 * (platform_front_x + rear_edge_margin - rear_x)
    )
    rear_score = torch.min(rear_contact * rear_ground_score * rear_edge_score, dim=1).values

    root = _robot(env, root_asset_cfg)
    height_progress = torch.clamp(
        (root.data.root_pos_w[:, 2] - start_height)
        / max(target_height - start_height, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    quiet_vertical = torch.exp(
        -torch.square(root.data.root_lin_vel_w[:, 2]) / vertical_speed_std**2
    )
    quiet_rotation = torch.exp(
        -torch.sum(torch.square(root.data.root_ang_vel_b[:, :2]), dim=1)
        / angular_speed_std**2
    )
    local_root_x = root.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    support_phase = torch.sigmoid(16.0 * (deactivate_root_x - local_root_x))
    reward = front_score * rear_score
    reward *= support_phase * (0.25 + 0.75 * height_progress)
    reward *= quiet_vertical * quiet_rotation
    reward *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return reward


def unsupported_ascent_l2(
    env: ManagerBasedRLEnv,
    max_supported_speed: float,
    minimum_contacts: int,
    deactivate_root_x: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize ballistic ascent before the body crosses the edge unless enough wheels support it."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contacts = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0
    insufficient_support = torch.sum(contacts, dim=1) < minimum_contacts
    asset = _robot(env, asset_cfg)
    local_root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    before_edge = local_root_x < deactivate_root_x
    ascent = torch.clamp(
        asset.data.root_lin_vel_w[:, 2] - max_supported_speed,
        min=0.0,
    )
    penalty = insufficient_support.float() * before_edge.float() * torch.square(ascent)
    penalty *= _climb_stage_gate(env, climb_stage_cfg, None, maximum_climb_stage)
    return penalty


def airborne_before_edge(
    env: ManagerBasedRLEnv,
    deactivate_root_x: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize all four wheels leaving support before the body reaches the platform."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    all_airborne = torch.all(
        torch.max(torch.norm(forces, dim=-1), dim=1)[0] <= 1.0,
        dim=1,
    )
    asset = _robot(env, asset_cfg)
    local_root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    penalty = all_airborne.float() * (local_root_x < deactivate_root_x).float()
    penalty *= _climb_stage_gate(env, climb_stage_cfg, None, maximum_climb_stage)
    return penalty


def platform_hold_quality(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_front_x: float,
    platform_height: float,
    wheel_height_std: float,
    target_root_height: float,
    root_height_std: float,
    xy_std: float,
    linear_speed_std: float,
    angular_speed_std: float,
    level_std: float,
    joint_targets: dict[str, float],
    joint_pose_std: float,
    wheel_asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_skill_id: int | None = None,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward a supported, quiet and symmetric nominal stance on the platform."""
    wheels = _robot(env, wheel_asset_cfg)
    wheel_pos = wheels.data.body_pos_w[:, wheel_asset_cfg.body_ids]
    wheel_x = wheel_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    x_score = torch.min(torch.sigmoid(12.0 * (wheel_x - platform_front_x)), dim=1).values
    z_score = torch.min(
        torch.exp(-torch.square(wheel_pos[..., 2] - platform_height) / wheel_height_std**2),
        dim=1,
    ).values

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact_score = torch.min(
        (torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0).float(), dim=1
    ).values

    root = _robot(env, root_asset_cfg)
    target = _target_pos_w(env, command_name)
    xy_error = torch.sum(torch.square(root.data.root_pos_w[:, :2] - target[:, :2]), dim=1)
    center_score = torch.exp(-xy_error / xy_std**2)
    height_score = torch.exp(
        -torch.square(root.data.root_pos_w[:, 2] - target_root_height) / root_height_std**2
    )
    tilt_error = torch.sum(torch.square(root.data.projected_gravity_b[:, :2]), dim=1)
    upright_score = torch.exp(-tilt_error / level_std**2)
    linear_score = torch.exp(
        -torch.sum(torch.square(root.data.root_lin_vel_w), dim=1) / linear_speed_std**2
    )
    angular_score = torch.exp(
        -torch.sum(torch.square(root.data.root_ang_vel_b), dim=1) / angular_speed_std**2
    )
    normalized_targets = tuple(sorted(joint_targets.items()))
    cache_key = ("platform_hold_pose", normalized_targets)
    if not hasattr(env, "_lighthw_platform_pose_cache"):
        env._lighthw_platform_pose_cache = {}
    if cache_key not in env._lighthw_platform_pose_cache:
        env._lighthw_platform_pose_cache[cache_key] = [
            (root.find_joints(joint_name)[0], target) for joint_name, target in normalized_targets
        ]
    pose_error = torch.zeros(env.num_envs, device=env.device)
    for joint_ids, target in env._lighthw_platform_pose_cache[cache_key]:
        pose_error += torch.mean(torch.square(root.data.joint_pos[:, joint_ids] - target), dim=1)
    pose_error /= max(len(normalized_targets), 1)
    pose_score = torch.exp(-pose_error / joint_pose_std**2)

    support_score = x_score * z_score * contact_score
    stability_score = (
        0.15 * center_score
        + 0.15 * height_score
        + 0.35 * upright_score
        + 0.15 * linear_score
        + 0.20 * angular_score
    )
    reward = support_score * (0.45 * pose_score + 0.55 * stability_score)
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, None)
    return reward


def platform_transition_attitude(
    env: ManagerBasedRLEnv,
    platform_front_x: float,
    platform_height: float,
    level_std: float,
    rate_std: float,
    wheel_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward body leveling once all wheels are clearing the platform edge."""
    wheels = _robot(env, wheel_asset_cfg)
    wheel_pos = wheels.data.body_pos_w[:, wheel_asset_cfg.body_ids]
    minimum_x = torch.min(
        wheel_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1),
        dim=1,
    ).values
    minimum_z = torch.min(wheel_pos[..., 2], dim=1).values
    transition_gate = torch.sigmoid(14.0 * (minimum_x - platform_front_x))
    transition_gate *= torch.sigmoid(12.0 * (minimum_z - platform_height))

    asset = _robot(env, asset_cfg)
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    level_score = torch.exp(-tilt_error / level_std**2)
    rate_error = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    rate_score = torch.exp(-rate_error / rate_std**2)
    reward = transition_gate * level_score * (0.90 + 0.10 * rate_score)
    reward *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return reward


def root_level_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward a horizontal trunk using roll and pitch gravity components."""
    asset = _robot(env, asset_cfg)
    error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward = torch.exp(-error / std**2)
    reward *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return reward


def root_level_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize trunk roll and pitch continuously."""
    asset = _robot(env, asset_cfg)
    penalty = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    penalty *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return penalty


def root_tilt_limit_l2(
    env: ManagerBasedRLEnv,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize only excessive trunk tilt, allowing the lean needed during climbing."""
    asset = _robot(env, asset_cfg)
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    penalty = torch.square(torch.clamp(tilt - max_tilt, min=0.0))
    penalty *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return penalty


def platform_touchdown_speed_l2(
    env: ManagerBasedRLEnv,
    max_downward_speed: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize hard downward motion at the instant any platform wheel makes contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    any_contact = torch.any(
        torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0,
        dim=1,
    ).float()
    asset = _robot(env, asset_cfg)
    excess_speed = torch.clamp(
        -asset.data.root_lin_vel_w[:, 2] - max_downward_speed,
        min=0.0,
    )
    penalty = any_contact * torch.square(excess_speed)
    penalty *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, None)
    return penalty


def knee_brace_body_lift(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    front_knee_sensor_cfg: SceneEntityCfg,
    rear_wheel_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward lifting the body while front knees brace and rear wheels push from the ground."""
    knee_sensor: ContactSensor = env.scene.sensors[front_knee_sensor_cfg.name]
    knee_forces = knee_sensor.data.net_forces_w_history[:, :, front_knee_sensor_cfg.body_ids]
    knee_contact = (
        torch.max(torch.norm(knee_forces, dim=-1), dim=1)[0] > 1.0
    ).float()
    knee_support = torch.mean(knee_contact, dim=1)

    rear_sensor: ContactSensor = env.scene.sensors[rear_wheel_sensor_cfg.name]
    rear_forces = rear_sensor.data.net_forces_w_history[:, :, rear_wheel_sensor_cfg.body_ids]
    rear_contact = torch.all(
        torch.max(torch.norm(rear_forces, dim=-1), dim=1)[0] > 1.0, dim=1
    ).float()

    asset = _robot(env, asset_cfg)
    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    reward = knee_support * rear_contact * height_progress
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return reward


def rear_ground_drive(
    env: ManagerBasedRLEnv,
    max_forward_speed: float,
    rear_ground_max_height: float,
    platform_front_x: float,
    rear_edge_margin: float,
    rear_asset_cfg: SceneEntityCfg,
    rear_sensor_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward forward body motion while both rear wheels are still driving on the ground."""
    rear = _robot(env, rear_asset_cfg)
    rear_pos = rear.data.body_pos_w[:, rear_asset_cfg.body_ids]
    rear_x = rear_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    rear_sensor: ContactSensor = env.scene.sensors[rear_sensor_cfg.name]
    rear_forces = rear_sensor.data.net_forces_w_history[:, :, rear_sensor_cfg.body_ids]
    rear_contact = (torch.max(torch.norm(rear_forces, dim=-1), dim=1)[0] > 1.0).float()
    rear_ground = torch.sigmoid(20.0 * (rear_ground_max_height - rear_pos[..., 2]))
    rear_before_edge = torch.sigmoid(10.0 * (platform_front_x + rear_edge_margin - rear_x))
    rear_support = torch.min(rear_contact * rear_ground * rear_before_edge, dim=1).values

    root = _robot(env, root_asset_cfg)
    forward_speed = torch.clamp(root.data.root_lin_vel_w[:, 0] / max_forward_speed, min=0.0, max=1.0)
    reward = rear_support * forward_speed
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return reward


def joint_position_targets_l2(
    env: ManagerBasedRLEnv,
    joint_targets: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    activation_root_height: float | None = None,
    root_height_sharpness: float = 20.0,
    deadband: float = 0.0,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize squared error from explicit joint-position targets."""
    asset = _robot(env, asset_cfg)
    error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in joint_targets.items():
        joint_ids = asset.find_joints(joint_name)[0]
        joint_error = torch.abs(asset.data.joint_pos[:, joint_ids] - target)
        if deadband > 0.0:
            joint_error = torch.clamp(joint_error - deadband, min=0.0)
        error += torch.mean(torch.square(joint_error), dim=1)
    error *= _root_height_gate(
        env, activation_root_height, root_height_sharpness, asset_cfg
    )
    error *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return error


def joint_position_targets_improvement(
    env: ManagerBasedRLEnv,
    joint_targets: dict[str, float],
    max_improvement: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward step-to-step reduction in explicit joint-target error."""
    asset = _robot(env, asset_cfg)
    error = torch.zeros(env.num_envs, device=env.device)
    normalized_targets = tuple(sorted(joint_targets.items()))
    for joint_name, target in normalized_targets:
        joint_ids = asset.find_joints(joint_name)[0]
        error += torch.mean(torch.square(asset.data.joint_pos[:, joint_ids] - target), dim=1)

    if not hasattr(env, "_lighthw_joint_target_error_cache"):
        env._lighthw_joint_target_error_cache = {}
    previous = env._lighthw_joint_target_error_cache.get(normalized_targets)
    if previous is None:
        previous = error.detach().clone()
    reset_mask = env.episode_length_buf <= 1
    previous = torch.where(reset_mask, error, previous)
    improvement = torch.clamp(
        (previous - error) / max(max_improvement, 1.0e-6),
        min=-1.0,
        max=1.0,
    )
    env._lighthw_joint_target_error_cache[normalized_targets] = error.detach()
    improvement *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return improvement


def target_approach_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_skill_id: int | None = None,
    target_speed: float | None = None,
    speed_std: float = 0.15,
    max_local_x: float | None = None,
    climb_stage_cfg: dict | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward controlled xy velocity toward the commanded target."""
    asset = _robot(env, asset_cfg)
    target = _target_pos_w(env, command_name)
    delta_xy = target[:, :2] - asset.data.root_pos_w[:, :2]
    direction = delta_xy / torch.clamp(torch.linalg.norm(delta_xy, dim=1, keepdim=True), min=1.0e-6)
    speed_to_target = torch.sum(asset.data.root_lin_vel_w[:, :2] * direction, dim=1)
    if target_speed is None:
        reward = torch.clamp(speed_to_target / max_speed, min=0.0, max=1.0)
    else:
        baseline = math.exp(-target_speed**2 / speed_std**2)
        reward = (
            torch.exp(-torch.square(speed_to_target - target_speed) / speed_std**2)
            - baseline
        ) / max(1.0 - baseline, 1.0e-6)
        reward = torch.clamp(reward, min=0.0, max=1.0)
    if max_local_x is not None:
        local_root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        reward *= (local_root_x < max_local_x).float()
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, None, maximum_climb_stage)
    return reward


def target_distance_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_progress: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_skill_id: int | None = None,
    max_local_x: float | None = None,
    climb_stage_cfg: dict | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward actual reduction of xy distance to the commanded target."""
    asset = _robot(env, asset_cfg)
    target = _target_pos_w(env, command_name)
    distance = torch.linalg.norm(target[:, :2] - asset.data.root_pos_w[:, :2], dim=1)

    if not hasattr(env, "_lighthw_prev_target_distance"):
        env._lighthw_prev_target_distance = distance.clone()

    reset_mask = env.episode_length_buf <= 1
    prev_distance = torch.where(reset_mask, distance, env._lighthw_prev_target_distance)
    progress = prev_distance - distance
    env._lighthw_prev_target_distance = distance.detach()

    reward = torch.clamp(progress / max_progress, min=-1.0, max=1.0)
    if max_local_x is not None:
        local_root_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        reward *= (local_root_x < max_local_x).float()
    reward *= _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, None, maximum_climb_stage)
    return reward


def xy_speed_limit(
    env: ManagerBasedRLEnv,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    """Penalize excessive horizontal root speed."""
    asset = _robot(env, asset_cfg)
    speed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    penalty = torch.square(torch.clamp(speed - max_speed, min=0.0))
    return penalty * _skill_gate(env, command_name, active_skill_id)


def bodies_over_platform_edge(
    env: ManagerBasedRLEnv,
    platform_front_x: float,
    min_height: float,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
    x_sharpness: float = 8.0,
    z_sharpness: float = 10.0,
    aggregation: str = "mean",
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward selected bodies being past the platform front edge and above the platform top."""
    asset = _robot(env, asset_cfg)
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_x_env = body_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    body_z = body_pos[..., 2]
    if aggregation == "minimum":
        body_x = torch.min(body_x_env, dim=1).values
        height = torch.min(body_z, dim=1).values
    elif aggregation == "mean":
        body_x = torch.mean(body_x_env, dim=1)
        height = torch.mean(body_z, dim=1)
    else:
        raise ValueError(f"Unsupported body aggregation: {aggregation}")
    x_score = torch.sigmoid(x_sharpness * (body_x - platform_front_x))
    z_score = torch.sigmoid(z_sharpness * (height - min_height))
    reward = x_score * z_score * _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return reward


def joint_pair_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str]],
    use_absolute: bool = False,
    deadband: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
    activation_root_height: float | None = None,
    root_height_sharpness: float = 20.0,
) -> torch.Tensor:
    """Penalize left-right joint-position mismatch."""
    asset = _robot(env, asset_cfg)
    normalized_pairs = tuple(tuple(pair) for pair in joint_pairs)
    cache_key = ("absolute" if use_absolute else "signed", normalized_pairs)
    if not hasattr(env, "_lighthw_joint_pair_symmetry_cache"):
        env._lighthw_joint_pair_symmetry_cache = {}
    if cache_key not in env._lighthw_joint_pair_symmetry_cache:
        env._lighthw_joint_pair_symmetry_cache[cache_key] = [
            (asset.find_joints(left_name)[0], asset.find_joints(right_name)[0])
            for left_name, right_name in joint_pairs
        ]

    penalty = torch.zeros(env.num_envs, device=env.device)
    for left_ids, right_ids in env._lighthw_joint_pair_symmetry_cache[cache_key]:
        left_pos = asset.data.joint_pos[:, left_ids]
        right_pos = asset.data.joint_pos[:, right_ids]
        if use_absolute:
            left_pos = torch.abs(left_pos)
            right_pos = torch.abs(right_pos)
        difference = torch.clamp(
            torch.abs(left_pos - right_pos) - deadband,
            min=0.0,
        )
        penalty += torch.mean(torch.square(difference), dim=1)
    penalty /= max(len(joint_pairs), 1)
    penalty *= _root_height_gate(
        env, activation_root_height, root_height_sharpness, asset_cfg
    )
    penalty *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return penalty


def joint_position_targets_exp(
    env: ManagerBasedRLEnv,
    joint_targets: dict[str, float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_local_x: float | None = None,
    max_local_x: float | None = None,
    gate_sharpness: float = 8.0,
    activation_asset_cfg: SceneEntityCfg | None = None,
    activation_front_x: float = 0.0,
    activation_min_height: float = 0.0,
    activation_sharpness: float = 10.0,
    activation_root_height: float | None = None,
    root_height_sharpness: float = 20.0,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Reward selected joints tracking explicit position targets."""
    asset = _robot(env, asset_cfg)
    error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in joint_targets.items():
        joint_ids = asset.find_joints(joint_name)[0]
        error += torch.mean(torch.square(asset.data.joint_pos[:, joint_ids] - target), dim=1)
    reward = torch.exp(-error / std**2)
    reward *= _local_x_window_gate(env, min_local_x, max_local_x, gate_sharpness)
    if activation_asset_cfg is not None:
        reward *= _bodies_over_edge_gate(
            env,
            activation_front_x,
            activation_min_height,
            activation_asset_cfg,
            activation_sharpness,
            activation_sharpness,
        )
    reward *= _root_height_gate(env, activation_root_height, root_height_sharpness)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, maximum_climb_stage)
    return reward


def raw_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """策略原始动作变化平方和；使用负权重抑制每步命令突变。

    它惩罚的是网络输出变化，不等同于关节速度或力矩。过强会让策略失去快速纠偏
    能力，所以通常在已经能站住后再逐步增强。
    """
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def root_vertical_speed_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    """Penalize root vertical speed to suppress ballistic jumping."""
    asset = _robot(env, asset_cfg)
    return torch.square(asset.data.root_lin_vel_w[:, 2]) * _skill_gate(env, command_name, active_skill_id)


def root_roll_pitch_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
    maximum_climb_stage: int | None = None,
) -> torch.Tensor:
    """Penalize root roll/pitch angular velocity for stable crawling."""
    asset = _robot(env, asset_cfg)
    penalty = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    penalty *= _skill_gate(env, command_name, active_skill_id)
    penalty *= _climb_stage_gate(
        env, climb_stage_cfg, required_climb_stage, maximum_climb_stage
    )
    return penalty


def trunk_platform_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_height: float,
    xy_tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_skill_id: int | None = None,
    platform_front_x: float | None = None,
    platform_height: float | None = None,
    wheel_asset_cfg: SceneEntityCfg | None = None,
    wheel_sensor_cfg: SceneEntityCfg | None = None,
    minimum_upright: float = 0.75,
    climb_stage_cfg: dict | None = None,
    required_climb_stage: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    target = _target_pos_w(env, command_name)
    xy_error = torch.linalg.norm(asset.data.root_pos_w[:, :2] - target[:, :2], dim=1)
    high_enough = asset.data.root_pos_w[:, 2] > min_height
    close_enough = xy_error < xy_tolerance
    upright = -asset.data.projected_gravity_b[:, 2] > minimum_upright
    success = high_enough & close_enough & upright
    if wheel_asset_cfg is not None and platform_front_x is not None and platform_height is not None:
        wheels = _robot(env, wheel_asset_cfg)
        wheel_pos = wheels.data.body_pos_w[:, wheel_asset_cfg.body_ids]
        wheel_x = wheel_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
        success &= torch.all(wheel_x > platform_front_x, dim=1)
        success &= torch.all(wheel_pos[..., 2] > platform_height, dim=1)
    if wheel_sensor_cfg is not None:
        sensor: ContactSensor = env.scene.sensors[wheel_sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, wheel_sensor_cfg.body_ids]
        success &= torch.all(
            torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0, dim=1
        )
    reward = success.float() * _skill_gate(env, command_name, active_skill_id)
    reward *= _climb_stage_gate(env, climb_stage_cfg, required_climb_stage, None)
    return reward


def flip_axis_progress(
    env: ManagerBasedRLEnv,
    axis: str,
    target_angle: float,
    cycle_time: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    """Phase-conditioned flip shaping around pitch or roll.

    Euler pitch cannot represent a full 360 degree rotation reliably, so this
    term rewards a sustained spin during takeoff and switches to upright landing
    shaping near the end of the cycle.
    """
    asset = _robot(env, asset_cfg)
    axis_id = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
    phase = torch.clamp(env.episode_length_buf.float() * env.step_dt / cycle_time, 0.0, 1.0)
    target_rate = target_angle / max(0.65 * cycle_time, 1.0e-6)
    spin_reward = torch.exp(torch.square(asset.data.root_ang_vel_b[:, axis_id] - target_rate).neg() / std**2)
    landing_reward = torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    return torch.where(phase < 0.70, spin_reward, landing_reward) * _skill_gate(env, command_name, active_skill_id)


def angular_velocity_window(
    env: ManagerBasedRLEnv,
    axis: str,
    target_rate: float,
    start_phase: float,
    end_phase: float,
    cycle_time: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    axis_id = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
    phase = env.episode_length_buf.float() * env.step_dt / cycle_time
    mask = (phase >= start_phase) & (phase <= end_phase)
    error = torch.square(asset.data.root_ang_vel_b[:, axis_id] - target_rate)
    return torch.exp(-error / std**2) * mask.float() * _skill_gate(env, command_name, active_skill_id)


def takeoff_height_window(
    env: ManagerBasedRLEnv,
    target_vertical_velocity: float,
    start_phase: float,
    end_phase: float,
    cycle_time: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    phase = env.episode_length_buf.float() * env.step_dt / cycle_time
    mask = (phase >= start_phase) & (phase <= end_phase)
    error = torch.square(asset.data.root_lin_vel_w[:, 2] - target_vertical_velocity)
    return torch.exp(-error / std**2) * mask.float() * _skill_gate(env, command_name, active_skill_id)


def upright_after_phase(
    env: ManagerBasedRLEnv,
    start_phase: float,
    cycle_time: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    phase = env.episode_length_buf.float() * env.step_dt / cycle_time
    reward = torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0) * (phase >= start_phase).float()
    return reward * _skill_gate(env, command_name, active_skill_id)


def wheel_contact_count_exp(
    env: ManagerBasedRLEnv,
    expected_contacts: int,
    std: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    in_contact = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0
    count_error = torch.square(torch.sum(in_contact.float(), dim=1) - expected_contacts)
    return torch.exp(-count_error / std**2) * _skill_gate(env, command_name, active_skill_id)


def low_root_speed(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    active_skill_id: int | None = None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    speed_error = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)
    return torch.exp(-speed_error / std**2) * _skill_gate(env, command_name, active_skill_id)
