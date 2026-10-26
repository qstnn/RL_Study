# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .rewards import _climb_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

# observation 是策略网络的输入。添加新观测会改变网络输入维度，旧 checkpoint
# 通常不能直接加载；只调 reward 权重则不会改变网络结构。


def skill_phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Return sin/cos phase for phase-conditioned acrobatic skills."""
    phase = env.episode_length_buf[:, None].float() * env.step_dt / cycle_time
    return torch.cat((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=-1)


def relative_target_position(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target xyz from command, expressed relative to the robot root in world axes."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_w = env.scene.env_origins + command[:, -4:-1]
    return target_w - asset.data.root_pos_w


def relative_target_position_body(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target xyz relative to the robot root, expressed in the robot frame."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_w = env.scene.env_origins + command[:, -4:-1]
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, target_w - asset.data.root_pos_w)


def climb_stage_observation(env: ManagerBasedEnv, stage_cfg: dict) -> torch.Tensor:
    """Return the normalized climb stage as a one-dimensional observation."""
    return (_climb_stage(env, stage_cfg).float() / 3.0).unsqueeze(-1)


def staged_approach_velocity_command(
    env: ManagerBasedEnv,
    stage_cfg: dict,
    forward_speed: float,
    stop_local_x: float,
) -> torch.Tensor:
    """Expose a rough-policy-compatible forward command only during approach."""
    stage = _climb_stage(env, stage_cfg)
    root = env.scene["robot"]
    local_root_x = root.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    approach_active = (stage == 0) & (local_root_x < stop_local_x)
    command = torch.zeros((env.num_envs, 3), device=env.device)
    command[:, 0] = approach_active.float() * forward_speed
    return command


def platform_transition_state(
    env: ManagerBasedEnv,
    platform_front_x: float,
    platform_height: float,
    wheel_asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Describe platform clearance and support during the airborne-to-landing transition."""
    wheels = env.scene[wheel_asset_cfg.name]
    wheel_pos = wheels.data.body_pos_w[:, wheel_asset_cfg.body_ids]
    min_x = torch.min(
        wheel_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1),
        dim=1,
    ).values
    min_z = torch.min(wheel_pos[..., 2], dim=1).values

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact_fraction = torch.mean(
        (torch.max(torch.norm(forces, dim=-1), dim=1)[0] > 1.0).float(),
        dim=1,
    )
    return torch.stack(
        (
            torch.clamp((min_x - platform_front_x) / 0.50, -1.0, 1.0),
            torch.clamp((min_z - platform_height) / 0.50, -1.0, 1.0),
            contact_fraction,
        ),
        dim=-1,
    )


def quasi_static_support_state(
    env: ManagerBasedEnv,
    platform_front_x: float,
    platform_height: float,
    rear_ground_max_height: float,
    front_asset_cfg: SceneEntityCfg,
    front_sensor_cfg: SceneEntityCfg,
    rear_asset_cfg: SceneEntityCfg,
    rear_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return front-platform and rear-ground support fractions."""
    front = env.scene[front_asset_cfg.name]
    front_pos = front.data.body_pos_w[:, front_asset_cfg.body_ids]
    front_x = front_pos[..., 0] - env.scene.env_origins[:, 0].unsqueeze(-1)
    front_sensor: ContactSensor = env.scene.sensors[front_sensor_cfg.name]
    front_forces = front_sensor.data.net_forces_w_history[:, :, front_sensor_cfg.body_ids]
    front_contact = (
        torch.max(torch.norm(front_forces, dim=-1), dim=1)[0] > 1.0
    ).float()
    front_support = torch.mean(
        front_contact
        * (front_x >= platform_front_x).float()
        * (front_pos[..., 2] >= platform_height - 0.05).float(),
        dim=1,
    )

    rear = env.scene[rear_asset_cfg.name]
    rear_pos = rear.data.body_pos_w[:, rear_asset_cfg.body_ids]
    rear_sensor: ContactSensor = env.scene.sensors[rear_sensor_cfg.name]
    rear_forces = rear_sensor.data.net_forces_w_history[:, :, rear_sensor_cfg.body_ids]
    rear_contact = (
        torch.max(torch.norm(rear_forces, dim=-1), dim=1)[0] > 1.0
    ).float()
    rear_support = torch.mean(
        rear_contact * (rear_pos[..., 2] <= rear_ground_max_height).float(),
        dim=1,
    )
    return torch.stack((front_support, rear_support), dim=-1)
