# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_trunk_climb_mixed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    platform_probability: float,
    approach_pose_range: dict[str, tuple[float, float]],
    approach_velocity_range: dict[str, tuple[float, float]],
    platform_root_range: dict[str, tuple[float, float]],
    platform_velocity_range: dict[str, tuple[float, float]],
    approach_joint_noise: float,
    platform_joint_noise: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset most environments below the trunk and a small subset in a valid platform stance."""
    asset = env.scene[asset_cfg.name]
    count = len(env_ids)
    platform_mask = torch.rand(count, device=asset.device) < platform_probability

    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, :3] + env.scene.env_origins[env_ids]
    orientations = root_states[:, 3:7].clone()
    velocities = root_states[:, 7:13].clone()

    approach_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    approach_ranges = torch.tensor(
        [approach_pose_range.get(key, (0.0, 0.0)) for key in approach_keys],
        device=asset.device,
    )
    approach_samples = math_utils.sample_uniform(
        approach_ranges[:, 0], approach_ranges[:, 1], (count, 6), device=asset.device
    )
    positions += approach_samples[:, :3]
    approach_delta = math_utils.quat_from_euler_xyz(
        approach_samples[:, 3], approach_samples[:, 4], approach_samples[:, 5]
    )
    orientations = math_utils.quat_mul(orientations, approach_delta)

    platform_ranges = torch.tensor(
        [platform_root_range.get(key, (0.0, 0.0)) for key in approach_keys],
        device=asset.device,
    )
    platform_samples = math_utils.sample_uniform(
        platform_ranges[:, 0], platform_ranges[:, 1], (count, 6), device=asset.device
    )
    platform_positions = env.scene.env_origins[env_ids] + platform_samples[:, :3]
    platform_orientations = math_utils.quat_from_euler_xyz(
        platform_samples[:, 3], platform_samples[:, 4], platform_samples[:, 5]
    )
    positions[platform_mask] = platform_positions[platform_mask]
    orientations[platform_mask] = platform_orientations[platform_mask]

    def sample_velocity(ranges_dict: dict[str, tuple[float, float]]) -> torch.Tensor:
        ranges = torch.tensor(
            [ranges_dict.get(key, (0.0, 0.0)) for key in approach_keys],
            device=asset.device,
        )
        return math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (count, 6), device=asset.device
        )

    velocities += sample_velocity(approach_velocity_range)
    platform_velocities = sample_velocity(platform_velocity_range)
    velocities[platform_mask] = platform_velocities[platform_mask]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos += math_utils.sample_uniform(
        -approach_joint_noise,
        approach_joint_noise,
        joint_pos.shape,
        asset.device,
    )
    platform_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    for joint_pattern, noise in platform_joint_noise.items():
        joint_ids = asset.find_joints(joint_pattern)[0]
        platform_joint_pos[:, joint_ids] += math_utils.sample_uniform(
            -noise,
            noise,
            platform_joint_pos[:, joint_ids].shape,
            asset.device,
        )
    joint_pos[platform_mask] = platform_joint_pos[platform_mask]
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    if not hasattr(env, "_lighthw_platform_reset_mask"):
        env._lighthw_platform_reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=asset.device)
    env._lighthw_platform_reset_mask[env_ids] = platform_mask
