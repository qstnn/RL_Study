# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Global curriculum scalar in [0, 1], updated from terrain-level mean.
gait_level: float = 0.0

def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
    """Update global gait_level from mean terrain level.

    Mapping rule:
    - mean <= 0.0 -> 0.0
    - 0.0 < mean < 3.0 -> 使用 exp 函数映射
    - mean == 3.0 -> 1.0
    - mean >= 3.0 -> 1.0
    """
    global gait_level

    mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
    if mean_tensor.numel() == 0:
        mean_val = 0.0
    else:
        mean_val = float(torch.mean(mean_tensor).item())

    if math.isnan(mean_val) or math.isinf(mean_val):
        mean_val = 0.0

    if mean_val <= 0.0:
        gait_level = 0.0
    elif mean_val < 3.0:
        # exp 映射：mean=0 时接近 0，mean=3 时恰好为 1
        gait_level = math.exp(mean_val - 3.0)
    else:  # mean_val >= 3.0
        gait_level = 1.0

    return gait_level

def get_gait_level_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return global gait_level as tensor matching environment batch size."""
    return torch.full((env.num_envs,), gait_level, device=env.device)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques (curriculum-scaled by gait_level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward * get_gait_level_tensor(env)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (curriculum-scaled by gait_level)."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward * get_gait_level_tensor(env)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact force violations (curriculum-scaled by gait_level)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    reward = torch.sum(violation.clip(min=0.0), dim=1)
    return reward * get_gait_level_tensor(env)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward * get_gait_level_tensor(env)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize L1 joint deviation from the default pose at standstill.

    目的：提供 M20 配置引用的零命令姿态项；激活条件：命令向量范数小于
    ``command_threshold``；返回值：选定关节默认角偏差绝对值之和，运动命令下为零。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    return deviation * (torch.linalg.norm(command, dim=1) < float(command_threshold)).float()

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def joint_mirror_default_offset(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    """Mirror joint deviations around the model-specific default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_default_offset_cache") or env.joint_mirror_default_offset_cache is None:
        env.joint_mirror_default_offset_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_default_offset_cache:
        joint_a = asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.default_joint_pos[:, joint_pair[0][0]]
        joint_b = asset.data.joint_pos[:, joint_pair[1][0]] - asset.data.default_joint_pos[:, joint_pair[1][0]]
        reward += torch.sum(torch.square(joint_a - joint_b), dim=-1)
    reward *= 1 / len(mirror_joints) if mirror_joints else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def joint_pair_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str]] | list[tuple[str, str, float]],
    use_absolute: bool = False,
    use_default_offset: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mismatch between selected joint pairs.

    A third value in each pair can be used as a sign relation. For example,
    ("FL_hip_joint", "FR_hip_joint", -1.0) enforces opposite deviations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    normalized_pairs = tuple(
        (str(pair[0]), str(pair[1]), float(pair[2]) if len(pair) > 2 else 1.0) for pair in joint_pairs
    )
    cache_key = (normalized_pairs, bool(use_absolute), bool(use_default_offset))
    if not hasattr(env, "joint_pair_symmetry_cache") or env.joint_pair_symmetry_cache is None:
        env.joint_pair_symmetry_cache = {}
    if cache_key not in env.joint_pair_symmetry_cache:
        env.joint_pair_symmetry_cache[cache_key] = [
            (asset.find_joints(left_name)[0], asset.find_joints(right_name)[0], sign)
            for left_name, right_name, sign in normalized_pairs
        ]

    penalty = torch.zeros(env.num_envs, device=env.device)
    for left_ids, right_ids, sign in env.joint_pair_symmetry_cache[cache_key]:
        left_pos = asset.data.joint_pos[:, left_ids]
        right_pos = asset.data.joint_pos[:, right_ids]
        if use_default_offset:
            left_pos = left_pos - asset.data.default_joint_pos[:, left_ids]
            right_pos = right_pos - asset.data.default_joint_pos[:, right_ids]
        if use_absolute:
            left_pos = torch.abs(left_pos)
            right_pos = torch.abs(right_pos)
            sign = 1.0
        penalty += torch.mean(torch.square(left_pos - sign * right_pos), dim=1)
    return penalty / max(len(normalized_pairs), 1)


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stand_contact_force_uniformity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.15,
    force_floor: float = 8.0,
    min_contact_force: float = 12.0,
    unload_weight: float = 0.8,
    rear_unload_weight: float = 1.0,
    rear_body_name_patterns: tuple[str, ...] = ("RL_wheel", "RR_wheel", "hl_wheel", "hr_wheel"),
    use_history: bool = False,
) -> torch.Tensor:
    """Penalize uneven wheel support only in static command states."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if use_history:
        force_z = torch.abs(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2])
        force_z = torch.max(force_z, dim=1)[0]
    else:
        force_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])

    mean_force = torch.mean(force_z, dim=1, keepdim=True)
    normalized_force = force_z / torch.clamp(mean_force, min=force_floor)
    uniformity = torch.mean(torch.square(normalized_force - 1.0), dim=1)

    unload = torch.clamp(min_contact_force - force_z, min=0.0) / min_contact_force
    unload_scale = torch.ones(force_z.shape[1], device=force_z.device, dtype=force_z.dtype)
    if rear_unload_weight != 1.0:
        body_names = getattr(contact_sensor, "body_names", [])
        patterns = tuple(pattern.lower() for pattern in rear_body_name_patterns)
        for local_index, body_id in enumerate(sensor_cfg.body_ids):
            body_name = body_names[int(body_id)] if int(body_id) < len(body_names) else ""
            if any(pattern in body_name.lower() for pattern in patterns):
                unload_scale[local_index] = float(rear_unload_weight)
    unload_penalty = torch.mean(torch.square(unload) * unload_scale.unsqueeze(0), dim=1)

    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return (uniformity + unload_weight * unload_penalty) * command_gate.float()


def stand_rear_contact_force_balance_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.15,
    force_floor: float = 8.0,
    min_rear_force: float = 22.0,
    difference_weight: float = 1.0,
    unload_weight: float = 0.8,
    rear_left_body_name_patterns: tuple[str, ...] = ("RL_wheel", "hl_wheel"),
    rear_right_body_name_patterns: tuple[str, ...] = ("RR_wheel", "hr_wheel"),
) -> torch.Tensor:
    """Penalize rear left/right support mismatch only in static command states."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_names = getattr(contact_sensor, "body_names", [])

    def _find_local_index(patterns: tuple[str, ...]) -> int | None:
        lowered_patterns = tuple(pattern.lower() for pattern in patterns)
        for local_index, body_id in enumerate(sensor_cfg.body_ids):
            body_name = body_names[int(body_id)] if int(body_id) < len(body_names) else ""
            if any(pattern in body_name.lower() for pattern in lowered_patterns):
                return local_index
        return None

    rear_left_index = _find_local_index(rear_left_body_name_patterns)
    rear_right_index = _find_local_index(rear_right_body_name_patterns)
    if rear_left_index is None or rear_right_index is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    rear_left_force = force_z[:, rear_left_index]
    rear_right_force = force_z[:, rear_right_index]
    rear_mean_force = torch.clamp(0.5 * (rear_left_force + rear_right_force), min=force_floor)

    balance_penalty = torch.square((rear_left_force - rear_right_force) / rear_mean_force)
    rear_forces = torch.stack((rear_left_force, rear_right_force), dim=1)
    unload = torch.clamp(min_rear_force - rear_forces, min=0.0) / min_rear_force
    unload_penalty = torch.mean(torch.square(unload), dim=1)

    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return (difference_weight * balance_penalty + unload_weight * unload_penalty) * command_gate.float()


def stand_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize static roll/pitch tilt without constraining locomotion posture."""
    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return flat_orientation_l2(env, asset_cfg=asset_cfg) * command_gate.float()


def stand_joint_limit_margin_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    margin: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize static joints before they sit on the soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    margin = max(float(margin), 1.0e-6)

    lower_penalty = torch.clamp(soft_limits[..., 0] + margin - joint_pos, min=0.0) / margin
    upper_penalty = torch.clamp(joint_pos - soft_limits[..., 1] + margin, min=0.0) / margin
    penalty = torch.mean(torch.square(lower_penalty) + torch.square(upper_penalty), dim=1)

    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return penalty * command_gate.float()


def stand_joint_pair_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str]] | list[tuple[str, str, float]],
    command_name: str,
    command_threshold: float = 0.15,
    use_absolute: bool = False,
    use_default_offset: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Apply selected joint symmetry only while the command is near zero."""
    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return joint_pair_symmetry_l2(
        env,
        joint_pairs=joint_pairs,
        use_absolute=use_absolute,
        use_default_offset=use_default_offset,
        asset_cfg=asset_cfg,
    ) * command_gate.float()


def _straight_motion_command_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (
        (command[:, 0] > min_lin_vel_x)
        & (torch.abs(command[:, 1]) < max_abs_lin_vel_y)
        & (torch.abs(command[:, 2]) < max_abs_ang_vel_z)
    )


def straight_motion_contact_force_uniformity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
    force_floor: float = 8.0,
    min_contact_force: float = 10.0,
    unload_weight: float = 0.6,
) -> torch.Tensor:
    """Penalize one wheel unloading during mostly straight forward motion."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_z = torch.abs(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2])
    force_z = torch.max(force_z, dim=1)[0]

    mean_force = torch.mean(force_z, dim=1, keepdim=True)
    normalized_force = force_z / torch.clamp(mean_force, min=force_floor)
    uniformity = torch.mean(torch.square(normalized_force - 1.0), dim=1)

    unload = torch.clamp(min_contact_force - force_z, min=0.0) / min_contact_force
    unload_penalty = torch.mean(torch.square(unload), dim=1)

    command_gate = _straight_motion_command_gate(
        env,
        command_name,
        min_lin_vel_x=min_lin_vel_x,
        max_abs_lin_vel_y=max_abs_lin_vel_y,
        max_abs_ang_vel_z=max_abs_ang_vel_z,
    )
    return (uniformity + unload_weight * unload_penalty) * command_gate.float()


def straight_motion_joint_pair_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str]] | list[tuple[str, str, float]],
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
    use_absolute: bool = False,
    use_default_offset: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Apply left-right joint symmetry only while rolling mostly straight forward."""
    command_gate = _straight_motion_command_gate(
        env,
        command_name,
        min_lin_vel_x=min_lin_vel_x,
        max_abs_lin_vel_y=max_abs_lin_vel_y,
        max_abs_ang_vel_z=max_abs_ang_vel_z,
    )
    return joint_pair_symmetry_l2(
        env,
        joint_pairs=joint_pairs,
        use_absolute=use_absolute,
        use_default_offset=use_default_offset,
        asset_cfg=asset_cfg,
    ) * command_gate.float()


def straight_motion_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """直行门控的偏航角速度误差平方。

    只在前进且侧向/yaw命令较小的样本上惩罚机身实际偏航速度，
    用于压制累计偏航漂移；不改变静止、侧移和转向命令的奖励。
    返回值为每个环境的 yaw 角速度误差平方，权重由配置中的负值给出。
    """
    command = env.command_manager.get_command(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_error = command[:, 2] - asset.data.root_ang_vel_b[:, 2]
    command_gate = _straight_motion_command_gate(
        env,
        command_name,
        min_lin_vel_x=min_lin_vel_x,
        max_abs_lin_vel_y=max_abs_lin_vel_y,
        max_abs_ang_vel_z=max_abs_ang_vel_z,
    )
    return torch.square(yaw_error) * command_gate.float()


def straight_motion_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch only during mostly straight forward rolling."""
    command_gate = _straight_motion_command_gate(
        env,
        command_name,
        min_lin_vel_x=min_lin_vel_x,
        max_abs_lin_vel_y=max_abs_lin_vel_y,
        max_abs_ang_vel_z=max_abs_ang_vel_z,
    )
    return flat_orientation_l2(env, asset_cfg=asset_cfg) * command_gate.float()


def _wheel_pos_in_base_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected wheel/body positions in the root-link frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    rel_pos_w = wheel_pos_w - asset.data.root_link_pos_w[:, None, :]
    num_wheels = len(asset_cfg.body_ids)
    return quat_apply_inverse(
        asset.data.root_link_quat_w[:, None, :].expand(-1, num_wheels, -1).reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_wheels, 3)


def _wheel_z_in_base_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected wheel/body z positions in the root-link frame."""
    return _wheel_pos_in_base_frame(env, asset_cfg=asset_cfg)[..., 2]


def _wheel_height_symmetry_penalty(
    wheel_z: torch.Tensor,
    height_scale: float,
    front_rear_weight: float,
    left_right_weight: float,
    variance_weight: float,
) -> torch.Tensor:
    wheel_z = wheel_z / height_scale
    front_z = torch.mean(wheel_z[:, 0:2], dim=1)
    rear_z = torch.mean(wheel_z[:, 2:4], dim=1)
    left_z = torch.mean(wheel_z[:, [0, 2]], dim=1)
    right_z = torch.mean(wheel_z[:, [1, 3]], dim=1)
    variance = torch.mean(torch.square(wheel_z - torch.mean(wheel_z, dim=1, keepdim=True)), dim=1)
    return (
        front_rear_weight * torch.square(front_z - rear_z)
        + left_right_weight * torch.square(left_z - right_z)
        + variance_weight * variance
    )


def stand_wheel_height_symmetry_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    height_scale: float = 0.08,
    front_rear_weight: float = 0.5,
    left_right_weight: float = 1.0,
    variance_weight: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize uneven wheel height in the base frame only while standing."""
    wheel_z = _wheel_z_in_base_frame(env, asset_cfg=asset_cfg)
    penalty = _wheel_height_symmetry_penalty(
        wheel_z,
        height_scale=height_scale,
        front_rear_weight=front_rear_weight,
        left_right_weight=left_right_weight,
        variance_weight=variance_weight,
    )
    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return penalty * command_gate.float()


def stand_wheel_xy_geometry_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    target_front_x: float = 0.235,
    target_rear_x: float = -0.243,
    target_left_y: float = 0.202,
    target_right_y: float = -0.202,
    x_scale: float = 0.12,
    y_scale: float = 0.08,
    x_weight: float = 0.5,
    y_weight: float = 1.0,
    rear_y_weight: float = 1.5,
    pair_symmetry_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the standing wheels in a symmetric rectangle in the base frame.

    The body order is expected to be FL, FR, RL, RR. This term targets the visual standing
    posture directly, instead of inferring it only from joint angles whose signs differ by leg.
    """
    wheel_pos = _wheel_pos_in_base_frame(env, asset_cfg=asset_cfg)
    target_x = torch.tensor(
        [target_front_x, target_front_x, target_rear_x, target_rear_x],
        dtype=wheel_pos.dtype,
        device=wheel_pos.device,
    )
    target_y = torch.tensor(
        [target_left_y, target_right_y, target_left_y, target_right_y],
        dtype=wheel_pos.dtype,
        device=wheel_pos.device,
    )

    x_error = torch.mean(torch.square((wheel_pos[..., 0] - target_x) / x_scale), dim=1)
    y_error = torch.mean(torch.square((wheel_pos[..., 1] - target_y) / y_scale), dim=1)

    rear_target_abs_y = 0.5 * (abs(target_left_y) + abs(target_right_y))
    rear_y = wheel_pos[:, 2:4, 1]
    rear_inward = torch.mean(torch.square(torch.clamp(rear_target_abs_y - torch.abs(rear_y), min=0.0) / y_scale), dim=1)

    front_pair_y = torch.square((wheel_pos[:, 0, 1] + wheel_pos[:, 1, 1]) / y_scale)
    rear_pair_y = torch.square((wheel_pos[:, 2, 1] + wheel_pos[:, 3, 1]) / y_scale)
    pair_symmetry = 0.5 * (front_pair_y + rear_pair_y)

    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    penalty = x_weight * x_error + y_weight * y_error + rear_y_weight * rear_inward + pair_symmetry_weight * pair_symmetry
    return penalty * command_gate.float()


def straight_motion_front_wheel_lower_than_rear_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
    allowed_front_lower: float = 0.015,
    height_scale: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize front wheels sitting noticeably lower than rear wheels during straight forward motion."""
    wheel_z = _wheel_z_in_base_frame(env, asset_cfg=asset_cfg)
    front_z = torch.mean(wheel_z[:, 0:2], dim=1)
    rear_z = torch.mean(wheel_z[:, 2:4], dim=1)
    front_lower = torch.clamp(rear_z - front_z - allowed_front_lower, min=0.0) / height_scale
    command_gate = _straight_motion_command_gate(
        env,
        command_name,
        min_lin_vel_x=min_lin_vel_x,
        max_abs_lin_vel_y=max_abs_lin_vel_y,
        max_abs_ang_vel_z=max_abs_ang_vel_z,
    )
    return torch.square(front_lower) * command_gate.float()


def straight_motion_wheel_height_symmetry_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_lin_vel_x: float = 0.2,
    max_abs_lin_vel_y: float = 0.35,
    max_abs_ang_vel_z: float = 0.35,
    height_scale: float = 0.08,
    front_rear_weight: float = 1.0,
    left_right_weight: float = 0.5,
    variance_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize front/rear and left/right wheel height imbalance in the base frame."""
    wheel_z = _wheel_z_in_base_frame(env, asset_cfg=asset_cfg)
    command_gate = _straight_motion_command_gate(
        env,
        command_name,
        min_lin_vel_x=min_lin_vel_x,
        max_abs_lin_vel_y=max_abs_lin_vel_y,
        max_abs_ang_vel_z=max_abs_ang_vel_z,
    )
    penalty = _wheel_height_symmetry_penalty(
        wheel_z,
        height_scale=height_scale,
        front_rear_weight=front_rear_weight,
        left_right_weight=left_right_weight,
        variance_weight=variance_weight,
    )
    return penalty * command_gate.float()


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _bernstein_torch(n: int, k: int, t: torch.Tensor) -> torch.Tensor:
    """Bernstein basis B_k^n(t) for tensor t in [0, 1]."""
    coeff = float(math.comb(n, k))
    return coeff * (1.0 - t) ** (n - k) * t**k


def _bezier_curve_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve points for batched parameter t.

    Args:
        control_points: Tensor of shape [m, 2].
        t: Tensor of shape [N, L] in [0, 1].

    Returns:
        Tensor of shape [N, L, 2].
    """
    n = control_points.shape[0] - 1
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n + 1):
        out = out + _bernstein_torch(n, k, t).unsqueeze(-1) * control_points[k]
    return out


def _bezier_curve_derivative_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate d/dt of Bezier curve points for batched parameter t."""
    n = control_points.shape[0] - 1
    delta_ctrl = n * (control_points[1:] - control_points[:-1])
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n):
        out = out + _bernstein_torch(n - 1, k, t).unsqueeze(-1) * delta_ctrl[k]
    return out


def phase_foot_trajectory_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
    command_threshold: float = 0.1,
    cycle_time: float = 0.4,
    phase_offsets: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0),
    gait_span: float = -0.008,
    gait_psi: float = 0.15,
    gait_delta: float = 0.03,
    x_offset: float = 0.0,
    stance_span: float = 0.20,
    stand_ref_z_offset: float = -0.2,
    velocity_weight: float = 0.5,
) -> torch.Tensor:
    """Track MuJoCo-style phase foot trajectory in body frame with exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_feet = len(body_ids)

    if num_feet == 0:
        return torch.zeros(env.num_envs, device=env.device)
    if len(phase_offsets) != num_feet:
        raise ValueError(f"phase_offsets length ({len(phase_offsets)}) must match tracked feet ({num_feet}).")

    # Build and cache base-fixed stand references from the first call.
    if (not hasattr(env, "phase_foot_ref_body")) or (env.phase_foot_ref_body.shape[1] != num_feet):
        rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
        foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
        for i in range(num_feet):
            foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        ref = foot_pos_b[0].detach().clone()
        ref[:, 2] += stand_ref_z_offset
        env.phase_foot_ref_body = ref.unsqueeze(0)

    stand_ref_body = env.phase_foot_ref_body.to(env.device).expand(env.num_envs, -1, -1)

    # Build phase S in [0, 2).
    phase_time = env.episode_length_buf.float() * env.step_dt
    phase_offsets_t = torch.tensor(phase_offsets, device=env.device, dtype=phase_time.dtype).unsqueeze(0)
    S = torch.remainder((2.0 * phase_time / max(cycle_time, 1e-6)).unsqueeze(1) + phase_offsets_t, 2.0)

    # MuJoCo-like piecewise trajectory in local (q, z).
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)
    stance_span = float(stance_span)
    stance_span = min(max(stance_span, 1e-6), 2.0 - 1e-6)

    q = torch.zeros_like(S)
    z = torch.zeros_like(S)
    dq_dS = torch.zeros_like(S)
    dz_dS = torch.zeros_like(S)

    stance_mask = S < stance_span
    if stance_mask.any():
        s_stance = S / stance_span
        q_stance = tau * (1.0 - 2.0 * s_stance)
        z_stance = torch.full_like(S, delta)
        dq_dS_stance = torch.full_like(S, -2.0 * tau / stance_span)
        dz_dS_stance = torch.zeros_like(S)

        q = torch.where(stance_mask, q_stance, q)
        z = torch.where(stance_mask, z_stance, z)
        dq_dS = torch.where(stance_mask, dq_dS_stance, dq_dS)
        dz_dS = torch.where(stance_mask, dz_dS_stance, dz_dS)

    swing_mask = ~stance_mask
    if swing_mask.any():
        t_bezier = torch.clamp((S - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=env.device,
            dtype=S.dtype,
        )
        qz_swing = _bezier_curve_torch(ctrl, t_bezier)
        dqz_dt = _bezier_curve_derivative_torch(ctrl, t_bezier)
        dt_dS = 1.0 / (2.0 - stance_span)

        q = torch.where(swing_mask, qz_swing[..., 0], q)
        z = torch.where(swing_mask, qz_swing[..., 1] + delta, z)
        dq_dS = torch.where(swing_mask, dqz_dt[..., 0] * dt_dS, dq_dS)
        dz_dS = torch.where(swing_mask, dqz_dt[..., 1] * dt_dS, dz_dS)

    dS_dt = 2.0 / max(cycle_time, 1e-6)
    dq_dt = dq_dS * dS_dt
    dz_dt = dz_dS * dS_dt

    ref_pos_b = stand_ref_body + torch.stack(
        [q + float(x_offset), torch.zeros_like(q), z],
        dim=-1,
    )
    ref_vel_b = torch.stack(
        [dq_dt, torch.zeros_like(dq_dt), dz_dt],
        dim=-1,
    )

    # Actual foot states in body frame.
    rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    rel_foot_vel_w = asset.data.body_lin_vel_w[:, body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    for i in range(num_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_vel_w[:, i, :])

    pos_offset = foot_pos_b - ref_pos_b
    vel_offset = foot_vel_b - ref_vel_b

    # Per-dimension (x, y, z) errors over feet for each environment.
    pos_err = torch.sum(torch.square(pos_offset), dim=1)
    vel_err = torch.sum(torch.square(vel_offset), dim=1)

    # Scalar total error for reward computation.
    total_err = torch.sum(pos_err, dim=1) + float(velocity_weight) * torch.sum(vel_err, dim=1)
    reward = torch.exp(-total_err / max(std, 1e-6) ** 2)

    # Command-gating follows full (x, y, yaw) command magnitude.
    command = env.command_manager.get_command(command_name)
    gate = torch.linalg.norm(command[:, :3], dim=1) > command_threshold

    # info for debugging
    pos_offset_xyz_mean = torch.mean(pos_offset, dim=(0, 1))
    vel_offset_xyz_mean = torch.mean(vel_offset, dim=(0, 1))
    # print(
    #     "Offset xyz mean | "
    #     f"pos(x,y,z)=({pos_offset_xyz_mean[0].item():.4f}, {pos_offset_xyz_mean[1].item():.4f}, {pos_offset_xyz_mean[2].item():.4f}) | "
    #     f"vel(x,y,z)=({vel_offset_xyz_mean[0].item():.4f}, {vel_offset_xyz_mean[1].item():.4f}, {vel_offset_xyz_mean[2].item():.4f})"
    # )
    # print("Reward:", reward * gate.float())
    return reward * gate.float() * get_gait_level_tensor(env)

def foot_impact_velocity(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    foot_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    downward_speed = torch.clamp(-foot_lin_vel[:, :, 2], min=0.0)
    downward_speed = torch.clamp(downward_speed - speed_threshold, min=0.0)

    penalty = torch.sum(first_contact * torch.square(downward_speed), dim=1)
    return penalty * get_gait_level_tensor(env)

# def stand_still_joint_deviation_l1(
#     env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize offsets from the default joint positions when the command is very small."""
    # command = env.command_manager.get_command(command_name)
#     # Penalize motion when command is nearly zero.
#     return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

# def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def stand_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """在零指令时惩罚机身水平速度，直接训练全身保持而非只收腿。

    仅当命令范数低于 ``command_threshold`` 时启用，返回每个环境的机身
    body-frame 水平速度平方和；非零运动命令不会受到该静止项的干扰。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * command_gate.float()


def stand_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """在零指令时惩罚机身横滚/俯仰角速度，抑制静止摇摆。

    仅当命令范数低于 ``command_threshold`` 时启用，返回每个环境的机身
    body-frame roll/pitch 角速度平方和；不约束高速前进时的正常姿态修正。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_gate = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * command_gate.float()


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_lin_xy_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by planar linear velocity command (x, y)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by planar linear velocity command only
    cmd_lin_xy = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= cmd_lin_xy > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_x_neg_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by negative x velocity command only."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate: x command must be negative and exceed magnitude threshold
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward *= cmd_x < 0.0

    return reward

def feet_air_time_ang_z_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by yaw angular velocity command (z)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by angular velocity command only
    cmd_ang_z = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    reward *= cmd_ang_z > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
