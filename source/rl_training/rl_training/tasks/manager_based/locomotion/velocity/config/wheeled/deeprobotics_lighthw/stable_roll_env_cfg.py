# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Stable low-speed wheeled LightHW route.

This route keeps the asymmetric 60/236 observation layout and the LightHW
asset, while adding only the corrections needed to make forward, reverse and
zero-command rolling easier to learn.  The original rough and blind routes
remain registered unchanged for comparison.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .adapted_rewards import (
    lighthw_stand_leg_velocity_l2,
    lighthw_stand_wheel_velocity_l2,
    lighthw_straight_yaw_rate_l2,
)
from .rough_env_cfg import FOOT_LINK_PATTERN, LEG_JOINT_NAMES, DeeproboticsLightHWRoughEnvCfg


@configclass
class DeeproboticsLightHWStableRollEnvCfg(DeeproboticsLightHWRoughEnvCfg):
    """Low-speed rolling route with explicit static and straight-line shaping."""

    def __post_init__(self):
        super().__post_init__()

        # Keep policy blind and critic privileged, exactly as the current
        # LightHW baseline: only the critic receives the rough height scan.
        self.observations.policy.height_scan = None

        # Stage A command envelope: learn forward/reverse rolling first, then
        # widen lateral and yaw ranges after the body remains stable.
        command = self.commands.base_velocity
        command.resampling_time_range = (8.0, 8.0)
        command.rel_standing_envs = 0.02
        command.rel_heading_envs = 0.0
        command.heading_command = True
        command.ranges.lin_vel_x = (-0.60, 0.60)
        command.ranges.lin_vel_y = (-0.10, 0.10)
        command.ranges.ang_vel_z = (-0.10, 0.10)
        command.ranges.heading = (0.0, 0.0)

        # Remove zero-weight placeholders before recreating the terms below;
        # this keeps the reward manager free of empty selectors.
        self.disable_zero_weight_rewards()

        self.rewards.stand_lin_vel_xy_l2 = RewTerm(
            func=mdp.stand_lin_vel_xy_l2,
            weight=-0.50,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_ang_vel_xy_l2 = RewTerm(
            func=mdp.stand_ang_vel_xy_l2,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_flat_orientation_l2 = RewTerm(
            func=mdp.stand_flat_orientation_l2,
            weight=-0.25,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_wheel_velocity_l2 = RewTerm(
            func=lighthw_stand_wheel_velocity_l2,
            weight=-0.01,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
            },
        )
        self.rewards.stand_leg_velocity_l2 = RewTerm(
            func=lighthw_stand_leg_velocity_l2,
            weight=-0.005,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            },
        )
        self.rewards.stand_contact_force_uniformity = RewTerm(
            func=mdp.stand_contact_force_uniformity,
            weight=-0.02,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_PATTERN]),
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "force_floor": 8.0,
                "min_contact_force": 12.0,
                "unload_weight": 0.8,
                "rear_unload_weight": 1.0,
                "use_history": False,
            },
        )
        self.rewards.straight_yaw_rate_l2 = RewTerm(
            func=lighthw_straight_yaw_rate_l2,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "min_abs_lin_vel_x": 0.15,
                "max_abs_lin_vel_y": 0.12,
                "max_abs_ang_vel_z": 0.12,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Lower actuator targets and add modest physical damping in this new
        # route.  The old rough/blind action mapping is left untouched.
        self.actions.joint_vel.scale = 30.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-30.0, 30.0)}
        self.scene.robot.actuators["joint"].damping = 3.0
        self.scene.robot.actuators["wheel"].damping = 1.0

        # Keep bad_orientation_2 disabled during this first stabilization run;
        # enable it only after the fixed-command posture test is repeatable.
        self.terminations.bad_orientation_2 = None

        print("[LightHW Stable Roll] dimensions: policy=60, critic=236, action=16")
        print("[LightHW Stable Roll] commands: vx=+/-0.60, vy=+/-0.10, wz=+/-0.10")

