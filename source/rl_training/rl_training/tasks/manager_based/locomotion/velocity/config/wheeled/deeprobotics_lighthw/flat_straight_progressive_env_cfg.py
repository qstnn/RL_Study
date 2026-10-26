"""Staged flat straight-line LightHW environment.

This is an isolated successor to ``Flat-Deeprobotics-LightHW-StraightHighSpeed-v0``.
The old task remains unchanged for comparison; this task keeps policy=60,
critic=236 and action=16 while fixing the wheel target headroom and command
curriculum used for the next training run.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

from .adapted_rewards import (
    lighthw_straight_yaw_rate_l2,
    lighthw_upright_orientation_l2,
    lighthw_wheel_contact_support,
    lighthw_wheel_contact_uniformity,
)
from .flat_straight_highspeed_env_cfg import (
    DeeproboticsLightHWFlatStraightHighSpeedEnvCfg,
)
from .progressive_commands import ProgressiveLightHWVelocityCommandCfg
from .progressive_rewards import track_lin_vel_xy_upright_exp_l1
from .rough_env_cfg import FOOT_LINK_PATTERN


@configclass
class LightHWFlatStraightProgressiveCommandsCfg(CommandsCfg):
    """Zero-lateral/zero-yaw commands with five forward-speed lessons."""

    base_velocity = ProgressiveLightHWVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        lateral_range=(0.0, 0.0),
        yaw_range=(0.0, 0.0),
        ranges=ProgressiveLightHWVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.2, 3.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class DeeproboticsLightHWFlatStraightProgressiveEnvCfg(
    DeeproboticsLightHWFlatStraightHighSpeedEnvCfg
):
    """Progressive flat route initialized from the completed model_2498 policy."""

    commands: LightHWFlatStraightProgressiveCommandsCfg = LightHWFlatStraightProgressiveCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.stage_start_steps = (0, 28_800, 57_600, 86_400, 115_200)
        command.stage_max_speeds = (1.0, 1.5, 2.0, 2.6, 3.2)
        command.zero_command_probability = 0.02
        command.minimum_abs_speed = 0.20
        command.anchor_probability = 0.40

        # The URDF wheel radius is about 0.0875 m.  Keep the raw scale=30 so
        # old actor outputs retain their meaning, but remove the artificial
        # 30 rad/s processed-action ceiling; the actuator itself is limited to
        # 37.7 rad/s and 37.0 leaves a small margin.
        self.actions.joint_vel.scale = 30.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-37.0, 37.0)}

        # Keep the 0.416 m reference pose and give the moving policy a stronger
        # incentive to recover it without allowing the base-height term to
        # replace velocity tracking.
        self.rewards.base_height_l2.weight = -0.80
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_upright_exp_l1
        self.rewards.track_lin_vel_xy_exp.params["std"] = 1.50
        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        # Make rear unloading expensive, while the speed curriculum and the
        # linear-error reward keep stopping in place from being attractive.
        self.rewards.wheel_contact_support.func = lighthw_wheel_contact_support
        self.rewards.wheel_contact_support.weight = 0.45
        self.rewards.wheel_contact_uniformity.func = lighthw_wheel_contact_uniformity
        self.rewards.wheel_contact_uniformity.weight = -0.08
        self.rewards.wheel_contact_uniformity.params["force_scale"] = 4.0
        self.rewards.straight_yaw_rate_l2.func = lighthw_straight_yaw_rate_l2
        self.rewards.straight_yaw_rate_l2.weight = -0.25

        # Discourage the hip limit and crouched leg configuration seen in the
        # model_2498 fixed-command replay.
        self.rewards.hipx_joint_pos_penalty.weight = -0.55
        self.rewards.joint_pos_limits.weight = -7.0
        self.rewards.upward.func = lighthw_upright_orientation_l2
        self.rewards.upward.weight = -0.12

        print(
            "[LightHW Flat Straight Progressive] policy=60, critic=236, action=16; "
            "speed_lessons=1.0->1.5->2.0->2.6->3.2, vy=0, wz=0, wheel_clip=37"
        )
