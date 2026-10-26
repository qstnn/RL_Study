"""Flat, straight-line LightHW transfer route.

This route is intentionally narrower than the rough high-speed experiment:
it keeps the proven low-speed action and actuator mapping, removes lateral and
yaw commands, and uses a plane so the policy can first learn an attractive,
four-wheel-supported straight roll before terrain or turning are reintroduced.
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
from .hierarchical_commands import HierarchicalLightHWVelocityCommandCfg
from .rough_env_cfg import FOOT_LINK_PATTERN
from .stable_roll_env_cfg import DeeproboticsLightHWStableRollEnvCfg
from .stable_roll_v3_rewards import (
    track_ang_vel_z_upright_exp,
    track_lin_vel_xy_upright_exp,
)


@configclass
class LightHWFlatStraightHighSpeedCommandsCfg(CommandsCfg):
    """Progressive forward/reverse speed commands with zero lateral/yaw input."""

    base_velocity = HierarchicalLightHWVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        stage_probabilities=(0.10, 0.30, 0.25, 0.35),
        lateral_range=(0.0, 0.0),
        yaw_range=(0.0, 0.0),
        low_speed_range=(-0.60, 0.60),
        mid_speed_range=(-1.50, 1.50),
        high_speed_min=2.00,
        high_speed_max=3.20,
        high_speed_anchor_ratio=0.50,
        ranges=HierarchicalLightHWVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.20, 3.20),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class DeeproboticsLightHWFlatStraightHighSpeedEnvCfg(DeeproboticsLightHWStableRollEnvCfg):
    """Warm-start-compatible flat straight-line high-speed environment."""

    commands: LightHWFlatStraightHighSpeedCommandsCfg = LightHWFlatStraightHighSpeedCommandsCfg()

    def __post_init__(self):
        # StableRoll supplies the reference low-speed pose, wheel scale=30,
        # wheel damping=1, policy=60/critic=236 and the existing stand terms.
        super().__post_init__()

        # Keep the critic's height scan so the old 60/236/16 low-speed
        # checkpoint can be fully warm-started.  The actor remains blind.
        self.observations.policy.height_scan = None

        # This route is a clean plane baseline.  The scanners remain present
        # for the privileged critic and simply observe the plane height.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None

        # Do not mix a high-speed transfer diagnosis with pushes or external
        # wrench events.  Mass, friction and actuator randomization remain
        # active so the policy still sees the same robot family as StableRoll.
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None

        command = self.commands.base_velocity
        command.resampling_time_range = (8.0, 8.0)
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.heading_command = False
        command.stage_probabilities = (0.10, 0.30, 0.25, 0.35)
        command.lateral_range = (0.0, 0.0)
        command.yaw_range = (0.0, 0.0)
        command.low_speed_range = (-0.60, 0.60)
        command.mid_speed_range = (-1.50, 1.50)
        command.high_speed_min = 2.00
        command.high_speed_max = 3.20
        command.high_speed_anchor_ratio = 0.50
        command.ranges.lin_vel_x = (-3.20, 3.20)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = None

        # Preserve the low-speed checkpoint's action semantics during the
        # transfer.  The failed rough high-speed route changed the wheel scale
        # from 30 to 40, so its policy output no longer had the same meaning.
        self.actions.joint_vel.scale = 30.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-30.0, 30.0)}
        self.scene.robot.actuators["joint"].damping = 3.0
        self.scene.robot.actuators["wheel"].damping = 1.0

        # Moving-speed tracking must also prefer a level body.  The straight
        # yaw term catches left/right drive asymmetry without asking for a
        # nonzero yaw command.
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=-0.15,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
        )
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_upright_exp
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_upright_exp
        self.rewards.upward = RewTerm(
            func=lighthw_upright_orientation_l2,
            weight=-0.08,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
        )
        self.rewards.straight_yaw_rate_l2 = RewTerm(
            func=lighthw_straight_yaw_rate_l2,
            weight=-0.15,
            params={
                "command_name": "base_velocity",
                "min_abs_lin_vel_x": 0.15,
                "max_abs_lin_vel_y": 0.05,
                "max_abs_ang_vel_z": 0.05,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.wheel_contact_support = RewTerm(
            func=lighthw_wheel_contact_support,
            weight=0.35,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_PATTERN]),
                "min_contact_force": 5.0,
            },
        )
        self.rewards.wheel_contact_uniformity = RewTerm(
            func=lighthw_wheel_contact_uniformity,
            weight=-0.02,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_LINK_PATTERN]),
                "min_contact_force": 5.0,
                "force_scale": 8.0,
            },
        )

        # The inherited static terms remain useful for the 10% standing stage;
        # their command gate prevents them from braking high-speed rolling.
        self.rewards.stand_contact_force_uniformity.params["command_threshold"] = 0.15

        print(
            "[LightHW Flat Straight HighSpeed] policy=60, critic=236, action=16; "
            "vx=(-3.2, 3.2), vy=0, wz=0; plane; four-wheel support reward"
        )


@configclass
class DeeproboticsLightHWFlatStraightHighSpeedFlatEnvCfg(
    DeeproboticsLightHWFlatStraightHighSpeedEnvCfg
):
    """Explicit flat alias used by smoke tests and playback tooling."""

    pass
