"""Conservative flat straight-line recovery route for LightHW.

This route is deliberately separate from the high-speed and progressive
experiments.  It starts from the validated low-speed checkpoint, keeps
``policy=60``/``critic=236``/``action=16``, and holds the command envelope at
``vx=+/-1.0 m/s`` until fixed-command playback passes posture, speed and
four-wheel-support checks.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

from .adapted_rewards import (
    lighthw_straight_yaw_rate_l2,
    lighthw_upright_orientation_l2,
    lighthw_wheel_contact_support,
    lighthw_wheel_contact_uniformity,
)
from .flat_straight_highspeed_env_cfg import DeeproboticsLightHWFlatStraightHighSpeedEnvCfg
from .progressive_commands import ProgressiveLightHWVelocityCommandCfg
from .progressive_rewards import track_lin_vel_xy_upright_exp_l1


@configclass
class LightHWFlatStraightRecoveryCommandsCfg(CommandsCfg):
    """Fixed +/-1 m/s commands with an explicit 2% zero-command slice."""

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
        stage_start_steps=(0,),
        stage_max_speeds=(1.0,),
        zero_command_probability=0.02,
        minimum_abs_speed=0.20,
        anchor_probability=0.50,
        ranges=ProgressiveLightHWVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class DeeproboticsLightHWFlatStraightRecoveryEnvCfg(
    DeeproboticsLightHWFlatStraightHighSpeedEnvCfg
):
    """Low-speed flat recovery environment; old high-speed tasks are untouched."""

    commands: LightHWFlatStraightRecoveryCommandsCfg = LightHWFlatStraightRecoveryCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.stage_start_steps = (0,)
        command.stage_max_speeds = (1.0,)
        command.zero_command_probability = 0.02
        command.minimum_abs_speed = 0.20
        command.anchor_probability = 0.50
        command.resampling_time_range = (8.0, 8.0)
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.heading_command = False
        command.lateral_range = (0.0, 0.0)
        command.yaw_range = (0.0, 0.0)
        command.ranges.lin_vel_x = (-1.0, 1.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = None

        # Use the less saturated linear-error kernel only in this recovery
        # route; it supplies a usable signal while the transferred policy is
        # still close to rest, without changing the old task's reward.
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_upright_exp_l1
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.75
        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        # Restore the conservative M20-compatible reward scales.  In
        # particular, force_scale=8 and weight=-0.02 avoid the 16x contact
        # variance amplification that destabilized route 06.
        self.rewards.base_height_l2.weight = -3.0
        self.rewards.hipx_joint_pos_penalty.weight = -0.40
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.upward.func = lighthw_upright_orientation_l2
        self.rewards.upward.weight = -0.08
        # The 100-update smoke playback reached the target speed but drifted
        # about 48 degrees in ten seconds; make this signal visible without
        # changing the command or action interface.
        self.rewards.straight_yaw_rate_l2.weight = -0.50
        self.rewards.wheel_contact_support.func = lighthw_wheel_contact_support
        # The same smoke playback kept all four wheels in forward motion but
        # intermittently unloaded the rear wheel in reverse.  This moderate
        # increase applies in both directions and leaves force variance at the
        # conservative M20 scale.
        self.rewards.wheel_contact_support.weight = 0.55
        self.rewards.wheel_contact_uniformity.func = lighthw_wheel_contact_uniformity
        self.rewards.wheel_contact_uniformity.weight = -0.02
        self.rewards.wheel_contact_uniformity.params["force_scale"] = 8.0

        # Keep the previous user-requested first-stage setting until a stable
        # fixed-command baseline has passed playback acceptance.
        self.terminations.bad_orientation_2 = None

        print(
            "[LightHW Flat Straight Recovery] policy=60, critic=236, action=16; "
            "vx=(-1.0, 1.0), vy=0, wz=0, zero=2%, wheel_clip=30"
        )
