"""Flat straight-line posture recovery route for LightHW.

Route 07 tracked +/-1 m/s well, but its final playback exposed three coupled
failures: the moving body crouched below the 0.416 m target, the zero-command
body rolled to one side, and forward motion accumulated yaw drift.  This
route keeps the same URDF, 60/236 observations and 16 actions, then gives the
policy a lower-speed posture lesson before any high-speed transfer.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .adapted_rewards import (
    lighthw_joint_pair_symmetry_l2,
    lighthw_straight_yaw_rate_l2,
    lighthw_upright_orientation_l2,
    lighthw_wheel_contact_support,
    lighthw_wheel_contact_uniformity,
)
from .flat_straight_recovery_env_cfg import (
    DeeproboticsLightHWFlatStraightRecoveryEnvCfg,
    LightHWFlatStraightRecoveryCommandsCfg,
)
from .progressive_commands import ProgressiveLightHWVelocityCommandCfg
from .rough_env_cfg import FOOT_LINK_PATTERN, LIGHTHW_SYMMETRY_PAIRS


@configclass
class LightHWFlatStraightPostureCommandsCfg(LightHWFlatStraightRecoveryCommandsCfg):
    """A low-speed lesson with enough zero-command data to learn equilibrium."""

    base_velocity = ProgressiveLightHWVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        lateral_range=(0.0, 0.0),
        yaw_range=(0.0, 0.0),
        stage_start_steps=(0,),
        stage_max_speeds=(0.60,),
        zero_command_probability=0.15,
        minimum_abs_speed=0.15,
        anchor_probability=0.70,
        ranges=ProgressiveLightHWVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.60, 0.60),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class DeeproboticsLightHWFlatStraightPostureEnvCfg(
    DeeproboticsLightHWFlatStraightRecoveryEnvCfg
):
    """Warm-start route that prioritizes posture and symmetry at +/-0.6 m/s."""

    commands: LightHWFlatStraightPostureCommandsCfg = LightHWFlatStraightPostureCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.stage_start_steps = (0,)
        command.stage_max_speeds = (0.60,)
        command.zero_command_probability = 0.15
        command.minimum_abs_speed = 0.15
        command.anchor_probability = 0.70
        command.resampling_time_range = (6.0, 6.0)
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.heading_command = False
        command.lateral_range = (0.0, 0.0)
        command.yaw_range = (0.0, 0.0)
        command.ranges.lin_vel_x = (-0.60, 0.60)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = None

        # The old policy reached +/-1 m/s, but its moving height averaged
        # 0.33 m while the target is 0.416 m.  Increase this term only in the
        # posture route; route 07 keeps the original checkpoint semantics.
        self.rewards.base_height_l2.weight = -8.0

        # Route 07's weighted joint-limit term stayed at -0.20, which is the
        # exact cost of one LightHW hip crossing the 0.9 soft limit.  Keep the
        # new action target below that limit and make the remaining violation
        # expensive enough to move the policy away from the hard +0.4 rad end.
        self.actions.joint_pos.clip[".*_hip_joint"] = (-0.32, 0.32)
        self.rewards.joint_pos_limits.weight = -12.0
        self.rewards.hipx_joint_pos_penalty.weight = -0.80

        # M20's generic mirror term is active, but its raw scale is too small
        # to correct the observed FL/RL load asymmetry.  Add the normalized
        # LightHW pair term so the narrow +/-0.4 rad hip span is not drowned
        # out by thigh/calf ranges.
        self.rewards.joint_pair_symmetry_l2 = RewTerm(
            func=lighthw_joint_pair_symmetry_l2,
            weight=-0.12,
            params={
                "joint_pairs": LIGHTHW_SYMMETRY_PAIRS,
                "asset_cfg": SceneEntityCfg("robot"),
                "use_default_offset": True,
            },
        )

        # Give the zero-command slice enough weight to teach a level,
        # four-wheel-supported equilibrium before speed is widened again.
        self.rewards.stand_lin_vel_xy_l2.weight = -1.0
        self.rewards.stand_ang_vel_xy_l2.weight = -0.15
        self.rewards.stand_flat_orientation_l2.weight = -1.0
        self.rewards.stand_contact_force_uniformity.weight = -0.05

        # Forward playback still turned about 25 degrees in its final ten
        # seconds.  Yaw-rate is observable by the policy, so strengthen this
        # local signal while keeping the command yaw exactly zero.
        self.rewards.straight_yaw_rate_l2 = RewTerm(
            func=lighthw_straight_yaw_rate_l2,
            weight=-1.50,
            params={
                "command_name": "base_velocity",
                "min_abs_lin_vel_x": 0.10,
                "max_abs_lin_vel_y": 0.05,
                "max_abs_ang_vel_z": 0.05,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.wheel_contact_support.func = lighthw_wheel_contact_support
        self.rewards.wheel_contact_support.weight = 0.60
        self.rewards.wheel_contact_uniformity.func = lighthw_wheel_contact_uniformity
        self.rewards.wheel_contact_uniformity.weight = -0.04
        self.rewards.wheel_contact_uniformity.params["force_scale"] = 8.0

        # Keep the user-requested termination policy for this corrective run;
        # a later validated route can turn bad_orientation_2 back on.
        self.terminations.bad_orientation_2 = None

        print(
            "[LightHW Flat Straight Posture] policy=60, critic=236, action=16; "
            "vx=(-0.6, 0.6), vy=0, wz=0, zero=15%, hip_clip=+/-0.32"
        )
