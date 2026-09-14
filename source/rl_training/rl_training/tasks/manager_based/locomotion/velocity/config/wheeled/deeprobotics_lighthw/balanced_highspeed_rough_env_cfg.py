"""LightHW high-speed route with an explicit zero-command damping gate.

This route keeps the M20 velocity-tracking skeleton, but separates static leg
velocity suppression from the moving command loss so the 3.5 m/s channel is not
penalized by a reward intended only for standing.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

from .adapted_rewards import lighthw_stand_leg_velocity_l2
from .hierarchical_commands import HierarchicalLightHWVelocityCommandCfg
from .rough_env_cfg import LEG_JOINT_NAMES
from .m20_reward_profile import apply_m20_reward_profile
from .rough_env_cfg import DeeproboticsLightHWRoughEnvCfg


@configclass
class LightHWBalancedHighSpeedCommandsCfg(CommandsCfg):
    """Balanced command mixture: enough standing, without starving 3.5 m/s."""

    base_velocity = HierarchicalLightHWVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(16.0, 16.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        stage_probabilities=(0.25, 0.20, 0.15, 0.40),
        lateral_range=(-0.05, 0.05),
        yaw_range=(-0.10, 0.10),
        low_speed_range=(-0.80, 0.80),
        mid_speed_range=(-1.80, 1.80),
        high_speed_min=2.50,
        high_speed_max=3.50,
        high_speed_anchor_ratio=0.50,
        ranges=HierarchicalLightHWVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.5, 3.5),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.10, 0.10),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class DeeproboticsLightHWBalancedHighSpeedRoughEnvCfg(DeeproboticsLightHWRoughEnvCfg):
    """Rough high-speed baseline with measurable zero-command quietness."""

    commands: LightHWBalancedHighSpeedCommandsCfg = LightHWBalancedHighSpeedCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.stage_probabilities = (0.25, 0.20, 0.15, 0.40)
        command.resampling_time_range = (16.0, 16.0)
        command.lateral_range = (-0.05, 0.05)
        command.yaw_range = (-0.10, 0.10)
        command.high_speed_min = 2.50
        command.high_speed_max = 3.50
        command.high_speed_anchor_ratio = 0.50

        # Keep the route-local terrain envelope fixed while the policy learns
        # the mixed command problem; high-speed rough recovery remains present.
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        # The parent route keeps M20-style tracking and posture roles.  These
        # terms are the LightHW-specific correction: only zero command gates
        # them, so the 3.5 m/s branch retains its leg recovery bandwidth.
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-100.0)
        # The parent calls disable_zero_weight_rewards(); recreate terms that
        # were intentionally zero in the generic M20-style reward schema.
        self.rewards.stand_lin_vel_xy_l2 = RewTerm(
            func=mdp.stand_lin_vel_xy_l2,
            weight=-0.8,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_ang_vel_xy_l2 = RewTerm(
            func=mdp.stand_ang_vel_xy_l2,
            weight=-0.3,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_flat_orientation_l2 = RewTerm(
            func=mdp.stand_flat_orientation_l2,
            weight=-0.35,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_wheel_velocity_l2.weight = -0.18
        self.rewards.stand_wheel_velocity_l2.params["command_threshold"] = 0.15
        self.rewards.stand_leg_velocity_l2 = RewTerm(
            func=lighthw_stand_leg_velocity_l2,
            weight=-0.01,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            },
        )

        # Keep high-speed actuator authority, but retain the additional joint
        # damping from the previous 3.5 m/s route to reduce contact limit cycles.
        self.scene.robot.actuators["joint"].damping = 3.0

        # Finalize this route with the same M20 reward profile as the base
        # LightHW route; command and actuator settings remain route-specific.
        apply_m20_reward_profile(self)


@configclass
class DeeproboticsLightHWBalancedHighSpeedFlatEnvCfg(DeeproboticsLightHWBalancedHighSpeedRoughEnvCfg):
    """Flat alias for quick smoke tests; the training route remains rough."""

    pass
