# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Playback-only compatibility environment for the original M20 biped policy."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.skills.mdp as skill_mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20RoughEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg


@configclass
class LegacyM20SkillCommandsCfg(CommandsCfg):
    """Minimal skill-command layout retained for legacy M20 playback."""

    skill = skill_mdp.SkillCommandCfg(
        resampling_time_range=(10.0, 10.0),
        num_skills=5,
        skill_ids=[0],
        cycle_time=2.0,
        target_pos_range=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        support_pairs=[0],
    )


@configclass
class LegacyM20BipedStandPlayEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Recreate the 71-observation M20 skill layout for legacy checkpoints."""

    commands: LegacyM20SkillCommandsCfg = LegacyM20SkillCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 6.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.env_spacing = 6.0
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.scene.robot.actuators["joint"].effort_limit = 260.0
        self.scene.robot.actuators["joint"].stiffness = 110.0
        self.scene.robot.actuators["joint"].damping = 3.5
        self.scene.robot.actuators["wheel"].effort_limit = 160.0

        self.commands.skill.skill_ids = [3]
        self.commands.skill.cycle_time = 4.0
        self.commands.skill.support_pairs = [0]
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Actor: 53 rough-observation values + 15 skill values + 3 target
        # values = 71. The critic drops velocity_commands but retains
        # base_lin_vel, which also gives 71 values.
        self.observations.policy.skill_command = ObsTerm(
            func=skill_mdp.generated_commands,
            params={"command_name": "skill"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.velocity_commands = None
        self.observations.critic.height_scan = None
        self.observations.critic.skill_command = ObsTerm(
            func=skill_mdp.generated_commands,
            params={"command_name": "skill"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.policy.relative_target = ObsTerm(
            func=skill_mdp.relative_target_position_body,
            params={"command_name": "skill", "asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.critic.relative_target = ObsTerm(
            func=skill_mdp.relative_target_position_body,
            params={"command_name": "skill", "asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=1.0,
        )

        # The legacy URDF uses a common negative Y axis for every sagittal
        # joint and wheel. These scales are the old-axis equivalents of the
        # later LightHW biped action mapping.
        self.actions.joint_pos.scale = {
            ".*_hipx_joint": 0.05,
            ".*_hipy_joint": 0.50,
            ".*_knee_joint": 0.55,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.scale = 0.8
        self.actions.joint_vel.clip = {".*": (-3.0, 3.0)}
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_rigid_body_material = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # Rewards are irrelevant during playback and many current reward terms
        # refer to LightHW link names. Removing them keeps this compatibility
        # task isolated from the active training configuration.
        for name in dir(self.rewards):
            reward = getattr(self.rewards, name)
            if reward is not None and hasattr(reward, "weight") and not callable(reward):
                setattr(self.rewards, name, None)

        for name in dir(self.terminations):
            termination = getattr(self.terminations, name)
            if (
                name != "time_out"
                and termination is not None
                and hasattr(termination, "func")
                and not callable(termination)
            ):
                setattr(self.terminations, name, None)

        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None
        self.curriculum.command_levels = None
