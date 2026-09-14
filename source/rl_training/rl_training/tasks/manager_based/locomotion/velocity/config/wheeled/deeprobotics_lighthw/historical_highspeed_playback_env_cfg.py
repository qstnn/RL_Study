"""Playback-only environment matching the 2026-09-08 high-speed baseline snapshot."""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLightHWRoughEnvCfg


@configclass
class DeeproboticsLightHWHistoricalHighSpeedPlaybackEnvCfg(DeeproboticsLightHWRoughEnvCfg):
    """Reconstruct the environment parameters saved with high-speed model_9500."""

    def __post_init__(self):
        super().__post_init__()

        # Restore the command distribution from params/env.yaml of
        # 2026-09-08_21-23-28; this is intentionally not the current baseline.
        command = self.commands.base_velocity
        command.resampling_time_range = (8.0, 8.0)
        command.rel_standing_envs = 0.05
        command.rel_heading_envs = 1.0
        command.ranges.lin_vel_x = (-3.5, 3.5)
        command.ranges.lin_vel_y = (-0.20, 0.20)
        command.ranges.ang_vel_z = (-0.40, 0.40)
        command.ranges.heading = (-math.pi, math.pi)

        # Restore reset uncertainty from the saved run.  The playback wrapper
        # is expected to disable pushes and external-force randomization; the
        # wrapper must target the actual ``randomize_push_robot`` event name.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-math.pi, math.pi),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.10, 0.10),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.05, 0.05),
        }

        # Restore the reward weights that were actually used by this model.
        self.rewards.is_terminated = None
        # These terms were added after the saved high-speed run.  Leaving them
        # alive with an empty sensor regex makes Isaac Lab reject the config.
        self.rewards.stand_lin_vel_xy_l2 = None
        self.rewards.stand_ang_vel_xy_l2 = None
        self.rewards.stand_flat_orientation_l2 = None
        self.rewards.stand_contact_force_uniformity = None
        self.rewards.stand_rear_contact_force_balance_l2 = None
        self.rewards.stand_leg_velocity_l2 = None
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.04
        self.rewards.flat_orientation_l2.weight = -0.12
        self.rewards.base_height_l2.weight = -0.45
        self.rewards.base_height_l2.params["target_height"] = 0.41
        self.rewards.body_lin_acc_l2 = None
        self.rewards.joint_torques_l2.weight = -1.2e-5
        self.rewards.joint_vel_l2 = None
        self.rewards.wheel_vel_penalty = None
        self.rewards.joint_acc_l2.weight = -8.0e-8
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_power.weight = -8.0e-6
        self.rewards.stand_still_without_cmd.weight = -1.2
        self.rewards.stand_still_without_cmd.params["command_threshold"] = 0.12
        for term in (self.rewards.hipx_joint_pos_penalty, self.rewards.knee_joint_pos_penalty, self.rewards.joint_pos_penalty):
            term.params["stand_still_scale"] = 4.0
            term.params["velocity_threshold"] = 0.35
            term.params["command_threshold"] = 0.12
        self.rewards.action_rate_l2.weight = -0.045
        self.rewards.undesired_contacts.weight = -0.9
        self.rewards.contact_forces.weight = -1.2e-4
        self.rewards.contact_forces.params["threshold"] = 40.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.4
        self.rewards.track_ang_vel_z_exp.weight = 0.85
        self.rewards.feet_contact_without_cmd.weight = 0.02
        self.rewards.upward.weight = 0.06
        self.rewards.joint_torques_wheel_l2.weight = -1.0e-5
        self.rewards.joint_acc_wheel_l2.weight = -1.5e-7
        self.rewards.joint_power_wheel_l1.weight = -8.0e-6
        self.rewards.stand_wheel_velocity_l2.weight = -0.04
        self.rewards.stand_wheel_velocity_l2.params["command_threshold"] = 0.12
        self.rewards.joint_pair_symmetry_l2.weight = -0.015

        # The snapshot predates the current route's extra zero-weight terms;
        # remove any remaining zero-weight placeholders with empty selectors.
        self.disable_zero_weight_rewards()
