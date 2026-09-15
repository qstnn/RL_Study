# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""LightHW velocity environment using the canonical M20 reward profile.

The M20 weights are applied at the end of ``__post_init__`` by
``m20_reward_profile.py``; this file supplies only LightHW morphology,
terrain, action and command settings.
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from .adapted_rewards import (
    lighthw_action_rate_l2,
    lighthw_base_height_l2,
    lighthw_contact_force_excess,
    lighthw_joint_acc_l2,
    lighthw_joint_pair_symmetry_l2,
    lighthw_joint_power_l1,
    lighthw_joint_torques_l2,
    lighthw_stand_wheel_velocity_l2,
    lighthw_stand_leg_velocity_l2,
)
from .asset_cfg import LIGHTHW_ROUTE_CFG
from .m20_reward_profile import apply_m20_reward_profile


BASE_LINK_NAME = "Base_link"
FOOT_LINK_PATTERN = ".*_wheel_link"

# The URDF order is FR, FL, RR, RL; each leg contains hip/thigh/calf/wheel.
# Keeping this order explicit is important for the wheel-free joint-position
# observation and for reproducible action/checkpoint dimensions.
ARTICULATION_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_wheel_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_wheel_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RR_wheel_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RL_wheel_joint",
]
LEG_JOINT_NAMES = [name for name in ARTICULATION_JOINT_NAMES if "wheel" not in name]
WHEEL_JOINT_NAMES = [name for name in ARTICULATION_JOINT_NAMES if "wheel" in name]
HIP_JOINT_NAMES = [name for name in LEG_JOINT_NAMES if "hip" in name]
THIGH_JOINT_NAMES = [name for name in LEG_JOINT_NAMES if "thigh" in name]
CALF_JOINT_NAMES = [name for name in LEG_JOINT_NAMES if "calf" in name]

# URDF axes are all expressed in the same signed coordinate convention.  Hip
# deviations therefore mirror with a minus sign, while thigh/calf deviations
# mirror with a plus sign around their model-specific defaults.
LIGHTHW_SYMMETRY_PAIRS = (
    ("FR_hip_joint", "FL_hip_joint", -1.0),
    ("RR_hip_joint", "RL_hip_joint", -1.0),
    ("FR_thigh_joint", "FL_thigh_joint", 1.0),
    ("RR_thigh_joint", "RL_thigh_joint", 1.0),
    ("FR_calf_joint", "FL_calf_joint", 1.0),
    ("RR_calf_joint", "RL_calf_joint", 1.0),
)


@configclass
class DeeproboticsLightHWActionsCfg(ActionsCfg):
    """Mixed position-leg and velocity-wheel actions."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.16,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=4.0,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class DeeproboticsLightHWRewardsCfg(RewardsCfg):
    """Extra wheel-specific terms used by the mixed action interface."""

    joint_torques_wheel_l2 = RewTerm(
        func=lighthw_joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    joint_acc_wheel_l2 = RewTerm(
        func=lighthw_joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    joint_power_wheel_l1 = RewTerm(
        func=lighthw_joint_power_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    stand_wheel_velocity_l2 = RewTerm(
        func=lighthw_stand_wheel_velocity_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
        },
    )
    stand_leg_velocity_l2 = RewTerm(
        func=lighthw_stand_leg_velocity_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    joint_pair_symmetry_l2 = RewTerm(
        func=lighthw_joint_pair_symmetry_l2,
        weight=0.0,
        params={
            "joint_pairs": LIGHTHW_SYMMETRY_PAIRS,
            "asset_cfg": SceneEntityCfg("robot"),
            "use_default_offset": True,
        },
    )


@configclass
class DeeproboticsLightHWRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Conservative rough-terrain starting point for LightHW."""

    actions: DeeproboticsLightHWActionsCfg = DeeproboticsLightHWActionsCfg()
    rewards: DeeproboticsLightHWRewardsCfg = DeeproboticsLightHWRewardsCfg()

    base_link_name = BASE_LINK_NAME
    foot_link_name = FOOT_LINK_PATTERN
    leg_joint_names = LEG_JOINT_NAMES
    wheel_joint_names = WHEEL_JOINT_NAMES
    hip_joint_names = HIP_JOINT_NAMES
    thigh_joint_names = THIGH_JOINT_NAMES
    calf_joint_names = CALF_JOINT_NAMES
    joint_names = ARTICULATION_JOINT_NAMES

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Scene ------------------------------
        self.scene.robot = LIGHTHW_ROUTE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # The chassis and wheel are smaller than M20; a shorter scan keeps the
        # terrain signal local and avoids spending policy capacity on distant rays.
        self.scene.height_scanner.pattern_cfg.resolution = 0.08
        self.scene.height_scanner.pattern_cfg.size = [1.2, 0.8]
        self.scene.height_scanner_base.pattern_cfg.resolution = 0.04
        self.scene.height_scanner_base.pattern_cfg.size = (0.12, 0.12)
        self.scene.terrain.max_init_terrain_level = 3

        terrain = self.scene.terrain.terrain_generator
        terrain.sub_terrains["pyramid_stairs"].proportion = 0.10
        terrain.sub_terrains["pyramid_stairs_inv"].proportion = 0.10
        terrain.sub_terrains["boxes"].proportion = 0.10
        terrain.sub_terrains["random_rough"].proportion = 0.40
        terrain.sub_terrains["hf_pyramid_slope"].proportion = 0.15
        terrain.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15
        terrain.sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.12)
        terrain.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.03, 0.12)
        terrain.sub_terrains["pyramid_stairs"].step_width = 0.28
        terrain.sub_terrains["pyramid_stairs_inv"].step_width = 0.28
        terrain.sub_terrains["boxes"].grid_height_range = (0.025, 0.10)
        terrain.sub_terrains["boxes"].grid_width = 0.35
        terrain.sub_terrains["random_rough"].noise_range = (0.008, 0.05)
        terrain.sub_terrains["random_rough"].noise_step = 0.01
        terrain.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.25)
        terrain.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.25)

        # --------------------------- Observations --------------------------
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = mdp.joint_pos_rel_without_wheel
            group.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            group.joint_pos.params["asset_cfg"].preserve_order = True
            group.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=self.wheel_joint_names, preserve_order=True
            )
            group.joint_vel.params["asset_cfg"].joint_names = self.joint_names
            group.joint_vel.params["asset_cfg"].preserve_order = True
            group.base_lin_vel.scale = 2.0
            group.base_ang_vel.scale = 0.5
            group.joint_pos.scale = 1.0
            group.joint_vel.scale = 0.05

        # ------------------------------ Actions ----------------------------
        # With raw policy actions near [-1, 1], hip motion stays inside the
        # narrow +/-0.4 rad joint range while still leaving room for balance
        # corrections; thigh/calf retain enough swing for terrain recovery.
        # The final clip is the physical target, not a raw policy clip.
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.08,
            ".*_(thigh|calf)_joint": 0.14,
        }
        self.actions.joint_pos.clip = {
            ".*_hip_joint": (-0.4, 0.4),
            "FL_thigh_joint": (-2.6952, 1.5613),
            "FR_thigh_joint": (-2.6952, 1.5613),
            "RL_thigh_joint": (-2.6952, 1.5613),
            "RR_thigh_joint": (-1.5613, 2.6952),
            ".*_calf_joint": (-2.5, 2.5),
        }
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_pos.preserve_order = True
        # The wheel scale is chosen so a unit policy action can reach the
        # updated 3.5 m/s command target with the 87.5 mm wheel radius.
        # The clip is kept slightly above the nominal 3.5 m/s angular speed
        # target so the policy does not saturate immediately at the boundary.
        self.actions.joint_vel.scale = 40.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-42.0, 42.0)}
        self.actions.joint_vel.joint_names = self.wheel_joint_names
        self.actions.joint_vel.preserve_order = True

        # ------------------------------- Events ----------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                # Keep the next run's initial attitude perturbation moderate:
                # it should learn to settle before handling full rough-terrain
                # disturbances.
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                # A smaller reset impulse gives the zero-command policy a
                # clean chance to learn the equilibrium around rest.
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.03, 0.03),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.4, 0.8)
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-4.0, 4.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-1.0, 1.0)
        self.events.randomize_reset_joints.params["position_range"] = (0.98, 1.02)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.05, 0.05)
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.events.randomize_push_robot.interval_range_s = (12.0, 18.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
        }
        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.5, 1.2]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.45, 1.1]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.3]

        # ------------------------------ Rewards ----------------------------
        # Keep every physical quantity unscaled by the shared gait_level in this
        # first route, so flat and rough runs have comparable reward semantics.
        # M20 reward profile (deeprobotics_m20/rough_env_cfg.py) is the
        # reference profile for this route.  Keep the coefficients identical
        # wherever the LightHW term has the same physical meaning; only the
        # joint/body selections and the wheel-specific terms remain platform
        # specific.  This makes checkpoints and reward curves comparable to
        # the established M20 baseline.
        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.func = lighthw_base_height_l2
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.func = lighthw_joint_torques_l2
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_l2.func = lighthw_joint_acc_l2
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_power.func = lighthw_joint_power_l1
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        # M20 explicitly disables wheel torque/acceleration penalties; keep
        # the LightHW terms declared for compatibility, but disabled in this
        # M20-profile route so wheel rolling is governed by tracking only.
        self.rewards.joint_torques_wheel_l2.weight = 0.0
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.joint_power_wheel_l1.weight = 0.0
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names

        # Legacy setup retained for subclass compatibility; the final M20
        # profile below selects the canonical stand_still term and clears this
        # alias to prevent duplicate zero-command penalties.
        self.rewards.stand_still_without_cmd.weight = -2.0
        self.rewards.stand_still_without_cmd.params = {
            "command_name": "base_velocity",
            "command_threshold": 0.20,
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
        }
        self.rewards.hipx_joint_pos_penalty.weight = -0.4
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hip_joint_names
        self.rewards.hipx_joint_pos_penalty.params["stand_still_scale"] = 6.0
        self.rewards.hipx_joint_pos_penalty.params["velocity_threshold"] = 0.50
        self.rewards.hipx_joint_pos_penalty.params["command_threshold"] = 0.20
        # Keep the inherited M20 ``hipy`` term alive: LightHW maps this role
        # to its thigh joints in ``apply_m20_reward_profile``.  Setting it to
        # ``None`` here would make the final profile application dereference
        # ``None.weight`` during every LightHW config instantiation.
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.calf_joint_names
        self.rewards.knee_joint_pos_penalty.params["stand_still_scale"] = 6.0
        self.rewards.knee_joint_pos_penalty.params["velocity_threshold"] = 0.50
        self.rewards.knee_joint_pos_penalty.params["command_threshold"] = 0.20
        # Reuse the generic term for thigh deviations; unlike the M20 naming,
        # LightHW has no separate hip-y joint.
        self.rewards.joint_pos_penalty.weight = -0.1
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.thigh_joint_names
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 6.0
        self.rewards.joint_pos_penalty.params["velocity_threshold"] = 0.50
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.20
        # M20 leaves wheel velocity damping disabled during motion.  The
        # LightHW stand-only wheel term remains available but disabled by
        # default so rolling is not accidentally penalized.
        self.rewards.stand_wheel_velocity_l2.weight = 0.0
        self.rewards.stand_wheel_velocity_l2.params["command_threshold"] = 0.20
        self.rewards.joint_pair_symmetry_l2.weight = 0.0

        self.rewards.action_rate_l2.func = lighthw_action_rate_l2
        self.rewards.action_rate_l2.weight = -0.01
        # M20 smoothness applies -2e-7 to leg acceleration and -1e-7 to wheel
        # acceleration.  Keep the LightHW adapter functions, but use the
        # reference magnitudes consistently.
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params = {
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Base_link", ".*_hip_link", ".*_thigh_link", ".*_calf_link"],
            ),
            "threshold": 0.5,
        }
        self.rewards.contact_forces.func = lighthw_contact_force_excess
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "threshold": 40.0,
        }

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[self.foot_link_name]
        )
        self.rewards.upward.weight = 0.08

        # Wheels should roll while commanded; M20-style air-time/gait terms
        # would encourage unnecessary wheel unloading on this platform.
        for name in (
            "action_mirror",
            "action_sync",
            "feet_air_time",
            "feet_air_time_lin_xy",
            "feet_air_time_x_neg",
            "feet_air_time_ang_z",
            "feet_air_time_variance",
            "feet_gait",
            "feet_slide",
            "feet_stumble",
            "feet_height",
            "feet_height_body",
            "feet_contact",
            "foot_impact_velocity",
            "phase_foot_trajectory_exp",
            "applied_torque_limits",
        ):
            setattr(self.rewards, name, None)

        # ---------------------------- Terminations -------------------------
        self.terminations.illegal_contact = None
        # Match the original M20 rough route: keep time-out and terrain-bound
        # termination, but do not terminate a rollout solely on orientation.
        self.terminations.bad_orientation_2 = None

        # ---------------------------- Curriculum ---------------------------
        # Keep terrain progression for rough training, but do not silently
        # expand command ranges before the low-speed baseline is understood.
        self.curriculum.command_levels = None

        # ------------------------------ Commands ---------------------------
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)
        # Keep the M20 zero-command ratio and command envelope for a direct
        # reward/PPO comparison.  z is the yaw-rate command in this interface.
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.ranges.lin_vel_x = (-3.5, 3.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.knee_joint_names = self.calf_joint_names

        # Final reward authority: apply the M20 profile after all inherited
        # LightHW tuning so every active LightHW route starts from one schema.
        apply_m20_reward_profile(self)

        if self.__class__.__name__ == "DeeproboticsLightHWRoughEnvCfg":
            self.disable_zero_weight_rewards()
