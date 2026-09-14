"""M20 reward profile applied to the LightHW morphology.

The weights below are copied from ``deeprobotics_m20/rough_env_cfg.py``.
Joint groups and body names are adapted to LightHW's three leg joints plus one
wheel joint per leg.  Terms whose physical units depend on the URDF limits are
normalized using the M20/LightHW effort and velocity ratios; the base-height
target and contact threshold follow LightHW's measured default pose and mass.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .adapted_rewards import (
    lighthw_action_rate_l2,
    lighthw_base_height_l2,
    lighthw_contact_force_excess,
    lighthw_joint_acc_l2,
    lighthw_joint_power_l1,
    lighthw_joint_torques_l2,
)

# URDF effort/velocity limits used to preserve comparable *normalized* costs.
# Isaac actuator configuration is the runtime authority; the M20 URDF XML
# advertises 260/160 Nm, while its loaded ArticulationCfg uses 76.4/21.6 Nm.
_M20_LEG_EFFORT = 76.4
_LIGHTHW_LEG_EFFORT = 48.0
_M20_LEG_VELOCITY = 22.4
_LIGHTHW_LEG_VELOCITY = 12.57
_M20_WHEEL_EFFORT = 21.6
_LIGHTHW_WHEEL_EFFORT = 24.0
_M20_WHEEL_VELOCITY = 79.3
_LIGHTHW_WHEEL_VELOCITY = 37.7
_LEG_TORQUE_NORM = (_M20_LEG_EFFORT / _LIGHTHW_LEG_EFFORT) ** 2
_LEG_POWER_NORM = (_M20_LEG_EFFORT * _M20_LEG_VELOCITY) / (
    _LIGHTHW_LEG_EFFORT * _LIGHTHW_LEG_VELOCITY
)
_LEG_ACC_NORM = (_M20_LEG_VELOCITY / _LIGHTHW_LEG_VELOCITY) ** 2
_WHEEL_ACC_NORM = (_M20_WHEEL_VELOCITY / _LIGHTHW_WHEEL_VELOCITY) ** 2


def apply_m20_reward_profile(cfg) -> None:
    """Apply the effective M20 reward terms to a LightHW environment config.

    目的：让 LightHW 与 M20 使用同一套奖励角色和权重，便于训练曲线和消融
    结果直接比较。激活条件：环境配置 ``__post_init__`` 的最后阶段；返回值：
    原地修改 ``cfg.rewards``，函数本身返回 ``None``。腿部关节按 hip/thigh/calf
    映射；轮关节不计入腿部姿态项，但针对 LightHW 的高轮惯量保留独立小权重阻尼。
    """

    rewards = cfg.rewards
    leg = cfg.leg_joint_names
    wheels = cfg.wheel_joint_names
    hips = cfg.hip_joint_names
    thighs = cfg.thigh_joint_names
    calves = cfg.calf_joint_names
    base = cfg.base_link_name
    feet = cfg.foot_link_name

    # Root terms: exact M20 values, with LightHW's base link and scanner.
    rewards.is_terminated.weight = 0.0
    rewards.lin_vel_z_l2.weight = -2.0
    # LightHW base rotational inertias are only about 22--24% of M20's, so
    # identical disturbances create much larger roll/pitch rates. Increase
    # this damping term conservatively while leaving M20's other root roles.
    rewards.ang_vel_xy_l2.weight = -0.04
    rewards.flat_orientation_l2.weight = 0.0
    # Use the LightHW adapter: its per-environment ray fallback avoids turning
    # one invalid height ray into a zero-height penalty for the whole batch.
    rewards.base_height_l2.func = lighthw_base_height_l2
    rewards.base_height_l2.weight = -0.5
    rewards.base_height_l2.params = {
        # LightHW's canonical default pose puts the chassis around 0.416 m;
        # retaining M20's 0.40 m target would create a constant crouch bias.
        "target_height": 0.416,
        "asset_cfg": SceneEntityCfg("robot", body_names=[base]),
        "sensor_cfg": SceneEntityCfg("height_scanner_base"),
    }
    rewards.body_lin_acc_l2.weight = 0.0

    # Joint terms: M20 weights, mapped to LightHW's leg groups.
    rewards.joint_torques_l2.func = lighthw_joint_torques_l2
    rewards.joint_torques_l2.weight = -2.5e-5
    rewards.joint_torques_l2.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=leg),
        "normalization": _LEG_TORQUE_NORM,
    }
    rewards.joint_acc_l2.func = lighthw_joint_acc_l2
    rewards.joint_acc_l2.weight = -2.0e-7
    rewards.joint_acc_l2.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=leg),
        "normalization": _LEG_ACC_NORM,
    }
    rewards.joint_power.func = lighthw_joint_power_l1
    rewards.joint_power.weight = -2.0e-5
    rewards.joint_power.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=leg),
        "normalization": _LEG_POWER_NORM,
    }
    rewards.joint_pos_limits.weight = -5.0
    rewards.joint_pos_limits.params["asset_cfg"].joint_names = leg
    rewards.joint_vel_limits.weight = 0.0
    rewards.joint_vel_limits.params["asset_cfg"].joint_names = wheels
    # M20 keeps a small wheel acceleration penalty in addition to leg
    # acceleration; preserve that exact term for LightHW's wheel actuators.
    if hasattr(rewards, "joint_acc_wheel_l2"):
        rewards.joint_acc_wheel_l2.func = lighthw_joint_acc_l2
        rewards.joint_acc_wheel_l2.weight = -1.0e-7
        rewards.joint_acc_wheel_l2.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=wheels),
            "normalization": _WHEEL_ACC_NORM,
        }
    # LightHW wheels are about 2.7x heavier than M20 wheels and have higher
    # reflected inertia. Add small wheel torque/power costs to prevent bursts.
    if hasattr(rewards, "joint_torques_wheel_l2"):
        rewards.joint_torques_wheel_l2.func = lighthw_joint_torques_l2
        rewards.joint_torques_wheel_l2.weight = -2.0e-6
        rewards.joint_torques_wheel_l2.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=wheels),
            "normalization": (_M20_WHEEL_EFFORT / _LIGHTHW_WHEEL_EFFORT) ** 2,
        }
    if hasattr(rewards, "joint_power_wheel_l1"):
        rewards.joint_power_wheel_l1.func = lighthw_joint_power_l1
        rewards.joint_power_wheel_l1.weight = -2.0e-6
        rewards.joint_power_wheel_l1.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=wheels),
            "normalization": (_M20_WHEEL_EFFORT * _M20_WHEEL_VELOCITY)
            / (_LIGHTHW_WHEEL_EFFORT * _LIGHTHW_WHEEL_VELOCITY),
        }

    rewards.stand_still.weight = -2.0
    rewards.stand_still.params["command_name"] = "base_velocity"
    rewards.stand_still.params["command_threshold"] = 0.06
    rewards.stand_still.params["asset_cfg"].joint_names = leg
    # Avoid counting the LightHW-only alias in addition to M20's canonical
    # stand-still term when this profile is applied after route tuning.
    if hasattr(rewards, "stand_still_without_cmd"):
        rewards.stand_still_without_cmd.weight = 0.0
    rewards.hipx_joint_pos_penalty.weight = -0.4
    rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = hips
    rewards.hipy_joint_pos_penalty.weight = -0.1
    rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = thighs
    rewards.knee_joint_pos_penalty.weight = -0.1
    rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = calves
    # ``joint_pos_penalty`` would duplicate the mapped thigh term.  Keep the
    # object at zero because legacy subclasses still configure its parameters.
    rewards.joint_pos_penalty.weight = 0.0

    # Action/contact/tracking terms: exact M20 weights and LightHW body names.
    # The mixed position/velocity action vector has a route-local scale.  Keep
    # action-rate evaluation free of the global gait curriculum so this cost
    # remains comparable between flat and rough LightHW runs.
    rewards.action_rate_l2.func = lighthw_action_rate_l2
    rewards.action_rate_l2.weight = -0.01
    rewards.undesired_contacts.weight = -1.0
    rewards.undesired_contacts.params = {
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[f"^(?!.*{feet}).*"],
        ),
        "threshold": 1.0,
    }
    rewards.contact_forces.func = lighthw_contact_force_excess
    rewards.contact_forces.weight = -1.5e-4
    rewards.contact_forces.params = {
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[feet]),
        # LightHW weighs 11.67 kg (about 28.6 N per wheel at rest); M20's
        # 100 N threshold would make this term effectively inactive.
        "threshold": 40.0,
    }
    rewards.track_lin_vel_xy_exp.weight = 2.0
    rewards.track_ang_vel_z_exp.weight = 1.0
    rewards.feet_contact_without_cmd.weight = 0.1
    rewards.feet_contact_without_cmd.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[feet]
    )
    rewards.upward.weight = 0.08
    # LightHW's rear default pose is the sign-mirrored counterpart of the
    # front pose, so use the maintained pair adapter while keeping M20's
    # mirror weight and role.
    rewards.joint_mirror.func = mdp.joint_pair_symmetry_l2
    rewards.joint_mirror.weight = -0.03
    rewards.joint_mirror.params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_pairs": [
            ("FR_hip_joint", "FL_hip_joint", -1.0),
            ("RR_hip_joint", "RL_hip_joint", -1.0),
            ("FR_thigh_joint", "FL_thigh_joint", 1.0),
            ("RR_thigh_joint", "RL_thigh_joint", 1.0),
            ("FR_calf_joint", "FL_calf_joint", 1.0),
            ("RR_calf_joint", "RL_calf_joint", 1.0),
        ],
        "use_default_offset": True,
    }

    # Keep the remaining LightHW-only gates at zero.  Wheel torque/power are
    # retained above because the LightHW wheel is substantially heavier than
    # M20's; the other gates would duplicate posture or standstill penalties.
    for name in (
        "stand_wheel_velocity_l2",
        "stand_leg_velocity_l2",
        "joint_pair_symmetry_l2",
        "stand_lin_vel_xy_l2",
        "stand_ang_vel_xy_l2",
        "stand_flat_orientation_l2",
        "stand_contact_force_uniformity",
        "stand_rear_contact_force_balance_l2",
    ):
        if hasattr(rewards, name):
            term = getattr(rewards, name)
            if term is not None and hasattr(term, "weight"):
                # Keep terms as zero-weight objects so downstream route
                # classes can safely adjust or replace them.
                term.weight = 0.0
