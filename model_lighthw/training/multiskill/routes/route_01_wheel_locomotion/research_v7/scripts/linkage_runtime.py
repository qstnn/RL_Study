"""Research-only A-frame replacement for the V6 single spring drawbar.

Imported after Isaac Sim starts. This module leaves the V6 source untouched;
``worker_a_frame.py`` selects it for one isolated physical comparison.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply

from linkage_variant import a_frame_forces


def current_linkage_state(env) -> dict[str, torch.Tensor]:
    """Read both bodies at the current physics substep, including point speeds."""
    robot, cart = env.scene["robot"], env.scene["trailer"]
    robot_point = quat_apply(robot.data.root_quat_w, env.r1_robot_hitch) + robot.data.root_pos_w
    cart_center = quat_apply(cart.data.root_quat_w, env.r1_cart_hitch) + cart.data.root_pos_w
    robot_com = robot.data.body_com_pos_w[:, env.r1_base_id]
    cart_com = cart.data.body_com_pos_w[:, env.r1_cart_base_id]
    robot_arm = robot_point - robot_com
    cart_arm = cart_center - cart_com
    robot_point_velocity = robot.data.root_com_lin_vel_w + torch.cross(
        robot.data.root_ang_vel_w, robot_arm, dim=-1
    )
    cart_center_velocity = cart.data.root_com_lin_vel_w + torch.cross(
        cart.data.root_ang_vel_w, cart_arm, dim=-1
    )
    if not hasattr(env, "r1_a_frame_local_side_axis"):
        env.r1_a_frame_local_side_axis = torch.tensor(
            [0.0, 1.0, 0.0], device=env.device, dtype=cart.data.root_pos_w.dtype
        ).repeat(env.num_envs, 1)
    side_axis = quat_apply(cart.data.root_quat_w, env.r1_a_frame_local_side_axis)
    result = a_frame_forces(
        robot_point, cart_center, robot_point_velocity, cart_center_velocity,
        cart.data.root_ang_vel_w, side_axis, robot_com, cart_com,
        env.r1_towed_kg > 0,
    )
    normal = env.r1_normal
    origin = env.scene.env_origins
    robot_clearance = ((robot_point - origin) * normal).sum(-1)
    cart_clearance = ((result["cart_points_w"] - origin[:, None, :]) * normal[:, None, :]).sum(-1)
    result["tip_clearance_m"] = torch.minimum(robot_clearance, cart_clearance.min(-1).values)
    result["ground_guard"] = (env.r1_towed_kg > 0) & (result["tip_clearance_m"] < 0.005)
    return result


def apply_a_frame_cart_forces(env) -> None:
    """Replace only the drawbar; retain original gravity and passive roll drag."""
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.cart_asset import RADIUS

    robot, cart = env.scene["robot"], env.scene["trailer"]
    active = env.r1_towed_kg > 0
    result = current_linkage_state(env)
    robot.set_external_force_and_torque(
        result["robot_force_w"][:, None, :], result["robot_torque_w"][:, None, :],
        body_ids=[env.r1_base_id], is_global=True,
    )
    # Identical 40 kg per-link gravity and four passive wheel rolling torques
    # to the original V6 implementation. No hidden cart or robot propulsion.
    forces = torch.zeros((env.num_envs, cart.num_bodies, 3), device=env.device)
    forces[:, :, 2] = -9.81 * env.r1_cart_masses * active[:, None]
    forces[:, env.r1_cart_base_id] += result["cart_force_w"]
    torques = torch.zeros_like(forces)
    torques[:, env.r1_cart_base_id] = result["cart_torque_w"]
    sensor = env.scene["trailer_contacts"]
    normals = (sensor.data.net_forces_w[:, env.r1_cart_sensor_ids] * env.r1_normal[:, None, :]).sum(-1).clamp_min(0.)
    speed = cart.data.joint_vel[:, env.r1_cart_joint_ids]
    resistance = -env.cfg.rolling_coefficient * normals * RADIUS * torch.tanh(speed / 0.2) * active[:, None]
    axis = quat_apply(cart.data.root_quat_w, env.r1_cart_axis)
    torques[:, env.r1_cart_wheel_ids] += resistance[:, :, None] * axis[:, None, :]
    cart.set_external_force_and_torque(forces, torques, is_global=True)
    env.r1_drawbar_length = result["center_length_m"]
    env.r1_drawbar_force = result["equivalent_tension_n"]
    env.r1_drawbar_guard |= result["guard"] | result["ground_guard"]
    env.r1_drawbar_peak = torch.maximum(env.r1_drawbar_peak, result["raw_tensions_n"].abs().sum(-1))
    env.r1_rolling_torque = resistance
    env.r1_a_frame_last = result
