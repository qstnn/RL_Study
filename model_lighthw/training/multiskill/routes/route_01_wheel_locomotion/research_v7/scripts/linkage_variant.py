"""研究用无动力 A 形双杆牵引；不修改 R1 原牵引杆或训练环境。

机器人端仍为一个球铰，拖车前缘左右各一个球铰。两根等长弹簧阻尼杆
在坡向的小位移刚度/阻尼上匹配现有单杆；这是一种待验证的新硬件几何，
不能凭静力计算推断 1 m 拖行成功。
"""

from __future__ import annotations

import math

import torch


REST_LONGITUDINAL_M = 1.0
HALF_WIDTH_M = 0.20
MATCHED_LONGITUDINAL_STIFFNESS_NPM = 2000.0
MATCHED_LONGITUDINAL_DAMPING_NSPM = 150.0
FORCE_GUARD_N = 1500.0


def matched_rod_parameters(
    longitudinal_rest_m: float = REST_LONGITUDINAL_M,
    half_width_m: float = HALF_WIDTH_M,
    longitudinal_stiffness_npm: float = MATCHED_LONGITUDINAL_STIFFNESS_NPM,
    longitudinal_damping_nspm: float = MATCHED_LONGITUDINAL_DAMPING_NSPM,
) -> tuple[float, float, float]:
    """Return each rod's rest length, axial stiffness and damping.

    At a straight, unloaded rest pose, ``2*k_rod*(L/rest_rod)**2`` equals
    the original single drawbar's longitudinal stiffness; likewise damping.
    """
    values = (longitudinal_rest_m, half_width_m, longitudinal_stiffness_npm, longitudinal_damping_nspm)
    if not all(math.isfinite(x) for x in values) or longitudinal_rest_m <= 0 or half_width_m <= 0 or longitudinal_stiffness_npm <= 0 or longitudinal_damping_nspm < 0:
        raise ValueError("A-frame dimensions, stiffness and damping must be finite and physical")
    rest = math.hypot(longitudinal_rest_m, half_width_m)
    multiplier = (rest / longitudinal_rest_m) ** 2 / 2
    return rest, longitudinal_stiffness_npm * multiplier, longitudinal_damping_nspm * multiplier


def a_frame_forces(
    robot_point_w: torch.Tensor,
    cart_center_point_w: torch.Tensor,
    robot_point_velocity_w: torch.Tensor,
    cart_center_velocity_w: torch.Tensor,
    cart_angular_velocity_w: torch.Tensor,
    cart_side_axis_w: torch.Tensor,
    robot_com_w: torch.Tensor,
    cart_com_w: torch.Tensor,
    active: torch.Tensor,
    *,
    longitudinal_rest_m: float = REST_LONGITUDINAL_M,
    half_width_m: float = HALF_WIDTH_M,
    longitudinal_stiffness_npm: float = MATCHED_LONGITUDINAL_STIFFNESS_NPM,
    longitudinal_damping_nspm: float = MATCHED_LONGITUDINAL_DAMPING_NSPM,
) -> dict[str, torch.Tensor]:
    """Compute equal/opposite body forces and torques for two passive rods.

    World-point velocities must already include each articulation's angular
    velocity at its center hitch. The cart side axis must be the cart body's
    unit local +Y axis rotated into world coordinates. Torques are about each
    body's COM; they are ready for ``set_external_force_and_torque``.

    Inputs have shape ``(N,3)`` except ``active`` with shape ``(N,)``. Rods
    support tension and compression like the original spring; no commanded
    speed, drive force, or terrain information enters this function.
    """
    vectors = (
        robot_point_w, cart_center_point_w, robot_point_velocity_w,
        cart_center_velocity_w, cart_angular_velocity_w, cart_side_axis_w,
        robot_com_w, cart_com_w,
    )
    n = robot_point_w.shape[0]
    if robot_point_w.shape != (n, 3) or any(x.shape != (n, 3) for x in vectors) or active.shape != (n,):
        raise ValueError("A-frame inputs must have shapes (N,3) and active (N,)")
    rest, stiffness, damping = matched_rod_parameters(
        longitudinal_rest_m, half_width_m, longitudinal_stiffness_npm, longitudinal_damping_nspm
    )
    side = cart_side_axis_w / cart_side_axis_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    offsets = half_width_m * torch.stack((side, -side), dim=1)
    cart_points_w = cart_center_point_w[:, None, :] + offsets
    cart_point_velocities_w = cart_center_velocity_w[:, None, :] + torch.cross(
        cart_angular_velocity_w[:, None, :].expand_as(offsets), offsets, dim=-1
    )
    delta = robot_point_w[:, None, :] - cart_points_w
    lengths = delta.norm(dim=-1).clamp_min(1e-6)
    directions = delta / lengths[..., None]
    length_rates = ((robot_point_velocity_w[:, None, :] - cart_point_velocities_w) * directions).sum(-1)
    raw_tensions = (stiffness * (lengths - rest) + damping * length_rates) * active[:, None]
    # Conservative global cap: sum of both axial magnitudes never exceeds
    # the original 1500 N numerical guard. A hit must fail behavioral review.
    axial_sum = raw_tensions.abs().sum(-1)
    scale = (FORCE_GUARD_N / axial_sum.clamp_min(1e-9)).clamp(max=1.0)
    cart_rod_forces_w = raw_tensions[..., None] * directions * scale[:, None, None]
    cart_force_w = cart_rod_forces_w.sum(1)
    robot_force_w = -cart_force_w
    center_delta = robot_point_w - cart_center_point_w
    center_length = center_delta.norm(dim=-1).clamp_min(1e-6)
    equivalent_tension = (cart_force_w * center_delta / center_length[:, None]).sum(-1)
    robot_torque_w = torch.cross(robot_point_w - robot_com_w, robot_force_w, dim=-1)
    cart_torque_w = torch.cross(
        cart_points_w - cart_com_w[:, None, :], cart_rod_forces_w, dim=-1
    ).sum(1)
    spring_energy_j = 0.5 * stiffness * (((lengths - rest) ** 2) * active[:, None]).sum(-1)
    damping_dissipation_w = damping * ((length_rates ** 2) * active[:, None]).sum(-1)
    spring_energy_rate_w = (stiffness * (lengths - rest) * length_rates * active[:, None]).sum(-1)
    mechanical_power_w = (
        (robot_force_w * robot_point_velocity_w).sum(-1)
        + (cart_rod_forces_w * cart_point_velocities_w).sum((1, 2))
    )
    return {
        "robot_force_w": robot_force_w,
        "robot_torque_w": robot_torque_w,
        "cart_force_w": cart_force_w,
        "cart_torque_w": cart_torque_w,
        "cart_rod_forces_w": cart_rod_forces_w,
        "robot_point_w": robot_point_w,
        "cart_center_point_w": cart_center_point_w,
        "cart_points_w": cart_points_w,
        "cart_point_velocities_w": cart_point_velocities_w,
        "lengths_m": lengths,
        "center_length_m": center_length,
        "equivalent_tension_n": equivalent_tension,
        "length_rates_mps": length_rates,
        "raw_tensions_n": raw_tensions,
        "spring_energy_j": spring_energy_j,
        "spring_energy_rate_w": spring_energy_rate_w,
        "damping_dissipation_w": damping_dissipation_w,
        "mechanical_power_w": mechanical_power_w,
        "guard": axial_sum >= FORCE_GUARD_N,
    }
