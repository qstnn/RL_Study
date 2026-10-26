"""Dissipative, chassis-reacted one-way clutch for an optional physical trailer variant."""

from __future__ import annotations

import torch


def passive_reverse_brake(joint_speed_rad_s: torch.Tensor, torque_limit_Nm: float,
                          speed_scale_rad_s: float = 0.03) -> torch.Tensor:
    """Positive wheel torque only while the passive wheel rotates downhill.

    The wheel is allowed to spin uphill without brake torque. At negative joint
    speed, torque * speed is nonpositive. This is a brake, never a drive target.
    """
    if torque_limit_Nm <= 0 or speed_scale_rad_s <= 0:
        raise ValueError("Brake torque limit and speed scale must be positive")
    reverse_speed = (-joint_speed_rad_s).clamp_min(0.0)
    torque = torque_limit_Nm * torch.tanh(reverse_speed / speed_scale_rad_s)
    if bool((torque * joint_speed_rad_s > 1e-7).any()):
        raise RuntimeError("Reverse brake produced positive joint power")
    if bool(((joint_speed_rad_s >= 0.0) & (torque != 0.0)).any()):
        raise RuntimeError("Reverse brake acted during uphill wheel rotation")
    return torque


def install_reverse_brake(cart_module, torque_limit_Nm: float,
                          speed_scale_rad_s: float = 0.03) -> None:
    """Add equal/opposite wheel and chassis torques to V6's passive 1 kHz forces.

    Isaac Lab stores world-frame external wrenches in articulation buffers after
    V6's original force function. Applying the axle reaction on cart_base is
    necessary: a wheel-only torque would be an unphysical external motor.
    """
    from isaaclab.utils.math import quat_apply

    original_apply = cart_module.apply_cart_forces

    def apply_with_brake(env):
        original_apply(env)
        cart = env.scene["trailer"]
        if not cart._use_global_wrench_frame:
            raise RuntimeError("Expected V6 cart external wrenches in world frame")
        speed = cart.data.joint_vel[:, env.r1_cart_joint_ids]
        active = env.r1_towed_kg > 0
        torque = passive_reverse_brake(speed, torque_limit_Nm,
                                       speed_scale_rad_s) * active[:, None]
        axis = quat_apply(cart.data.root_quat_w, env.r1_cart_axis)
        wheel_torque_w = torque[:, :, None] * axis[:, None, :]
        # Apply the wheel torque and its equal, opposite axle reaction on the
        # chassis. These additions preserve V6's gravity, drawbar and rolling
        # buffers; the 1 kHz PhysX step consumes them together.
        body_reaction_w = -wheel_torque_w.sum(1)
        if not torch.allclose(wheel_torque_w.sum(1) + body_reaction_w,
                              torch.zeros_like(body_reaction_w), atol=1e-6, rtol=0):
            raise RuntimeError("Passive clutch has nonzero net axle torque")
        cart._external_torque_b[:, env.r1_cart_wheel_ids] += wheel_torque_w
        cart._external_torque_b[:, env.r1_cart_base_id] += body_reaction_w
        env.r1_reverse_brake_torque_Nm = torque
        env.r1_reverse_brake_joint_power_W = torque * speed
        env.r1_reverse_brake_joint_velocity_pre_rad_s = speed.clone()
        env.r1_reverse_brake_body_reaction_w_Nm = body_reaction_w
        if bool((env.r1_reverse_brake_joint_power_W > 1e-7).any()):
            raise RuntimeError("Passive clutch supplied energy")

    cart_module.apply_cart_forces = apply_with_brake
