"""Reward terms whose units are made explicit for the LightHW baseline.

The common velocity task contains useful M20-era terms, but several of them
are multiplied by a global ``gait_level`` curriculum scalar.  That is useful
for a mature rough-terrain curriculum and surprising for a new flat baseline:
the same weight can silently evaluate to zero.  These small adapters retain
the same physical quantities while keeping their scale deterministic across
flat and rough routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lighthw_joint_torques_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalization: float = 1.0
) -> torch.Tensor:
    """Return the squared applied torque of selected joints.

    目的：抑制轻量机身上的无谓大力矩；激活条件：每个仿真控制步；返回值：
    每个环境的 ``sum(tau**2)``，不含奖励权重和时间步长。
    """

    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return float(normalization) * torch.sum(torch.square(torque), dim=1)


def lighthw_joint_acc_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalization: float = 1.0
) -> torch.Tensor:
    """Return squared joint acceleration without a hidden terrain curriculum."""

    asset = env.scene[asset_cfg.name]
    acceleration = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return float(normalization) * torch.sum(torch.square(acceleration), dim=1)


def lighthw_joint_power_l1(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), normalization: float = 1.0
) -> torch.Tensor:
    """Return mechanical power magnitude ``sum(abs(tau * qdot))``.

    轻量机器人不需要照搬 M20 的功率惩罚量级；函数只返回物理量，具体的
    ``Nm * rad/s`` 到奖励的缩放放在环境配置中，便于路线间比较。
    """

    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return float(normalization) * torch.sum(torch.abs(torque * velocity), dim=1)


def lighthw_action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return squared change of the complete mixed action vector."""

    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1)


def _lighthw_selected_raw_actions(
    env: "ManagerBasedRLEnv", action_term_name: str, joint_names: Sequence[str]
) -> torch.Tensor:
    """Select raw policy actions by resolved action-term joint name.

    The policy action order and PhysX articulation order are different for
    LightHW.  Looking up the resolved action-term names keeps this diagnostic
    tied to the configured position-action order instead of a guessed tensor
    index.
    """

    term = env.action_manager.get_term(action_term_name)
    resolved_names = list(getattr(term, "_joint_names", ()))
    selected_ids = [index for index, name in enumerate(resolved_names) if name in set(joint_names)]
    if not selected_ids:
        raise ValueError(
            f"No configured joints from {tuple(joint_names)!r} were found in action term "
            f"{action_term_name!r}: {resolved_names!r}"
        )
    return term.raw_actions[:, selected_ids]


def lighthw_joint_action_saturation_l1(
    env: "ManagerBasedRLEnv",
    action_term_name: str,
    joint_names: Sequence[str],
    raw_limit: float = 3.0,
) -> torch.Tensor:
    """Penalize selected raw joint actions beyond a configured safe margin.

    目的：对超出安全幅度的raw输出给出奖励代价，避免物理目标贴边后raw继续
    增大；激活条件：每个控制步；返回值：所选关节
    ``relu(abs(raw)-raw_limit)`` 的平均值。关节选择由路线配置指定，既可用于
    hip，也可用于thigh/calf；不直接修改目标角。``raw_limit`` 使用策略原始
    动作单位，不含奖励权重。
    """

    raw = _lighthw_selected_raw_actions(env, action_term_name, joint_names)
    return torch.mean(torch.relu(torch.abs(raw) - float(raw_limit)), dim=1)


def lighthw_joint_action_saturation_fraction(
    env: "ManagerBasedRLEnv",
    action_term_name: str,
    joint_names: Sequence[str],
    raw_limit: float = 3.0,
) -> torch.Tensor:
    """Report the fraction of selected raw actions outside the safe margin.

    目的：把策略输出饱和率作为可读的回合指标；激活条件：每个控制步；返回值：
    所选动作中 ``abs(raw) >= raw_limit`` 的比例，范围为 ``[0, 1]``。
    """

    raw = _lighthw_selected_raw_actions(env, action_term_name, joint_names)
    return torch.mean((torch.abs(raw) >= float(raw_limit)).float(), dim=1)


def lighthw_contact_force_excess(
    env: "ManagerBasedRLEnv", threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize wheel contact force above a LightHW-specific threshold.

    目的：允许约 28.6 N/轮的静态载荷，同时抑制小轮半径带来的撞击峰值；
    激活条件：选定轮体的历史最大法向/合力超过 ``threshold``；返回值：
    各轮超额力的和。历史窗口只用于检测峰值，不改变接触传感器本身。
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_norm = torch.linalg.vector_norm(force_history, dim=-1)
    peak_force = torch.amax(force_norm, dim=1)
    excess = torch.clamp(peak_force - float(threshold), min=0.0)
    return torch.sum(excess, dim=1)


def lighthw_wheel_contact_support(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    min_contact_force: float = 5.0,
) -> torch.Tensor:
    """Reward the fraction of LightHW wheels carrying a real normal load.

    目的：直线滚动时保持四轮支撑，避免策略通过抬起后轮或单侧滚动来
    换取速度奖励；激活条件：每个控制步；返回值：四个目标轮中法向力
    不低于 ``min_contact_force`` 的比例，范围为 ``[0, 1]``，不含奖励权重。
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    normal_force = torch.clamp(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2], min=0.0)
    supported = normal_force >= float(min_contact_force)
    return torch.mean(supported.float(), dim=1)


def lighthw_wheel_contact_uniformity(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    min_contact_force: float = 5.0,
    force_scale: float = 8.0,
) -> torch.Tensor:
    """Penalize unequal or unloaded wheel normal forces.

    目的：让四轮载荷分布平滑，减少前后轮长期悬空和单轮过载；激活条件：
    每个控制步；返回值：归一化法向力方差与未加载轮比例之和，数值越小越好。
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    normal_force = torch.clamp(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2], min=0.0)
    mean_force = torch.mean(normal_force, dim=1, keepdim=True)
    variance = torch.mean(torch.square(normal_force - mean_force), dim=1)
    variance = variance / max(float(force_scale) ** 2, 1.0e-6)
    unloaded_fraction = torch.mean((normal_force < float(min_contact_force)).float(), dim=1)
    return variance + unloaded_fraction


def lighthw_upright_orientation_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize both tilt and inversion relative to upright gravity.

    目的：让平地直线策略保持正立，避免旧 ``upward`` 项把倒置状态当成
    低损失；激活条件：每个控制步；返回值：投影重力与 ``[0, 0, -1]``
    的平方距离，正立为零，完全倒置为四。
    """

    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.sum(torch.square(gravity[:, :2]), dim=1) + torch.square(gravity[:, 2] + 1.0)


def lighthw_base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize terrain-relative base height with per-environment fallback.

    目的：约束 Base_link 原点相对地面的高度到配置的 target_height；激活条件：配置了
    ``sensor_cfg`` 时使用当前环境的射线平均高度，否则使用世界 z；返回值：
    ``(base_z - terrain_z - target_height)**2``。无效射线只影响对应环境，
    不会把整批环境的高度目标一起替换掉。
    """

    asset = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]
    if sensor_cfg is None:
        terrain_z = torch.zeros_like(base_z)
    else:
        sensor = env.scene.sensors[sensor_cfg.name]
        ray_z = sensor.data.ray_hits_w[..., 2]
        finite = torch.isfinite(ray_z)
        valid = finite.all(dim=1) & (torch.abs(ray_z) < 1.0e6).all(dim=1)
        safe_ray_z = torch.where(finite, ray_z, torch.zeros_like(ray_z))
        measured = torch.mean(safe_ray_z, dim=1)
        terrain_z = torch.where(valid, measured, torch.zeros_like(measured))
    return torch.square(base_z - terrain_z - float(target_height))


def lighthw_stand_wheel_velocity_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel spin only when the commanded body should be standing.

    轮子在运动命令下必须自由滚动；零命令时才对其速度平方施加约束，避免把
    M20 的轮速惩罚误用成全程滚动阻尼。
    """

    asset = env.scene[asset_cfg.name]
    wheel_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(wheel_velocity), dim=1)
    command = env.command_manager.get_command(command_name)
    stand_gate = torch.linalg.norm(command, dim=1) < float(command_threshold)
    return penalty * stand_gate.float()


def lighthw_stand_leg_velocity_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """在零命令时惩罚腿部关节速度平方。

    目的：抑制 LightHW 静止时围绕默认角的高频腿部摆动；激活条件：
    ``||command|| < command_threshold``；返回值：选定腿关节的
    ``sum(qdot**2)``，不包含奖励权重。运动命令下返回零，避免削弱
    3.2 m/s 所需的腿部姿态修正。
    """

    asset = env.scene[asset_cfg.name]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(joint_velocity), dim=1)
    command = env.command_manager.get_command(command_name)
    stand_gate = torch.linalg.norm(command, dim=1) < float(command_threshold)
    return penalty * stand_gate.float()


def lighthw_straight_yaw_rate_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    min_abs_lin_vel_x: float = 0.15,
    max_abs_lin_vel_y: float = 0.12,
    max_abs_ang_vel_z: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw-rate error during forward or backward straight rolling.

    目的：前进和后退都保持直行，抑制左右轮/执行器不对称造成的累计偏航；
    激活条件：``abs(vx)`` 足够大且侧向、yaw 命令较小；返回值：实际
    yaw 角速度与命令之差的平方，不含奖励权重。
    """

    command = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    straight_gate = (
        (torch.abs(command[:, 0]) > float(min_abs_lin_vel_x))
        & (torch.abs(command[:, 1]) < float(max_abs_lin_vel_y))
        & (torch.abs(command[:, 2]) < float(max_abs_ang_vel_z))
    )
    yaw_error = command[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return torch.square(yaw_error) * straight_gate.float()


def lighthw_joint_pair_symmetry_l2(
    env: "ManagerBasedRLEnv",
    joint_pairs: Sequence[tuple[str, str, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_default_offset: bool = True,
) -> torch.Tensor:
    """Compare mirrored joints after normalizing by their soft-limit span.

    ``sign=-1`` is used for LightHW hip joints because all four URDF hip axes
    point along +X while the left/right mounting locations are mirrored.
    Thigh/calf pairs use ``sign=+1``.  Normalizing by each pair's average
    half-range prevents the narrow +/-0.4 rad hip from being numerically
    drowned out by the much wider thigh/calf limits.
    """

    asset = env.scene[asset_cfg.name]
    normalized_pairs = tuple((str(left), str(right), float(sign)) for left, right, sign in joint_pairs)
    cache_key = (normalized_pairs, bool(use_default_offset))
    if not hasattr(env, "lighthw_symmetry_joint_cache"):
        env.lighthw_symmetry_joint_cache = {}
    if cache_key not in env.lighthw_symmetry_joint_cache:
        resolved = []
        for left_name, right_name, sign in normalized_pairs:
            left_ids = asset.find_joints(left_name, preserve_order=True)[0]
            right_ids = asset.find_joints(right_name, preserve_order=True)[0]
            if len(left_ids) != 1 or len(right_ids) != 1:
                raise ValueError(
                    f"LightHW symmetry pair must resolve to one joint: {left_name}, {right_name}"
                )
            resolved.append((int(left_ids[0]), int(right_ids[0]), sign))
        env.lighthw_symmetry_joint_cache[cache_key] = tuple(resolved)

    penalty = torch.zeros(env.num_envs, device=env.device)
    for left_id, right_id, sign in env.lighthw_symmetry_joint_cache[cache_key]:
        left = asset.data.joint_pos[:, left_id]
        right = asset.data.joint_pos[:, right_id]
        if use_default_offset:
            left = left - asset.data.default_joint_pos[:, left_id]
            right = right - asset.data.default_joint_pos[:, right_id]
        soft_limits = asset.data.soft_joint_pos_limits
        left_half_range = 0.5 * (soft_limits[:, left_id, 1] - soft_limits[:, left_id, 0])
        right_half_range = 0.5 * (soft_limits[:, right_id, 1] - soft_limits[:, right_id, 0])
        scale = torch.clamp(0.5 * (left_half_range + right_half_range), min=1.0e-3)
        penalty += torch.square((left - sign * right) / scale)
    return penalty / max(len(normalized_pairs), 1)
