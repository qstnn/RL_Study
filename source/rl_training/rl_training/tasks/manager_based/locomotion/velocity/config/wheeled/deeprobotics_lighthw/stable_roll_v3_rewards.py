"""Orientation-aware rewards for LightHW StableRoll V3 only."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_upright_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Velocity tracking, scaled by upright alignment; inverted states earn zero."""
    data = env.scene[asset_cfg.name].data
    error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - data.root_lin_vel_b[:, :2]), dim=1
    )
    upright = torch.clamp(-data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    return torch.exp(-error / std**2) * upright


def track_ang_vel_z_upright_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw-rate tracking with the same upright factor as linear tracking."""
    data = env.scene[asset_cfg.name].data
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - data.root_ang_vel_b[:, 2])
    upright = torch.clamp(-data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    return torch.exp(-error / std**2) * upright


def stand_upright_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared distance to gravity [0, 0, -1], gated by the standing command.

    Returns 0 upright, 2 on the side and 4 fully inverted. Near upright it
    retains the old xy-gravity penalty's small-angle scale. No weight or dt
    is applied here; the reward manager owns both.
    """
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    penalty = torch.sum(torch.square(gravity[:, :2]), dim=1) + torch.square(gravity[:, 2] + 1.0)
    standing = torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    return penalty * standing.float()
