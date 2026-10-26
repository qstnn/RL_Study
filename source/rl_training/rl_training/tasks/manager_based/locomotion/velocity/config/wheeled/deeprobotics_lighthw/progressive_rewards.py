"""Rewards used only by the staged LightHW straight-line route."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def track_lin_vel_xy_upright_exp_l1(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track horizontal velocity with an absolute-error exponential kernel.

    The old squared-error kernel becomes almost zero while a new policy is
    stationary under a 3.2 m/s command.  Dividing the norm by ``std`` instead
    of its square preserves a useful learning signal at large errors, while
    the upright gate still prevents a fallen body from earning speed reward.
    """

    data = env.scene[asset_cfg.name].data
    command = env.command_manager.get_command(command_name)[:, :2]
    error = torch.linalg.vector_norm(command - data.root_lin_vel_b[:, :2], dim=1)
    upright = torch.clamp(-data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    return torch.exp(-error / max(float(std), 1.0e-6)) * upright

