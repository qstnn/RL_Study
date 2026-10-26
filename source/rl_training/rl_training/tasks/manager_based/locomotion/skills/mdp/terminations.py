# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

# termination 函数返回 bool 张量：True 的并行环境会结束当前 episode。
# 它本身不产生奖励；对应惩罚由环境配置中的 ``is_terminated.weight`` 决定。


def root_below_height(env, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身低于绝对安全高度时结束回合，主要过滤完全倒地状态。"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def root_too_far(env, max_xy_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身离开所属环境原点太远时结束，防止不同并行环境相互干扰。"""
    asset = env.scene[asset_cfg.name]
    root_pos_env = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.linalg.norm(root_pos_env, dim=1) > max_xy_distance

