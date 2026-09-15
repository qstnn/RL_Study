# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""LightHW rough environment with no terrain height observation."""

from __future__ import annotations

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLightHWRoughEnvCfg


@configclass
class DeeproboticsLightHWBlindRoughEnvCfg(DeeproboticsLightHWRoughEnvCfg):
    """Rough LightHW task with blind policy and critic observations.

    All inherited terrain, reward, action and asset settings remain unchanged;
    only the two ``height_scan`` observation terms are removed.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # The base class only performs this cleanup for its exact class name;
        # repeat it for this registered subclass so zero-weight terms with
        # empty selectors cannot reach the reward manager.
        self.disable_zero_weight_rewards()

        # Keep the expected layout visible in both train and play startup logs.
        print("[LightHW Blind Rough] dimensions: policy=60, critic=60, action=16")
