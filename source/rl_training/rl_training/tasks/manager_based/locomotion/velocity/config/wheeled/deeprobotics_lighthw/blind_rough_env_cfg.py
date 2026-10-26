# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""LightHW rough environment with policy-blind and critic-height observations."""

from __future__ import annotations

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLightHWRoughEnvCfg


@configclass
class DeeproboticsLightHWBlindRoughEnvCfg(DeeproboticsLightHWRoughEnvCfg):
    """Rough LightHW task with a blind policy and privileged critic height scan.

    All inherited terrain, reward, action and asset settings remain unchanged;
    only the policy ``height_scan`` term is removed.  The critic keeps the
    inherited height scan so this layout follows the M20 asymmetric setup.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.height_scan = None
        # The base class only performs this cleanup for its exact class name;
        # repeat it for this registered subclass so zero-weight terms with
        # empty selectors cannot reach the reward manager.
        self.disable_zero_weight_rewards()

        # Keep the expected layout visible in both train and play startup logs.
        print("[LightHW Blind Rough] dimensions: policy=60, critic=236, action=16")
