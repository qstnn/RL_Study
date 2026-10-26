# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLightHWRoughEnvCfg


@configclass
class DeeproboticsLightHWFlatEnvCfg(DeeproboticsLightHWRoughEnvCfg):
    """Flat-ground LightHW route that keeps the same reward semantics."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve the M20-style reward roles, but remove the terrain sensor
        # path so the flat route becomes a clean baseline instead of a hidden
        # terrain curriculum.
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "DeeproboticsLightHWFlatEnvCfg":
            self.disable_zero_weight_rewards()
