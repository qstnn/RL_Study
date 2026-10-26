# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Legacy M20 skill playback environments."""

import gymnasium as gym


gym.register(
    id="M20-BipedStand-Legacy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.legacy_m20_env_cfg:LegacyM20BipedStandPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_ppo_cfg:LegacyM20BipedStandPPORunnerCfg"
        ),
    },
)
