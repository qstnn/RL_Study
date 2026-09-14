# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-LightHW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLightHWFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWFlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsLightHWRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWRoughPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-BalancedHighSpeed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.balanced_highspeed_rough_env_cfg:"
            "DeeproboticsLightHWBalancedHighSpeedRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "DeeproboticsLightHWBalancedHighSpeedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Rough-Deeprobotics-LightHW-HistoricalHighSpeedPlayback-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.historical_highspeed_playback_env_cfg:"
            "DeeproboticsLightHWHistoricalHighSpeedPlaybackEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLightHWRoughPPORunnerCfg"
        ),
    },
)
