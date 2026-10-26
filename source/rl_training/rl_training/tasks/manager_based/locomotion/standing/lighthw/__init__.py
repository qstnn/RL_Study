"""Register the independent LightHW standing baseline."""

import gymnasium as gym

gym.register(
    id="Stand-Deeprobotics-LightHW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LightHWStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:LightHWStandPPORunnerCfg",
    },
)
