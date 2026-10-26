# 配置唯一来源

环境：`source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_env_cfg.py`。
奖励：同包`WheelRewardsCfg`及`rewards.py`。PPO：`ppo_cfg.py`。任务/版本/维度：`catalog.py`。

`WheelStage1EnvCfg`继承旧Rough环境，在父类初始化后替换本实例奖励/命令/重置/地形；
复用canonical资产和修正后的按名称关节观测。不会改旧Rough类、公共资产或旧注册。
转换缓存由worker在每个run下指定。只有显式`catalog.register_tasks()`才注册新任务。

配置声明57/236/16，真实值由`runtime_check.json`在实例化后确认。
目前完整Isaac Sim不可用，真实配置导入/实例化尚未验证。每次实际运行保存最终env/agent YAML。
