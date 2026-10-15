# R6配置来源

环境：`source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/payload_env_cfg.py`。
PPO：同目录`payload_ppo_cfg.py`；持载观测/代价：`payload_mdp.py`；契约：`payload_spec.py`。
载荷生成：`payload_asset.py`，在原Base_link惯量上按平行轴定理添加固定刚性载荷。
共享R1基础定义在本分支中只读继承，R6构造实例后替换零命令、奖励、载荷观测和终止设置。
有效配置每次写入run的`params/`；初始质量由CLI `--payload-kg`确定并在PhysX初始化后核对。

`payload_spawn.py`显式启用URDF惯量导入；运行时检查质量/质心/惯量，不依赖默认导入选项。
