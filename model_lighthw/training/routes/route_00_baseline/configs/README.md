# Configs

放这一条训练线专用的环境、奖励和 PPO 配置。

当前使用的 LightHW 配置入口是：

- `deeprobotics_lighthw/__init__.py`：注册 `Flat-Deeprobotics-LightHW-v0`、原始 `Rough-Deeprobotics-LightHW-v0` 和独立的 `Rough-Deeprobotics-LightHW-Blind-v0`
- `deeprobotics_lighthw/rough_env_cfg.py`：M20 风格奖励骨架，但按 LightHW 重新分配动作尺度、接触阈值和高度目标
- `deeprobotics_lighthw/blind_rough_env_cfg.py`：从 policy 组移除 `height_scan`，critic 保留高度扫描，观测为 60/236 维
- `deeprobotics_lighthw/flat_env_cfg.py`：关掉地形扫描和地形课程，作为首轮平地基线
- `deeprobotics_lighthw/agents/rsl_rl_ppo_cfg.py`：PPO 默认超参

当前这条线的关键适配点：

- `hip` 关节动作尺度更小，避免把窄限位打得过猛。
- `thigh/calf` 保留中等动作尺度，方便恢复和地形起步。
- `wheel` 速度尺度按约 0.8 m/s 的命令范围重定。
- `base_height` 目标约 0.42 m，接近 LightHW 的自然机身高度。
- 接触惩罚以约 40 N 的轮接触阈值为起点。
