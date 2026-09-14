# LightHW route_04_balanced_highspeed

这一版使用 `m20_reward_profile.py` 的 M20 奖励权重，按 LightHW 的 hip/thigh/calf
关节组和 wheel body 名称做映射；专用 LightHW 静止门控置零，避免与 M20 的
`stand_still` 重复计分。

命令比例为 standing/low/mid/high=`25%/20%/15%/40%`，高速层范围 `2.5--3.5 m/s`，
其中一半为精确 `±3.5 m/s` 直行锚点。rough 地形初始等级限制为 1，保持固定地形
难度，避免训练同时追逐地形课程和静止目标。轮速动作保持 `scale=40`、clip=`±42`
和 3.0 的腿部 PD damping，不牺牲高速执行器通道。

建议从 route_00 的高速候选 `model_9500.pt` 做 actor-only transfer，使用新策略噪声
配置；不要沿用已经失败的 route_03 Stage A checkpoint。

训练完成后，必须分别固定回放 `0 0 0`、`0.8 0 0`、`3.5 0 0`，再判断模型是否可用。
