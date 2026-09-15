# route_00_baseline

这是第一条正式训练线的起点。这里保留与基线相关的配置、脚本、日志和结果。

这条路线先做 LightHW 的 M20 风格迁移，但不会照搬旧参数：

- flat 入口用于先验证站立、低速跟踪和动作维度是否正确。
- rough 入口用于带地形的适配训练。
- `Rough-Deeprobotics-LightHW-v0` 保留原始 236 维 policy/critic height-scan 基线。
- `Rough-Deeprobotics-LightHW-Blind-v0` 是独立的 policy 无高度、critic 保留高度扫描任务，policy/critic 为 60/236 维，action 为 16 维；其默认 run_name 为 `lighthw_rough_blind`，从头训练。
- 训练日志写到本路线下的 `logs/rsl_rl/<experiment_name>/`。

- `configs/`：环境、奖励、PPO 配置。
- `scripts/`：训练、播放、评估入口。
- `logs/`：TensorBoard、终端日志、checkpoint。
- `results/`：结果说明入口；评估摘要、表格和结论统一写入 [training_journal.md](../../records/training_journal.md)。
