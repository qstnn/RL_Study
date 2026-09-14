# Training Layout

- `routes/`：每条训练路线独立放置配置、脚本、日志和结果。
- `records/`：跨路线的训练记录与结果索引，主档为 [records/training_journal.md](records/training_journal.md)。

## 记录约定

- 训练过程、结果摘要和验收结论统一写入 [records/training_journal.md](records/training_journal.md)。
- 更早的长篇 LightHW 历史分析保留在 [reports/lighthw/LIGHTHW_TRAINING_HISTORY.md](../../reports/lighthw/LIGHTHW_TRAINING_HISTORY.md)，作为背景归档。

当前保留 `route_00_baseline` 历史基线和 `route_04_balanced_highspeed` 当前
M20 奖励迁移路线；失败的分层、静态和阶段 A 路线已从运行树删除；包含：

- `configs/`：LightHW 入口、flat/rough 环境和 PPO 配置。
- `scripts/`：route 本地训练和播放脚本，运行时日志写到本 route 的 `logs/`。
- `results/`：评估摘要说明；具体结果统一写入 [records/training_journal.md](records/training_journal.md)。
