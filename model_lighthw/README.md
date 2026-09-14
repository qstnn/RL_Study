# LightHW 新训练工作空间

这个目录用于 LightHW 新机器狗的独立训练线，和现有训练资产分开管理。

## 结构

- `lighthw_urdf/`：当前唯一 canonical LightHW URDF；GitHub 分支只上传 URDF，mesh 保留在本地以避免大文件和训练产物污染仓库。
- `training/routes/route_00_baseline/`：历史基线，保留用于对照。
- `training/routes/route_04_balanced_highspeed/`：当前唯一可运行的 M20 奖励迁移路线。
- `training/routes/route_01_template/`：后续新路线的复制模板。
- `training/records/`：训练过程与结果的统一记录，主档是 [training/records/training_journal.md](training/records/training_journal.md)。

## 记录约定

- 路线级结果不再拆成多份日期 md，统一写入 `training/records/training_journal.md`。
- 更早的长篇历史分析保留在 [reports/lighthw/LIGHTHW_TRAINING_HISTORY.md](../reports/lighthw/LIGHTHW_TRAINING_HISTORY.md)，作为归档背景，不作为日常增补入口。

## 入口

- `Flat-Deeprobotics-LightHW-v0`
- `Rough-Deeprobotics-LightHW-v0`
- `Rough-Deeprobotics-LightHW-BalancedHighSpeed-v0`

## 约定

- 新路线先复制 `route_01_template`，再改配置和脚本。
- 训练日志、checkpoint 和中间产物放在对应路线的 `logs/` 下。
- 结果摘要统一写入主记录，不再拆成多份路线结果 md。
