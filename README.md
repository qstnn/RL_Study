# LightHW R1–R6 reference

2026-10-26 的文本参考快照：汇总当前工作区的多技能环境参数、奖励函数、观测/动作、训练与评估配置、相关 Python/Shell 源码，以及统一到 **22 kg** 的机器人 URDF。此分支使用独立根提交，不携带原分支的大文件历史。

## 入口与来源

路线目录位于 [`model_lighthw/training/multiskill/routes`](model_lighthw/training/multiskill/routes)，公共环境/奖励实现位于 [`source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw`](source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw)。保留当前版本及已有实验分支的文本定义，未重新设计奖励或训练参数。

| 路线 | 主要目录 | 工作区来源与当前范围 |
|---|---|---|
| R1 轮式运动 | `route_01_wheel_locomotion/` | `rl_training`，含 research/exploration 的现有源码与配置 |
| R2 跑动/台阶 | `route_02_legged_stairs/` | `rl_training`，保留当前规划，尚无已实现的训练任务 |
| R3 高障碍 | `route_03_high_obstacle/` | `rl_training`，保留当前可行性规划 |
| R4 双腿平衡 | `route_04_two_leg_balance/` | `rl_training`，含 split_v4 前/后腿分支与 sustain 等版本 |
| R5 特技 | [`route_05_acrobatics/`](model_lighthw/training/multiskill/routes/route_05_acrobatics/README.md) | R5A 的 Auto/Guided、教师搜索、起跳实验、专用脚本和说明统一放在 `backflip/`；R5B 位于 `sideflip/` |
| R6 负载 | `route_06_heavy_payload/` | `rl_training_multiskill`，含负载环境、奖励及惯性处理 |

逐文件来源、复制时哈希与本分支哈希见 [`reference/source_manifest.json`](reference/source_manifest.json)。共享目录的选择记录见 [`reference/overlay_resolution.json`](reference/overlay_resolution.json)。R5 目录迁移前后对应关系见 [`reference/layout_migration.json`](reference/layout_migration.json)。根目录不再外置 R5 专用目录、脚本和说明；框架共享源码仍按 R1/R4/R6 的组织方式放在 `source/`。

## 22 kg 模型

- [URDF](model_lighthw/lighthw_urdf/urdf/lighthw.urdf)：17 个 link，16 个关节；整机本体质量 22 kg。
- 原总质量 11.668111486857054 kg，所有 link 质量和惯性张量六分量统一乘以 **1.8854807845109101**；局部质心、连接、几何、关节限位与电机参数保持原值。
- [模型元数据](model_lighthw/lighthw_urdf/model_profile.json)及 R1/R4/R5A/R6 合同记录新的质量/URDF 哈希；R2/R3/R5B 仍引用公共模型，开发状态不变。
- R6 的载荷为额外质量，例如 65 kg 载荷对应总质量 **87 kg**。
- 旧质量下的训练效果不能作为 22 kg 模型的行为验证。详见 [模型变更说明](model_lighthw/training/multiskill/MODEL_22KG.md)。

## 上传范围与使用边界

仅包含文本。没有上传 checkpoint/模型权重、STL 网格、USD、数据集、训练输出、视频、离线 wheel 或仿真安装目录。保留的小型数值测试 fixture 用于 CPU 回归，不代表本轮新增仿真证据。历史报告中的本地输出链接可能指向未上传的文件。

这是供查看、比较和后续复现的配置快照。直接运行仿真还需要自行准备 Isaac Lab/Isaac Sim、Python 依赖及 URDF 引用的 17 份 mesh；涉及既有策略或数据的入口还需要对应文件。外部资产的路径和哈希见 [portable_assets.json](model_lighthw/training/multiskill/records/portable_assets.json)。`offline/` 仅保留安装逻辑和依赖声明，不是完整离线安装包。

## 验证

克隆后可用 Python 3.11+、Git、Bash 执行不依赖 mesh/仿真环境的文本检查：

```bash
python reference/check_snapshot.py
```

首次快照使用已有依赖和临时提供的 mesh 完成了 100 项 CPU 测试，以及 R1/R4A/R5A/R6 的 `check` 入口；目录整理另做了 R5 路径、导入和 CPU 回归检查。临时 mesh 不在提交中。未启动 GPU 仿真或正式训练。验证命令和边界见 [`reference/VALIDATION.md`](reference/VALIDATION.md)。
