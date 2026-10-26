# R1 V4：迁移后的单 Actor 混合训练

**2026-10-19更新：200次诊断已完成，冻结验收通过数14→12→9→6，建议先修正能力保留问题，不从model199原样续训。** 第50次model49与初始化均为14/16。最新结果与正确回放路径见 [200次审查](results/2026-10-19_200_update_review/README.md)。下方保留初始化设计及原运行复现命令，不能将17日启动准入记录当作200次模型验收。

本路线接续 2026-10-16 的迁移校准，使用 Route38 权重初始化，再将诊断教师的动作拟合到一个可部署观测的 MLP。正式入口默认加载 `initialization/selected/model_init.pt`，重新建立 Critic、Adam 和探索噪声。不是原 V3/model499 的续训。

当前起点在固定初态的完整 16 项协议中 **15 项完成、14 项全指标通过**；已通过原先固定的迁移预筛。尚待 PPO 改善：空载 +25° 实际速度约 2.916 m/s；40 kg 拖车 +25° 行进在 4.96 s 牵引净高失败。初始化检查不等于最终 R1 验收，也不代表随机扰动或实机通过。 实际 PPO 2 次后仍为 14/16；精确恢复再更新 1 次后为 13/16，40 kg 上坡零指令四轮支撑降到 91.2%，但原迁移预筛仍通过。完整证据见 [2026-10-17 报告](results/2026-10-17_transfer_readiness/README.md)。

## 原200次诊断复现命令（当前不建议原样重复）

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v4/scripts/train.py \
  --headless --num_envs 1024 --max_iterations 200 --eval_interval 50
```

这是已于2026-10-19完成的 200 次更新小预算 R1 对照的复现入口；每 50 次暂停 PPO，独立冻结评估相同 16 项，结束后继续训练。默认模型通过 `selected/readiness.json` 及 SHA256 核对后才能启动。脚本打印本次输出目录，自动保存源码/资产哈希与生效 YAML；该轮由用户启动，结果见本页顶部审查链接。

全部 64 个许可工况从第一个训练 batch 同时出现，共享一个 Actor、一个 Critic 和一个 PPO 优化器，无课程切换。15 kg 机身载荷与 40 kg 拖车互斥；速度要求仅空载；−25° 无任何拖车，保留空载高速和 15 kg 下坡。其他允许坡度继续包含拖车。

## 物理与训练条件

| 项目 | V4 默认值与含义 |
|---|---|
| 地面 | 摩擦系数 4.0，滚阻 0.02；这是高附着实验假设，未验证实物路面可实现 |
| 牵引点 | 名义出生时距坡面 0.07 m；运行中净高单独记录，失败阈值仍 0.005 m |
| 电机限幅 | 腿 96 N·m、轮 48 N·m，关节输出轴口径；本轮未提高力矩上限 |
| 轮控制 | 速度增益 2，目标限制 ±37 rad/s；四轮刚体上限 3000 deg/s，其余刚体 1000 deg/s |
| 腿控制 | PD 80/3，动作缩放 hip 0.08、thigh/calf 0.14，URDF 关节限位保持 |
| 时序 | 物理 1 kHz，策略 50 Hz，训练 episode 上限 20 s |
| PPO | 固定学习率 1e−5，初始动作标准差 0.05，默认 200 更新/每 50 更新评估 |
| Actor | 300 输入、16 输出、ELU 512/256/128；输入含 5 帧历史和操作员给定的名义载荷/拖车信息 |

Actor 不读真实线速度、地形法向、接触力或仿真实际质量；这些可用于 Critic、奖励和离线教师标签。部署须提供同样的 IMU、编码器、指令和已知任务信息。上一动作的 80 个输入列（Python 切片 `205:285`）在初始化及 PPO 中永久零权重，梯度和 Adam 动量检查保证它们不会重新进入控制；其他 220 列可训练，导出仍是标准单个 300→16 MLP。

增强了机身坡面高度惩罚（−100→−200）与停车漂移惩罚（−0.5→−2）。上坡相对俯仰允许前倾 0～10°；坡面不使用回默认腿姿态惩罚。离线教师对无拖车上坡采用最多约 5° 前倾；40 kg 上坡使用向前约 6 cm 的支撑布局标签。最终训练/回放由单个 Actor 输出动作，不运行教师控制器，不按工况切换多个策略。

## 在哪里看奖励、配置与日志

- 奖励权重、环境和动作：[env_cfg.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v4/env_cfg.py)。计算实现为同目录 `mdp.py`，PPO 为 `ppo_cfg.py`，完整约束为 `spec.py`。
- 每次运行写入本目录 `runs/<时间>/`。实际生效配置是 `params/env.yaml` 与 `params/agent.yaml`；`runtime_check.json` 记录真实维度、质量、力矩、轮刚体上限及工况覆盖。
- 终端记录：`console.log`。训练曲线与模型：`training/`，包括 TensorBoard events、`condition_metrics.csv`、`motor_metrics.csv`、`model_*.pt`。
- 行为判断：`periodic_evaluations/update_0050/evaluation.json` 等。每工况 50 Hz CSV 与 1 kHz `*_motor_trace.npz`、`*_motor.json` 保留首次失败前数据。
- 正常结束导出：`exported/actor.pt`、`contract.json`、`verification.json`。训练结束状态仅表示流程完成，是否通过任务看冻结评估。

```bash
tensorboard --logdir /home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v4/runs --port 6006
```

不要把日志中的平均已结束 episode 时长除以 20 当存活概率。重点看 15 kg 六项是否保留、40 kg 上坡是否能真正走满 1 m 并停稳、空载上坡实际速度是否达到 3 m/s；检查倾斜、航向、四轮支撑、漂移及终止原因。每 50 次评估会增加运行时间。

## 回放与恢复

**默认自由视角（2026-10-19）：** 当前V4回放仅启动时定位一次镜头，之后脚本不再跟随或重置相机。可在Isaac Sim窗口自由旋转、平移、缩放；需要跟随时自行在窗口设置。命令无需增加参数，已运行进程需退出后重新启动才能加载修改。

以下命令可直接回放已验证的默认初始化，显示 15 kg 的 +25° 上坡、行进至少 1 m 后停车。无显示环境添加 `--headless`。初始化模型没有 Adam 状态，不能作为 `--resume` 的输入。

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v4/scripts/play.py \
  --checkpoint model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v4/initialization/selected/model_init.pt \
  --slope_deg 25 --payload_kg 15 --towed_kg 0 --speed 0.3 --seed 43
```

训练后的回放把 `--checkpoint` 换为该次 `runs/<时间>/training/model_*.pt`。精确恢复网络和 Adam 使用同一个 train 入口的 `--resume <实际checkpoint路径>`；`--max_iterations` 是额外更新次数。仅支持相同 V4 合同及物理参数；环境状态重新出生，不声称逐步复现中断前物理轨迹。

## 电机需求的边界

1 kHz 记录包含实际/请求力矩、峰值、P95/P99/P99.9、机械 RMS、限幅比例、连续高负荷时长和机械功率。40 kg 上坡稳停已成功，但上坡行进仍未通过。失败数据只能诊断；当前结果不能给出完成全部 R1 的最低电机额定值，也没有热模型或减速器效率换算。后续先完成任务，再按成功轨迹及饱和证据决定是否扩大限幅、重新对照，避免把控制失败解释为电机不足。
