# LightHW 训练总记录

说明：本文件合并训练过程、结果摘要与验收结论；route 级结果不再拆成多份 md。更早的长篇背景分析仍保留在 [LIGHTHW_TRAINING_HISTORY.md](../../../reports/lighthw/LIGHTHW_TRAINING_HISTORY.md) 作为归档，不再作为当前增补入口。

## 训练过程

| 日期 | 路线 | 配置 / 任务 | 轮次 / 步数 | 状态 | 备注 |
|---|---|---|---|---|---|
| 2026-09-07 | route_00_baseline | LightHW flat / rough 基线 | 0 | 已建档 | URDF 已独立绑定，奖励与动作已做 LightHW 适配 |
| 2026-09-07 | route_00_baseline | LightHW rough 首轮训练 | 12,000 iterations | 已完成训练 | seed=42；末 500 轮平均回报 24.143；checkpoint 11999 有效且参数 finite |
| 2026-09-08 | route_00_baseline | LightHW rough 第二轮前分析 | 5,156 iterations | 已停训 / 待重训 | 当前 run mean_reward 21.462；末步约 21.185；action_rate_l2 和 joint_pos_limits 基本平台，准备更稳的第二版参数 |
| 2026-09-08 | route_00_baseline | LightHW rough 第二版参数训练 | 7,730 iterations | 建议停训 / 待回放 | 当前 run mean_reward 38.585（last500）；末步约 37.789；action_rate_l2 和 joint_pos_limits 已横盘，XY / yaw 误差仍只小幅改善 |
| 2026-09-08 | route_00_baseline | LightHW 高速基线重配 | 0 | 已改参待训 | command x 上限改为 3.5 m/s；wheel 动作幅度同步放大；play.py 支持 fixed_command；旧 mean_reward 阈值待新曲线重标定 |
| 2026-09-09 | route_00_baseline | LightHW 高速基线训练 | 12,000 iterations | 已完成 / 待回放 | run 2026-09-08_21-23-28；best trailing-500 在 model_9500 附近，mean_reward 21.217；final last500 19.974，后段回退 |
| 2026-09-09 | route_00_baseline | LightHW 静止保持参数-only 高速重训 v1 | 0 | 已改参待训 | 不修改 reward 函数实现；standing、静止 gate、平滑、PPO 探索和 reset 参数已调整；建议从 model_9500 warm-start |
| 2026-09-09 | route_00_baseline | LightHW 静止保持参数-only 高速重训 v1 检查 | 4,940 iterations | 建议停止 / 待重配 | run 2026-09-09_14-04-17_param_static_hold_v1；last500 mean_reward 17.316（仅约 +0.51/500 轮），mean_episode_length 657.5/1000；bad_orientation_2 约 3.20、time_out 约 2.87，存活能力已平台 |
| 2026-09-09 | route_01_hierarchical | LightHW 分层速度命令训练 | 0 | 已建档待训 | 保留 `±3.5 m/s` 高速层；standing 20%、low 20%、mid 20%、high 40%，高速层含精确 `±3.5 m/s` 直行锚点；恢复旧高速基线的全局动态平滑/姿态惩罚量级；route_00 保持不变 |
| 2026-09-10 | route_01_hierarchical | LightHW 分层速度命令训练结果 | 12,000 iterations | 已完成 / 存活未达标 | run 2026-09-09_16-32-38_hierarchical_speed_hold_v1；last500 mean_reward 19.714，model_11500 的 trailing-500 约 19.866；mean_episode_length 656.0/1000；bad_orientation_2 3.107，约 6,000 轮后横盘；优先回放 model_11500 与 model_11999 |
| 2026-09-10 | route_02_static_speed3p5 | LightHW 零指令稳定 + 3.5 m/s 执行器 envelope 参数-only 重训 | 6,000 | 已完成 / 训练端不达标，待重配 | run 2026-09-10_16-21-12_static_speed3p5_from_route01；末 500 轮 mean_reward 8.529、mean_episode_length 441.8/1000；bad_orientation_2 5.94；固定命令回放尚未验收 |
| 2026-09-14 | route_03_stage_a_static | LightHW 阶段 A：零指令全身平衡预训练 | v1≈1,149 / v2≈800 | 已提前停止，待重配 | v1 在 rough level 0 上横盘；v2 改为 plane、70% 零命令后仍未改善，保留 v2 `model_800.pt` 作对照，不再延长本轮 |
| 2026-09-14 | route_04_balanced_highspeed | 重新审查 M20 迁移后的高速 + 零命令稳定路线 | 0 | 已建档待训 | 保留 M20 速度跟踪骨架；新增零命令门控腿关节速度惩罚，standing/low/mid/high=25%/20%/15%/40%，高速仍含精确 ±3.5 m/s 锚点 |

## 结果摘要

### 首轮 rough mean reward 预期

按当前 rough 奖励结构和已见过的结果口径，这一轮把 `mean_reward ≥ 24.0` 记为“优秀”，`24.5+` 记为“很强”；若长期低于 `22.5`，就不算这轮的好模型。

说明：这里只在同一条 route、同一套 reward 口径内比较，不把不同奖励版本直接横向硬比。

### 第二版 rough mean reward 预期

按当前第二版 rough 口径，这一轮把 `mean_reward ≥ 38.5` 记为“优秀”，`39.0+` 记为“很强”；若长期低于 `38.0`，就不算这轮的好模型。

说明：这条线只适用于现在这版参数，不要和上一版 24 分的口径混用。

### 高速基线预期

当前已经把 command x 上限和轮速动作幅度切到 3.5 m/s 目标；不沿用上面的 24.0 / 38.5 门槛。根据第一轮高速训练日志，先把 `mean_reward ≥ 21.0` 记为“训练端优秀候选”，`22.0+` 记为“很强候选”；最终是否优秀必须以 3.5 m/s 固定命令回放为准。

### rough 首轮结果

训练已跑满 12,000 iterations，seed 为 42。最终检查点为 `model_11999.pt`。模型状态字典共 16 个动作输出，观测输入为 236 维，所有权重为有限值。

| 指标 | 前 500 轮平均 | 后 500 轮平均 | 最后一轮 |
|---|---:|---:|---:|
| mean reward | -2.162 | 24.143 | 23.741 |
| mean episode length（控制步） | 595.85 | 744.06 | 735.31 |
| XY 速度误差（m/s） | 0.406 | 0.357 | 0.357 |
| yaw 速度误差（rad/s） | 1.842 | 0.782 | 0.800 |
| base z（m） | 0.429 | 0.381 | 0.405 |
| terrain level | 0.650 | 5.314 | 5.278 |
| bad orientation 终止统计 | 51.13 | 2.26 | 1.96 |

补充：末 500 轮 `joint_pos_limits` 约 `-0.244`、`action_rate_l2` 约 `-0.562`，说明动作平滑和限位仍是下一轮重点。

### rough 第一轮平台期判断（旧参数）

当前这一轮 rough 训练在 5,156 steps 左右已进入长尾阶段。

| 窗口 | mean_reward | XY 误差 | yaw 误差 |
|---|---:|---:|---:|
| 2500-3000 | 20.432 | 0.320 | 0.803 |
| 3000-3500 | 20.792 | 0.313 | 0.765 |
| 3500-4000 | 21.087 | 0.306 | 0.743 |
| 4000-4500 | 21.370 | 0.301 | 0.725 |
| 4500-5000 | 21.326 | 0.299 | 0.728 |
| 5000-5500 | 21.649 | 0.298 | 0.726 |

补充：`action_rate_l2` 基本维持在 `-0.60` 附近，`joint_pos_limits` 基本横盘在 `-0.042` 左右，`terrain_levels` 早期上升后回落。当前 run 的末段 mean_reward 约 `21.462`，已经只剩长尾微增益，适合停训并切第二轮更稳参数。

### rough 第二版平台期判断（当前参数）

当前这轮 rough 训练已经进入更明显的长尾平台：

| 窗口 | mean_reward | XY 误差 | yaw 误差 |
|---|---:|---:|---:|
| 5000-5500 | 37.662 | 0.323 | 0.609 |
| 5500-6000 | 37.834 | 0.316 | 0.605 |
| 6000-6500 | 38.126 | 0.313 | 0.595 |
| 6500-7000 | 38.274 | 0.309 | 0.589 |
| 7000-7500 | 38.471 | 0.308 | 0.584 |
| latest snapshot | 38.585 | 0.302 | 0.584 |

补充：`action_rate_l2` 约 `-0.118`、`joint_pos_limits` 约 `-0.198`，基本横盘；`terrain_levels` 还有轻微回落。当前收益还能涨一点，但已经属于很慢的尾段增益，足够停训做回放检查。

### 高速基线训练结果

高速 baseline run：`logs/rsl_rl/deeprobotics_lighthw_rough/2026-09-08_21-23-28`。训练配置确认 `lin_vel_x=(-3.5, 3.5)`，轮速动作 `scale=40.0`、`clip=±42.0`，从零训练 12,000 iterations。

| 指标 | 最佳附近 model_9500 trailing-500 | 最终 model_11999 trailing-500 | 最后一轮 |
|---|---:|---:|---:|
| mean reward | 21.217 | 19.974 | 19.934 |
| mean episode length（控制步） | 868.6 | 880.1 | 926.5 |
| XY 速度误差（m/s） | 0.638 | 0.639 | 0.685 |
| yaw 速度误差（rad/s） | 0.985 | 0.989 | 1.045 |
| base z（m） | 约 0.337 | 0.346 | 0.327 |
| terrain level | 5.989 | 5.973 | 6.023 |
| bad orientation 终止统计 | 0.966 | 0.834 | 0.542 |
| action_rate_l2 | -0.402 | -0.464 | -0.496 |
| joint_acc_wheel_l2 | -0.175 | -0.182 | -0.197 |

结论：训练端不是崩溃，地形课程已经推到较高等级，但 9000-10000 轮之后总奖励回退，动作变化和轮端加速度惩罚继续变差。优先回放 `model_9500.pt`，再对比 `model_10000.pt` 和 `model_11999.pt`；不要默认最终 checkpoint 最好。

### 高速基线回放结论（前进可用，静止保持不合格）

本轮回放的行为结论是：前进学习已经有基本效果，但零指令时机器人持续乱动/漂移，不能保持固定姿态和固定位置。因此本轮只能标记为 **B：初步探索**，不能作为“前进 + 静止保持”完整基线验收。若回放没有使用 `--fixed_command 0 0 0`，应先用固定零指令复核；固定零指令下仍出现该现象时，可排除命令重采样造成的假象。

| 证据 | 当前观察 | 对静止保持的含义 |
|---|---:|---|
| 零指令回放 | 持续乱动，静止保持失败 | 行为验收未通过 |
| `rel_standing_envs` | 0.05 | 训练中只有约 5% 环境专门得到站立指令，静止经验明显不足 |
| `stand_still_without_cmd`（末 500） | -0.008 | 只惩罚腿部关节偏离默认位姿，批量平均几乎不可见，不能约束机身漂移 |
| `stand_wheel_velocity_l2`（末 500） | -0.085 | 只抑制零指令轮速，没有直接抑制 base 的水平速度、偏航速度或位置漂移 |
| `action_rate_l2`（末 500） | -0.464 | 动作变化惩罚在后段变差，和回放中的抖动相符 |
| `joint_acc_wheel_l2`（末 500） | -0.182 | 轮端加速度惩罚后段变差，说明轮速/接触控制仍偏激进 |
| `mean_reward` | model_9500 约 21.217；最终约 19.974 | 后段回退，最终检查点不是本轮最佳 |
| 末 500 速度误差 | XY 约 0.639 m/s；yaw 约 0.989 rad/s | 前进方向“能走”但跟踪精度尚未达到强基线水平 |
| `mean_noise_std` | 初始约 0.60，末段约 0.625 | 策略后期没有稳定收敛；播放虽为确定性推理，但反映训练仍在高不确定性区域 |

根本原因按影响从大到小排列：

1. **静止数据占比过低。** 高速命令范围扩大到 `[-3.5, 3.5]` m/s 后，仍只保留 5% standing env，策略容量自然优先服务高速行驶。
2. **静止奖励约束错位。** `stand_still_without_cmd` 只看腿部关节位置；轮速项只看轮关节速度。两者都没有直接要求 `root_lin_vel≈0`、`root_ang_vel≈0`、base 水平位置不漂移或航向不变化，所以“腿大致收住但机身仍在滚/晃”在当前奖励下是可能的。
3. **轮动作尺度对零点精度不友好。** 为覆盖 3.5 m/s，轮动作 scale 已设为 40；这对高速能力合理，但归一化动作很小的偏置也会转换成明显轮速，静止时更容易出现持续微动。
4. **训练末段发生退化。** 9000-10000 轮后总回报下降，`action_rate_l2` 和轮端加速度惩罚继续恶化；最终 checkpoint 不应直接视为最好模型。
5. **缺少分阶段课程。** 本轮从零同时覆盖高速 rough 与静止保持，没有先把零指令稳态学牢，再逐级提高速度上限，导致两类目标互相争夺优化空间。

上一轮诊断建议（尚未实施的结构性方向）：

- 保留 `model_9500.pt` 作为当前“运动候选”，不要默认使用 `model_11999.pt`。
- 把 standing 比例先提高到约 0.20（可在 0.15-0.25 范围试验），并增加连续零指令保持时段。
- 增加零指令下的机身水平/偏航速度、位置漂移和轮速约束；现有腿部默认位姿项保留但不应作为唯一静止约束。
- 轮动作 scale 暂不再增大；通过静止专用动作平滑/轮速约束和较低探索强度改善零点，而不是牺牲 3.5 m/s 能力。
- 采用“静止 → 0.8 → 1.5 → 2.5 → 3.5 m/s”的分阶段或 warm-start 训练，再比较固定命令回放。
- 验收必须拆开：固定 `0 0 0` 检查 18-20 s 的位置/速度/航向漂移与四轮支撑；固定 `3.5 0 0` 检查速度误差、抖动和地形通过率。

### 下一轮参数方案（已应用，未修改 reward 函数实现）

本轮只调整环境/奖励权重参数、命令分布、reset 扰动和 PPO 超参数；`adapted_rewards.py` 与公共 reward 函数实现没有改动。训练仍保留 `lin_vel_x=(-3.5, 3.5)`，以便同时检验静止和高速前进。

| 参数组 | 旧值 | 新值 | 目的 |
|---|---:|---:|---|
| `rel_standing_envs` | 0.05 | 0.20 | 提高零指令样本占比，给静止策略足够训练信号 |
| 静止 `command_threshold` | 0.12 | 0.20 | 让现有静止项覆盖更宽的近零命令区间 |
| `stand_still_without_cmd.weight` | -1.2 | -2.0 | 加强已有腿部默认位姿约束 |
| `stand_still_scale` | 4.0 | 6.0 | 零指令且低速时更强地保持腿部姿态 |
| 静止项 `velocity_threshold` | 0.35 | 0.50 | 低速 settling 阶段仍使用静止姿态约束 |
| `stand_wheel_velocity_l2.weight` | -0.04 | -0.12 | 抑制零指令轮端持续转动 |
| `action_rate_l2.weight` | -0.045 | -0.06 | 降低动作高频变化 |
| `joint_acc_l2.weight` | -8.0e-8 | -1.0e-7 | 降低腿部关节冲击 |
| `joint_acc_wheel_l2.weight` | -1.5e-7 | -2.5e-7 | 降低轮端加速度和滚动抖动 |
| `ang_vel_xy_l2.weight` | -0.04 | -0.06 | 增加机身角速度阻尼 |
| `flat_orientation_l2.weight` | -0.12 | -0.18 | 减少静止时的机身摇摆 |
| `joint_pair_symmetry_l2.weight` | -0.015 | -0.025 | 减少左右不一致造成的微动 |
| `feet_contact_without_cmd.weight` | 0.02 | 0.04 | 鼓励零指令时保持四轮支撑 |
| reset roll/pitch | ±0.15 rad | ±0.10 rad | 先让策略学会稳态，再承受较大初始倾斜 |
| reset base velocity | ±0.10 m/s | ±0.05 m/s | 减少起步瞬间的无意义漂移 |
| reset angular velocity | roll/pitch ±0.03，yaw ±0.05 | roll/pitch ±0.02，yaw ±0.03 | 降低静止起步的扰动 |
| `lin_vel_y` range | ±0.20 m/s | ±0.15 m/s | 把容量优先给前进和静止 |
| `ang_vel_z` range | ±0.40 rad/s | ±0.30 rad/s | 减少与静止目标冲突的高速转向样本 |
| PPO `init_noise_std` | 0.60 | 0.40 | 从零训练时降低初始探索幅度 |
| PPO `entropy_coef` | 0.004 | 0.0015 | 减少后期策略继续保持高随机性的倾向 |
| PPO `learning_rate` | 3.5e-4 | 2.0e-4 | 减少后段回退和参数震荡 |
| PPO `desired_kl` | 0.01 | 0.008 | 限制单次策略更新幅度 |
| experiment name | `deeprobotics_lighthw_rough` | `deeprobotics_lighthw_static_hold_hs_v1` | 让新 TensorBoard 曲线与旧高速 run 分开 |

推荐使用旧高速 run 的 `model_9500.pt` 做完整 warm-start，保留已经学到的前进能力，同时用新参数塑造静止保持。注意：warm-start 会继承 checkpoint 中的策略噪声状态；配置里的 `init_noise_std=0.40` 主要对从零训练生效。由于本轮重点是行为验收，不能只以 mean reward 判断成败。

本轮验收门槛暂定为：

- 固定 `0 0 0` 回放 18-20 s：机身不应持续向任一方向漂移，平均水平速度尽量低于 `0.05 m/s`，偏航速度尽量低于 `0.05 rad/s`，四轮支撑率目标 `>99%`。
- 固定 `3.5 0 0` 回放：保持前进能力，重点看速度误差、动作高频抖动和轮端接触；不要求 mean reward 与旧权重版本直接相等。
- 训练端先观察前 2,000-4,000 iterations 的 `mean_reward`、`action_rate_l2`、`joint_acc_wheel_l2` 和 episode length 是否同步改善，再决定是否继续到满轮次。

推荐启动命令（从上一轮运动表现最好的 `model_9500.pt` 完整 warm-start）：

```bash
cd /home/user/RL_lab/rl_training/model_lighthw/training/routes/route_00_baseline && TERM=xterm ./scripts/train_rough.sh --headless --seed 42 --max_iterations 12000 --run_name param_static_hold_v1 --warm_start_checkpoint /home/user/RL_lab/rl_training/model_lighthw/training/routes/route_00_baseline/logs/rsl_rl/deeprobotics_lighthw_rough/2026-09-08_21-23-28/model_9500.pt
```

### 回放验证状态

- GPU 固定指令回放尚未完成。
- Isaac Sim 启动时报告 `No CUDA devices found` / `NVML_ERROR_DRIVER_NOT_LOADED`。
- kit Python 还缺 `flatdict`、`gymnasium`、`rsl-rl-lib`。

结论：这部分不能当作真实回放成功证据。

### 静止保持 v1 存活率与平台期检查

本轮从上一轮高速模型 warm-start，实际曲线记录到 iteration 4939（共 4,940 个标量点），当前没有继续运行的训练进程。最大回合长度为 20 s，即 1,000 个控制步。

| 指标 | 近 500 轮 | 解释 |
|---|---:|---|
| mean_reward | 17.316 | 仍有微小增益，但不能代表存活能力改善 |
| mean_episode_length | 657.5 / 1000（65.8%） | 回合大多未活满，低于当前 70% 存活时长门槛 |
| bad_orientation_2 | 3.199 | 近 500 轮基本横盘，是主要失败终止 |
| time_out | 2.868 | 超时终止没有明显上升 |
| terrain_out_of_bounds | 0.191 | 占比较小，不是首要原因 |
| 终止计数口径 | — | `bad_orientation_2` 是 terminated；`time_out` 与 `terrain_out_of_bounds` 均是 truncated，三项计数可能重叠，不能直接当作生存概率 |
| terrain level | 5.973 | 已长期停在较高 rough 难度 |
| XY / yaw 速度误差 | 0.478 / 0.593 | 近 500 轮基本横盘，跟踪能力也没有明显突破 |

结论：本轮应停止并保留 `model_4900.pt` 作为诊断检查点，不建议继续沿用当前参数跑到 12,000 轮。奖励曲线还在缓慢上升，因此这是“存活平台期”，不是严格意义上的总回报完全平 plateau；但 `bad_orientation_2` 和回合长度已经满足停止条件。下一轮应先降低 rough 难度或采用分阶段课程，再逐步恢复 3.5 m/s、高外力和完整地形随机化；优先调整参数，不改奖励函数实现。

### route_01 分层训练结果

本轮使用分层命令和上一轮高速 `model_9500.pt` warm-start，实际跑满 12,000 iterations。所有 TensorBoard 标量均为有限值，最终 `model_11999.pt` 的 16 维策略权重也通过有限值检查。

| 指标 | 前 1,000 轮 | 后 1,000 轮 | 结论 |
|---|---:|---:|---|
| mean_reward | 2.273 | 19.790 | 训练完成，但约 6,000 轮后进入慢速平台；末 500 轮为 19.714 |
| mean_episode_length | 565.9 | 655.9 | 只达到 65.6% 的 1,000 步上限，存活仍不合格 |
| bad_orientation_2 | 4.357 | 3.107 | 前期下降，6,000 轮后约 3.1 横盘，是首要失败终止 |
| time_out | 2.464 | 2.839 | 超时计数略升，不能抵消姿态终止问题 |
| terrain_out_of_bounds | 0.288 | 0.301 | 配置中标记为 truncated 的地形越界，规模较小 |
| stand_wheel_velocity_l2 | -0.430 | -0.074 | 零指令轮速明显收敛 |
| action_rate_l2 | -0.309 | -0.201 | 动作变化惩罚改善后趋于横盘 |
| joint_acc_wheel_l2 | -0.117 | -0.093 | 轮端加速度惩罚改善有限 |

`Metrics/base_velocity/error_vel_xy` 和 `error_vel_yaw` 是命令管理器的累计误差代理，按固定 8 s 命令窗口记录，不能直接当作完整 episode 的 m/s 或 rad/s MAE。`Metrics/base_velocity/hierarchy_stage` 是发生 reset 的环境中最后一次层级编号均值，也不是各层占比；本轮未记录每个层级独立的成功率。

训练端最值得保留的是 `model_11500.pt`（trailing-500 mean_reward 约 19.866，接近滚动窗口峰值）和最终 `model_11999.pt`。两者都只能称为“分层训练候选”，不能仅凭 mean_reward 判定优秀；必须用固定 `0 0 0` 与 `3.5 0 0` 回放检查存活、速度误差、四轮支撑和航向漂移。

按行为拆分的当前验收状态：

| 行为 | 训练端证据 | 当前状态 |
|---|---|---|
| 静止 `0 0 0` | standing 配置层为 20%；`stand_wheel_velocity_l2` 收敛到约 `-0.073`，`stand_still_without_cmd` 约 `-0.032` | B：静止相关奖励改善；机身漂移、轮支撑率和姿态仍需固定回放 |
| 低速 `0.8 m/s` | low 层配置范围为 `±0.8 m/s`；混合误差代理末 500 轮约 `0.501` | C：没有按层级拆分的速度/存活曲线，不能确认本轮低速能力 |
| 高速 `3.5 m/s` | high 层占 40%；其中约 10% 的全部命令为精确 `±3.5 m/s` 直行锚点；轮速动作接口保持 `scale=40`、`clip=±42` | C：只证明样本覆盖和接口保持，尚无本轮固定高速回放的速度误差、支撑和倒地证据 |

### route_01 回放诊断：静止乱动与高速右前轮卸载（2026-09-10）

本次用户回放观察为：零命令时腿部持续大幅乱动；低速前进基本可用；高速时右前轮（FR）出现悬空。该观察不能被当前总回报推翻，原因是训练曲线只记录批量标量，尚未记录固定命令下的机身速度、逐轮当前接触力和 reset 前状态。

#### 自动 reset 的影响

`play.py` 在 `env.step()` 后丢弃 `terminated/truncated`，而 ManagerBasedRLEnv 会在同一个 `step()` 内自动 reset 已结束环境；因此回放窗口会继续运行，看起来像机器人“倒下后又复活”。`play_motor_monitor.py` 也会继续记录，只有 `episode_step` 下降才能在 CSV 中分割回合。这个机制会掩盖视觉上的失败，但不能单独解释所有乱动。

当前分层 run 的末 500 轮 TensorBoard 精确值如下。终止项是每次 rollout 的计数均值，不是百分比；`time_out` 和 `terrain_out_of_bounds` 是 truncated，且计数可能重叠。

| 指标 | 末 500 轮均值 | 口径 |
|---|---:|---|
| `Train/mean_reward` | 19.713585 | 训练总回报，约 9,000 轮后已基本横盘 |
| `Train/mean_episode_length` | 655.965740 控制步 | 约 13.12 s；不是 65.6% 生存率 |
| `bad_orientation_2` | 3.116750 | 主要失败终止，约 6,000 轮后仍在 3 左右 |
| `time_out` | 2.837333 | truncated 计数 |
| `terrain_out_of_bounds` | 0.292417 | truncated 计数，规模较小 |

因此，确实存在频繁 reset，且“无限重新复活”会让肉眼只看到持续运行窗口；但现有日志不能把它换算成精确存活百分比。静止失败是行为层真实问题，不应再用 `mean_reward` 或 `mean_episode_length/1000` 解释成已通过。

#### 奖励信号审计

`stand_still_without_cmd` 只计算 12 个腿关节相对默认角的绝对偏差，并用命令阈值门控；它不约束机身水平速度、偏航速度、位置漂移、姿态角或四轮载荷。`stand_wheel_velocity_l2` 只在近零命令时惩罚四个轮子的角速度。`feet_contact_without_cmd` 只奖励“首次接触事件”的数量，同样只在近零命令启用，不是持续支撑率。

末 500 轮相关项为：`stand_still_without_cmd=-0.032466`、`stand_wheel_velocity_l2=-0.073238`、`action_rate_l2=-0.199891`、`joint_acc_wheel_l2=-0.093088`、`joint_pos_limits=-0.159253`。这说明静止专用项数值很小，而动作变化、轮端加速度和限位项更大；但它们都是 reset-batch 的 episode 平均，不能直接等同于视频中的抖动幅度。

右前轮悬空暂不能归因于能耗。能耗项只有 `joint_power=-0.000710`、`joint_power_wheel_l1=-0.001827`，相对于速度跟踪 `track_lin_vel_xy_exp=1.358210`、`track_ang_vel_z_exp=0.425341` 很弱。更符合代码和数据的解释是：运动阶段没有四轮最低承载/持续接触奖励，`contact_forces` 只惩罚历史窗口中超过 40 N 的冲击峰值（末 500 轮仅 `-0.009120`），对“轮子离地”没有惩罚；同时关节对称项只看角度，不看负载。因此策略可能用右前轮卸载来换取速度、姿态或冲击代价下降。这是高可信假设，不是当前数据已经证明的唯一因果。

另一个已经确认的硬约束是执行器速度上限：训练动作接口为 `scale=40`、裁剪 `±42 rad/s`，而 LightHW URDF/执行器的 wheel `velocity_limit=37.7 rad/s`。按轮半径约 `0.0875 m`，3.5 m/s 需要约 `40 rad/s`，比执行器上限高约 6.1%，对应的无滑移线速度上限约为 `3.30 m/s`。因此当前“3.5 m/s”是命令目标，不是执行器可保证的物理速度；在负载、地形或左右受力稍有不对称时，FR 先卸载或速度跟不上是合理结果。分层命令还每 8 s 突然切换一次 standing/low/mid/high，增加了从高速到静止的瞬态冲击。

#### 下一步修改顺序（本次未改参数）

1. 先做可重复的固定命令监控：对 `0 0 0`、`0.8 0 0`、`3.5 0 0` 分别运行 `play_motor_monitor.py`，使用 `--disable_payload_motor_profile --monitor_leg all --monitor_all_body_contacts --physics_substep_log --no_plots --max_steps 1200`。用 `episode_step` 下降切分回合，统计每回合长度、机身速度/角速度、姿态、FR 当前 `Fz` 低于 5 N 的比例、四轮同时承载率和力矩饱和率。
2. 先处理高速目标与执行器上限的一致性：用监控 CSV 检查 `*_wheel_joint_vel_target` 是否长期超过 `37.7 rad/s`。若要保留“3.5 m/s 命令”但接受物理上限，应把它作为超额目标并记录实际可达速度；若要真正保留 3.5 m/s 的物理能力，则必须先核实轮半径和执行器/URDF速度上限，这属于动力学/资产参数变更，不是奖励权重调整。
3. 在上限问题明确后，参数优先只改命令过渡和扰动暴露：延长 8 s 重采样时间或采用平滑过渡，降低训练早期 push/rough 难度，按静止→低速→高速 warm-start；不先增大能耗项，也不先压低轮速动作尺度。
4. 如果上述监控确认静止机身仍漂移，说明现有参数调节无法补足观测目标；下一轮再增加零命令机身速度/角速度约束。若确认 FR 在高速持续低载，再增加高速直行的逐轮最低接触/负载均衡约束。

本次状态：静止行为 **未通过**；低速行为依据用户视频为 **初步可用、尚无 CSV 量化**；高速 FR 支撑为 **未通过待量化**。在完成三组固定命令 CSV 前，不把该 checkpoint 标为优秀模型，也不继续沿当前奖励结构长训。

### fixed-command 监测结果：高速回放（2026-09-10）

本次三次监测启动时间约为 `14:54:11`、`14:55:13`、`14:56:34`。`play_motor_monitor.py` 每次都以写模式覆盖同一个 `motor_monitor/motor_log.csv` 和 `physics_substep_log.csv`，因此当前保留下来的 1200 个控制步只代表最后一次运行；CSV 没有保存 `fixed_command` 字段。根据最终数据的机身速度约 3.2 m/s 和轮速目标，当前文件与最后一次 `3.5 0 0` 高速回放一致；静止、低速两次的独立 CSV 已不可恢复，不能用当前文件替代它们。

当前证据文件：

- `motor_log.csv`：1200 个控制步；
- `physics_substep_log.csv`：4800 个物理子步；
- `episode_step` 在全局第 1089 行下降一次，说明发生过一次自动 reset；连续的第一段只有 1089 步（约 21.78 s），后面 111 步是 reset 后的新回合，不能合并为 24 s 的连续成功。

#### reset 前第一段（高速行为的有效窗口）

| 指标 | 结果 | 解释 |
|---|---:|---|
| 机身 body-frame `v_x` 均值 | 3.201 m/s | 速度能力接近 3.5 m/s |
| 去掉前 200 步后的 `v_x` 均值 | 3.257 m/s | 起步后速度跟踪更接近目标 |
| 相对 3.5 m/s 的平均绝对误差 | 0.302 m/s | 速度不是当前首要失败点 |
| body-frame `v_y` 均值 / 绝对值 P95 | 0.013 / 0.317 m/s | 机体坐标侧向速度不大 |
| 根部 yaw 起点→reset 前 | -0.026→-1.699 rad | 约 -1.67 rad（-95.8°）偏航漂移，世界坐标因此出现明显侧移 |
| `root_x_env` 位移 | +46.951 m | 运行窗口内持续向前漂移，接近粗糙地形局部边界 |
| `root_y_env` 位移 | -36.934 m | 与偏航漂移/世界坐标侧移一致 |
| `root_z` 均值 / 最小值 | 0.366 / 0.041 m | 多次明显下沉或塌腿；不能称为稳定站立 |
| roll 绝对值 P95 / pitch 绝对值 P95 | 0.239 / 0.238 rad | 姿态波动很大 |

reset 前最后一个有效控制步（全局第 1088 步）为 `root_z=0.289 m`、`roll=-0.095 rad`、`pitch=-0.042 rad`、`projected_gravity_xy` 范数约 `0.104`、`projected_gravity_z=-0.995`。这一个时刻没有达到 `bad_orientation_relaxed` 的 `0.95` 倾斜阈值；但机身高度已经很低，且只有 RR 轮有约 188 N 当前竖向接触力。当前 CSV 没保存 termination term mask，因此不能把这次 reset 只归因于姿态、地形越界或 timeout；下一次需直接记录 `termination_manager` 的逐项布尔值。

#### 轮端支撑

| 轮 | 当前 `Fz<5 N` 比例（reset 前） | 最长连续卸载 |
|---|---:|---:|
| FL | 62.81% | 0.48 s |
| FR | **98.71%** | **5.20 s** |
| RL | 79.06% | 0.34 s |
| RR | 53.90% | 0.26 s |

四个轮子同时达到 `Fz>=5 N` 的比例为 **0/1089 = 0%**；每步平均只有 1.055 个轮子达到 5 N（0 个轮子同时支撑 340 步、1 个 370 步、2 个 358 步、3 个 21 步、4 个 0 步）。这不是“右前轮偶尔悬空”，而是 FR 基本整段失去承载，整体也没有形成四轮支撑。

reset 后剩余 111 步中，FR 当前接触力为 **100%** 低于 5 N，`root_z` 均值仅 0.195 m；这部分是复活后的失败状态，不能用于提高存活率或支撑率。

#### 轮速目标与执行器上限

当前运行记录到的 wheel velocity target 经常超过 URDF/执行器 `37.7 rad/s` 限制：

| 轮 | 目标绝对值超过 37.7 rad/s 的比例 | 最大目标 |
|---|---:|---:|
| FL | 83.65% | 42.0 rad/s |
| FR | 40.86% | 42.0 rad/s |
| RL | 25.62% | 42.0 rad/s |
| RR | 89.44% | 42.0 rad/s |

因此 3.5 m/s 是策略命令目标，不是所有轮子都能在真实执行器边界内无滑移保证的速度。FR 长期卸载更符合“速度目标过高 + 没有持续最低承载约束 + 左右受力不均”的组合解释，不支持把它首先归因于能耗奖励。当前 `joint_power_wheel_l1` 权重为 `-8e-6`，而 `contact_forces` 只惩罚超过 40 N 的历史峰值；这两项都没有直接奖励四轮持续着地，且没有对 FR 低载施加惩罚。

另外，reset 前有约 **4.96%** 的控制步出现非轮体接触（最高约 562 N），主要发生在 calf link；`Base_link` 没有持续碰撞。说明问题更像腿部塌落/擦地和载荷转移，而不是底盘一直压在地面上。

#### 本次结论

- **高速运动能力：部分通过。** Reset 前 body-frame 前进速度均值 3.20 m/s，去掉起步后为 3.26 m/s，说明策略确实学到了接近 3.5 m/s 的前进；但偏航累计漂移约 96°、高度多次塌低，不能称为稳定高速。
- **高速支撑：未通过。** FR 98.71% 时间低载，四轮同时支撑率 0%，与用户观察一致。
- **复活掩盖问题：确实存在。** 1200 步窗口包含一次 reset，后 111 步不能作为前一回合的延续成功；但 reset 前本身已经出现低高度和单轮承载，失败是真实的，不只是视觉错觉。
- **静止/低速：本次没有保留下来的独立 CSV，不能用当前高速文件代替。** 当前状态继续保持“静止未通过、低速仅有视频层面的初步可用”。

下一轮仍应先做参数/诊断层面的处理：保存三条命令的独立 CSV，记录逐项 termination mask；再针对高速命令与 37.7 rad/s 执行器上限的一致性、命令切换过渡和持续四轮承载进行调整。当前不把 `model_11999.pt` 标为优秀模型，也不以 reset 后的 111 步提升其存活评价。

### route_02_static_speed3p5 参数-only 调整（2026-09-10）

本轮针对上一轮已经量化的三个问题建立独立路线：零命令时全身乱动、3.5 m/s
高速阶段 FR 长期卸载/姿态塌低，以及 wheel 执行器 `37.7 rad/s` 上限低于
3.5 m/s 所需的约 `40 rad/s`。不修改奖励函数实现，不改变观测和 16 维动作接口，
也不回写 route_00/route_01。

#### 修改清单

| 类别 | route_01 | route_02 | 目的 |
|---|---:|---:|---|
| standing / low / mid / high | 20 / 20 / 20 / 40 % | **35 / 15 / 10 / 40 %** | 给零指令足够样本，同时保留完整高速层 |
| high-speed exact anchor | 25% | **35%** | 高速层内精确 `±3.5 m/s` 直行样本约从 10% 提到 14% |
| command hold | 8 s | **12 s** | 减少高速与静止之间的突变切换 |
| lateral / yaw range | ±0.15 / ±0.30 | **±0.05 / ±0.20** | 第一轮先隔离直行稳定性 |
| wheel actuator `velocity_limit` | 37.7 rad/s | **42.0 rad/s** | 使 `40 rad/s ≈ 3.5 m/s` 不再被仿真上限截断 |
| wheel actuator `velocity_limit_sim` | 未显式覆盖 | **42.0 rad/s** | 同步 PhysX DOF 速度上限 |
| leg damping | 2.0 | **3.0** | 减少姿态修正后的高频摆动 |
| terrain curriculum | active | **frozen**，初始 level≤1 | 先解决闭环稳定，再恢复 rough 难度 |
| reset roll/pitch | ±0.10 | **±0.05 rad** | 让策略从近似平衡点学习零指令保持 |
| reset base velocity | ±0.05 | **±0.02 m/s** | 减少复位后的无意义漂移 |
| reset angular velocity | roll/pitch ±0.02，yaw ±0.03 | **±0.01 / ±0.02 rad/s** | 降低起步瞬态 |
| push interval / velocity | 12--18 s / ±0.25 m/s | **20--28 s / ±0.12 m/s** | 保留扰动但不让静态学习被频繁打断 |
| startup uncertainty | mass ±等旧范围 | **mass 0.95--1.05、COM ±0.015 m、gain 0.98--1.02** | 先收窄动力学随机化 |
| `stand_still_without_cmd` | -2.0 | **-3.5** | 增强零命令腿部默认位姿约束 |
| static leg pose weights | hip -0.30；thigh/calf -0.12 | **hip -0.45；thigh/calf -0.18** | 增大静止状态下偏离默认姿态的代价 |
| static scale | 6.0 | **7.0** | 仅在低命令/低速门控内增强 |
| `stand_wheel_velocity_l2` | -0.12 | **-0.30** | 抑制零命令轮子自转 |
| symmetry | -0.025 | **-0.04** | 抑制左右关节修正不一致 |
| body attitude | ang xy -0.04；flat -0.12 | **-0.06；-0.20** | 降低静止摇摆和俯仰/横滚偏差 |
| base height | -0.45 | **-0.55** | 减少塌腿、擦地和高速载荷转移 |
| velocity tracking | lin 2.4；yaw 0.85 | **lin 2.6；yaw 1.0** | 零命令也直接受益，同时保持高速跟踪压力 |
| action / acceleration | action -0.045；leg acc -8e-8 | **action -0.06；leg acc -1e-7** | 抑制高频动作变化；wheel acc 保持 -2.5e-7 |

执行器速度上限的改动只发生在
`speed3p5_asset_cfg.py` 的 route-local 深拷贝中，旧路线的 37.7 rad/s 不变。
轮半径按现有 mesh 约 `0.0875 m` 估算，`40 rad/s` 对应 `3.50 m/s`，42 rad/s
只提供约 5% 的跟踪余量。这个仿真设置不能替代真实执行器规格确认：如果实机
loaded speed 仍只有 37.7 rad/s，物理上的 3.5 m/s 仍然不可保证。

#### 本轮 mean reward 预期

由于 standing 比例、静态惩罚和 terrain curriculum 都变化，route_02 的
`mean_reward` 只在本路线内部比较，不与 route_00 的 24/38.5 或 route_01 的
19.7 直接横比。暂定训练端目标为：

- trailing-500 `mean_reward >= 21.0`：优秀候选；
- trailing-500 `mean_reward >= 23.0`：很强候选；
- 但只有固定命令回放同时通过，才记为优秀模型。

#### 启动与首个检查点

建议从 route_01 的 `model_11500.pt` 完整 warm-start，先跑 6,000 iterations；
前 2,000--3,000 iterations 检查静态专用奖励和存活指标，若
`bad_orientation_2`、`mean_episode_length` 没改善，先停在最近 checkpoint，
不要盲目跑满。

```bash
cd /home/user/RL_lab/rl_training/model_lighthw/training/routes/route_02_static_speed3p5 && TERM=xterm ./scripts/train_rough.sh --headless --seed 42 --max_iterations 6000 --run_name static_speed3p5_from_route01 --warm_start_checkpoint /home/user/RL_lab/rl_training/model_lighthw/training/routes/route_01_hierarchical/logs/rsl_rl/deeprobotics_lighthw_hierarchical_rough/2026-09-09_16-32-38_hierarchical_speed_hold_v1/model_11500.pt
```

#### 固定命令验收门槛

只统计 reset 前的连续回合，不把自动 reset 后的“复活”片段并入成功率：

- `0 0 0`：连续 20 s 无 reset；水平速度均值 `<0.05 m/s`，偏航速度
  `<0.05 rad/s`，航向漂移 `<5°`，四轮 `Fz>=5 N` 同时支撑率 `>99%`；
- `0.8 0 0`：低速前进无持续高频摆动；
- `3.5 0 0`：`v_x>=3.3 m/s`，平均绝对速度误差 `<=0.25 m/s`，航向漂移
  `<10°`，wheel target 不超过 `42 rad/s`，FR 不再长期低载。

当前这些是验收目标，不是已经完成的 GPU 回放结果；训练结束后仍需保存三条
固定命令的独立 CSV 和逐项 termination mask。

### route_02_static_speed3p5 中途训练诊断（2026-09-10，TensorBoard 快照 4,761 iterations）

本节只记录训练端诊断；训练进程仍在写入，尚未运行固定命令回放。`Episode_Reward/*`
是按最长 20 s 回合归一化的加权贡献，因此早期终止会同时压低正、负奖励，不能把
它直接当作存活状态下的逐步原始物理量。

| 指标 | 2,500--3,000 平均 | 最近 500 轮平均（至 4,761） | 结论 |
|---|---:|---:|---|
| `Train/mean_reward` | 7.155 | 8.041 | 远低于本路线优秀候选 21.0，只有微弱改善 |
| `Train/mean_episode_length`（控制步） | 439.5 | 441.9 / 1000 | 存活没有实质改善 |
| `bad_orientation_2` | 6.13 | 6.06 | 姿态失败未下降 |
| `error_vel_xy` | 约 0.30 | 0.303 | 命令混合下没有继续收敛 |
| `error_vel_yaw` | 约 0.21 | 0.233 | 偏航跟踪没有继续收敛 |

最近 500 轮的正贡献为 `+1.413`，负贡献为 `-1.014`，净值 `+0.398`；乘以 20 s
约为训练端 mean reward `7.97`，与 TensorBoard 的 `8.04` 一致，说明低分并非遗漏
日志项。最大负项如下（括号为按 20 s 折算的平均回报损失）：

| 负项 | 最近 500 轮 | 回报损失 | 含义 |
|---|---:|---:|---|
| `action_rate_l2` | -0.180 | -3.61 | 策略输出仍有持续高频变化 |
| `joint_pos_limits` | -0.127 | -2.53 | 腿部经常靠近软关节限位 |
| `hipx_joint_pos_penalty` | -0.112 | -2.25 | 髋部默认位姿/静态姿态偏差很大 |
| `stand_wheel_velocity_l2` | -0.102 | -2.04 | 零命令门控内轮子仍明显自转 |
| `joint_acc_wheel_l2` | -0.098 | -1.95 | 轮速反复加减速 |
| `joint_acc_l2` | -0.092 | -1.85 | 腿部高频加速度仍大 |
| `ang_vel_xy_l2` | -0.085 | -1.71 | 机身仍有横滚/俯仰角速度 |
| `stand_still_without_cmd` | -0.062 | -1.25 | 零命令时腿部没有回到默认静止构型 |

上述八项约占全部负回报的 85%。其中 `stand_still_without_cmd` 和
`stand_wheel_velocity_l2` 都是直接零命令门控；髋/膝/大腿姿态项在静止且低速时会
被放大 7 倍。它们在 3,000 轮后只有很小改善，说明当前权重已经在施压，但没有建立
稳定平衡点；简单降低这些权重只会掩盖抖动，并不能改善行为。

正项主要来自 `track_lin_vel_xy_exp=+0.952`、`track_ang_vel_z_exp=+0.352` 和
`upward=+0.104`。它们的绝对值偏低有一部分来自平均回合只有约 44% 的最长时长，
所以现有日志不能把它单独解释为速度跟踪失败；但也没有证据表明 3.5 m/s 的固定命令
已经通过。当前混合命令曲线无法按 standing/low/high 分层验收。

`feet_contact_without_cmd=+0.00081` 几乎没有贡献，并非连续四轮支撑奖励：它只在
轮子从非接触变为首次接触时给分，增加权重可能鼓励反复落轮，不能用来修复静止稳定。
此外，`is_terminated` 在本路线的有效配置中为 `null`，因此 `bad_orientation_2`
终止没有直接负回报，只通过提前失去后续回报间接受罚。这解释了姿态失败长期不下降。

本次中途结论：训练优化本身没有数值崩溃（value loss 已处于稳定量级、策略噪声约
0.35），但行为目标没有收敛。下一次参数调整应优先处理终止代价、零命令机身速度/
角速度和连续四轮承载的可观测性；不能继续仅靠增加腿位姿或首次接触奖励权重。当前
checkpoint 只能作为诊断候选，不能标为优秀模型。

### route_02_static_speed3p5 完成训练诊断（2026-09-10，TensorBoard 6,000 iterations）

本轮已写出 `model_5999.pt`，训练进程结束。以下结果来自最终 event 文件的完整
6,000 个标量点；`Episode_Reward/*` 仍是按最长 20 s 回合归一化的加权贡献，
不是固定命令下的物理量，也不能代替回放验收。

| 指标 | 2,500--2,999 平均 | 5,500--5,999 平均 | 末步 5,999 | 结论 |
|---|---:|---:|---:|---|
| `Train/mean_reward` | 7.159 | 8.529 | 4.743 | 后段只有缓慢增益，远低于优秀候选 21.0 |
| `Train/mean_episode_length`（控制步） | 439.6 | 441.5 | 367.5 | 完成回合平均约 8.83 s（上限 20 s），没有随训练改善，末步还回落 |
| `bad_orientation_2` | 6.04 | 5.94 | 6.42 | 姿态失败长期横盘 |
| `time_out` | 2.07 | 2.14 | 1.46 | 不能据此证明完成回合 |
| `terrain_out_of_bounds` | 1.21 | 1.18 | 0.92 | 与姿态终止同时存在，需逐项 mask 才能分离 |
| `error_vel_xy` | 约 0.31 | 0.317 | 0.293 | 混合命令下仅有小幅改善，未证明 3.5 m/s |
| `error_vel_yaw` | 约 0.23 | 0.232 | 0.218 | 偏航跟踪仍未收敛到可验收水平 |

末 500 轮的奖励贡献按绝对负回报排序如下。括号内是按 20 s 折算后的平均
回报损失，便于看出其对总分的实际影响：

| 负项 | Episode_Reward | 20 s 折算损失 | 诊断 |
|---|---:|---:|---|
| `action_rate_l2` | -0.1781 | -3.56 | 高频动作变化仍是第一大损失 |
| `joint_pos_limits` | -0.1289 | -2.58 | 腿关节长期靠近软限位 |
| `hipx_joint_pos_penalty` | -0.1123 | -2.25 | 髋关节偏离默认构型，静止姿态没有锁住 |
| `joint_acc_wheel_l2` | -0.0951 | -1.90 | 轮速在反复加减速 |
| `stand_wheel_velocity_l2` | -0.0945 | -1.89 | 零命令门控内仍存在轮子自转 |
| `joint_acc_l2` | -0.0897 | -1.80 | 腿部修正动作仍然剧烈 |
| `ang_vel_xy_l2` | -0.0826 | -1.65 | 机身横滚/俯仰角速度未被压下 |
| `stand_still_without_cmd` | -0.0612 | -1.22 | 零命令腿部仍没有回到默认姿态 |

这八项约占负回报绝对值的 85%。相较第 3,000 轮，`track_lin_vel_xy_exp`
由 `0.808` 升到 `0.955`、`track_ang_vel_z_exp` 由 `0.307` 升到 `0.354`，
说明策略仍在提高“跟命令”的部分得分；但 `action_rate_l2`、髋位姿、关节/轮端
加速度和限位惩罚没有下降，反而略增，因此总分增长主要是跟踪项增加，不是稳定性
改善。`stand_wheel_velocity_l2` 是少数有改善的静止项，但幅度不足以改变行为。

正项末 500 轮为 `track_lin_vel_xy_exp=+0.955`、`track_ang_vel_z_exp=+0.354`、
`upward=+0.105`，合计约 `+1.414`；负项合计约 `-0.981`，每步净贡献约
`+0.433`，与 `mean_reward` 约 `8.5` 一致。`feet_contact_without_cmd=+0.00088`
仍然只是首次接触事件奖励，不是连续四轮支撑奖励，不能解释或修复静止乱动。

综合结论：本轮参数-only warm-start 没有达到预期。训练算法本身没有出现 NaN、
value loss 爆炸或策略噪声失控，但策略陷入“能获得一部分速度跟踪分，同时持续抖动、
靠近限位并频繁姿态失败”的局部解。`model_5999.pt` 可以保留作对照检查，不能标记
为优秀模型；本轮不建议继续延长迭代，下一轮应先增加固定命令下的机身速度/角速度、
四轮连续承载和逐项 termination mask 记录，再针对终止代价与零命令全身稳定性重新配参。

### LightHW 静止乱动根因复审与下一轮方案（2026-09-14）

本次复审重新读取了 route_01/route_02 的 TensorBoard、环境快照、奖励实现和
warm-start 加载路径。route_02 没有保存 `model_5999.pt` 的静止固定命令 CSV，
所以以下是“代码 + 训练轨迹 + route_01 固定回放”的高可信根因；route_02 静止
行为仍需用同一 checkpoint 做固定回放确认。

#### 根因排序

1. **静止奖励与目标错位（最高可信）**：`stand_still_without_cmd` 只惩罚腿部
   关节偏离默认角，`stand_wheel_velocity_l2` 只惩罚轮关节速度，
   `feet_contact_without_cmd` 只奖励首次接触事件。它们没有直接约束静止时的
   机身横滚/俯仰角速度、航向漂移、位置漂移和当前帧四轮持续承载。通用
   `track_lin_vel_xy_exp` 虽然对零命令有速度跟踪压力，但它是混合命令下的指数项，
   不能替代上述静止专用约束。
2. **姿态失败缺少即时训练代价（最高可信）**：route_02 的
   `rewards.is_terminated` 实际为 `null`。`bad_orientation_2` 主要只让回合提前
   结束，策略没有在失败步收到足够明确的负反馈；平均回合只有约 8.83 s，正负
   奖励的有效学习窗口进一步缩短。
3. **warm-start 噪声与轮速尺度不匹配（高可信）**：`warm_start_training()` 会
   完整加载旧策略及其动作噪声，配置里的 `init_noise_std=0.40` 不会重置旧噪声。
   route_02 末段 `mean_noise_std≈0.342`，按轮速 `scale=40` 折算为约
   `13.7 rad/s` 的一倍标准差探索量；这对零命令轮速精度非常不利。它不会直接
   证明确定性回放必然抖动，但会让策略在训练中更难形成零速平衡点。
4. **多目标样本竞争（高可信）**：route_02 同时使用 rough 地形和
   `35%/15%/10%/40%` 的 standing/low/mid/high 命令。高速跟踪正项在末 500 轮
   已升到 `+0.955`，但动作变化、限位、髋偏移和轮端加速度惩罚仍居前，说明优化
   更偏向“继续跟命令”，没有形成静止平衡解。12 s 命令保持时间减少了切换次数，
   但不能替代静止阶段课程。
5. **自动 reset 是观测混淆，不是抖动根因**：回放会在 `env.step()` 内自动复位，
   视觉上像“倒下后重新站起”；route_01 固定回放已经显示 reset 前仍存在真实的
   零命令乱动，因此不能用 reset 现象解释全部问题。

#### 推荐调整顺序

先保留 `stand_still_without_cmd=-3.5`、静态髋/膝/大腿项和 `track_lin_vel_xy_exp=2.6`，
不要继续无差别放大它们。下一轮应增加静止专用的机身和支撑反馈：

| 项目 | 建议初值 | 门控/用途 |
|---|---:|---|
| 零命令机身水平速度平方 | `-1.0 ~ -2.0` | `||cmd||<0.15`，直接压制静止漂移 |
| 零命令机身横滚/俯仰角速度平方 | `-0.3 ~ -0.6` | `||cmd||<0.15`，压制身体摇摆 |
| `stand_flat_orientation_l2` | `-0.4 ~ -0.8` | 只在零命令启用，避免影响高速姿态 |
| `stand_contact_force_uniformity` | `-0.3 ~ -0.6` | 当前帧四轮载荷均衡与最低承载 |
| `stand_rear_contact_force_balance_l2` | `-0.15 ~ -0.30` | 防止单侧后轮卸载 |
| `is_terminated` | `-100 ~ -200` | 只惩罚非 timeout 的姿态终止，先从 `-100` 试 |
| `feet_contact_without_cmd` | 保持 `+0.04` | 不再增加；它是首次接触事件，不是持续支撑率 |

其中前两项需要增加已有奖励接口的零命令门控版本；连续支撑项可复用当前
`stand_contact_force_uniformity`，不应把首次接触奖励当作支撑替代品。

参数上采用两阶段：阶段 A 在 flat/最低 rough 等级训练约 2,000--3,000 轮，命令
比例设为 standing `60%`、low `25%`、mid `10%`、high `5%`，high 样本固定为
`±3.5 m/s` 直行锚点；阶段 B 从阶段 A checkpoint 恢复到
`35%/15%/10%/40%`，逐步恢复 rough 随机化和高速样本。这样保留 3.5 m/s 学习通道，
又先让策略形成零命令平衡点。两个阶段都保留轮速 `scale=40`、仿真上限 `42 rad/s`，
不通过降低动作尺度掩盖高速能力问题。

warm-start 时应加入显式的策略噪声重置，建议阶段 A 设为 `0.12--0.18`，阶段 B
设为 `0.18--0.25`；仅修改配置中的 `init_noise_std` 对当前完整 warm-start 不生效。
命令保持时间可设为 `16--20 s`，push 在阶段 A 延迟到至少 `30 s` 或暂时关闭，
阶段 B 再恢复。复位姿态当前已经较窄，不需要继续收窄。

#### 验证条件

每阶段都必须保存独立的 `0 0 0`、`0.8 0 0`、`3.5 0 0` 回放 CSV，并在 reset 前
统计机身线/角速度、航向漂移、当前帧四轮 `Fz` 和逐项 termination mask。静止通过
条件仍为连续 20 s 无 reset、水平速度 `<0.05 m/s`、偏航速度 `<0.05 rad/s`、
航向漂移 `<5°`、四轮同时承载率 `>99%`；训练端 mean reward 只能作为同一 reward
口径内的筛选指标，不能替代这个验收。

### route_03_stage_a_static 启动前检查（2026-09-14）

阶段 A 已注册为 `Rough-Deeprobotics-LightHW-StageA-Static-v0`，v1 训练在约 1,100
轮提前停止，v2 仍使用同一任务入口和独立 run_name。
静态检查通过：新增奖励函数、环境配置、PPO 配置均通过 `py_compile`，训练/回放脚本
通过 `bash -n` 和 `git diff --check`。CPU 张量门控检查确认零命令项返回形状为
`(N,)`、数值有限且非负，均匀承载惩罚为 0、卸载时为正；权重使用负值。

阶段 A v1 的命令采样为 standing/low/mid/high=`60%/25%/10%/5%`；v2 调整为
`70%/20%/5%/5%`，low/mid
不会落入 `0.15` 静止门限，高速样本全部为 `±3.5 m/s` 且 lateral/yaw 为 0。由于
当前 shell 未激活含 Isaac Lab/Omniverse 的运行环境，本次未把静态检查冒充 GPU
仿真验证；正式训练时由训练命令完成完整 manager/sensor 初始化检查。

### route_03_stage_a_static 启动阻塞：NVIDIA 驱动 API mismatch（2026-09-14）

阶段 A 尚未进入任务加载。启动时的 `cudaGetDeviceCount()` / CUDA 804 来自系统
驱动状态，不是环境配置或奖励函数：用户态 `libcuda.so`、`nvidia-smi` 客户端和已安装
软件包为 `580.178.04`，但当前运行中的内核模块为 `580.173.02`。内核日志多次报告
`NVRM: API mismatch`，同时 `nvidia-smi` 无法连接驱动、`torch.cuda.is_available()`
为 `False`。

处理顺序：先退出 Isaac/训练进程并重启系统，让内核模块与已安装的 580.178.04
用户态组件同步；重启后必须先确认 `nvidia-smi` 正常、`torch.cuda.is_available()`
为 True，再重新执行阶段 A 命令。若重启后仍出现版本不一致，再按同一版本重新安装
`nvidia-driver-580`/`nvidia-dkms-580` 并重启；不要通过修改奖励、任务或设置 CPU
设备绕过该错误。

### route_03_stage_a_static v2 失败复盘与提前停止（2026-09-14）

v2 日志已经写到约 800 轮，当前没有活动训练进程；因此本轮按用户观察提前停止，
最新 `model_800.pt` 只作为回归对照，不作为可用模型。TensorBoard 的末段窗口显示：
mean reward 约 4.5--4.7，mean episode length 约 300 控制步（约 6 s），
`bad_orientation_2` 仍约 8.5；`is_terminated` 约 -0.14。与 v1 末段约 396 步、
`bad_orientation_2` 约 6.2 相比，v2 的平面地形和 70% 零命令没有带来生存改善。

这次曲线把根因范围进一步缩小了：

1. **抖动发生在“腿关节层”，不是机身姿态层。** v2 的
   `stand_flat_orientation_l2` 约 -0.0013、`stand_lin_vel_xy_l2` 约 -0.013，
   说明机身平均姿态/平移量已经很小；但回放仍然乱动，且
   `stand_still_without_cmd` 约 -0.081 没有变好。这是“围绕默认角小幅高频摆动”
   的典型特征：该项只看关节位置偏差的 L1，不看腿关节速度，也不看动作到执行器
   的高频变化，因此即使视觉抖动很大，奖励仍可接近 0。
2. **静止支撑项没有提供有效梯度。** `stand_contact_force_uniformity` 约 -0.0071、
   `stand_rear_contact_force_balance_l2` 约 -0.0041，v2 前后几乎不变；它们既没有
   把四轮持续承载变成强信号，也不能解释“姿态平但腿在动”。此前的
   `feet_contact_without_cmd` 仍是首次接触事件，不能当作持续支撑率。
3. **平面化不是根治手段。** v2 比 rough v1 的生存更差，证明主要瓶颈不在地形
   随机化。当前 LightHW 动作仍是腿位置 PD（`scale=0.16`）+ 轮速动作（Stage A
   继承 `scale=40`）；零命令时轮速探索和腿部 PD 接触回弹仍可产生高频极限环。
4. **v2 的强终止代价和奖励重构只放大了失败后果，没有增加可辨识的纠错信号。**
   `is_terminated=-200` 后终止均值仍约 -0.14，而新静止项几乎为平坦信号；这会
   让 actor 在旧高速策略基础上更快避开失败，却没有学到“怎样把四条腿安静地压住”。
   v2 使用的是 actor-only transfer，critic 为新初始化；这也使奖励口径突变后的
   早期价值估计更不稳定，但不是唯一根因。

下一轮先不要再调 command 比例或继续加大 `stand_still_without_cmd`。应先做一个
**可观测性/执行器对齐小实验**：固定 `0 0 0`，记录每条腿 `q`、`qdot`、动作、PD
力矩、四轮当前帧 `Fz` 和 termination mask。若确认“q 偏差小而 qdot/动作大”，再增加
零命令门控的腿关节速度（或动作幅度）惩罚，并把 `joint_acc_l2` 从当前极小量级提高
到可见但不压死探索的范围；同时将关节 PD damping 作为独立消融变量。若确认某轮持续
卸载，再只调接触最低力/载荷均衡，不要同时改全部奖励。

下一轮的停止/验收门槛保持不变：固定零命令连续 20 s 无 reset，水平速度和偏航角速度
均 <0.05，航向漂移 <5°，四轮同时承载率 >99%；没有这些回放证据时，任何 mean
reward 或“看起来没复位”都不记为静止学会。

### route_04_balanced_highspeed：M20 迁移复审后的改动（2026-09-14）

route_03 的失败说明，问题不是“静止样本还不够多”，而是 M20 的静止项在 LightHW
上观测错了对象：默认关节位置项不能识别高频腿部摆动，首次接触项也不是连续支撑率。
因此新路线保留 M20 的速度跟踪、姿态、限位、接触和动作平滑角色，但新增
`lighthw_stand_leg_velocity_l2`，只在 `||base_velocity|| < 0.15` 时惩罚腿关节
速度平方；运动命令下该项严格为零，不削弱 3.5 m/s 的腿部恢复动作。

本路线的命令比例为 standing/low/mid/high=`25%/20%/15%/40%`，高速层一半是精确
`±3.5 m/s` 直行锚点。与失败的 Stage A 相比，不再用 70% 零命令和 plane 地形掩盖
问题；rough 初始等级固定为 1，腿部 PD damping 保持 3.0，轮速动作保持
`scale=40`、仿真上限 `42 rad/s`。新增静止项初始权重为 `-0.01`，终止代价从
`-200` 降为 `-100`，避免终止梯度压过可学习的动作信号。

训练建议从 route_00 高速候选 `model_9500.pt` 做 actor-only transfer；route_03
`model_800.pt` 已标记为失败对照，不作为初始化。该路线必须分别固定回放
`0 0 0`、`0.8 0 0`、`3.5 0 0`，并记录腿部 `q/qdot`、动作、PD 力矩、四轮 `Fz`
和 termination mask。只有静止连续 20 s 无 reset、水平/偏航速度均 `<0.05`、
航向漂移 `<5°`、四轮同时承载率 `>99%`，以及高速达到 `v_x≥3.3 m/s` 时，才算
同时保留高速能力和零命令稳定性。

启动修复：父类 `disable_zero_weight_rewards()` 会把通用静止机身项置为 `None`；
route_04 已在子类中显式重建 `stand_lin_vel_xy_l2`、`stand_ang_vel_xy_l2` 和
`stand_flat_orientation_l2` 后再赋权，配置已重新通过 `py_compile` 与
`git diff --check`。

回放一致性修复：发现旧模型回放默认使用当前 `Rough-Deeprobotics-LightHW-v0`，而当前
配置已改变 `rel_standing_envs`、静止奖励权重、reset roll/pitch 范围等参数。现新增
`Rough-Deeprobotics-LightHW-HistoricalHighSpeedPlayback-v0`，按高速 run
`2026-09-08_21-23-28/params/env.yaml` 重建命令、reset 和奖励参数，专门用于复核
`model_9500.pt`；普通 `play_rough.sh` 不再作为该旧模型的严格对照入口。

历史回放兼容修复：历史 run 不包含后续新增的零命令机身/接触奖励；回放配置现显式将
这些新增项设为 `None`，避免空 body regex 被 Isaac Lab 传感器解析器拒绝。
随后发现旧 run 的 `wheel_vel_penalty` 也为 `null`；历史配置已显式关闭并在参数恢复
后统一清理零权重占位项，避免空 joint regex 继续触发解析错误。

### 旧高速模型回放前后差异复审（2026-09-14）

当前回放仍出现零命令乱动/重置，不能直接归因于 checkpoint 退化。与训练时的环境和
旧回放方式相比，现有 `play.py` 有以下会改变测试语义的差异：

1. `--fixed_command 0 0 0` 是新的确定性零命令测试；旧命令不带该参数时，命令管理器
   每 8 s 随机重采样，模型并没有被真正固定在零命令上。训练日志里零命令样本仅约 5%，
   因此“旧回放看起来稳定”不能证明零命令保持已学会。
2. 回放统一把 `terrain.max_init_terrain_level` 改为 `None`，并将 terrain generator
   缩成 5x5；训练快照是 generator、10x20、`max_init_terrain_level=3`。这会改变初始
   地形行/难度，固定零命令时可能直接落在更难地形上。
3. 固定命令模式把 `time_out` 设为 `None`，所以姿态终止后会持续自动 reset；视觉上的
   “一直重置”被放大，但真正触发 reset 的仍是 `bad_orientation_2`。
4. `play.py` 关闭策略观测 corruption，而训练快照 `enable_corruption=true`。这通常
   是合理的确定性推理设置，但它不是训练分布的完全复现，需在 A/B 对照中保持一致。

因此目前最可信的解释是：固定零命令首次暴露了模型原本没有学好的静止行为，同时回放
又使用了不同的地形采样；不是 `fixed_command` 数值本身把一个已稳定策略“改坏”。
下一步应先做同一 checkpoint 的三组对照：历史参数+随机命令、历史参数+固定零命令、
历史参数+固定零命令但固定 plane。只有第二组失败而第一组成功，才是静止能力缺失；
若第三组成功而第二组失败，主因是回放地形差异；若三组都失败，再继续检查动作/URDF
和 reset 状态。

### 三组回放结果复核（2026-09-14）

用户实际观察为：第一组（历史参数、随机命令、rough）稳定；第二组（同一模型、
`--fixed_command 0 0 0`、rough）乱动并重置；第三组（同一模型、固定零命令、plane）
仍乱动并重置。该结果排除了 rough 地形是唯一原因，但还不能把根因唯一归结为策略的
静止能力：第二/三组同时改变了命令观测注入、单环境设置和 timeout 语义。

另外，当前 `play.py` 误将 `events.push_robot` 置空；LightHW 配置真正的事件名是
`events.randomize_push_robot`，所以回放中的随机推力实际上未关闭。固定单环境下这会
更容易暴露摔倒和自动 reset，尤其不能用来判断“无外力静止能力”。下一次对照应使用
原生命令生成器强制 standing，而不是替换 `policy.velocity_commands`，并显式将
`env.events.randomize_push_robot=null`；在该对照前不改奖励、不重训。

### 原生命令对照启动诊断（2026-09-14）

在当前主机复现该命令时，Isaac Sim 已完成场景和观测管理器创建，但主机的
`nvidia-smi` 报告 `NVIDIA driver is not loaded`，随后默认 `cuda:0` 在加载模型时失败。
使用 `--device cpu` 后，环境可创建到模型加载阶段，但 checkpoint 内部张量仍标记为
CUDA，当前 `play.py` 的 `ppo_runner.load()` 没有 `map_location`，因此 CPU 回放仍会失败。
这属于运行时 GPU/CPU 加载问题，不是策略或奖励结论；恢复 NVIDIA 驱动后应使用 CUDA
回放。CPU 仅作为临时启动诊断，不能替代 PhysX/GPU 行为验收。
