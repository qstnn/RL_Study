# R1 V5：完整任务循环与受控PPO

**当前状态（2026-10-19 18:42）：协同控制验证完成，重拖行进仍未准入正式训练。** 6批GPU、78条任务/588780个1kHz时刻；39次移动均未完成1m＋停车，最大正进度4.56cm。低增益IMU姿态反馈静止带噪3/3通过，但轮载分配与腿反作用前馈组合仍不稳定；扩大补偿未解决。原Actor、生产控制及96/48Nm上限保持，正式PPO新增0。[完整验证、数值敏感性说明及下一步](results/2026-10-19_coordinated_towing/README.md)。

**前次样本补采状态（2026-10-19 17:55）：可靠样本和控制改动已完成本轮验证，正式训练仍未就绪。** 三个观测／动作噪声流筛得原 Actor 13 工况、36,294 帧、51 组成功参考；另有 +25°／40 kg 静止教师示范 3,000 帧，3/3 完整通过。重拖行进仍未完成 1 m；辅助控制最大正向进度约 2.60 cm。单 Actor 离线拟合候选未通过完整保持验证，未替换原模型。[本轮数据、控制及重拖方案](results/2026-10-19_towing_recovery/README.md)。

新增拖车全阶段轮载裕量奖励、参考组等权抽样与最大组误差约束、受保护工况／阶段覆盖检查及教师数据误用拦截。81 项测试＋11 子测试通过，336 条冻结协议复算；其中错误反馈探针和未实际生效的高精度对照已注明。新增正式／GPU 集成 PPO 均 0，不能把静态教师的成功当成训练后的单 Actor 能力。默认 `readiness.json` 已设为未就绪，旧启动命令会明确给出缺口。

原 `selected/model_init.pt` 与旧缓存、历史训练和评估均保留。正式训练日期仍为 2026-10-19，最新 200 更新 run 为 `2026-10-19_15-00-24.955621`，固定验收 15→14→14/16；重置采样修复有效，重拖静止退化触发停止。[200 更新审查](results/2026-10-19_post_reset_200_review/README.md) · [重置修复](results/2026-10-19_reset_fix/README.md) · [原准备报告](results/preparation/README.md)。

## 任务与实验条件

全部64个许可工况从第一个训练batch混合，最终部署一个 **300输入→16输出Actor**。输入保留5帧历史、IMU/编码器、指令及已知名义载荷信息；真实线速度、地形、接触和停车位置仅供奖励、Critic或诊断。

- 15 kg机身负载与40 kg拖车互斥。负载行进需走满1 m并稳定停车，无速度最低要求。
- 至少3 m/s仅要求空载。−25°下坡保留空载速度和15 kg负载任务，不安排拖车。
- 轮速目标上限 **±50 rad/s**；轮关节数值速度上限 **60 rad/s**；四轮刚体角速度上限 **4000 deg/s**，其余刚体1000 deg/s。
- 高附着实验：摩擦系数 **4.0**、滚阻系数 **0.02**、名义牵引点离坡面 **0.07 m**。腿/轮输出轴力矩上限仍为 **96/48 N·m**；这些条件不等于实机或电机额定能力验收。

每轮移动任务先零指令3秒。负载达到1 m净行进距离后减速；空载行进目标保持至本轮第11秒后减速。**实际指令归零后，完整经过2秒过渡＋5秒保持，才采下一任务**；静止任务持续20秒。episode上限65秒，物理1 kHz、策略50 Hz。起步前漂移不计入行进距离，负载进度奖励使用固定0.2 m/s平滑尺度，不除随机参考速度。

## 默认训练配置

| 项目 | 设置 |
|---|---|
| 环境与总预算 | 1024环境、1000轮更新 |
| 预热 | 前50轮仅更新Critic；余950轮允许Actor更新 |
| Critic迁移 | 252输入；从V4迁移时保留前242列，新增10列权重置零；新建Adam |
| 新增Critic状态 | 停车沿线/横向位移、保持计时、任务目标、阶段、阶段计时、剩余距离 |
| PPO采样 | 每环境64步；`gamma=.995`，`lambda=.95`；3 epochs×4 minibatches |
| 学习率 | Actor固定`1e-5`；Critic固定`1e-4` |
| 已测训练扰动 | 原观测噪声幅值×0.1；raw动作初始/最大标准差：腿0.005、轮0.01；已做64项噪声模式回放，停车和重拖仍有失败 |
| 成功行为保持 | 绑定SHA的 Actor成功缓存；工况／阶段组等权，辅助损失0.1，最大组归一化MSE≤0.02；完整保护阶段必须覆盖，教师标签拒绝直接安装 |
| Actor提案保护 | 全rollout KL≤0.01且全部参考组MSE≤0.02；回退后仍不满足则拒绝Actor/Adam更新，Critic继续 |
| 冻结评估 | 开始前、每100轮及末尾逐工况检查；同一受保护工况连续两次退化则停止 |

以上为代码中的待重新准入配置，本轮尚无新GPU PPO集成。拖车支撑裕量奖励权重−3，移动和停车均激活；实际收益尚待训练验证。950轮是允许Actor更新的预算，实际接受的优化步数以日志为准。停车奖励、碰撞和牵引净高保护保留；默认腿姿奖励仅在平地停车激活，上坡允许相对坡面前倾。

## 终端操作

以下为此前 200 更新运行的历史启动命令，保留用于追溯；当前默认入口会明确拒绝未就绪状态，请先阅读本轮报告中的剩余验收要求。它从校准初始化创建新run和新Adam，检查修复准入及初始化/源码SHA：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
cd /home/user/RL_lab/rl_training
export TERM=xterm
python -u model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v5/scripts/train.py \
  --headless --num_envs 1024 --max_iterations 1000 --eval_interval 100 --seed 42
```

启动时会先执行16工况冻结验收，看到`Baseline before further updates`后，主终端可能暂时没有新输出。2026-10-19正式启动实测该阶段约2分19秒，随后自动进入训练；耗时随机器负载变化。子任务日志在`periodic_evaluations/update_0000_launcher.log`和`periodic_evaluations/update_0000/console.log`，完成结果在该目录`evaluation.json`。主日志出现`Learning iteration`、`training/optimizer_metrics.csv`开始增加，才表示开始训练更新。前50轮仅Critic更新是预设预热。被中断的旧run可能残留`status=running`，判断是否活跃应结合实际进程和最新日志。

本命令创建新run；不通过准入检查时入口会拒绝启动。`--initialize_from`用于已知V4/V5网络迁移并新建Adam；`--resume`仅用于配置签名一致的V5模型，恢复网络、Adam、优化阶段及随机数状态，仿真状态会重新初始化。正式负载行进距离固定1 m。

下面仅为回放接口示例，要求模型与当前契约／配置一致；旧契约模型须使用对应归档源码，不能直接作为当前精确续训模型。以下参数表示15 kg上坡行进：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v5/scripts/play.py \
  --checkpoint /本次run的绝对路径/quality/best_eligible.pt \
  --slope_deg 25 --payload_kg 15 --towed_kg 0 --speed 0.3
```

默认World自由视角，脚本只设置首次构图，不在循环中锁定镜头；需要跟随时在Isaac Sim窗口自行设置。使用原模型记录的摩擦、力矩、牵引点等参数，入口会检查签名。

## 配置、奖励与输出位置

- [环境及奖励权重](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/env_cfg.py)、[奖励和Critic输入函数](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/mdp.py)。
- [完整任务状态机](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/commands.py)、[PPO参数](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/ppo_cfg.py)、[PPO实现](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/ppo.py)。
- [冻结验收协议](scripts/protocol.py)、[逐工况质量检查](scripts/quality.py)。

每次运行写入`runs/<时间>/`：

| 路径 | 用途 |
|---|---|
| `request.json`、`params/env.yaml`、`params/agent.yaml` | 启动参数和实际实例化配置 |
| `source_snapshot/`、`source_manifest.json` | 本次运行源码及SHA |
| `console.log`、`run.json`、`training_result.json` | 控制台、进程状态、实际更新数和输出选择 |
| `training/model_*.pt` | 网络、优化器和训练状态 |
| `training/condition_metrics.csv` | 分工况/分阶段状态与完整任务循环计数 |
| `training/optimizer_metrics.csv` | KL、回退/拒绝、实际优化步数、Critic和探索统计 |
| `training/motor_metrics.csv` | 1 kHz采样汇总的实际力矩与限幅统计 |
| `periodic_evaluations/` | 各次冻结验收JSON、50 Hz CSV和1 kHz力矩轨迹 |
| `quality/best_eligible.pt`、`quality/quality_state.json` | 保住受保护工况的最佳候选及保护集合 |
| `exported/actor.pt` | 仅当前末次模型通过受保护工况检查时生成的单Actor导出 |

`cycle_complete_events`等列表示完整协议次数，**不代表行为通过数或存活率**；`*_total`是本环境实例累计数，`*_episode_mean`是采样状态中的episode计数均值。存活时长、提前终止原因和完整任务通过情况，应查看冻结评估原始轨迹及逐项检查，不能用训练平均episode时长除以上限65秒估算存活率。`best_eligible`也不自动表示16项全部达标。

查看TensorBoard：

```bash
tensorboard --logdir model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v5/runs --port 6006
```
