# 已执行教师动作的精确重放对照

2026-10-23 新增 `open_loop_teacher_probe.py`，共用 `student_worker.run_simulation` 的显式诊断分支。它只回答：同一套普通初态、GPU16/4、μ3物理执行通道，逐帧施加成功教师已经执行过的动作，能否复现完整任务。没有 Actor 推理、在线教师反馈、状态写入、首帧修正、插值或恢复拟合。

此前学生首帧误差缩小仍未自主通过，因此本对照可区分动作近似误差与名义轨迹本身的复现问题。重放成功不是自主策略成功；重放失败也不能直接推出重拖物理不可行或 PPO 无法学习。应继续比较相同索引下的初态、第一处分叉、1 kHz 接触/速度/外力与实际腿轮目标。

## 来源与执行

App 前调用原 `read_teacher_bundle`，独立重审普通GPU上下坡完整成功教师的物理、源快照、原始观测/动作及完整1 m、滚动、减速、2+5秒。两方向物理 profile 必须一致，选定坡度只能对应一个源回合。本轮使用 `ordinary_gain600_gpu` 与 `ordinary_downroll040_gpu`；每条原始 `executed_action` 均为 float32 `[2000,16]`，原时间网格为 `0.02 × frame_index`。

保留原绝对动作索引和原 float32 字节，不增加网络输入或加载网络。共享物理通道沿用相应任务固定准备时钟：上坡3.00秒、下坡7.48秒。机器人、拖车、μ3、GPU16/4、原80/3腿伺服与96/48 Nm限矩、牵引几何、普通零速 reset 均复用原学生实现。原动作 manager 的缩放和限幅仍生效，保存序列输入、manager输入、腿轮目标及逐项映射误差。

新增模块只在显式 `r1_open_loop_teacher_sequence_v1` 分支加载；默认学生路径继续调用原 `policy(observation)`。没有把序列包装成假 Actor。实际记录仍使用原 `recorded_wheel_action_type` / `attach_recorders`，包含原生1 kHz CoP、牵引子步前状态、终止前指标和电机伺服估计；最终仍运行未改门槛的 `real_cart_audit`。

## 结果身份

- `artifact_kind=diagnostic_open_loop_teacher_action_replay`，`open_loop_replay=true`。
- `actor_calls=0`，`teacher_calls=0`；真实序列消耗量单列为 `sequence_calls`。
- `open_loop_complete_protocol_pass` 保留真实完整行为结果；`autonomous_anchor_pass`、`autonomous_behavior_accepted`、`formal_ppo_authorized` 始终 false。
- `actor_inference=disabled_open_loop`，`training_labels_allowed=false`。现有普通教师读取、恢复读取和自主准入入口均拒绝此身份，不能把重放结果当新的成功教师或恢复标签。
- 输出 `open_loop_source_manifest.json`、`open_loop_behavior_review.json` 及标准原始 CSV/三组 NPZ；序列动作字段是 `recorded_teacher_action`，不会声称它是 `actor_action`。

当前默认学生执行仍冻结176源，诊断增加自身模块，共177源。修改共享worker会改变后续执行 SHA，历史自主回放保留原快照，不重签。原 BC10、constrained14和研究 scripts 源未修改；模型文件保持不变。

## 验证与待调度命令

52项CPU测试通过，覆盖 float32 字节/负零保持、50 Hz索引、跳帧/重复/时钟错误拒绝、两类源污染拒绝、无Actor对象、明确共享分支及三类数据/准入拒绝。仅测试与实际教师源的只读预检，没有启动仿真、Actor推理或优化器。预检证据位于持续目录 `admission_inventory/open_loop_teacher_probe_precheck.json`。

下面是待主线程调度的诊断命令，不是训练命令：

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
source /home/user/RL_lab/isaac-sim/setup_conda_env.sh
R1_RESEARCH=model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/research_v7
R1_CURRENT="$R1_RESEARCH/results/2026-10-23_admission_continuation"
python "$R1_RESEARCH/migration_v1/open_loop_teacher_probe.py" \
  --teacher-runs "$R1_CURRENT/ordinary_gain600_gpu" "$R1_CURRENT/ordinary_downroll040_gpu" \
  --slope-deg 25 \
  --output "$R1_CURRENT/open_loop_teacher_up_gpu"
```

另一方向必须选择 `--slope-deg -20` 和新的独立输出目录；其7.48秒起步从真实源自动读取，不接受人为提前时钟。诊断结束前没有任何复现成功结论。
