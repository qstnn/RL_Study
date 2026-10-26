# R4a 正式 PPO 策略的独立自主评估

本入口只评估已保存的 PPO checkpoint；**不会训练、恢复优化器、抽样 log_std 或构造教师**。实际策略输入保持 raw v7 305 维，加因果记忆 2、触发前实际动作 16、本体几何/滚动估计 2，共 325 维；动作是完整 PPO checkpoint 中 `actor.*` 的确定性均值。BC Actor 只用于核对同一网络拓扑和冻结起身前缀，PPO 权重严格从 checkpoint 加载，结果类型为 `ppo_checkpoint_mean_autonomous_evaluation`，不会伪装成 `structured_feedback_bc`。

入口先核对训练 run 的请求、结果、原源快照/依赖清单、最终 checkpoint SHA、原 BC 来源 SHA 与 325/88/16 合同，然后在新目录封存自身及原物理来源。仿真沿用 `worker_sustain.setup`：两槽从四轮初态、1 kHz 物理/50 Hz 控制、无环境噪声和 reset 随机化、站立有效、不自动返回；到捕获后所选连续观察时长才由 `observation_complete` 结束，最长技术时限为 hold+15 秒。1 kHz 轨迹按 20,000 tick 有界 GPU 缓冲转 CPU，避免 300 秒实验逐帧留存 Python/GPU 小张量对象。

独立 `verify_eval.py` 复用原 `verify_sustain.analyse` 从原始物理数据重算四轮入场、后轮支撑、COM 速度≤0.12m/s、前轮接触/净空、姿态、角速度、真实 HF、连续 2 秒捕获、首次中断永久锁存以及额外 10/60/300 秒。另以独立 NumPy/URDF 重算本体反馈和因果记忆、冻结前缀/结构化反馈公式、Actor 均值与实际执行动作，核对教师与接管帧均为零。`result.json` 的 `observation_passes` 只是环境记录；以 `verification.json` 的每例物理复算为行为证据。运行中 `progress.json` 每 10 仿真秒原子更新当前 seed、控制步、任务时钟、捕获后持续时间、首次失稳和活动槽位，便于观察长评估。固定初态和三个 seed 不代表扰动或硬件鲁棒性。

先做两槽10秒screen。以下为本轮实际运行命令；该目录已有记录，复测时须将`--output`改成新的目录，入口拒绝覆盖原结果：

```bash
cd /home/user/RL_lab/rl_training
R4_BASE=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5
R4_TRAIN="$R4_BASE/ppo_integration_v1/runs/2026-10-23_15-13-34_formal"
bash "$R4_BASE/evaluation_ppo_v1/eval.sh" evaluate \
  --checkpoint "$R4_TRAIN/checkpoints/model_001034.pt" \
  --training-run "$R4_TRAIN" \
  --output "$R4_BASE/evaluation_ppo_v1/runs/2026-10-23_final_screen10" \
  --seeds 43 --hold-seconds 10
```

本轮screen通过后，分别以`--seeds 43 --hold-seconds 60`及`--seeds 43 --hold-seconds 300`运行`2026-10-23_final_repeat60`和`2026-10-23_final_endurance300`，每种时长两槽。[本轮实际结果与状态](../results/2026-10-23_formal_1000_review/README.md)。入口也支持`--seeds 43 44 45`做重复，但无随机化时不能将不同seed称为不同扰动覆盖；未运行的回合不计入结果。

每次评估独立从四轮初态开始，不能将训练中已保持的时长接续计算。`progress.json`的`seed_completed`是在环境结束后写入，部分状态已因自动reset归零；最终连续保持时间以`verification.json`为准。测试结束后的reset不计入有效轨迹，也不是策略主动返回。单次结果的`robot_behavior_accepted=false`保留“未自动晋级”语义，整体验收由独立结果审查另行记录。`eval.sh verify <run>`可在CPU上重新执行完整验证，不会启动GPU。

CPU 接口测试：

```bash
cd /home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/evaluation_ppo_v1
PYTHONDONTWRITEBYTECODE=1 /home/user/miniconda3/envs/lab/bin/python -m unittest discover -s tests -v
```
