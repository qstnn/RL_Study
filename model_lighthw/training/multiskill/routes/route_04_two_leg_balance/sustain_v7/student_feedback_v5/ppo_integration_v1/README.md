# R4a 结构化学生的隔离 PPO 入口（待真实 GPU 短验）

此入口只使用原 `SustainBalanceEnv`/RSL wrapper、原物理/奖励/305维原观测及88维 Critic；部署 Actor 输入是原305＋因果记忆2＋捕获触发前动作16＋本体 FK2＝325。起身前缀冻结，捕获反馈头由 PPO 更新。环境观测噪声、reset 随机化关闭；PPO 采样标准差显式设为0.0005。`stop_on_observation=False`，iteration 结束不调用环境 reset；只有实际物理失败/边界或技术时限触发原环境自动 reset。跨进程续训恢复模型、Adam、学习率、计数和 RNG，但从新的物理回合开始，不能拼接两次站立时间。

所有 GPU 命令先复核**指定** `verification.json`、`request/result/session`、当前 Actor SHA、六个 seed43/44/45×双槽教师全关额外10秒及完整轨迹独立重建。该六例尚未完成时，入口在启动 Isaac Sim 前拒绝；CPU 合成测试和源封存不算机器人 PPO 验收。

```bash
PPO_DIR=/home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/ppo_integration_v1
STUDENT_DIR=/home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/structured_v1
ACTOR="$STUDENT_DIR/runs/2026-10-23_full_01/actor.pt"
SIX="$STUDENT_DIR/runs/2026-10-23_full_01_six/verification.json"
bash "$PPO_DIR/ppo.sh" check --actor "$ACTOR" --six-case-verification "$SIX"
```

在上述命令通过后，先由协作线程审查并按有限预算运行短验；不自动开启正式训练。两槽 smoke 默认为512控制步/槽×2次更新、20秒技术时限，用于真实 timeout 的**重置前** Critic 价值只补偿一次，并记录物理终止不补偿、1kHz轨迹、PPO/Adam和iteration间不reset：

```bash
bash "$PPO_DIR/ppo.sh" smoke --actor "$ACTOR" --six-case-verification "$SIX" \
  --num-envs 2 --output "$PPO_DIR/runs/SMOKE_N2_UNIQUE"
```

完成两槽检查后，正式准入还需真实16槽的同配置 smoke 和**新进程**一次更新续训；64槽为可选扩展，不替代16槽证据：

```bash
bash "$PPO_DIR/ppo.sh" smoke --actor "$ACTOR" --six-case-verification "$SIX" \
  --num-envs 16 --output "$PPO_DIR/runs/SMOKE_N16_UNIQUE"
bash "$PPO_DIR/ppo.sh" resume-smoke --actor "$ACTOR" --six-case-verification "$SIX" \
  --num-envs 16 --smoke-run "$PPO_DIR/runs/SMOKE_N16_UNIQUE" \
  --output "$PPO_DIR/runs/RESUME_N16_UNIQUE"
```

20秒时限仅用于强制真实 timeout 短验。正式配置使用16槽、32控制步/槽每次更新、360秒技术时限。配置差异须先做有限的 `formal-smoke`：从已核查的16槽 timeout smoke checkpoint 显式迁移**仅** rollout 长度512→32，保留模型、Adam、RNG、计数并核对 SHA，另开新物理回合做32次更新（20.48秒），保存完整16槽1kHz轨迹。末态每槽自主捕获后额外10秒、无中断只是行为候选，必须独立审查速度、接触、高频和完整轨迹后才能批准正式训练。

短验 `physics_trace.npz` 逐1ms/逐槽封存世界系COM、四轮中心（`wheel_pos_w`按原四轮顺序展平12列）、出生/保持原点、机身高度、COM速度、角速度、重力方向、目标重力、四轮法向力/净空、12腿角、4轮速、机身碰撞力、扭矩峰值、阶段/模式/起身开始时刻/站立命令，以及原状态机累计量和实际动作。审查可从这些**原始列**独立重算支撑线、净空、姿态、速度与四项1kHz高通RMS；预存的 `line_error_m`、`clearance_m`、`jitter_rms` 只作为运行时读回对照，不能代替复算。

`wheel_pos_w` 对应原规格的 FR、FL、RR、RL；`com_w` 是质量加权的实际物理 COM。`target_gravity` 和 `phase` 是该 1ms tick 状态更新后的 `command(t)` 读回；首次设置 `start_time_s` 的单个 tick，原状态机用于本 tick 的参考是更新前值，独立复算须按 `start_time_s` 处理这一拍因果差异。

```bash
bash "$PPO_DIR/ppo.sh" formal-smoke --actor "$ACTOR" --six-case-verification "$SIX" \
  --num-envs 16 --smoke-run "$PPO_DIR/runs/SMOKE_N16_UNIQUE" \
  --output "$PPO_DIR/runs/FORMAL_SMOKE_N16_UNIQUE"
```

正式入口还要求独立审查生成的 `review.json`，其中须绑定16槽 timeout smoke、1更新续训、32×32/360秒短验的 `result.json` SHA、两份1kHz轨迹 SHA，且 `technical_review_pass`、`behavior_review_pass` 均为 true。`behavior_cases` 要逐槽列出0–15的 `extra10s_pass`、`velocity_pass`、`support_pass`、`high_frequency_pass`、`uninterrupted`，全部真实通过。审查可由当前协作线程基于实际证据完成，不是新的用户审批。进入正式训练后只记录实时采样和每100更新/末尾 checkpoint，**不自动做冻结评估**；`--max-updates` 是本进程新增预算，例如经审查决定1000时：

```bash
bash "$PPO_DIR/ppo.sh" train --actor "$ACTOR" --six-case-verification "$SIX" \
  --review "$PPO_DIR/reviews/REVIEWED_UNIQUE/review.json" \
  --max-updates 1000 --num-envs 16 --output "$PPO_DIR/runs/TRAIN_UNIQUE"
```

每个运行目录封存原R4a/学生源和资产、本入口所有新源、通用PPO驱动、本机RSL PPO/ActorCritic/storage及Isaac wrapper/DirectRLEnv代码，保存 SHA 清单。当前源码或依赖变化会让旧证据失效。结果中的 `technical_pass` 仅表示实际更新/timeout/恢复接口工作，`behavior_candidate_pass` 也不等于独立机器人验收；正式训练的行为结果仍待训练结束后用户通知再审查。

CPU 软件检查（**无 Isaac Sim、非机器人行为证据**）：

```bash
/home/user/miniconda3/envs/lab/bin/python -m unittest discover \
  -s "$PPO_DIR/tests" -v
bash -n "$PPO_DIR/ppo.sh"
```
