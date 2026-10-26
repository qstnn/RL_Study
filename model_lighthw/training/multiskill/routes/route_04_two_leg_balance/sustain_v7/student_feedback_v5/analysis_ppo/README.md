# R4a PPO 1 kHz 独立物理审查

`review_physics.py` 只读已完成的 `ppo_integration_v1/runs/<run>/physics_trace.npz`、运行 JSON、PPO 更新日志及 checkpoint，不启动仿真，也不导入环境状态机。该目录未进入 PPO 执行入口的源码清单。首次 GPU 运行前可用以下命令验证脚本本身：

```bash
/home/user/miniconda3/envs/lab/bin/python -m unittest discover -s /home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/analysis_ppo/tests -v
```

GPU smoke 完成后，从本目录运行：

```bash
/home/user/miniconda3/envs/lab/bin/python review_physics.py \
  ../ppo_integration_v1/runs/<formal-smoke-run> \
  --output results/<formal-smoke-run>_physics_review.json
```

输出逐槽、逐物理回合记录。物理回合按每槽 `physics_time_s` 回退分段，检查每段从 1 ms 开始、无漏采样。独立重算 rear RR/RL 有限支撑线、前轮净空及接触门、COM 速度、根角速度、目标姿态、机身碰撞、摔倒/地面边界、5 Hz 二阶高通及 0.1 s 指数 RMS 的四项门。根据真正的 1 kHz 序列重建 0.5 s 四轮入场、局部 3 s 起身、连续 2 s capture、首次中断永久锁存和额外连续 10 s。所有阈值采用 v7 原值；记录值 `line_error_m`、`clearance_m`、`jitter_rms` 和状态标志只用于独立结果对照，不能替代原始物理量来放行。

`formal_smoke_16_slot_five_field_gate` 仅在 16 槽各自全程保持一个未重置物理回合，且末段五项均通过时才为真：连续 2 s capture、capture 后连续 10 s、没有首中断、没有失败/边界、末 tick 仍满足全部稳定门。状态机或实际综合物理门与独立重算不一致会使该槽失败。高频四个单项的浮点跨阈值差异逐帧保留在 `hf_limit_disagreement_details`，但若该帧综合平滑门两侧均为失败，不能据此伪造综合门或捕获状态差异；离线物理门始终采用原阈值，没有容差放行。`behavior_cases` 给出每槽供后续 `review.json` 采用的摘要；20 s timeout smoke 选达到最长 capture 后观察的已完成回合，formal-smoke 只选末回合并要求未重置。脚本还核对真实 Critic Adam step、每步一次特征推进、前缀冻结、Actor 系数变化、checkpoint 继承和截断末值只补偿一次的更新日志，并将运行来源清单 SHA 写入报告。

本 JSON 供后续独立证据审查据 1 kHz 轨迹形成 PPO `review.json`；本脚本不自行作正式准入决定。正式 `train` 模式没有保存完整 1 kHz trace，因此此脚本用于 smoke、resume-smoke 和 formal-smoke。旧 trace 若缺原始 COM、轮位置、base_z、腿角或轮速会直接拒绝审查。
