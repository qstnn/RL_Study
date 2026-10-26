# R4a 正式 PPO 已完成：启动记录与审查

更新：2026-10-23 18:04（Asia/Shanghai）。训练日期2026-10-23。**本轮1000次新增更新与结果审查均已完成，最终global1034；冻结模型固定条件额外10/60/300秒各2/2通过，当前无需再次启动训练。** [本次完整审查](results/2026-10-23_formal_1000_review/README.md) · [最终证据绑定](results/2026-10-23_formal_1000_review/review.json)。 [历史准入验证报告](VALIDATION.md) · [绑定结果与轨迹的准入审查](ppo_integration_v1/reviews/2026-10-23_admission/review.json) · [只读启动预检](ppo_integration_v1/reviews/2026-10-23_admission/formal_preflight.json)。

## 查看已完成的正式训练记录

正式进程已于2026-10-23 16:38:38完成。原终端静默的问题已核实为进度写入文件；以下只读命令现在显示 completed、local1000、global1034：

```bash
cd /home/user/RL_lab/rl_training
R4_BASE=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5
/home/user/miniconda3/envs/lab/bin/python "$R4_BASE/monitoring/monitor.py" \
  --run "$R4_BASE/ppo_integration_v1/runs/2026-10-23_15-13-34_formal"
```

默认每5秒刷新，Ctrl+C只退出监控。首次启动已完成106轮后被中断，本次完成的run重新从global34开始，两次计数不合并。[诊断与监控说明](monitoring/README.md)。

## 本轮历史启动命令：当前无需再次执行

```bash
cd /home/user/RL_lab/rl_training
R4_BASE=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5
bash "$R4_BASE/ppo_integration_v1/ppo.sh" train \
  --actor "$R4_BASE/structured_v1/runs/2026-10-23_full_01/actor.pt" \
  --six-case-verification "$R4_BASE/structured_v1/runs/2026-10-23_full_01_six/verification.json" \
  --review "$R4_BASE/ppo_integration_v1/reviews/2026-10-23_admission/review.json" \
  --num-envs 16 --max-updates 1000 \
  --output "$R4_BASE/ppo_integration_v1/runs/$(date +%Y-%m-%d_%H-%M-%S)_formal"
```

脚本自动设置本机lab Python和Isaac Sim运行环境，无需另外激活。输出目录必须是新目录，时间戳避免覆盖。该命令已完成参数、六例独立重建、源合同及正式审查预检；用户已据此完成本轮正式长训练。

它先验证BC来源和六例，随后从审查绑定的 **model_000034.pt** 恢复实际PPO模型、Critic、Adam、学习率、计数与RNG。不会重新从BC权重开始正式学习。新增1000次更新后预期global1034、Adam2068；新进程物理回合重新从四轮初态开始，不能与短验站立时间拼接。

| 设置 | 本轮固定值与含义 |
|---|---|
| 槽位与批次 | 16槽，每更新32控制步/槽，新增512000个环境转移；每槽累计640秒采样 |
| 持续任务 | 50Hz控制、1kHz物理；360秒仅为技术截断，iteration/保存/捕获后观察完成均不reset，不自动返回 |
| 环境条件 | 原URDF/质量/惯量/腿48Nm与轮24Nm；无观测噪声、初态/域随机化 |
| 学习部分 | 冻结起身前缀，更新26个有效反馈系数、Critic和策略标准差；Actor325/Critic88/动作16 |
| PPO | 初始采样std0.0005、学习率3e−6、每更新2个Adam步、gamma0.995、lambda0.95、clip0.2；恢复时沿用已保存值 |
| 奖励 | 原v7持续积分奖励不变；2秒捕获和观察截止不发完成大奖；≤0.12m/s慢漂移及真实HF门保留 |
| 保存与审查 | 每新增100更新及末尾保存，只记录在线训练指标；不做每100冻结评估或末尾自动评估，用户完成后通知审查 |

已完成在线回合/奖励分项审查及最终冻结策略的10/60/300秒独立实测：300秒组两槽连续300.008/300.001秒，无首次失稳、碰撞或高频超门；共783020个原1kHz样本独立复核。最终模型已列为固定条件长稳基线。原BC60秒与PPO60秒同条件比较有改善也有轻微退化，不能宣称PPO全面优于BC；BC尚无300秒对照。下一步先规划确定性初态偏差/有限外扰恢复诊断，再决定训练调整，不直接追加同类预算。

这轮保持原奖励和固定环境，建立了最终模型五分钟自主长稳证据；尚未进行抗扰动或无限时长验证。R4b、主动返回平滑和实机能力仍独立规划。执行源目录内的README已随短验封存，保留当时“待GPU”的标题；当前结论以本页及本轮审查为准，VALIDATION保留训练前准入证据。
