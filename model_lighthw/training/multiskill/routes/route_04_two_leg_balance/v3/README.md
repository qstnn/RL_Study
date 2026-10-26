# R4 v3：固定条件连续训练

**2026-10-20正式1000轮已完成审查，完整目标未通过。** 最终前0/3、后0/3，双腿严格保持0秒，保持窗口仍几乎全程四轮支撑；300/600对照也未形成目标动作。观测噪声与初态扰动关闭已生效。本轮不原样续训，下一步先调整转移/卸载与辅助奖励的取舍。[完整审查及曲线](../results/2026-10-20_v3_1000_update_review/README.md)。

本轮关闭观测噪声和初始姿态扰动，固定初态、质量/摩擦/PD/地形及零执行器延迟。沿用v2的动作、奖励、事件时钟和完整成功条件；前/后两种模式仍由同一Actor学习。PPO保留分阶段、受控的非零动作探索。

按用户要求：**训练期间及训练结束时都不自动冻结评估。全部训练结束后，等用户通知再审查。** 原计划中的训练前模型对照和中途冻结频率安排不再自动执行。

## 本轮命令（已完成，保留复现）

```bash
cd /home/user/RL_lab/rl_training
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/v3/scripts/r4_v3.sh train \
  --num_envs 512 \
  --max_iterations 1000 \
  --save_every 100 \
  --seed 42
```

入口自动使用本机`lab`环境，以headless模式从零初始化v3 Actor/Critic；不加载旧v2的300轮模型或本轮短验模型。正式训练由用户启动。

`max_iterations`表示新增优化尝试数，KL回退的尝试也计入预算；保留更新数以`updates.jsonl`的`accepted`为准。512环境×32步×1000尝试为16,384,000个环境转移。

训练由一个worker连续完成全部预算，保持同一仿真环境，不再每100轮关闭环境并启动评估。`--save_every 100`只在完成第100、200……次尝试后保存模型；可按需要设置其他正整数。`--eval_every`已从v3入口移除，不要复制旧版参数。

## 保存与后续审查

输出目录：`v3/runs/<时间戳>/`，结束时终端打印`[R4 V3 REVIEW DIRECTORY]`。

- `train_001000/final.pt`：完成默认1000尝试后的最终模型、Adam、LR和随机数状态。
- `train_001000/training/model_000100.pt`等：中途保存点，编号为实际已完成尝试数。
- `train_001000/updates.jsonl`、`episodes.jsonl`及TensorBoard：训练日志；保留原分阶段诊断，不触发冻结评估或行为审查。
- `result.json`：完成数量、最终模型路径/SHA；`frozen_reviews=[]`、`review_status=awaiting_user_request`。训练完成不自动标记行为通过。

显式续训只接受同一v3执行源码和合同；`--checkpoint`指定保存点时，`max_iterations`仍表示新增尝试数。恢复模型、Adam、LR与RNG，物理状态重新从固定初态开始，不声称跨进程恢复完整仿真轨迹。原v1/v2模型与源码保留，不静默跨版本续训。

训练结束后用户通知，再对最终模型进行独立前/后任务评估、轨迹复算与必要的回放。`evaluate/play`保留为显式操作，训练入口不会调用。

## 保持的任务与物理

Actor/Critic/动作维度305/88/16；12腿位置＋4轮速、50Hz策略/1kHz物理；质量11.668111kg、腿/轮48/24Nm及12.57/37.7rad/s、PD80/3、μ1、自碰撞和陀螺力保持。奖励沿用v2，未增加外力、预设抬腿轨迹或降低成功门槛。

完整任务仍要求初始真实连续站稳0.5秒，指定双轮支撑、另一对净空≥3cm，并满足COM线误差≤6cm、速度≤0.12m/s、角速≤0.25rad/s、目标姿态误差≤12°及保持漂移≤10cm，连续双腿保持≥2秒；随后终末四轮站稳连续≥1秒。

环境噪声关闭与PPO探索分开：静止阶段0/2/4的动作std仍为学习std的0.03倍，转移/返回阶段保留完整std。固定环境不等于确定性优化；无噪声训练能否完成目标须等正式训练后验证。

## 交付时的软件验证（历史）

见[验证记录](validation/README.md)。以下记录为交付时的软件/运行时短验；本次正式训练审查见顶部链接。

```bash
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/v3/scripts/r4_v3.sh check
```
