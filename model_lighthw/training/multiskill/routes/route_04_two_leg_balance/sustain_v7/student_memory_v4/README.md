# R4a student_memory_v4：捕获过程记忆单因素对照

2026-10-23，Asia/Shanghai。已完成独立实现、示范输入重建、两组各6000次BC更新和各两槽自主GPU测试。**记忆屏蔽组0/2，记忆暴露组0/2，合格保持均0秒，PPO仍不准入。** 模型仅作为研究候选保留，不作为已通过的PPO起点。

[完整结果与精度复核](VALIDATION.md) · [下一步方向](../NEXT_DIRECTION.md) · [单因素最终核验](validation/final_checks.json)。

## 本轮修改

保留v7原305维本体观测，在末尾追加两维可部署状态：俯仰≤−5°后的触发锁存、触发后0.8秒平滑混合进度。新Actor输入307维，环境原输入305维和Critic88维不变。状态每个50Hz控制输入推进一次，reset/入场/返回清空，40步后进度永久为1；不含绝对回合年龄、观察截止时间或教师COM/接触特权量。这里的触发不等于“连续合格2秒完成捕获”。

两组均从上一轮observed模型显式迁移，保留原305列权重并给新增两列置零，初始输出完全相同；两组容量均181664参数。共享主干307→256→256→128、双16维输出头，原瞬时俯仰路由不变。zero组将两维输入置零，exposed组提供实际记忆，除此之外初始化、数据、逐批采样及预算相同。

7726个动作标签及原8个数据数组逐值保留；按完整原始控制序列重建记忆后再选原训练帧，避免被排除片段造成计时断档。没有新增教师动作、失败动作或未执行查询标签。

原持续奖励、URDF/质量/惯量、腿/轮48/24Nm、动作映射、1kHz物理/50Hz控制不变，无环境噪声或随机化；允许≤0.12m/s慢漂移，原实际高频门保留。站立指令有效时不自动返回。

## 实测摘要

| 条件 | BC更新 | 训练集MSE | 两槽终止时间 | 自主额外10秒 | 合格保持 |
|---|---:|---:|---|---|---|
| 记忆屏蔽zero | 6000 | 5.43555e−7 | 5.14 / 5.06秒，倾倒保护 | 0/2 | 0 / 0秒 |
| 记忆暴露exposed | 6000 | 5.44116e−7 | 5.28秒非轮碰撞 / 5.14秒倾倒保护 | 0/2 | 0 / 0秒 |

每组仅seed43两槽筛查；失败后没有扩展seed44/45或运行PPO。不能把小MSE、晚终止或正确触发当作自主站稳。

原评估器的记忆数值容差1e−7小于实测GPU/NumPy最大差1.19209e−7，两个原session保留failed状态。物理回放完整；独立[精度复核器](analysis/verify_memory_v4_1.py)以3e−7只核对连续记忆数值，锁存/实际输入尾部逐值相等，全部物理门槛不变。四例20,620个有效物理样本、动作、来源已复核；不需要重跑这四例。

## 只读复核命令

下面复查已有证据，不启动BC、GPU回放或PPO：

```bash
cd /home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_memory_v4
/home/user/miniconda3/envs/lab/bin/python analysis/verify_memory_v4_1.py runs/2026-10-23_zero_screen
/home/user/miniconda3/envs/lab/bin/python analysis/verify_memory_v4_1.py runs/2026-10-23_exposed_screen
/home/user/miniconda3/envs/lab/bin/python analysis/final_checks.py
```

`memory.sh check`为6项CPU检查；`prepare`只允许创建新数据目录；`train --mode zero|exposed --output <新目录> --device cuda:0`为固定6000步BC实验；`evaluate --actor <模型> --output <新目录> --seeds 43`为自主筛查。后两项本轮已完成，不建议原样重跑。冻结的evaluate仍使用原精度容差，需另行运行上面的独立复核器，不能仅按进程退出码判定机器人行为。全部输出目录禁止覆盖已有结果。

当前无正式PPO启动入口。307维状态接入PPO还需在后续独立版本中实现并验证向量环境reset、rollout连续状态、末状态价值及保存/续训；不能把原305维Runner直接套用。

下一项有限工作是支撑方向反馈可预测性检查：整条来源轨迹及两槽共同留出，评估本体输入对有符号COM支撑偏差和沿支撑前向速度的估计误差，并独立检查本轮失败状态。预测准确仍需真实自主复验；新动作标签只收实际执行且成功恢复的教师段。准入仍需两槽×seed43/44/45自主捕获后额外10秒全部通过，再做真实GPU PPO更新/保存/续训短验；最终目标为自主60秒重复与300秒耐久。
