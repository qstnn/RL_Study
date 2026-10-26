# R4a 教师捕获反馈与第三轮纠偏

**归档结果：** 本版4例完整成功0，最长严格保持1.988秒，未生成第三轮数据/Actor。继续在独立 `takeover_v3` 检查更强角速度阻尼；原2秒门不变。下面采集/训练命令只保留为流程说明，不用于失败诊断。

`takeover_v1`的20条接管时刻对照最长严格保持1.928秒，没有完整成功。此处独立封存新入口，原guided_v5的22份执行源码、takeover_v1、旧Actor和全部物理资产保持；新增4份源码与原22份共同核验。

新参数只影响教师捕获反馈，仍经原16维合法动作接口执行。`combined`把COM速度反馈1.2→2.0、俯仰角速度反馈0.05→0.10；不改变质量、摩擦、力矩、PD、观测、奖励、动作边界和返回参数。原教师文件不修改，候选参数实际写入诊断结果并独立重算。还保留显式`base`、仅`velocity`、仅`rate`选项；存在选项不表示已经完成对照或验收。

教师从reset起以影子模式持续更新内部状态，到指定时间才接管；物理、观测历史、时钟及漂移参考全程连续。每条接管前缀与第二轮自主回放核对，新增成功示范只能来自通过原2秒严格保持＋终末四轮1秒的实际教师后缀。失败、Actor前缀及reset后帧不收作成功示范。旧1512帧反事实数据保留并单独标注。

第三轮依旧采用原305→256→256→128→16 ELU网络，seed42、从零400 epoch、原阶段/切换邻域/纠偏权重；没有Critic或Adam迁移。完成后不自动评估/PPO，等待用户通知。实际验证状态见 [VALIDATION.md](VALIDATION.md)。

## 运行入口

各命令独立，不自动串联；已存在的输出目录不能覆盖。已执行的命令以验证记录为准，下面同时保留复现说明。

```bash
cd /home/user/RL_lab/rl_training
R4_GUIDED=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/guided_v5
R4_TAKEOVER="$R4_GUIDED/takeover_v2"
bash "$R4_TAKEOVER/r4a_takeover.sh" check

bash "$R4_TAKEOVER/r4a_takeover.sh" diagnose \
  --actor "$R4_GUIDED/runs/2026-10-22_bc_corrected_02/actor.pt" \
  --teacher "$R4_GUIDED/runs/2026-10-22_teacher13_pair_final" \
  --baseline "$R4_GUIDED/runs/2026-10-22_bc_corrected_02_autonomous" \
  --seed 43 --switch-times 4.8 4.9 --feedback combined \
  --output "$R4_TAKEOVER/runs/2026-10-22_takeover_feedback_4"
```

仅在上述诊断存在完整成功且通过轨迹核验后：

```bash
bash "$R4_TAKEOVER/r4a_takeover.sh" collect \
  --dataset "$R4_GUIDED/runs/2026-10-22_corrected_dataset_02_final" \
  --diagnosis "$R4_TAKEOVER/runs/2026-10-22_takeover_feedback_4" \
  --output "$R4_TAKEOVER/runs/2026-10-22_corrected_dataset_03"

bash "$R4_TAKEOVER/r4a_takeover.sh" pretrain \
  --dataset "$R4_TAKEOVER/runs/2026-10-22_corrected_dataset_03" \
  --epochs 400 --seed 42 \
  --output "$R4_TAKEOVER/runs/2026-10-22_bc_corrected_03"
```

数据分为`sample_kind=0`原成功教师、`1`历史反事实、`2`本轮完整成功接管后缀。每次加载都会从证据重新组装并逐数组核对；保持完整诊断/数据/来源目录。

第三轮训练完成，用户通知后才执行独立自主回放：

```bash
bash "$R4_TAKEOVER/r4a_takeover.sh" evaluate \
  --actor "$R4_TAKEOVER/runs/2026-10-22_bc_corrected_03/actor.pt"
```

新Actor必须使用本目录的`r4a_takeover.sh`，旧`r4a_guided.sh`及takeover_v1入口不接受。教师接管成功不能当作新Actor成功，拟合误差也不能代替自主验收。该入口没有PPO模式，原自主2秒保持及无物理失败的PPO前提保持。
