# R4a 教师接管与第三轮纠偏

**2026-10-22归档状态：** 原教师接管时刻诊断共20例，完整成功0，最长严格保持1.928秒；未生成本目录第三轮数据或Actor。当前转入独立 [takeover_v2](../takeover_v2/README.md) 检查捕获反馈，下面命令仅为原流程记录。[本版证据](VALIDATION.md)。

这是 `guided_v5` 的独立纠偏入口，原22份执行源码、教师、奖励和模型保持。当前物理与验收合同仍为后腿支撑、原305维观测/16维动作、无噪声/随机化、4–7秒窗口严格保持2秒、终末四轮稳定1秒。新增4份执行源码独立封存，共26份；该入口不包含PPO，不与旧入口隐式混用模型。

## 阶段与数据规则

1. `diagnose`：同原两槽布局和固定seed43，分别在3.6/4.0/4.4/4.8秒接管，共8例。从初态运行第二轮Actor；影子教师从reset起一直读取当前状态并更新自身触发器，到指定时刻才执行教师动作。接管时不重置物理、历史、时钟或教师。每条接管前的物理/观测/动作与已存自主回放核对。
2. 诊断完整记录1kHz物理和50Hz控制；独立复算原协议，并重算Actor、教师动作及实际执行裁剪。只有完成原严格保持和完整返回的回合才可用于下一阶段。失败或仅短暂保持的接管不计成功示范。
3. `collect`：保留第二轮累计数据中的3600成功教师帧及1512历史反事实标签，只增加完整成功接管回合中**实际执行教师的后缀**。不把接管前Actor动作模仿为正确标签，不收入自动reset后帧，不新增未经物理验证的离线反事实标签。
4. 新字段 `sample_kind` 区分原成功教师0、旧反事实1、成功接管2；`is_correction`仍保留供原损失使用。数据加载时重新从原始物理证据组装并逐数组比较，不能只改元数据成功标记。
5. `pretrain`：原305→256→256→128→16 ELU网络，从零监督拟合400 epoch，seed42；沿用原阶段平衡、切换邻域和纠偏样本权重，不恢复旧Adam/Critic。只初始化新Actor，结束不自动评估。
6. `evaluate`为显式、独立命令。第三轮Actor行为必须另行验证，拟合误差/教师成功都不能记作自主策略通过。按用户要求，预训练完成后等通知再审查；PPO继续等待准入及运行/保存/续训短验。

## 命令

各阶段显式执行，不自动连跑；输出目录必须不存在。`check`已完成10项CPU检查，不等同于物理验收。

```bash
cd /home/user/RL_lab/rl_training
R4_GUIDED=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/guided_v5
R4_TAKEOVER="$R4_GUIDED/takeover_v1"
bash "$R4_TAKEOVER/r4a_takeover.sh" check
```

本次8例诊断：

```bash
bash "$R4_TAKEOVER/r4a_takeover.sh" diagnose \
  --actor "$R4_GUIDED/runs/2026-10-22_bc_corrected_02/actor.pt" \
  --teacher "$R4_GUIDED/runs/2026-10-22_teacher13_pair_final" \
  --baseline "$R4_GUIDED/runs/2026-10-22_bc_corrected_02_autonomous" \
  --seed 43 --switch-times 3.6 4.0 4.4 4.8 \
  --output "$R4_TAKEOVER/runs/2026-10-22_takeover_8"
```

诊断经核对并存在完整成功后，才允许收集和训练；若诊断无成功，会在创建输出目录前拒绝采集。

```bash
bash "$R4_TAKEOVER/r4a_takeover.sh" collect \
  --dataset "$R4_GUIDED/runs/2026-10-22_corrected_dataset_02_final" \
  --diagnosis "$R4_TAKEOVER/runs/2026-10-22_takeover_8" \
  --output "$R4_TAKEOVER/runs/2026-10-22_corrected_dataset_03"

bash "$R4_TAKEOVER/r4a_takeover.sh" pretrain \
  --dataset "$R4_TAKEOVER/runs/2026-10-22_corrected_dataset_03" \
  --epochs 400 --seed 42 \
  --output "$R4_TAKEOVER/runs/2026-10-22_bc_corrected_03"
```

第三轮完成、用户通知后再执行：

```bash
bash "$R4_TAKEOVER/r4a_takeover.sh" evaluate \
  --actor "$R4_TAKEOVER/runs/2026-10-22_bc_corrected_03/actor.pt"
```

该新Actor必须使用本目录入口，旧 `r4a_guided.sh` 不接受它。保留整个来源/数据/诊断链，不能只取出Actor后删除训练证据。

## 验证状态

初始静态检查：10项CPU测试、Python编译和shell语法通过，原22源码/18资产保持。覆盖失败/部分成功拒收、只收已执行教师后缀、排除reset帧、接管不可回退、前缀变化拒绝、数据与预算准入、无自动评估/PPO。

实际接管结果、数据量、第三轮完成情况以本目录的 `VALIDATION.md` 及相应 `runs/*/result.json` 为准。运行时启动通过不代表自主技能成功。
