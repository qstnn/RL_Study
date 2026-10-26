# R4a 成功接管数据与第三轮纠偏

**后续当前版本：[smooth_v6](../../smooth_v6/README.md)。** 用户确认允许稳定段≤0.12m/s漂移，新版增加实际抖动验收与奖励；第四轮自主六例均失败，PPO未准入，下一步诊断起身/捕获退化。下文保留第三轮旧协议历史；本目录执行源码和模型没有改动。

**当前状态（2026-10-22自主审查后）：PPO未准入。** 第三轮六例自主严格保持/完整成功0，均约6.42秒机身碰撞；新增八例实际状态接管全部跑满12秒，但保持漂移12.44–38.92cm，严格保持/完整成功仍0。下一步先修捕获回中，再收通过原协议的双槽示范；本轮新增训练/样本/PPO均0。[完整审查、图表及下一步](results/2026-10-22_bc_corrected_03_review/README.md)。下文保留第三轮数据与原诊断历史。

当前接管恢复示范来自本目录。原教师的20条时刻对照均未完整成功；速度反馈2.0、角速度反馈0.10的4例对照最长保持1.988秒。进一步将角速度反馈设为0.20后，4.8秒接管的槽1达到严格保持2.010秒、终末四轮稳定2.001秒，完成原完整任务；槽0未通过。

**这是一条固定seed43、教师接管的恢复成功，不能记为自主Actor成功或鲁棒性通过。** 保持时长只比2秒多约10ms，余量很小。原观测/动作、质量/摩擦/电机/PD、奖励、保持窗口/漂移原点和验收标准均保持。原guided_v5的22份源码、takeover_v1/v2和旧模型完整保留。

本版`strong`教师参数为COM速度反馈2.0、俯仰角速度反馈0.20；其余沿用原返回候选13，返回参数不变。`drift`是预留对照选项，本次未运行，不能当作已验证控制器。新4份执行源码与原22份共同封存；新Actor只使用本目录入口。

## 第三轮数据与训练

- 数据共 **5472帧、19个来源回合**：原3600成功教师帧＋1512历史反事实纠偏帧＋360实际成功接管后缀。仅新增槽1的4.8–11.98秒教师动作。
- 失败槽、接管前Actor动作及reset后帧不作新增示范；`sample_kind=0/1/2`分别标识上述三类，不能将5472帧全部称为成功示范。
- 保持原305维观测，不增加COM/接触特权输入；教师真值仅用于诊断和动作标签。
- 原305→256→256→128→16 ELU网络，从零拟合400 epoch、seed42，保持原阶段平衡、切换邻域和纠偏2倍权重。旧Critic/Adam/Actor权重不迁移。
- 本次按用户要求执行第三轮预训练；完成后不自动评估、不自动PPO，等待用户通知审查。实际完成证据见 [VALIDATION.md](VALIDATION.md)。

## 显式命令

脚本自动配置lab Python及Isaac Sim环境；GPU阶段要求CUDA。每条命令独立执行，输出目录必须不存在，下面已完成命令保留用于追溯，不能原样覆盖现有结果。

```bash
cd /home/user/RL_lab/rl_training
R4_GUIDED=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/guided_v5
R4_TAKEOVER="$R4_GUIDED/takeover_v3"
bash "$R4_TAKEOVER/r4a_takeover.sh" check

bash "$R4_TAKEOVER/r4a_takeover.sh" diagnose \
  --actor "$R4_GUIDED/runs/2026-10-22_bc_corrected_02/actor.pt" \
  --teacher "$R4_GUIDED/runs/2026-10-22_teacher13_pair_final" \
  --baseline "$R4_GUIDED/runs/2026-10-22_bc_corrected_02_autonomous" \
  --seed 43 --switch-times 4.8 --feedback strong \
  --output "$R4_TAKEOVER/runs/2026-10-22_takeover_strong_2"

bash "$R4_TAKEOVER/r4a_takeover.sh" collect \
  --dataset "$R4_GUIDED/runs/2026-10-22_corrected_dataset_02_final" \
  --diagnosis "$R4_TAKEOVER/runs/2026-10-22_takeover_strong_2" \
  --output "$R4_TAKEOVER/runs/2026-10-22_corrected_dataset_03"

bash "$R4_TAKEOVER/r4a_takeover.sh" pretrain \
  --dataset "$R4_TAKEOVER/runs/2026-10-22_corrected_dataset_03" \
  --epochs 400 --seed 42 \
  --output "$R4_TAKEOVER/runs/2026-10-22_bc_corrected_03"
```

第三轮自主回放已按用户通知完成；以下保留独立复现入口（默认创建新目录）：

```bash
cd /home/user/RL_lab/rl_training
R4_TAKEOVER=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/guided_v5/takeover_v3
bash "$R4_TAKEOVER/r4a_takeover.sh" evaluate \
  --actor "$R4_TAKEOVER/runs/2026-10-22_bc_corrected_03/actor.pt"
```

`evaluate`仅执行原Actor均值，无教师接管和训练更新，默认两槽×seed43/44/45。本次实际审查未通过，不能按拟合误差宣称成功。正式PPO仍需自主重复严格保持且无物理失败，随后在独立版本显式接入PPO并完成更新/保存/续训短验；此目录没有PPO模式，不自动推进。

完整保留诊断、数据、源快照和模型链；不要只移动`actor.pt`后删除上游证据。旧`r4a_guided.sh`及takeover_v1/v2均不接受本版Actor。
