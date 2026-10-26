# R4a guided_v5：示范初始化，再自主训练

**后续已转入独立[smooth_v6](../smooth_v6/README.md)。** 用户确认允许≤0.12m/s缓慢漂移，新增实际抖动门；第四轮自主审查六例均失败，PPO未准入，下一步诊断起身/捕获退化。以下保留原10cm协议与guided_v5阶段记录，旧Actor和源码保持；新模型不使用本目录旧入口。

由 `soft_transfer_v2` 的 200 次审查转向“先取得可重复成功动作，再训练原观测 Actor”。R4a 仅后腿站立；R4b 等待。最新验证结果见 [VALIDATION.md](VALIDATION.md)。

**当前状态（2026-10-22后续自主审查）：PPO未准入，先修捕获回中。** 第三轮5472帧/400 epoch的Actor已按用户通知审查：六例均约6.42秒机身碰撞、严格保持0；追加八例实际状态教师接管均跑满12秒，但漂移超10cm、严格保持/完整成功0。旧单槽成功恢复未覆盖新Actor的访问状态；本轮新增训练/数据/PPO均0，不继续原样追加epoch。下一步先做保持原点漂移反馈的小预算对照，通过后再准备第四轮纠偏。[最新审查与计划](takeover_v3/results/2026-10-22_bc_corrected_03_review/README.md) · [当前入口](takeover_v3/README.md)。新Actor只能使用takeover_v3入口；下面为原guided_v5流程说明，旧源码、模型和历史结果保留。

## 实现与阶段

1. `teacher-search`：256 个局部捕获候选；`--family return` 固定已筛出的捕获动作，检查 64 个返回候选。教师通过原 16 维腿位置/轮速接口工作，用 COM/速度反馈追上并制动后轴，加入 PD 静力位置偏置；返回时平滑改变目标 COM/后轴间距。真值仅供教师。
2. `teacher-replay`：先原布局单槽、再两槽，分别重复 seed 43/44/45；保存完整 1 kHz 物理量、50 Hz 原 305 维观测与动作，独立 NumPy 复算协议。固定初态下多 seed 只是重复性检查，不能称作鲁棒性覆盖。
3. `collect`：必须两个布局均完成完整任务，才收录成功示范；失败或仅保持片段不进入此版完整任务数据集。
4. `pretrain`：仅初始化 Actor，按阶段平衡损失，提高起身/保持及 1/4/7/10 s 切换邻域权重。原 305→256→256→128→16 MLP 不增加 COM/接触输入。监督误差不能代替闭环行为。
5. `evaluate`：关闭教师，从四轮固定初态独立运行 Actor 均值，完整回合记录。至少六例均严格保持 2 s 且完整回合无物理失败，才能进入 PPO；完整任务验收仍需末段四轮 1 s。
6. `train`：新 Critic/Adam，以合格 BC Actor 开始 100–200 次自主 PPO。初始基础 std 腿 0.05、轮 0.03，保留原按阶段缩放和 KL/非有限保护。高斯 PPO 仍需非零探索，这不同于观测噪声或环境随机化。每 50 次只保存；中途和末尾都不自动冻结，不自动晋级。

所有阶段独立目录、来源哈希、资产快照和合同；跨旧版本、被改动数据/Actor、缺少自主证据均拒绝。续训只允许同版本已开始的 PPO 模型，恢复 Actor/Critic、Adam、学习率、计数与 RNG，累计尝试不得超过首轮 200 次预算；物理状态不存档，续训重新从固定初态开始。

若独立回放偏离，使用 `collect --validation` 从真实 Actor 访问状态补教师标签，再显式预训练/回放。这是离线重算反事实动作标签，没有在失败轨迹中实际施加教师动作，不能称为成功恢复示范。记录 `is_correction`，训练损失给予纠偏样本 2 倍权重；不引入观测噪声、不增加 Actor 输入。既有累计数据用 `--dataset` 显式合并，不能丢掉前轮纠偏样本。

仅这个只读纠偏导入允许学习脚本版本变化：必须核对原归档源码、21份物理/观测/教师源码、资产、合同和 Actor 动作重算。旧 Actor 权重不迁移，旧 Adam/Critic 不恢复。实际训练/自主评估仍要求当前完整版本。

保留示范、验证及历轮纠偏来源目录，不能只复制 `actor.pt` 后删除这些证据；启动时会重新核对其哈希与轨迹。

本版沿用 `soft_transfer_v2` 奖励、物理、48/24 Nm、原动作限幅与严格 2 s/末段 1 s 验收；无观测噪声、初态扰动、域随机化。本次主要改变训练初始化与数据路径。未完成保持时提前终止仍为后续独立对照，不混入此次实现。

## 命令

从仓库根目录执行；脚本自动配置 `lab` Python 和 Isaac Sim/Isaac Lab。可通过 `R4_PYTHON`、`R4_ISAAC_SIM`、`R4_ISAACLAB` 指定本地安装位置。GPU 阶段要求可见 CUDA。

```bash
cd /home/user/RL_lab/rl_training
R4_GUIDED=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/guided_v5
bash "$R4_GUIDED/r4a_guided.sh" check
```

从头重复当前返回候选验证的示例；各 `--output` 必须尚不存在。选定候选后应逐阶段检查结果，不要将这些命令作为自动串联晋级脚本。

```bash
bash "$R4_GUIDED/r4a_guided.sh" teacher-search --family return --output "$R4_GUIDED/runs/return_search"
bash "$R4_GUIDED/r4a_guided.sh" teacher-replay --search "$R4_GUIDED/runs/return_search" --candidate-id 13 --layout original --output "$R4_GUIDED/runs/teacher_original"
bash "$R4_GUIDED/r4a_guided.sh" teacher-replay --search "$R4_GUIDED/runs/return_search" --candidate-id 13 --layout pair --reference "$R4_GUIDED/runs/teacher_original" --output "$R4_GUIDED/runs/teacher_pair"
bash "$R4_GUIDED/r4a_guided.sh" collect --teacher "$R4_GUIDED/runs/teacher_pair" --output "$R4_GUIDED/runs/demonstrations"
bash "$R4_GUIDED/r4a_guided.sh" pretrain --dataset "$R4_GUIDED/runs/demonstrations" --epochs 200 --output "$R4_GUIDED/runs/bc_200"
bash "$R4_GUIDED/r4a_guided.sh" evaluate --actor "$R4_GUIDED/runs/bc_200/actor.pt" --output "$R4_GUIDED/runs/bc_autonomous"
```

若上述 Actor 自主回放失败，可做一轮显式纠偏；每轮先查看行为结果再决定是否继续，不自动循环：

```bash
bash "$R4_GUIDED/r4a_guided.sh" collect \
  --teacher "$R4_GUIDED/runs/teacher_pair" \
  --dataset "$R4_GUIDED/runs/demonstrations" \
  --validation "$R4_GUIDED/runs/bc_autonomous" \
  --output "$R4_GUIDED/runs/corrected_dataset_01"
bash "$R4_GUIDED/r4a_guided.sh" pretrain --dataset "$R4_GUIDED/runs/corrected_dataset_01" --epochs 400 --output "$R4_GUIDED/runs/bc_corrected_01"
bash "$R4_GUIDED/r4a_guided.sh" evaluate --actor "$R4_GUIDED/runs/bc_corrected_01/actor.pt" --output "$R4_GUIDED/runs/bc_corrected_01_autonomous"
```

通过后，下面 PPO 命令中的 `--actor` 和 `--validation` 应同时换成同一轮已合格产物。不能只换其中一个。

仅在对应自主验证通过后使用以下 PPO 入口；不使用旧 `soft_transfer_v2/final.pt`，也不将教师当成自主 Actor。若 BC 独立回放失败，先定位偏离并补充 Actor 实际访问状态上的教师标签，不能凭拟合损失下降直接加长 PPO。

```bash
bash "$R4_GUIDED/r4a_guided.sh" train \
  --actor "$R4_GUIDED/runs/bc_200/actor.pt" \
  --validation "$R4_GUIDED/runs/bc_autonomous" \
  --num-envs 512 --updates 200 --save-every 50 \
  --output "$R4_GUIDED/runs/ppo_200"
```

训练完成后等用户通知再显式 `evaluate --checkpoint .../final.pt` 并审查；该命令不会被训练入口自动调用。尚无合格 Actor 时，训练入口会拒绝启动，保留实际验证边界。
