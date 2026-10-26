# R4 双腿平衡：R4a后腿、R4b前腿

**2026-10-23 18:04 审查完成：R4a正式1000次新增更新、最终global1034；冻结最终模型在固定条件下自主额外10/60/300秒各2/2通过。** 300秒组连续保持300.008/300.001秒，无首次失稳/碰撞，高频门全部合格；三组共783020个1kHz样本独立复核。最终模型列为固定条件长稳基线，下一步规划确定性偏差/外扰恢复诊断，尚不追加训练；R4b等待。下方运行状态均为历史记录。[完整审查与图表](sustain_v7/student_feedback_v5/results/2026-10-23_formal_1000_review/README.md)。

**2026-10-23 15:24 运行状态更新：用户已启动正式训练，当前第107/1000次新增更新（global141），仍在推进。** 原终端静默是进度写入文件所致；已新增[独立只读监控](sustain_v7/student_feedback_v5/monitoring/README.md)，请保持现有训练运行。以下14:58的“正式训练未启动”是准入审查时的历史状态。

**2026-10-23 14:58 R4a PPO准入已通过，正式长训练未启动。** 本体估计＋可学习反馈学生保留原物理/奖励，教师全关六例6/6、两槽额外60秒2/2；真实更新/保存/新进程续训及16槽正式配置32轮短验16/16通过。当前使用325维结构化学生及独立PPO入口，从经审查global34继续；旧v4/v6模型不作为当前启动点。[完整本轮证据](sustain_v7/student_feedback_v5/VALIDATION.md) · [正式1000新增更新命令](sustain_v7/student_feedback_v5/TRAINING.md) · [下一步](NEXT_TRAINING_PLAN.md)。

以下保留此前阶段及失败记录，当前准入结论以上方为准。

**最新捕获记忆对照已完成（2026-10-23）：** 独立`student_memory_v4`显式提供触发锁存/0.8秒混合进度，与同容量屏蔽组各6000次BC、同7726帧/初始化/采样。两组自主均0/2、合格保持0秒；暴露记忆两槽5.28/5.14秒失败，PPO仍未准入。4例20,620个物理样本已独立复核，原报告数值容差失败保留并由离线v4.1复核，物理门未变。下一步检查支撑方向偏差/速度的本体预测能力。[最新报告](sustain_v7/student_memory_v4/VALIDATION.md) · [当前研究入口](sustain_v7/student_memory_v4/README.md) · [下一步](sustain_v7/NEXT_DIRECTION.md)。

以下保留持续任务定义和此前各阶段记录，当前学生结论以上方为准。

**当前阶段信息与路由对照已完成（2026-10-23）：** 独立`student_phase_v3`两组同容量/初始化/数据/逐批采样，各6000次BC；实际俯仰路由改善起身方向，但两槽5.26/5.98秒碰撞，合格保持0，自主0/2。固定平均对照亦0/2，未扩展六例，PPO仍未准入。下一步隔离捕获触发记忆/混合进度，不原样增训。[最新完整报告](sustain_v7/student_phase_v3/VALIDATION.md) · [当前研究入口](sustain_v7/student_phase_v3/README.md) · [下一步方向](sustain_v7/NEXT_DIRECTION.md)。以下短段接管及旧版本结果保留作历史。

**当前短段接管纠偏已完成（2026-10-23）：** 独立`sustain_v7/student_segments_v2_1`取得4/4教师恢复额外10秒成功，新增2686帧教师执行、总7726帧，学生03完成6000次BC更新；关闭教师自主六例0/6、保持0，两槽4.74/5.72秒碰撞，PPO仍未准入。前段更贴近参考但起身到捕获的转换仍失败，下一步优先阶段表达/分支对照，不原样追加epoch。[本轮完整报告](sustain_v7/student_segments_v2_1/VALIDATION.md) · [当前工作流](sustain_v7/student_segments_v2_1/README.md) · [下一步方向](sustain_v7/NEXT_DIRECTION.md)。此前学生01/02及各六例失败记录保留。

持续环境/奖励和此前教师额外30秒已通过并保持冻结；明确返回过程冲击仍单独优化。[持续任务与奖励](sustain_v7/README.md) · [此前教师证据](sustain_v7/VALIDATION.md)。

**上一训练版本`smooth_v6`（保留）：允许稳定段≤0.12m/s缓慢漂移，取消10cm累计位移门，增加实际机身/关节高频振动门和抑振奖励。** 第四轮7112帧/200 epoch纠偏后的六例自主审查已完成：保持/完整成功0，5.78/6.28秒机身碰撞，起身和支撑较第三轮退化，PPO未准入。先诊断当前起身/捕获状态，不继续原样训练。[最新自主评估](smooth_v6/results/2026-10-22_bc_corrected_04_review/README.md)。

**持续任务之前的教师诊断：** 角速度反馈对照未取得实质改善后，已固定3.2秒接管/反馈0.20比较触发角−20°/−10°/−5°。三组保持1.334/1.241、1.610/1.570、1.809/1.813秒，提前捕获有改善，但旧协议完整成功0/6。−5°捕获角速度峰值升高、后轮短暂卸载；后续混合对照和长稳证据见页首v7报告，本段旧结果不改判。新增数据/训练/PPO均0。[最新触发角对照与图表](smooth_v6/diagnostics/capture_trigger04/README.md)。

[旧v6环境与回放入口](smooth_v6/README.md) · [第四轮模型与旧验证](smooth_v6/VALIDATION.md) · [下一步计划](NEXT_TRAINING_PLAN.md)。旧源码/模型保留，R4a后腿优先、R4b前腿等待。旧第四轮Actor使用smooth_v6专用入口，不能作为v7直接续训模型；正式训练中途/末尾不自动冻结。

---

以下为guided_v5及更早版本的历史说明，当前状态以上文为准。

**2026-10-22：当前使用独立 `guided_v5`，优先 R4a 后腿支撑。** 训练改为教师闭环验证→原305维观测示范→Actor模仿初始化→自主回放→小预算PPO。R4b为前腿支撑，继续等待。此前文档中“R4a前腿/R4b后腿”的建议已被用户指定命名覆盖。

[当前训练方法与分阶段命令](guided_v5/README.md) · [本轮验证与阶段状态](guided_v5/VALIDATION.md) · [下一轮计划](NEXT_TRAINING_PLAN.md)。教师成功、监督拟合与自主策略验收分别记录。

**2026-10-22 15:39更新：** 教师接管诊断及第三轮纠偏400 epoch已完成，当前专用入口为[guided_v5/takeover_v3](guided_v5/takeover_v3/README.md)。26条对照后取得一条完整成功接管轨迹（严格保持2.010秒、终末四轮2.001秒），只补360帧成功后缀，累计5472帧；第三轮自主回放尚未运行，PPO0，等待用户通知审查。[完成证据](guided_v5/takeover_v3/VALIDATION.md)。旧guided_v5入口不接受该新Actor。

**2026-10-22 15:00审查更新：** 第二轮纠偏400 epoch已完成，自主六例严格保持最长0.160秒、完整成功0/6；一个槽5.44秒倾角失败，另一个跑满12秒并返回四轮。起身改善，下一步先做偏离状态教师接管及捕获/制动数据验证，PPO尚未启动。[审查与图表](guided_v5/results/2026-10-22_bc_corrected_02_review/README.md)。

```bash
cd /home/user/RL_lab/rl_training
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/guided_v5/r4a_guided.sh check
```

固定环境、无观测噪声/初态随机化；保留非零PPO探索、原物理和严格保持2秒/终末四轮1秒。正式PPO仅保存，不做中途或末尾自动冻结；完成后用户通知再审查。新输出位于`guided_v5/runs/`，新版本不直接恢复旧模型/Adam。

`split_v4`及公共`--route 04/04a/04b`仍保留为旧版复现入口，当前新流程使用上面的专用脚本及[分阶段命令](guided_v5/README.md)，不再推荐旧版从零1000次命令。[旧奖励与历史结果](split_v4/README.md)。

## 联合版本历史

- v3正式1000尝试、997保留，前后均0/3成功、严格保持0秒；300/600/1000抽查均四轮小幅倾身。观测噪声/初态扰动已关闭，不能再归因为环境噪声。[审查](results/2026-10-20_v3_1000_update_review/README.md) · [原版复现说明](v3/README.md)。
- v2在300完整评估后用户停止，日志到337，前后均未学成。[审查](results/2026-10-20_v2_300_stop_review/README.md)。
- v1卡在初始站稳入场，下文保留其原方案和命令用于历史追溯，**不作为当前训练入口**。


## v1实现与首轮训练记录（归档）

更新：2026-10-20，Asia/Shanghai。验收任务 **1-10：空载前腿或后腿双支撑静止，并返回四轮**。
这是多技能 Route04；旧 `training/routes/route_04_balanced_highspeed` 是另一条高速路线。

当前环境、奖励与训练链路已实现并验证；2026-10-20用户完成首轮512环境/1000更新，最终前0/3、后0/3，10次冻结0/60，最长有效双腿保持0秒。25,608个已结束带噪训练回合均未通过初始站稳入场条件；不建议原样续训或直接重跑下方首轮命令。下一步先诊断入场与转移奖励，`deployment_ready=false`。[正式训练审查、曲线及复算](results/2026-10-20_1000_update_review/README.md)。

此前17项CPU/框架检查、8环境0-PPO及512环境1+1次PPO短验只证明运行链路，不代表技能成功。

## 启动

以下保留本次首轮训练命令用于追溯；该配置已完成1000更新且未通过行为验收，暂不建议重复执行：

```bash
cd /home/user/RL_lab/rl_training
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/r4.sh train \
  --num_envs 512 --max_iterations 1000 --eval_every 100 --seed 42
```

默认使用 `lab` Python、本机 `../Isaaclab` 和 `../isaac-sim`。路径不同可设置 `R4_PYTHON`、`R4_ISAACLAB`、`R4_ISAAC_SIM`；不改shell全局配置。训练headless；每100次PPO保存并退出该段，随后独立进程用冻结均值Actor复评seed43/44/45和前后模式。下一段精确恢复模型、Adam、学习率和RNG；物理环境重新reset，不声称恢复物理中途状态。预算为新增1000更新，不自动延长或切换课程。

每段32步rollout，默认512环境，总环境转移16,384,000。`max_iterations`是新增PPO更新次数，不是回合数。没有固定总耗时承诺。

输出位于本目录 `runs/<时间戳>/`；终端最后打印 `[R4 REVIEW DIRECTORY]`。保留整个目录：`progress.json`、每段 `episodes.jsonl`、`updates.jsonl`、TensorBoard、`final.pt`、冻结 `evaluation.json` 和 `trace_seed*.npz` 都用于审查。不存在自动挑选“成功模型”或自动部署。

辅助命令：

```bash
# CPU几何、动作范围与成功反例检查
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/r4.sh check
# 0-PPO短验：包含一个完整自然超时以及局部reset检查
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/r4.sh smoke --num_envs 8
# 恢复：必须显式给出同一R4契约/源码的checkpoint；新增预算由参数控制
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/r4.sh train --checkpoint /绝对路径/final.pt --max_iterations 100
# GUI回放或headless评估（使用冻结均值Actor）
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/r4.sh play --checkpoint /绝对路径/final.pt
bash model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/r4.sh evaluate --checkpoint /绝对路径/final.pt
```

公共入口 `multiskill/run.py --route 04` 已转交R4专用启动器。不要把R1、R5A、旧高速模型传入；相同动作维度不代表合同兼容。`--support_mode front/rear`用于从零分训对照，默认 `both` 是同一条件Actor；不同模式和保持时长的checkpoint不会静默互载。

## 路线规划与边界

1. **A0 几何/物理和软件准入（本轮）**：规范URDF、接触和动作范围审查；17刚体质量/COM/惯量、摩擦、关节限幅运行时读回；重置、PPO更新和恢复链路。
2. **A1 完整短任务（首轮命令）**：从空载四轮初态开始，前后模式共同训练，连续双支撑至少2秒，最终返回四轮连续稳定至少1秒。读取分模式成功率、最长保持、碰撞、漂移和饱和，而非只看reward。
3. **A2 长保持（后续，当前不自动执行）**：A1通过后再做5秒保持。代码可用 `--hold_seconds 5`从零训练；2秒模型到5秒合同的迁移须单独验证，当前明确拒绝隐式迁移。先固定多seed与reset，避免目标改变掩盖旧能力退化。
4. **A3 扰动和技能接口（未来未实现）**：初态/IMU噪声、材料误差、有限外推扰动和可测接触估计；只在返回四轮稳定后允许技能切换。需明确实机执行器、延迟和可观测性，不能把本轮软件准入作为实机验收。

候选晋级标准：前后模式分别至少20个冻结回合、跨不少于3个seed，各成功率≥95%，无NaN/Inf/非法接触；审查力矩和动作饱和、姿态、速度和漂移。每段默认6例是学习趋势筛查，数量不足以独立宣告晋级。前后模式若明显负迁移，先用相同预算做单模式对照，再决定分Actor。

## 环境与动作

独立源码：`source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/two_leg_balance/`。
Gym ID：`MultiSkill-LightHW-TwoLegBalance-v1`，由R4 worker在AppLauncher之后显式注册。

- 平坦静态地面、空载、无牵引；URDF总质量11.668111kg；重力9.81，静/动摩擦均1.0，恢复系数0。导入显式保留URDF惯量，启用自碰撞、陀螺力，solver8/2；物理1kHz，策略50Hz。刚体数值角速度保护为100000deg/s，避免旧1000deg/s非物理截断；电机关节和策略轮速限制按下述值保留。
- 腿位置PD80/3，腿48Nm/12.57rad/s；轮速度阻尼1，轮24Nm/37.7rad/s，轮armature .00243216；无动作延迟。这些是仿真模型限制，不是实机持续额定能力。
- 动作16维，FR/FL/RR/RL顺序的12腿位置和4轮速度。零动作对应原站姿；腿从中性角分段线性映射到保守URDF区间，轮目标±15rad/s。Actor自主产生控制目标，没有预设腿运动轨迹、根速度注入、外力扶持或旧策略自动初始化。
- 出生后0.2秒零动作消除落地瞬态；之后网络接管。reset只回到四轮初态，不从倾斜双支撑状态采样来替代真实进入过程。
- 12秒短任务：0–1秒四轮稳定；1–4秒重心/姿态转移；4–7秒双支撑窗口（要求其中连续2秒）；7–10秒返回；10–12秒恢复（结束时连续≥1秒）。所有阶段时钟基于真实物理步，不随机错置episode计时。

前腿模式0的目标俯仰+52.137°，后腿模式1为−56.119°。候选由原站姿关节角的全身COM对齐所选轮轴计算，目标重力方向在转移/返回时平滑变化。这是奖励参考，不是强制机身姿态。

几何静力筛查：前/后最大理想腿力矩10.113/11.021Nm；非轮碰撞网格最低离地45.13/45.33mm；悬空轮净空532.62/560.09mm；两支撑轮各约57N。仅一个姿态族的FK及理想竖直力平衡，未证明动态可达、非线性稳定、自碰撞无干涉或实机可行。原始结果见 `validation/cpu/geometry.json`。

## 观测

Actor每帧61维，5帧历史共305维：角速度3、重力3、12腿相对角、16关节速度、上一动作16、模式one-hot2、目标重力3、阶段one-hot5、归一化时间1。IMU/关节加小噪声，历史在reset时全部重填；无根线速度、COM、接触力或净空真值。

Critic88维：61维清洁帧加27维特权状态（全身COM、COM/机身速度、四轮力/净空、高度、支撑线误差、漂移、任务计时和事件）。独立检查Critic接口，防止PPO退化为无特权Critic或误喂Actor。

## 奖励

连续项按实际控制dt=.02缩放；事件项每回合只计一次。权重和实现集中在 `spec.py`、`env.py`，验收状态在 `core.py`。

| 项目 | 权重 | 激活与含义 |
|---|---:|---|
| 姿态 | +3 | 全阶段跟踪所给目标重力，不把全程水平当目标 |
| 支撑 | +2 | 转移/保持鼓励指定双轮；站立/返回鼓励四轮 |
| 悬空净空 | +1.5 | 转移/保持且指定双轮接触；按8cm饱和 |
| COM支撑线 | +2 | 转移/保持且指定双轮接触；指数宽度10cm，使用有限线段 |
| 静止 | +2 | 双支撑窗口须指定两轮+悬空轮；恢复段需四轮 |
| 返回腿姿 | +1 | 初始/返回/恢复阶段，弱参考原站姿 |
| 严格双支撑保持 | +6 | 必须同时满足下方严格判据，连续计时 |
| 严格四轮恢复 | +4 | 已完成双支撑后，末段恢复计时 |
| 首次达到保持时长 | +10一次 | 必须有初始四轮稳定，不奖励反复进出 |
| 终末完整成功 | +20一次 | timeout时已经保持且最后连续恢复≥1秒 |
| 动作变化/腿力矩/腿速度/静止轮速 | −.025/−.03/−.03/−.015 | 按归一化平方均值，轮速只约束静止段 |
| 动作边界/超.6m漂移/耗时 | −.05/−1/−.1 | 连续成本，鼓励局部完成 |
| 实际失败 | −8一次 | 非轮碰撞、跌倒或越界 |

## 严格成功和终止

先在初始阶段连续四轮稳定≥.5秒；双支撑必须同时满足：所选两轮竖直力各>5N，另两轮力绝对值<2N且保守净空≥3cm；COM投影距有限支撑线段≤6cm；COM速度≤.12m/s、机身角速度≤.25rad/s；相对目标姿态误差≤12°；相对双支撑窗口首帧漂移≤10cm。保持每1ms检查，中断清零，不累加多个短片段。

返回时要求四轮各>5N、水平姿态误差≤12°、机身高度.416±.07m、同样速度门槛、相对恢复段首帧漂移≤10cm；必须在回合最后连续保持≥1秒。先前恢复过、最后又失稳不能算成功。

非轮接触>5N（仅出生.15秒宽限）、倾角>80°、根高<.18m或总COM位移>1.5m终止；NaN/Inf直接报错停止。前后成功分别统计；用20ms平均接触不能替代1ms事件记录。悬空轮净空用轮心减外包半径，保守处理轮倾斜。

## 验证与资料

最终交付验证记录见 `validation/README.md`。冻结轨迹可用 `scripts/verify_trace.py <eval目录>`重新计算连续保持和终末成功；回合汇总不会作为复算输入指标。

API实现以本机Isaac Lab 2.2源码和RSL-RL安装版本为准；[Isaac Lab Direct环境官方说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/tutorials/03_envs/create_direct_rl_env.html)说明了动作、终止、奖励和reset职责；[RSL-RL官方组件说明](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/overview.rst)说明runner的训练、保存和恢复接口。后续版本文档只作设计参考，不替代本机运行时核对。
