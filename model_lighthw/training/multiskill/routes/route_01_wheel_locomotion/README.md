# Route01：轮式运动与坡地

**2026-10-20方向调整：用户已授权以[直接PPO V8](direct_rl_v8/README.md)作为下一次训练主线。** 第一阶段固定条件、关闭观测噪声与额外随机化，保留统一Actor和64工况；WBC教师作为诊断参考，不再作为RL训练前置。V8已通过25项CPU和512环境12＋4次GPU更新/恢复短验，初始化15/18；训练入口准入通过，完整重拖仍待学习，正式训练由用户启动。以下V7准入结论是历史教师迁移流程状态，不是直接RL的禁止条件。

**2026-10-20 15:03：R1执行审计和五组教师修正对照已实施，仍未达到正式RL准入。** 23项CPU通过，8次GPU/46轨迹包含12条中止和4条短验；默认控制与原基线逐值一致，五组实验配置均未通过完整教师。μ3.0下旧Actor新评估15/18：原空载/背载边界项通过，+25°重拖移动及−20°重拖静止/制动失败。无新学生Actor/PPO，正式训练继续由用户启动。详见[实施证据与下一步](research_v7/results/2026-10-20_servo_support/README.md)和[准入状态](research_v7/results/2026-10-20_servo_support/readiness.json)。

**2026-10-20 12:14：当前研究入口为[R1 V7](research_v7/README.md)，正式RL暂不准入。** 保持高附着μ3.0、单Actor和64工况路线；固定行推理后，G0物理8次短验/20轨迹/18比较逐值一致。修正瞬时零冲量接触误判及人工力矩变化硬限额，18项CPU检查通过。完整教师仍0/4：有效拖车进度+25°7.85cm、−20°17.68cm，−20°静止20s支撑率85.6%<95%。先解决动态支撑、完成1m＋减速＋2s＋5s，再迁移统一Actor和受控PPO。18边界/128全条件已含−20°/40kg制动；不续训599。详见[本轮物理/教师诊断](research_v7/results/2026-10-20_contact_diagnosis/README.md)。

以下保留旧阶段记录与历史命令。

**2026-10-16 16:25任务范围修订：−25°下坡排除20/40 kg拖车，保留空载≥3 m/s与15 kg机身负载下坡（无速度要求）；当前64训练组合、16项固定验收。此前18项行为结果为修订前历史，启动命令保持。**

**2026-10-16 16:16：当前入口为[R1 V3](unified_v3/README.md)，500次诊断训练待用户启动。** 18项CPU检查、真实GPU启动、2+1次PPO、保存/精确恢复/导出/周期评估/单环境回放已验证。40 kg/±25°能力仍未通过；最终18工况评估12/18完整、1/18满足全部新指标，不能称任务达标或给出最小电机额定值。

V3沿固定路线计速度分、加强航向/净高和稳停、加入限加速度与5 cm停车漂移验收。上坡25°允许相对坡面低头0～10°并取消坡面默认腿姿惩罚，为前腿收缩后腿伸长留空间。仍为单Actor、全部64种工况从首批参加；15 kg与40 kg互斥，≥3 m/s仅空载。μ2.3、滚阻.02、初始牵引点.10 m、腿/轮96/48 N·m和PD80/3保留。默认仅迁移V2 model3900 Actor，重建Critic/Adam/探索噪声；每100次冻结16工况评估，500次结束后再判断后续。

**V2历史：[unified_v2/](unified_v2/README.md)，已于2026-10-16由用户停止。** 原run 10:19启动，最后日志4238、最新保存model4200；原来源、模型、日志保持。[停止核对与提案](unified_v2/records/2026-10-16_v2_stopped_next_plan.json)、[中期行为审查](unified_v2/results/2026-10-16_14-45-55_live_training_review/README.md)。

**V1历史：[unified/](unified/README.md)。** 2026-10-15～16的5000次训练已完成，原15+40 kg全速度组合未达标；模型、源码、日志保留用于回放和Actor初始化。[V1最终审查](unified/results/2026-10-16_09-30-37_final_training_review/README.md)。

**以下为原Stage1历史入口：平地、空载、无牵引，.2～.6m/s前进和25%零指令。** `multiskill/run.py --route 01`和本目录原`scripts/`仍指向Stage1。
**2026-10-15：首轮正式1000次PPO已完成，实际57/236/16与训练保存链路通过；固定行为验收待完成。**
训练15:43:42～16:38:20，最终`model_999.pt`；[训练审查、曲线及准确回放命令](results/2026-10-15_stage1_training_review/README.md)。
训练完成和CPU模型检查不等于已通过站立/行走验收。

task：`MultiSkill-LightHW-Wheel-Stage1-v0`；Actor/Critic/action声明为 **57/236/16**。
Actor仅角速度、重力、指令、关节角/速度与上一动作；Critic额外有线速度和地形高度。
旧60维Actor不兼容；训练只从零，`--checkpoint`仅用于本路线的回放和评估。
位置腿12+轮速4不变；线性腿scale .08/.14、URDF限位，轮scale5/clip37。
静止专用接触/姿态/速度约束在运动时关闭；平地正立与高度奖励仅适用于Stage1。

## 直接命令

工作目录 `/home/user/RL_lab/rl_training`，完整Isaac Sim对应的`lab`环境：

```bash
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/run.py check --route 01
python model_lighthw/training/multiskill/run.py smoke --route 01 --headless --num_envs 32
```

环境短验通过后，2次PPO更新检查日志与checkpoint保存，独立输出：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py --headless --num_envs 32 --max_iterations 2
```

本次已完成的正式Stage1训练命令（重跑会从零创建新run，不会续训model999）：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

训练结束会在`runs/<上海时间戳>/training_result.json`记录本次最终模型的准确路径。
将下面`/absolute/path/to/model.pt`替换为该文件中的checkpoint字段，不自动搜索旧模型：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/play.py --checkpoint /absolute/path/to/model.pt
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/evaluate.py --headless --checkpoint /absolute/path/to/model.pt --eval_seeds 43 44 45
```

播放是固定18秒起停GUI序列；评估是3seed×3case、共162秒仿真，无PPO更新。
每case失败立即记录并结束该case；不会在重置后继续拼成成功轨迹。
相同确定性初态下不同seed可能完全相同，因此不把多seed运行冒称域随机化鲁棒性验收。

## 文件与输出

- [配置来源](configs/README.md)：环境、独立奖励、PPO及控制契约，不复制公共资产。
- [评估协议](evaluation/README.md)：固定零/.3/.6起停、完整性与分项指标。
- `runs/`：每次独立PPO，`training/`含TensorBoard/checkpoint。
- `validation/`：100步无PPO短验。
- `playbacks/`、`evaluations/`：独立冻结回放和评估。
- [技能声明](skill.json)：候选进入/退出条件；未准备实机部署。

原规划曾采用逐级课程；用户已选择在`unified_v3/`从首批同时训练修订后的三类R1工况。原Stage1不会自动迁入联合训练，两个入口保持独立。
零指令四轮支撑要求与运动中的动态接触变化分开，移动低载只记录，不直接判失败。
