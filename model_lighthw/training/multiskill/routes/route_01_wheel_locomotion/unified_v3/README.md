# R1 V3：路线跟踪、坡面调姿与稳停诊断

更新：2026-10-16，Asia/Shanghai。**V3 的第500次结果已审查，原运行已由用户停止；目前处于迁移校准验证阶段，全任务仍未达标。** 最新证据与下一步见 [校准审查](results/2026-10-16_calibration_review/README.md)。本轮发现旧路线四轮刚体3000 deg/s覆盖没有进入原R1；隔离校准已恢复该设置并对照轮增益1/3，原生产配置保留作历史复现，不能把下方旧命令当作已批准的迁移训练入口。

**本次任务范围修订：−25°下坡不训练也不验收拖车（包括20/40 kg）。** 该坡度保留空载实际速度≥3 m/s，以及15 kg机身负载下坡（无移动速度门槛）；其他坡度的牵引设置保留。训练组合64种，固定验收16项。本轮校准继续使用这一任务范围。

## 历史V3训练命令（保留复现，当前不建议原样续训）

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v3/scripts/train.py \
  --headless --num_envs 1024 --max_iterations 500 --seed 42 --eval_interval 100
```

默认只迁移V2正式run `2026-10-16_10-19-29.659784/training/model_3900.pt` 的Actor；新建Critic、Adam与0.25探索标准差。该模型在8秒控制对照中的停车表现优于model4200，但并未通过全部任务。GPU逐样本验证迁移前后Actor函数最大误差0。没有继承V2累计更新次数/优化器，也不使用本轮仅3次更新的测试模型作为正式起点。

历史命令中的500是新增PPO更新数。每100次暂停主训练，用同一已保存Actor进行seed43的16工况冻结评估，记录后自动恢复主训练；没有第二个学习策略。正常行为不通过会记录失败，不中断训练；若评估程序本身报错则训练报错停止，避免悄悄漏检。最多执行500次，结束后查看分工况结果再决定后续，不自动追加长训。评估期间终端显示`Pause PPO`属于正常流程，详细输出见`periodic_evaluations/update_0100_launcher.log`等。

## 任务与物理条件

- 同一Gym环境定义、一个290/242/16 ActorCritic和一个PPO优化器。64种坡度/质量组合从第一批参加，无课程；重置后有1秒零速落地窗口，随后按斜率限制启动。
- 15 kg机身载荷和40 kg有轮被动拖车互斥；中间质量0/5/10/15与0/20/40也参加。坡面0/±5/±10/±15/±20/±25°；−25°的20/40 kg拖车组合排除，其他坡度的拖车保留。
- ≥3 m/s仅为空载要求，参考最高3.1；有载/拖车的.2～.6 m/s是训练参考，不是最低速度验收条件。
- 高附着假设μ=2.3，滚阻系数.02，初始牵引点安装净高.10 m，腿/轮上限96/48 N·m，腿PD80/3。均为实验条件，未作真实路面标定。动态牵引净高随姿态和下沉变化。
- 当前仍为长直恒坡与直行/停车，不包含坡道接缝、转向、横坡和实际地形随机化验收。
- 原规范URDF、17个mesh及V2全部213份来源文件保持；旧模型、日志和R6工作区不改。

## 修改了什么

| 项目 | V3实现 | 原因 |
|---|---|---|
| 速度方向 | 固定路线切向/横向计分，独立于机身朝向 | 防止偏航后沿错误方向仍获满跟踪分 |
| 正跟踪奖励 | 姿态、航向、高度、牵引净高连续缩放，最低保留.05 | 鼓励安全推进，所有负代价仍生效 |
| 有载移动 | 有界前向进度奖励，横向/后退惩罚；不要求精确跟上参考速度 | 与有载无最低速度要求一致 |
| 起停 | 空载1.0、有载.3 m/s²；目标与实际指令分别记录 | 减少骤起骤停，停车奖励等实际归零再激活 |
| 航向/净高 | 航向权重-2；牵引8 cm提前惩罚、权重-6 | 失败前持续提供纠正信号；5 mm保护边界不变 |
| 稳停 | 运动权重-6、四轮支撑+2；增加坡面漂移代价 | 全身零指令保持，不只固定腿角度 |
| 坡面姿态 | 上坡25°允许机身相对坡面低头0～10°；坡面取消默认腿姿惩罚 | 为前腿收缩、后腿伸长留空间；不锁死关节姿态 |

上坡姿态使用**俯仰容许区间**，并非把所有关节拉向一组新默认角度。25°上坡对应机身相对世界约15～25°，保留横滚约束、高度、承载、碰撞和牵引净高要求。坡度较小时容许区间按比例缩小。下坡只给相对坡面±5°以内的调节空间，没有强制把上坡姿态机械反向；固定姿态对照中这种反向调整会降低后牵引点。

Actor观测仍为原来的五帧角速度、重力、指令、关节角/速度、动作和航向，不新增真实坡度、质量、世界位置或线速度。坡面姿态参考和漂移真值用于训练奖励/评估，不能因此声称部署端已有地形或定位传感器。动作仍以原默认腿角度为偏置，但坡面没有回默认角度的奖励惩罚。

## 能力验证结果与边界（修订前历史）

以下探针包含原来的−25°拖车工况，原始记录保留；该工况已从当前训练与验收要求中移除。

先做了72个独立GPU条件探针，均PPO0、全重力、原保护边界、首次终止截断；每项最多8秒：

| 对照 | 数量 | 结果 |
|---|---:|---|
| model3900/model4200/默认姿态，PD80/3、120/4.5、160/6，牵引安装.10/.15 m | 42 | 36个40 kg陡坡全部提前失败；6个平地牵引/坡面15 kg控制项存活8秒 |
| 静力估算前馈及轮制动反馈，仅作真值诊断 | 12 | 40 kg/±25°全部未稳住8秒；不能把静力可解当动态可用 |
| 固定四轮支撑位置，前短后长0/5/10°，空载/15 kg/40 kg、±25° | 18 | 空载和15 kg共12项存活8秒；6项40 kg仍失败；15 kg仍有约.06 m/s滑移 |

几何上，5°低头能使前轮相对机身的垂直距离缩短约3.0 cm、后轮增加约2.8 cm；在名义.416223 m根部高度下，后部牵引净高由约8.62 cm增至12.06 cm。真实仿真出生根高仍为.43 m，着地后会下沉；几何估算不等于动态净高保证。10°姿态也可达，但40 kg上坡没有因此稳定，不能认定越前倾越好。

因此本版开放调姿，并未采用更高PD或.15 m安装高度。上述失败没有证明当前机器人必然做不到，也没有证明加大电机即可解决。**96/48 N·m暂保留；只有完成任务的1 kHz轨迹才可作为该策略力矩需求证据，仍不是最小电机额定值。**

原始证据：`validation/capability_01/`、`capability_static_02/`、`posture_03/`，均有CSV、JSON、1 kHz NPZ和实际参数。静力/逆运动学探针放在`diagnostics/`，没有接入生产Actor或额外驱动拖车。

## 本次任务范围修订检查

22项CPU检查通过；168个真实GPU环境运行100控制步、0次PPO，实际64工况且−25°拖车数为0，坡度/质量/控制维度保持。零动作20次终止只用于检查有限步进和重置，不能当作下坡能力通过。新16项验收矩阵已做静态核对，本轮未重跑完整行为评估、未启动正式训练。当前生产源码与该GPU短验快照一致；213份V2来源保持。[修订记录](records/2026-10-16_downhill_scope_revision.json)。

## 修订前已完成的集成验证（历史）

- 18项CPU回归：真实奖励方向/净高调用点、坡面姿态、默认腿姿退出、实际命令状态机、完整停车窗口、负载无速度门槛、Actor函数与噪声隔离。
- `validation/smoke_05/`：168环境100步，66工况、290/242/16、质量/COM/惯量/摩擦/真实电机限幅核对通过。零动作发生31次终止，只证明有限步进及重置链路。
- `validation/train_06/`：1024环境2次PPO、Adam40，真实触发18工况周期评估，父训练物理状态与checkpoint哈希保持，随后正常保存/导出。
- 该冻结评估12/18完成协议时段，**0/18通过全部新指标**；40 kg陡坡仍早期失败，停车漂移/四轮承载等仍不达标。不能用2次更新判断奖励学习效果，且新旧协议/初始化不同，不能直接比较通过率。
- `validation/play_step_08/`：单环境突变命令回放完整跑完，250帧稳停窗口；漂移未达标。
- `validation/evaluate_final_09/`：最终代码再次完成18工况评估，实际6空载/6载荷/6拖车、0/±25°和物理质量逐项核对；12/18完整、1/18全指标通过。
- `validation/resume_07/`：精确恢复网络、Adam40与学习率，仅追加1次到Adam60、model2；实际Actor导出最大误差5.25e-6。

上列2+1次PPO和18项验收是修订前证据，原始快照、模型和指标均保留在`records/validation.json`及各run。当前64组合/16项验收的改动与短验记录见`records/2026-10-16_downhill_scope_revision.json`；本次没有重新进行PPO或正式行为验收。

## 验收协议

固定16项：平地6项、+25°上坡6项、−25°下坡4项。平地/上坡覆盖空载、15 kg机身载荷、40 kg拖车的零指令/行进后停；−25°只覆盖空载和15 kg机身载荷。

- 零指令：20秒，取15～20秒完整250帧停车窗口。
- 空载行进：先静止3秒，目标3.1至11秒，再限加速度归零；实际指令归零后过渡2秒，再稳停5秒，30秒预算。全速指令建立并再稳定1秒后计算巡航速度，要求实际≥3.0 m/s、MAE≤.15。
- 有载行进：先静止3秒，以.3参考前进至少1 m，减速归零后过渡2秒+稳停5秒，60秒预算。没有最低移动速度，也没有有载速度MAE门槛；仅原地站住不能算行进通过。
- 稳停均值速度≤.05 m/s、角速≤.05 rad/s、四轮承载比例≥.95，新增5秒坡面内漂移≤.05 m。
- 姿态P95≤15°、航向绝对P95≤10°、高度误差P95≤.08 m；不得发生非轮接触、牵引保护或其他提前终止。5 mm净高和45°翻覆保护保持。
- 首次终止即截断，不把自动重置后的轨迹接成成功；记录完整50 Hz行为与1 kHz各关节力矩。固定初态多seed可能相同，只是筛查，不宣称随机鲁棒性。

## 奖励、配置与输出在哪里

- [环境与奖励权重](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v3/env_cfg.py)、[奖励函数](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v3/mdp.py)、[指令与起停状态机](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v3/commands.py)。
- [PPO参数](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v3/ppo_cfg.py)、[模型/动作/物理合同](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v3/spec.py)。
- 训练输出：`unified_v3/runs/<上海时间戳>/`，终端首行会打印绝对路径。
- `training/`：TensorBoard事件、checkpoint、`condition_metrics.csv`、`motor_metrics.csv`；`console.log`为控制台日志。
- `periodic_evaluations/update_0100/`等：各case行为CSV、力矩NPZ、指标JSON及`evaluation.json`；`index.jsonl`汇总每100次结果。
- `params/env.yaml`、`params/agent.yaml`是本次实际实例化配置；`request.json`记录CLI和实验条件；`source_snapshot/`与`source_manifest.json`记录来源。
- `training_result.json`给最终checkpoint准确路径；`exported/actor.pt`及其合同是带5帧输入要求的Actor。

TensorBoard可用：

```bash
tensorboard --logdir model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v3/runs
```

## 回放、完整评估与续训

将下面checkpoint替换为正式run里`training_result.json`或周期评估记录中的准确路径。GUI回放默认15 kg、25°、.3参考：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v3/scripts/play.py --checkpoint /absolute/path/to/model.pt
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v3/scripts/evaluate.py --headless --checkpoint /absolute/path/to/model.pt --eval_seeds 43
```

`--payload_kg 0 --towed_kg 40`选择拖车，`--payload_kg 0 --towed_kg 0 --speed 3.1`选择空载。可追加`--command_profile step`执行原突变指令压力检查；该结果单独标记，不代替正常限加速度协议。更改过力矩/摩擦/牵引/加速度的模型，回放需显式传入其保存参数，签名不同会拒绝。

同版断点续训使用`--resume /absolute/path/to/model.pt --max_iterations N`，N是**额外**更新数，恢复网络、Adam和学习率，但仿真环境/RNG重新初始化，不能称逐步完全相同的轨迹复现。V2及本次范围修订前的V3合同不能作为当前`--resume`；可用`--initialize_from`只迁移Actor。默认V2 model3900初始化仍可用。下一步是否续训需先看100/200/300/400/500的行为趋势，不能只看总reward或训练完成状态。
