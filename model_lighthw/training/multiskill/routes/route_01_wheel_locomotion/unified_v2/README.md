# R1 V2：互斥载荷/拖车与电机力矩需求实验

**2026-10-16 15:22状态更新：用户已停止本轮训练。** 已确认启动/训练进程均退出，日志到4238（共4239次），最新保存model4200（4201次更新）；最后100次平均episode9.088 s。原run.json仍写running，属于手动停止后的旧状态，原始记录保留。本轮只确认停止并提出下一步方案，没有启动V3或新PPO。[停止核对与调整提案](records/2026-10-16_v2_stopped_next_plan.json)。

本版本按2026-10-16用户确认的目标运行：**空载速度要求至少3 m/s；15 kg机身载荷与40 kg有轮拖车分别训练，不同时施加；三类工况均包含±25°坡面**。一个Actor、一个Critic、一个PPO优化器，所有工况从第0次更新参加训练。

任务为`MultiSkill-LightHW-Wheel-MotorSizing-v0`。Actor/Critic/action仍为290/242/16；V1的代码、模型及审查保留在[unified/](../unified/README.md)。V2正式训练于2026-10-16 10:19启动，现已由用户停止，尚未通过完整新目标。

**2026-10-16 15:05中期检查：** 正式V2已于2026-10-16 10:19启动，截至此记录完成4122/5000次，主训练继续运行。最新100次平均episode约8.61/20 s；冻结model3900的18个标准工况中11个活满20 s、7个按现有协议通过，另加平地低速1个通过。主要缺口为40 kg拖车坡面初始净高、空载高速跑偏/刹停，以及下坡停车支撑；没有完整R1达标结论。[中期审查与曲线](results/2026-10-16_14-45-55_live_training_review/README.md)。

**2026-10-16 10:14实现与集成验证记录：** 15项CPU回归、20个Python文件编译、168环境采样短验，以及最终1024环境2次PPO+1次恢复更新通过；Adam40→60、导出与GPU Actor最大误差≤4.77e-6。冻结旧Actor两档力矩38个case只用于诊断。生产回放入口已核对失败截断，以及15 kg上坡完成1 m后的250帧稳停窗口；该完整轨迹仍因高度失败，3次更新集成模型也未达到任务要求。[完整验证记录](records/validation.json)。

| 工况 | 正式1024环境占比 | 训练移动指令 | 验收目标 |
|---|---:|---|---|
| 空载、无拖车 | 40% | 0.2～3.1 m/s，移动样本40%锚定3.1 | 高速实速≥3.0 m/s及稳定起停 |
| 仅机身载荷5/10/15 kg | 30% | 0.2～0.6 m/s参考 | 15 kg完成行进与稳停，不设移动速度门槛 |
| 仅拖车20/40 kg，机身不加负载 | 30% | 0.2～0.6 m/s参考 | 40 kg完成行进与稳停，不设移动速度门槛 |

每类都含0、±5、±10、±15、±20、±25°；平地增加采样，各类20%环境锚定其最大质量及±25°。1024环境包含66个互斥工况，数量410/307/307。命令每6～8秒重采样，25%为停车。负载低速指令提供学习方向，既不是最低速度要求，也不作为验收误差门槛。

## 力矩与物理条件

实验默认**每个腿关节96 N·m、每个轮关节48 N·m**，原实验为48/24 N·m。这里是显式PD执行器的关节输出侧限幅；提高上限供任务学习和需求测量使用，不能直接当作已经确定的最小电机额定力矩。后续可以用`--leg_torque_limit`、`--wheel_torque_limit`继续调整。

用户允许的实验假设明确记录为：**路面静/动摩擦系数2.3，初始牵引点离坡面法向高度0.10 m，拖车滚阻系数0.02**。这些值可分别通过`--ground_friction`、`--hitch_height`、`--rolling_coefficient`调整，每次写入配置与checkpoint签名。

机器人空载自重约11.6681 kg。按40 kg拖车、25°、滚阻0.02、牵引力平行坡面的理想稳态估算，所需摩擦系数约2.13，原1.3不足；降低速度和增加电机力矩都不改变这个静力约束。2.3仅是获准测试的高附着假设，尚未实测标定，也不证明动态抗倾覆成立。

牵引点局部坐标为机器人(-0.38,0,-0.33)m、拖车(0.35,0,-0.20)m；0.10 m是初始高度，运动中会变化。新增两端牵引点离坡面不足5 mm时终止，防止用地下连接点获得虚假稳定性。牵引杆仍为1 m、2000 N/m弹簧及150 N·s/m阻尼的力学模型，没有杆体/安装支架碰撞网格；实机安装、结构与热能力另需验证。

载荷质量/质心/惯量合成、四轮无驱动拖车、滚阻与下坡推车机制沿用V1。12腿位置+4轮速、腿PD80/3、轮PD阻尼3、动作缩放和轮速目标±37 rad/s保持原控制定义。物理1 kHz、策略50 Hz、训练episode20 s。

## 奖励与记录

奖励函数在[mdp.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v2/mdp.py)，权重及环境配置在[env_cfg.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v2/env_cfg.py)，PPO在[ppo_cfg.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v2/ppo_cfg.py)。每次运行的`params/env.yaml`与`params/agent.yaml`保存实际实例化配置。

修复effort奖励和饱和统计：按真实`actuator.effort_limit`及其关节索引归一化，避免误用PhysX的1e9求解器保护值。effort权重仍为-0.05，保持轻度正则，让策略可以使用必要承载力矩。速度粗/精跟踪4/2、速度误差-0.5、坡面姿态-2等保留，针对冻结轨迹中载荷下沉与低牵引点触地增加三项调整：

| 调整 | 权重 | 依据与实际含义 |
|---|---:|---|
| 坡面法向高度平方误差 | -5 → -100 | 目标仍0.416 m；8 cm误差的瞬时成本从0.032增至0.64 |
| 牵引点间隙缺额平方 | -2 | 仅有拖车且净高不足4 cm时连续惩罚，5 mm失败边界保留 |
| 非超时失败 | -100 | 乘20 ms后每次失败扣2；正常episode超时不扣 |

共19项奖励。新版同时修正任务采样，不把负载硬推至3.1 m/s。这些是根据失败证据作出的学习信号调整，效果需要正式训练后的行为验收。

`training/motor_metrics.csv`包含探索、失败和重置邻近样本，只用于诊断；电机需求证据取冻结且通过验收的轨迹。

每次运行的输出相互隔离：

- `runs/<时间>/training/`：TensorBoard、checkpoint；`condition_metrics.csv`按空载/载荷/牵引、坡向、移动/停车统计。
- `training/motor_metrics.csv`：每次PPO更新、每组工况、16个关节的1 kHz峰值、RMS、限幅前请求、饱和/裁剪比例与累计时间、转速、机械功率/功。
- `playbacks/`、`evaluations/`：50 Hz行为CSV，逐关节1 kHz原始`*_motor_trace.npz`及`*_motor.json`，含P95/P99/P99.9、最长连续高力矩时间。
- `runtime_check.json`、`request.json`、`contract.json`、`source_manifest.json`与`source_snapshot/`：实际执行器限幅、物理条件、工况和代码来源；`console.log`是完整控制台输出。
- `validation/`：集成短验和冻结对照，不混入正式训练。

**仅完整通过行为验收的轨迹才设置`requirement_evidence_valid=true`。** 失败轨迹的翻倒/碰撞峰值保留作诊断。峰值贴近上限会标记`peak_censored_by_limit`，限幅前请求不等于实际输出。数据是关节输出侧力矩；没有减速比/效率换算，机械RMS也不是已验证的持续热额定值。

## 正式训练命令

以下为本轮已启动的正式训练命令；再次执行会创建另一个实验，不会接入当前作业。需新开实验时，从仓库根目录的完整`lab`环境执行：

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v2/scripts/train.py \
  --headless --num_envs 1024 --max_iterations 5000 --seed 42 \
  --leg_torque_limit 96 --wheel_torque_limit 48 \
  --ground_friction 2.3 --hitch_height 0.10 --rolling_coefficient 0.02
```

默认从V1正式`2026-10-15_19-03-12.895963/training/model_4999.pt`迁移Actor与log_std，**Critic及Adam重新初始化**。目标、奖励和动力学已变化，这是新实验，不是原5000次的精确续训。`--fresh`可从零；`--initialize_from /absolute/model.pt`接受已知同维度V1/V2模型。无需启动旧Stage1。

同一V2实验中断恢复用`--resume /absolute/model.pt`，恢复Actor、Critic、Adam和迭代号，物理签名必须一致。`--max_iterations`是本次追加更新数，续训时填剩余预算。若提高力矩上限，使用`--initialize_from`开新run并重置Critic/Adam；不能用`--resume`把不同上限的结果混为同一个实验。

## 回放与任务验收

训练完成后，读取该run的`training_result.json`获取最终checkpoint。下例将`CHECKPOINT`改为**V2正式训练模型**的绝对路径：

```bash
CHECKPOINT=/absolute/path/to/v2/model_4999.pt
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v2/scripts/play.py \
  --checkpoint "$CHECKPOINT" --slope_deg 25 --payload_kg 0 --towed_kg 40 --speed 0.3
```

这条是40 kg拖车上坡GUI回放。15 kg上坡改为`--payload_kg 15 --towed_kg 0`；空载高速用`--payload_kg 0 --towed_kg 0 --speed 3.1`；下坡用`--slope_deg -25`。回放物理参数必须与checkpoint签名一致，使用上述默认值时无需重复；若训练改过，回放也必须传入对应值。

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v2/scripts/evaluate.py \
  --headless --checkpoint "$CHECKPOINT" --eval_seeds 43 44 45
```

每seed评估18个case：0/±25° × 空载/15 kg载荷/40 kg拖车 × 零指令/移动。空载为18 s起停，移动目标3.1 m/s；载荷/拖车先静止3 s，以0.3 m/s参考指令行进**1 m**后停车7 s，最后5 s评估稳停。移动速度不设门槛，`--eval_distance`可以改变目标距离；60 s是一次筛查预算，超时只判本次未完成，不能解释为用户新增最低速度要求。

同时检查完整窗口、姿态、航向、高度、四轮静止支撑、非轮碰撞、牵引保护及离地间隙，失败后立即结束该轨迹。1 m恒坡筛查不等于整段坡道、坡面过渡、长期耐久或随机鲁棒性验收。相同确定性初态的多seed结果可能相同。

验证记录见[records/validation.json](records/validation.json)，力矩对照结论见[records/motor_caps_review.md](records/motor_caps_review.md)。冻结旧Actor的两档上限对照入口为`scripts/compare_motor_caps.py`，它不做PPO更新，不替代V2正式训练后的验收。
