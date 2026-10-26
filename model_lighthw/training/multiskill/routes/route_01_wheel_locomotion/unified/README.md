# R1：负载、速度、坡面、牵引联合 PPO

**历史V1说明。2026-10-16起，当前任务修订与电机需求实验移至[unified_v2/](../unified_v2/README.md)。** 下文保留本次旧实验的叠加载荷、全工况速度要求和已确认的指标问题；新目标、修复及提高的力矩上限均在V2，V1源码/模型/原始日志不回写。

本版本按用户要求把 R1 的站立、至少3m/s轮式行走、15kg机身载荷、25°坡面和40kg有轮拖车放入**同一个任务、同一个ActorCritic、同一个PPO优化器**。所有工况从第0次更新就参加训练，没有先训练低速再逐级开放的时间课程，也没有按工况切换Actor。旧Stage1的配置、模型和入口保留。

**2026-10-16当前状态：正式联合训练已完成5000次更新，最终model4999；目标行为筛查未通过。** 训练2026-10-15 19:03～2026-10-16 01:08，约6小时5分钟；末100次平均回合寿命4.041秒/20秒。最终模型seed43固定24个标准工况中12个完成18秒、2个通过全部指标（均为平地机身空载零指令）；另加平地0.3m/s起停通过。平地无拖车空载/15kg高速实速2.264/2.059m/s，15kg+40kg起步及±25°组合仍失败，尚未达到实际≥3m/s目标。

**发现需修正的指标与奖励实现：** `mdp.effort_usage`和力矩饱和日志误用1e9 Nm的PhysX求解器上限，导致effort正则几乎为零、饱和率漏报；真实PD仍按腿48Nm/轮24Nm限幅。独立GPU读取已确认，本次仅按真实上限重算诊断并保留历史模型/配置。不能依据原饱和率为0断言未触顶，也未证明这是全部失败的唯一原因。

[最终训练审查、曲线、25组回放与准确GUI命令](results/2026-10-16_09-30-37_final_training_review/README.md) · [结构化行为结果](results/2026-10-16_09-30-37_final_training_review/behavior_review.json)。实现阶段的10项CPU检查、17个Python编译、GPU短验及原Stage1保护记录仍见[原验证记录](records/validation.json)；validation中的3次更新模型是历史集成检查点。当前保留正式model4999待后续改进，不自动延长训练或标记为可部署。

## 同时训练的工况

| 项目 | 实际配置 |
|---|---|
| 坡面 | 0、±5、±10、±15、±20、±25°；11条真实长坡面 |
| 机身载荷 | 0、5、10、15kg；固定在Base_link局部(0,0,0.20)m，等效0.30×0.25×0.20m刚性载荷 |
| 有轮拖车 | 不连接、20、40kg；四个无驱动轮，真实质量、惯量和坡面重力 |
| 速度 | 非零指令0.2～3.1m/s；3.1用于为实际至少3.0m/s留出跟踪余量 |
| 停车 | 每次命令采样25%概率为零，6～8秒重采样 |
| 批次组成 | 80%按完整坡度×载荷×牵引混合；20%固定15kg+40kg+±25°目标组合 |
| 高速采样 | 普通移动样本40%锚定3.1；上述最大组合的移动样本70%锚定3.1 |
| 控制 | 12腿位置+4轮速、1kHz物理/50Hz策略；20秒episode |

完整混合部分保留更多平地样本。正式默认1024个并行环境，不同工况的rollout进入同一个PPO批次。命令仍为前进/停车，当前没有横坡、横移、转向或平坡过渡训练；坡面长度覆盖本轮20秒回合，不把不同任务分别训练后拼成“统一Actor”。

载荷通过平行轴定理合成机身质量、质心和惯量，并把主轴旋转写入PhysX；回读真实属性核对。没有只增质量而忽略惯量。载荷当前为刚性惯性模型，没有额外载荷碰撞外形。

拖车是独立Articulation，40kg名义质量由36kg车体和4个1kg被动轮组成，20kg工况按比例缩放。1m牵引杆用2000N/m弹簧、150Ns/m阻尼连接两端，每个1ms物理步施加等大反向力及牵引点力矩。支持压缩，所以会体现下坡拖车推车；不把40kg等同为固定392.4N阻力。

滚阻系数默认`0.02`，通过接触法向载荷计算每个被动轮的阻力矩`Crr×N×r×tanh(轮速/0.2)`，可用`--rolling_coefficient`修改。拖车重力按每个link真实质量在每个物理步施加；0kg牵引样本的拖车不连接、关闭重力并停放在坡道外。拖车没有Actor动作、轮速驱动或按目标速度产生的辅助推力。

## 明确的物理假设

默认接触摩擦系数为`1.3`，可用`--ground_friction`修改。这是尚未标定的高附着工况，不代表实机已具有该摩擦。机器人自重约11.6681kg，在15kg载荷、40kg拖车、25°上坡、滚阻0.02时，理想稳态所需摩擦系数约1.20；1.0不足以满足该组合，改变奖励不能消除这个约束。真实拖车的滚阻、牵引点和路面条件仍需标定。

牵引点高度、载荷重心和支撑范围也需满足抗倾覆力矩约束。已有初始姿态静力估算和最终模型实际回放均提示陡坡牵引稳定性不足；摩擦需求核对没有覆盖这一约束，详见最新审查。

只在本统一版本将轮速PD的damping由Stage1的1.0改为3.0，以改善重载上坡的稳态轮速误差。轮目标仍限制±37rad/s、轮力矩仍24Nm，腿48Nm与PD80/3、腿scale/关节限位不变，没有放大硬件上限。该改变会改变闭环响应，网络参数迁移不等于旧闭环行为逐值不变。

牵引杆1500N保护触发、拖车翻倒/脱离、机器人非轮碰撞/相对坡面倾倒、离开坡道分别记录失败。牵引保护检查所有1ms子步；其余默认时序统计为50Hz，不宣称已经验证热、结构或持续力矩能力。

## 观测与奖励

Actor输入为**290维**：原57维本体观测加1维相对初始航向，保留5帧历史。历史按观测项分块、从旧到新排列。只使用IMU、关节编码器、命令和历史动作；航向需要实机的相对航向估计。地形法向、真实载荷、牵引质量和仿真线速度仅供**242维Critic**，不送入Actor。两者均为512-256-128的标准MLP，动作16维。PPO每轮24步，5个epoch×4个minibatch，初始LR1.5e-4并按KL自适应；保留Stage1迁移的log_std（只有`--fresh`使用初始标准差0.25）。

奖励主要调整如下，实际函数和权重在源码中独立定义：

| 奖励项 | 权重 | 设计目的 |
|---|---:|---|
| 沿坡切向速度粗跟踪 | +4 | 高速严重欠速时仍有学习信号 |
| 沿坡切向速度精跟踪 | +2 | 接近目标后提高精度 |
| 平滑L1速度误差 | -0.5 | 避免远离目标时指数核信号过弱 |
| 绕坡面法向的yaw跟踪 | +1 | 抑制持续旋转 |
| 相对初始航向误差 | -0.5 | 保持直线，误差在Actor中可见 |
| 相对坡面倾斜 | -2 | 替代世界水平姿态约束 |
| 坡面法向离地高度 | -5 | 目标0.416m，不惩罚爬坡导致的世界高度变化 |
| 坡面法向速度 / 翻滚角速 | -1 / -0.05 | 抑制跳动，同时允许正常上坡竖直运动 |
| 归一化力矩使用率 | -0.05 | 轻度正则，避免强惩罚必要的承载力矩 |
| 关节限位 / 目标越界 / 动作变化 | -5 / -0.1 / -0.005 | 约束越界和抖动，不直接禁止使用合法边界目标 |
| 停车机身运动 / 轮转速 / 腿姿偏差 | -3 / -0.01 / -0.05 | 仅零指令启用；减弱固定腿姿约束，允许负载调整 |
| 停车四轮支撑 | +1 | 按坡面法向力、随负载缩放阈值；移动时关闭 |

上表是本轮配置的权重与设计目的。2026-10-16审查确认effort实现的归一化分母错误，其实际贡献近乎为零；应从显式actuator读取48/24Nm上限进行下一版修复。本次没有追改历史训练或重新训练。

支撑项使用四轮中的最小承载分数，避免仅靠三轮高载取得接近满分；不把运动中动态载荷转移当作失败。训练还按平地/上坡/下坡/15kg/40kg/停车/高速/最难组合分别记录实际速度、误差、接触、倾角、力矩、目标限幅、牵引力和终止。

配置/奖励/PPO/拖车/契约：[env_cfg.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified/env_cfg.py)、[mdp.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified/mdp.py)、[ppo_cfg.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified/ppo_cfg.py)、[cart.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified/cart.py)、[spec.py](../../../../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified/spec.py)。源码目录：`source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified/`，其中`env_cfg.py`是权重和环境配置，`mdp.py`是奖励/载荷/状态采集，`cart.py`与`cart_asset.py`是拖车，`ppo_cfg.py`是PPO参数，`spec.py`是版本契约。参考[PhysX Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)，并以本机GPU实际读写探针和规范URDF交叉验证轴系：先设置质心主惯量，再设置COM主轴姿态；本机getter返回沿link轴表达的完整惯量。启动时逐一回读质量/COM/惯量，并确认非机身连杆惯量保持不变。

## 一次联合训练命令

在完整激活的lab环境、仓库根目录执行。默认读取原Stage1正式`model_999.pt`，迁移Actor/Critic和log_std；最新观测列保留原映射，新历史/航向/特权输入列初始为零。**优化器重新初始化**，这是新任务的网络初始化，不是原任务完整续训。旧文件只读。

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified/scripts/train.py \
  --headless --num_envs 1024 --max_iterations 5000 --seed 42 \
  --ground_friction 1.3 --rolling_coefficient 0.02
```

这是一个有界的5000次联合训练预算，所有目标工况从开始就在同一环境中出现；5000不是通过验收的保证。预算结束保存并退出，不自动晋级、续训或评估。若要全新随机初始化，加`--fresh`。`--warm_start`只接受原57维Stage1模型；本统一任务后续使用`--resume /absolute/path/to/model.pt`恢复网络、Adam、实际学习率和更新编号，另开run，物理状态重新reset。

## 回放与评估

统一模型不能通过旧Stage1的play.py加载。训练结束读取新run的`training_result.json`取得准确路径，然后：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified/scripts/play.py \
  --checkpoint /absolute/path/to/unified/model.pt \
  --slope_deg 25 --payload_kg 15 --towed_kg 40 --speed 3.1 --seed 43
```

单环境GUI跟随视角，固定18秒：零命令3秒、前进8秒、停车7秒；下坡用`--slope_deg -25`，平地用0。不带`--headless`。若训练使用其它摩擦/滚阻参数，回放需显式传入相同值，防止换工况后误标为同任务结果。

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified/scripts/evaluate.py \
  --headless --checkpoint /absolute/path/to/unified/model.pt --eval_seeds 43 44 45
```

固定评估为平地/±25°、0/15kg载荷、0/40kg牵引、零/3.1m/s命令，共24工况×3seed。每个工况独立判定，最大1296秒仿真；提前终止保留失败，不能拼接自动重置后的帧。高速组合要求实际巡航均速≥3.0m/s，同时检查误差、姿态、航向（绝对误差P95≤10°）、停车、支撑和碰撞。相同确定性初态的多seed不等于随机扰动鲁棒性验收。

## 输出位置

所有新输出位于本目录下：

- `runs/<时间戳>/`：联合训练；`training/`含checkpoint、TensorBoard和`condition_metrics.csv`。
- `validation/`：100步零PPO物理短验，以及显式`--validation`的至多5次PPO集成检查；不作为正式训练候选。
- `playbacks/`、`evaluations/`：冻结模型的独立CSV、分工况JSON与`evaluation.json`。
- 每run均保存实际env/agent YAML、请求/版本、源码快照、机器人/拖车资产哈希、真实物理检查、起止状态和完整console。
- `exported/actor.pt`：一个290输入、16输出的TorchScript Actor及其契约；必须按约定构建5帧观测历史。导出检查使用实际GPU观测与原策略比较，不包含Critic或其它Actor。

原Stage1文件位于上级目录的`runs/2026-10-15_15-43-42_148523/`；本目录不会覆盖它。
