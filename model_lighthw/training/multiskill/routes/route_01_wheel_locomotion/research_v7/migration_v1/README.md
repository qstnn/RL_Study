# R1 migration_v1：部署合同、成功数据读取与隔离策略接口

2026-10-23：这是独立的数据与迁移接口，代码存在、CPU测试和离线拟合都不代表PPO准入。本目录不修改生产 V6/V8、研究 worker/controller 或全局进展。普通初态GPU16/4双向教师及严格bundle读取已通过，主流程已完成两个CPU拟合候选。实际命令起点允许按任务固定为上坡3.00 s、下坡7.48 s，执行前必须匹配所选方向全部成功教师的实际时钟；跨不同任务同起点的额外限制已删除，真实行为门保持。学生自主结果仍以独立运行档案为准，本项接口审查没有启动学生仿真或PPO。

当前实现 [contract.py](contract.py)、[dataset.py](dataset.py)、[policy.py](policy.py)、[schedule.py](schedule.py)、[bc.py](bc.py)、可选[准备阶段采样](preparation_sampling.py)、[student.py](student.py)、独立 [student_worker.py](student_worker.py) 及其 [记录接线](student_recording.py)/[运行核验](runtime_checks.py)，并有 CPU 测试。数据读取成功也固定输出 `student_training_ready=false`、`unified_actor_accepted=false`、`formal_ppo_authorized=false`。离线BC可使用完整双向教师标签；学生执行另要求所选方向的全部实际起点等于 `--preparation-seconds`，不会用teacher ready决定命令时机。没有正式PPO入口。

## 部署合同从哪里来

`contract.checked_contract()` 实际读取仓库中的 `wheel_unified_v6/spec.py` 和 LightHW `reference_observations.py`，核对历史长度、具名关节次序、动作执行器与历史动作 mask。它不依赖记忆中的列编号，也不导入 Isaac Sim。V7 记录的实际接口是 Policy 300、Critic 252、Action 16；新学生的 Critic 合同尚未实现，不能冒用 V8 的 268 维。

以下索引是 Python 半开区间；每一观测项单独存五帧，旧帧在前、当前帧在后：

| 观测项 | 全部五帧 | 当前帧 | 缩放/语义 |
| --- | --- | --- | --- |
| 机身角速度 | 0:15 | 12:15 | 机身坐标系，×0.5 |
| 投影重力 | 15:30 | 27:30 | 世界重力单位方向转到机身 |
| 速度命令 | 30:45 | 42:45 | 当前可部署命令 |
| 关节相对位置 | 45:125 | 109:125 | 相对默认位姿；四个轮位置置零 |
| 关节速度 | 125:205 | 189:205 | ×0.05 |
| 上一动作 | 205:285 | 269:285 | **80 列全部保留原始档案，但策略输入屏蔽** |
| 相对初始航向 | 285:290 | 289:290 | rad |
| 已知任务上下文 | 290:300 | 298:300 | 名义背载 kg/15、名义拖车 kg/40 |

观测关节顺序为每条腿的 `hip/thigh/calf/wheel`，依次 FR、FL、RR、RL；动作顺序为四腿的 12 个关节，然后四个轮关节。两者映射为 `[0,1,2,12,3,4,5,13,6,7,8,14,9,10,11,15]`。不能按动作顺序直接解释观测关节块。

动作保持原原生接口：腿 raw action ×`0.08/0.14/0.14` 后加默认角，按 URDF 范围限幅，PD=80/3；轮 raw action ×5，目标限于 ±50 rad/s，关节速度上限 60 rad/s，阻尼 2。物理步长 1 ms、控制步长 20 ms。教师的受力/接触/质心真值仅用于离线校验，绝不追加进 Actor 的 300 维输入。

`masked_observations()` 返回独立副本，保留原始观测供逐项核验。未来拟合可调用 `install_actor_mask(actor, optimizer)`，同时清零输入层 205:285 列的权重、梯度和已有 Adam 同形状状态；`validate_actor_mask()` 再检查。该函数本身不执行优化器更新。未来 checkpoint 必须保存此合同及 mask 状态，不能只靠输入数据偶然为零。

## 教师数据的接受条件

`read_episode()` 只接受完整已结束的物理档案，不接受失败回合中的局部好帧。它核验：

1. 请求与真实初态均为 `ordinary`：默认关节、坡面对齐姿态、零初速、0.416223 m 名义初始高度、40 kg 真实拖车、episode step 0。允许进入后的低姿态和部分轮支撑。
2. 请求为 GPU，运行档案中的机器人 Tensor 确认在 CUDA；机器人、拖车实际 USD 求解迭代均为 16/4，场景上限匹配。
3. 物理线保持 μ=3、滚阻系数 0.02、圆胎被动拖车、牵引杆高 0.04 m、机器人牵引点偏移 0.20 m、原机器人质量/质心、原执行器；关闭观测噪声，没有 Actor 推理或 PPO 更新。上下坡必须具有相同 `execution_physics`（仅设备字段排除比较）、几何、资产 SHA、实际牵引点与求解器设置。
4. 两份源清单非空，关键执行源都已封存，全部声明源 SHA 重算一致，机器人 URDF 匹配原合同，拖车资产与出处 checkpoint 记录 SHA。保留历史执行源，**不要求旧教师的控制器文件等于当前编辑中的控制器**。
5. 从原始 CSV、1 kHz 物理与力矩轨迹重新运行当前 `real_cart_audit.analyze(write_report=False)`，要求 `move_brake_hold_pass=true`。不能信任已保存的通过布尔值。完整标准仍是实际机器人与拖车净拖行至少 1 m、减速、2 s 过渡、5 s 保持，以及原滚动、姿态、接触/终止等验收；没有添加四轮同时支撑要求或 −0.5 m/s 诊断硬门。
6. 由初态和每控制步前的实际物理状态重建最新角速度、重力、关节位置/速度、命令、航向、载荷上下文，误差不超过 `1e-5`；同时核对每项五帧历史移位、上一执行动作与动作标签均有限。数据读取期间原始档案 SHA 变化也拒绝。

`read_teacher_bundle()` 还要求同时有 **+25°/40 kg 与 −20°/40 kg**，同物理配置且不重复输入同一回合。提供的任一回合失败，整批拒绝，不静默丢掉坏回合。当前读取器对应 V7 的 `seed43_case00` 单槽档案格式；要增加种子/槽位，先显式扩展文件格式解析，不把文件名改名当作新增证据。

导出包含原始 `observation`、`observation_masked`、实际执行的 `action`、`stage`、`episode_id`、`group_id`、`case_key`。阶段包括准备、等待、起步、运动、制动、过渡、保持。数据标记为 `teacher_labels_for_offline_fit_only`，不是已部署 Actor 的成功缓存。按**完整回合**划分训练/验证集，不能随机拆相邻帧冒充独立验证；本模块尚不分配该划分。

## 已执行的验证与只读命令

在 `research_v7` 目录执行 CPU 测试：

```bash
cd /home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/research_v7
/home/user/miniconda3/envs/lab/bin/python -m pytest migration_v1/tests -q
```

当前完整目录136 项 CPU 测试通过，覆盖实际观测片段重建、具名关节映射、80 列 mask 权重/梯度/Adam 状态、观测污染/非有限动作、GPU/求解器/普通初态核验、来源篡改/必要文件缺失、缺少上下坡或物理不一致、失败轨迹拒绝，以及冻结基线/残差、64工况覆盖、BC回合划分/均衡、学生接口、worker拒绝门/动作映射和按任务固定准备时钟。准备采样17项测试覆盖阶段边界/时钟/来源、默认数学行为保持和完整回合留出后的实际组采样。本项接口审查没有执行真实数据优化/PPO/学生仿真。最新实际候选只读prepare见[任务时钟核验](../results/2026-10-23_admission_continuation/admission_inventory/task_specific_preparation_validation.json)，准备细分验证见[采样记录](../results/2026-10-23_admission_continuation/admission_inventory/preparation_sampling_validation.json)；此前[双向与时钟核验](../results/2026-10-23_admission_continuation/admission_inventory/ordinary_dual_teacher_and_clock_review.json)、[worker验证](../results/2026-10-23_admission_continuation/admission_inventory/migration_v1_student_worker_validation.json)和[BC/学生验证](../results/2026-10-23_admission_continuation/admission_inventory/migration_v1_bc_student_validation.json)作为历史记录保留。合成优化测试只验证接口，不计为机器人BC。

`tests/fixtures/ordinary_observation_excerpt.*` 是实际 `ordinary_fast020_cpu` 的前 16 个控制步，仅供观测语义核验。该运行未完成拖行；fixture 元数据明确标记用途，不能用作成功教师标签。人工 GPU 元数据只存在于单元测试临时对象，不作为真实成功证据保存。

已对真实 `ordinary_fast020_cpu` 和 `ordinary_fast020_gpu` 调用读取器：前者因非 GPU 拒绝，后者经真实行为重审因未开始命令运动、未完成净 1 m/滚动等完整条件拒绝；没有生成训练数据。详情见 [负例复核记录](../results/2026-10-23_admission_continuation/admission_inventory/migration_v1_negative_checks.json)。

已对 `ordinary_gain600_gpu` 与 `ordinary_downroll040_gpu` 完成只读双向核验：同μ3物理profile、资产/求解器/普通初态/来源/观测重建全部通过，共4000帧，没有导出数据或拟合。可以用下列**数据读取**命令导出经重审的完整双向档案（不是训练命令）；部署时钟一致性仍由学生入口另审：

```bash
/home/user/miniconda3/envs/lab/bin/python migration_v1/dataset.py \
  --runs /绝对路径/已通过的普通初态上坡GPU档案 /绝对路径/已通过的普通初态下坡GPU档案 \
  --output /绝对路径/新的教师数据目录
```

输出目录必须不存在；拒绝时退出码 2，且不创建目录。通过时写 `teacher_data.npz` 与 `manifest.json`，后者记录数据、读取器、观测源、审计源及输入档案 SHA；它仍将所有训练/Actor/PPO门保持为 false。

## 冻结旧输出的独立策略容器

`policy.FrozenBaseTowingResidualActor(base_actor)` 保持单一部署图300输入/16输出，内部是冻结旧base加拖载residual，**不是旧V8单MLP**。构造器复制已核验架构和80列mask的base、冻结其参数/eval并记录实际张量SHA；残差末层初始化为零。名义拖载当前上下文>0才选base+residual；空载/15 kg直接选原base输出，不通过速度命令或教师真值切换。空载/背载在同输入/批处理方式下严格保留输出，但新物理环境下的行为仍须重新验证。

`trainable_parameters()`只返回残差，`project_constraints(optimizer)`保护残差权重/梯度/Adam mask；`assert_invariants(optimizer)`拒绝base变动、解冻、训练态或被纳入优化器。恢复要求版本/观测源/实现源/原base张量签名一致，不允许checkpoint更换受保护base。容器没有Gaussian、Critic、RSL-RL接线、导出器或真实checkpoint来源加载器，`checkpoint_provenance_verified`保持false。CPU测试中的手工残差扰动只验证保护不变量，不是已训练学生。

## 从首批铺满64工况

`schedule.permitted_catalog()`实际读取生产V8训练声明，并与真实full128项验收列表去重交叉核验为相同64种：空载11、背载33、拖载20，排除−25°拖载且载荷互斥。公共`catalog.py`只提供路线入口，没有64组合，因此不修改它，也不把无关R4入口变更纳入本模块的许可工况来源。

`make_schedule(n)`先铺全部64，再对余槽按来源声明的类别权重和重拖上下坡锚点确定性加权。默认模式n<64拒绝；明确`mode='diagnostic'`可生成少量接线配置，但永远不能准入。原V8配额代码实际在n=64时仅覆盖37种、n=128仅57种；新模块的64/128/256配置均覆盖64，未修改生产V8。

未来worker还必须将第一个真实动作前的`(slope,payload,tow,phase)`批次传给`validate_pre_action_conditions()`，检查实际64集合、单位、合法组合与阶段。配置表完整不代表运行已覆盖；当前测试输入均为合成元数据，`runtime_pre_action_verified`和执行来源认证保持false。详见 [环境/PPO接口合同](ENV_PPO_INTERFACE.md)。

## 后续仍需实现和实际验证

离线BC、严格学生加载与独立学生仿真worker代码现已实现，详见[离线迁移与学生接口](BC_STUDENT_INTERFACE.md)。worker仅实现两个重拖锚点，先做 strict load_student 和实际教师时钟核验，成功后才允许 AppLauncher 导入/启动和创建输出；没有教师接管或动作修正。固定准备默认3 s，旧来源下坡显式选择 `--preparation-seconds 7.48`，仅用从reset起算的时钟偏移；选择8 s需要所选方向真实教师实际同为8 s。保持原1.2 m制动协议，仍按实际1 m/减速/2+5 s及滚动验收。源码与记录接线的初次静态审查见[worker审查](../results/2026-10-23_admission_continuation/admission_inventory/migration_v1_student_worker_review.md)，当前按任务配置见[任务时钟审查](../results/2026-10-23_admission_continuation/admission_inventory/task_specific_preparation_review.md)。

旧时钟CPU pilot已由主流程完成并独立审查，保留为历史工件。新`--balance-preparation`默认关闭；启用时单独提高短暂进入/捕获组的采样机会，仍不改变输入、标签、base/mask或整回合划分。该更改新增一份签名源，旧pilot应连同已封存的9份源复核，不能在当前10份源下直接重签成新候选。详见[准备采样审查](../results/2026-10-23_admission_continuation/admission_inventory/preparation_sampling_review.md)。

后续仍须完成学生自主回放、旧能力成功参考缓存、统一环境奖励/状态机适配、64 工况实际rollout/PPO、准入签名与正式启动入口。已有完整双向普通 GPU 教师和拟合审查，下一步按各任务匹配的固定时钟检查教师关闭后的自主行为。与此同时，单独适配拖载低姿态、部分支撑、新拖车/牵引点与滚动验收，保留空载高速和 15 kg 背载门；然后检查从首批混合全部 64 工况的受控 PPO 接线。拟合误差降低或静态单元测试通过都不能代替自主行为验收。

上述改造应继续留在独立版本中，详见 [准入与最小迁移清单](../results/2026-10-23_admission_continuation/admission_inventory/README.md)。正式训练由用户启动；只有真实证据达到相应准入门后，主流程才提供经过验证的启动命令。
