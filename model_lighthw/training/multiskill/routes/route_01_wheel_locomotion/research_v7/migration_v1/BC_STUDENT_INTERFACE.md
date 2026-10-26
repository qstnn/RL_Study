# 离线残差拟合与学生独占动作接口

当前已实现离线BC代码与CPU接口测试；普通上下坡GPU完整教师及严格双向4000帧读取均通过，主流程已完成两个CPU拟合候选。真实拟合结果仍不代表自主通过。实际准备时钟允许按任务预先固定：上坡3.00 s、下坡7.48 s，执行前必须匹配所选方向全部教师的实际起点。跨不同任务强制同起点是初版额外限制，现已删除；没有改变任何真实拖行/制动/滚动验收。后续自主结果以独立运行档案为准。

旧时钟2000步CPU pilot经[独立只读审查](../results/2026-10-23_admission_continuation/admission_inventory/bc_clock_pilot_independent_review.md)：来源/mask/base/row拟合一致，但最初降姿误差明显高于总体loss。随后同参数启用准备细分采样的[独立比较](../results/2026-10-23_admission_continuation/admission_inventory/bc_preparation_pilot_comparison.md)证实早期误差降低，也记录了部分命令后误差增大；没有学生仿真或PPO结论。旧pilot及其9份源快照保留为历史工件，不重签。当前严格加载器拒绝旧pilot与当前10份源码签名不一致，这是版本保护，不是把旧拟合结果改判成物理失败。

## 真实入口的顺序

`bc.fit_from_runs(runs, baseline_checkpoint, output, ...)`首先调用`read_teacher_bundle`重审所有输入原始轨迹：同物理、ordinary、实际GPU16/4、+25°/40 kg与−20°/40 kg、完整1 m/制动/2+5 s及滚动等既有门。任一失败或缺方向都会在创建输出、加载基线和优化前退出，不能拿一个上坡教师或成功局部帧绕过。

随后校验每份教师出处的同一baseline文件SHA、具名300维观测/16维动作合同、ELU层形状、80列历史mask、有限float32权重；只读Actor，不恢复旧Critic或Adam。每次实际拟合只更新拖载residual，按`case×stage`均匀选组再均匀选帧，避免长保持窗掩盖起步/制动。损失是16维raw动作的平方误差，默认各维scale=1，记录所有scale；不按低loss直接宣布物理成功。

可选`--balance-preparation`调用独立`preparation_sampling.py`，按已认证回合中的实际控制前时钟、进入路径时长、prepared标志与下坡捕获阶段，细分为路径进入、捕获（下坡再分roll/brake/settle）、准备完成后的等待。命令开始后的起步/运动/制动/过渡/保持组不变。默认关闭时返回原数组对象、沿用原采样器及随机数序列；启用时只替换离线`group_id`，原始300维观测、动作标签、stage、成功认证和完整episode划分全部保持。教师内部阶段仅用于离线采样，绝不进入Actor输入。

启用后仍按所有`case×stage`组等概率采样，并非先对上下坡各取一半。实际已核验的上坡8组、下坡10组对应总权重44.44%/55.56%；每组概率及完整回合留出后的真实训练概率都写入`sampling_configuration`。元数据还逐组记录源文件SHA、成员回合、每个帧索引和实际时间，避免仅靠组名宣称起步获得足够训练。评估同时保留原阶段分组的`original_stage_final_train`，便于与旧pilot比较；仍须另看最初降姿的关节目标/轮速/力矩估计误差，不能只比较新组MSE。

细分前逐项核对记录长度、20 ms时钟、CSV状态/命令对齐、路径时钟、原始观测与执行动作一致性；来源文件读前读后重新验SHA。[准备阶段采样核验](../results/2026-10-23_admission_continuation/admission_inventory/preparation_sampling_review.md)使用真实`ordinary_up_fixed8_gpu`和旧`ordinary_downroll040_gpu`只读验证18组；该历史记录中的全bundle时钟拒绝仍可用默认无case检查复现。当前worker按所选任务核对对应实际起点。

用户许可在证明μ不足时研究提高到4，但当前读取器仍是严格μ3物理版本。若未来选择μ4，须显式修改并签新物理合同，重新取得同μ4普通GPU上下坡成功证据，不能将μ3上坡与μ4下坡混合用于BC；本接口尚未作该扩展，也没有改变实际摩擦参数。

划分只接受完整`episode_id`。若提供验证回合编号，整个回合的所有阶段一起留出，训练仍须包含上下坡；禁止邻帧随机拆分。若当前每方向只有一个成功回合，可全部用于拟合，此时报告`held_out=null`和`unavailable_no_independent_holdout`，不能伪造独立验证，也不额外增加“必须再跑某数量教师”的物理硬门。决定学生能力的是随后教师关闭的自主回放。

训练前/后核对迁移代码、审计源和所有输入轨迹SHA。成功执行拟合才写新目录的`student.pt`和`manifest.json`，不覆盖已有结果。checkpoint只有部署Actor（冻结base+residual）及出处/拟合元数据，**不含Critic、优化器或任何追加给Actor的特权观测**；元数据中的原始物理参数不是网络输入。训练权重拟合结果始终标记`offline_fitted_student_candidate`、`autonomous_behavior_accepted=false`、`formal_ppo_authorized=false`。

CLI位于`bc.py`，可先执行`python migration_v1/bc.py --help`查看参数。当前不提供可以绕过证据的开关或正式PPO启动命令。真实双向教师完成后，再由主流程选择完整episode数据与拟合参数，执行用户已授权的离线迁移。

## 冻结的基线是什么

来源为教师request指定的`unified_v5/initialization/selected/model_init.pt`，文件SHA：

```
3cd3eac8bf1f33ac915d03feb73feb1c2c215065ade56ff00320d3c1aabb1d30
```

它是历史14/16边界通过的能力候选，文件有效且已有300/16接口与历史mask；它不是当前新物理路线的通过证明。其较早`selection.json`写过`training_ready=true`，但同目录稍后的`readiness.json`明确为false，记录高坡40 kg拖行失败、噪声下空载不稳定、恢复候选保留能力未全通过。BC代码不读取旧selection的true来授权任何阶段。

冻结该base可保护已有输出函数，不能证明新μ3/新拖车/16/4/无噪声环境中空载高速和15 kg行为已通过。输出严格相同仅在相同输入及相同row/batch执行方式下成立；新学生默认row，与旧研究回放一致，显式batch需另行记录/审查。保留能力仍要实际新环境回放。

## 学生独占执行接口

`student.load_student(checkpoint, device=..., inference_mode='row')`检查相邻manifest与checkpoint文件SHA、嵌入元数据一致、当前源签名、重新审查双向教师以及原baseline出处，再严格恢复冻结base和residual。没有“跳过来源检查”选项。

返回的`StudentActionInterface`每次只接收`observations[N,300]`并返回原始16维动作；没有env/time/teacher/fallback参数，也不添加轮速boost、QP/解析动作、腿偏置或教师接管。运行记录区分Actor调用数、teacher调用数和执行方式。独立`student_worker.py`现已接入此接口，但未实际仿真；不能凭接口的`teacher_calls=0`声称自主行为已经通过。

`environment_reconstruction_contract()`列出同物理重建/实际读回需要的字段。独立worker直接调用该动作接口和原生action manager，复用观测/记录实现，不调用`loaded_cart_worker`的教师action-transform。`student_recording.py`继承原生`TowingWheelAction`并只读记录其执行后、积分前的牵引外力；原弹簧/滚阻公式完全复用，没有修改原module全局。环境、身体/拖车资产、mass/inertia/hitch、材料、实际16/4、ordinary初态、300/16语义和1 kHz原生CoP均安排实际读回核验；没有修改现有研究worker或生产catalog。

`student_worker.prepare()`先调用上述严格学生加载和双向来源审查，再核对所选坡度全部episode中由实际`real_cart_audit.move_window_s[0]`重算、随来源一起验证的起步时刻，要求都等于所选`--preparation-seconds`。未知方向、同方向多个不一致起点或错误CLI时长仍拒绝；完整双向bundle要求保持。成功前不导入AppLauncher，也不创建输出目录。随后冻结原生环境/动作/观测历史/执行器、具体物理基类与记录器、资产以及独立学生模块的源SHA；启动前与结束时复查。两个锚点的真实执行只允许row方式，每步保存原300观测、Actor raw、wrapper限幅后的实际动作、腿/轮原生目标和逐项映射误差，并核对Actor调用数等于执行帧数、teacher调用数为0。真实回放标记`artifact_kind=autonomous_student_execution`，单锚点通过也始终保持统一Actor及PPO准入false。当前CPU通过的是接线/拒绝门测试，尚无本项审查执行的真实学生档案。

旧上坡成功教师在1.60 s首次prepared、3.00 s实际起步；新通过的旧协议下坡在4.48 s首次prepared、7.48 s实际起步，虽然其request声明move_start仍为3。下坡3 s时机器人尚以约0.46 m/s沿坡运动，不能让学生这时提前接到源教师未见过的运动命令，再把失败归咎BC。

学生固定准备默认3 s，旧教师来源的下坡须显式`--preparation-seconds 7.48`；其命令子类只设置`cycle_start_step=round((prep-3)/dt)`再调用原状态机，没有teacher/contact/QP或ready条件。每次Actor前核对真实elapsed/phase/command，并记录实际起步时刻。坡度/时长只用于任务与命令协议的离线配置，不追加到300维Actor输入、不切换策略或修正动作。同一个新候选已通过上坡3.00 s与下坡7.48 s的只读prepare，尚不证明实际自主成功；8 s仅在所选方向的全部源教师实际同为8 s时允许。`check_source_preparation`无case参数时仍保留全bundle检查，历史拒绝结论不被重写。详见[任务时钟审查](../results/2026-10-23_admission_continuation/admission_inventory/task_specific_preparation_review.md)。1.2 m制动阈值、实际净1 m及2+5 s验收不变。

时间对齐并不保证现有300维MLP一定能学会，仍可能需要可部署状态或因果记忆。真实学生失败必须完整记录，再决定是否需要执行成功教师恢复采集，不能用理论教师建议动作给失败轨迹伪造成功标签。

## 已执行验证与未执行边界

CPU全套测试覆盖数据来源/拒绝门、冻结策略、工况覆盖、基线SHA/观测语义、完整回合划分、阶段均衡、残差优化/保存恢复及无接管接口。每次完整测试只有3步合成数值核优化与1步合成临时打包优化，数据、源请求和临时产物都标明`synthetic_interface_test_only`，不算真实机器人训练。真实模型仅只读校验，没有对其执行优化。

实际对“仅ordinary_gain600_gpu上坡”执行BC入口拒绝测试，退出2，原因缺完整下坡，目标目录不存在。详见[BC/学生验证记录](../results/2026-10-23_admission_continuation/admission_inventory/migration_v1_bc_student_validation.json)。

本项接口审查不执行独立学生worker或学生自主上下坡/旧能力审查；实际CPU拟合候选由主流程另行调度并记录。新奖励/状态机的64工况真实rollout与受控PPO、准入签名和正式训练仍须另行实现/核验。没有把离线loss、冻结函数等价或CPU测试通过升级为这些阶段的证据。独立worker入口可用`python migration_v1/student_worker.py --help`只读查看参数；本目录不据当前接口证据提供最终训练启动命令。
