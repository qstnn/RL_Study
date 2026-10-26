# 学生偏离后的真实教师恢复数据与新候选

2026-10-23。当前两个自主学生回放均失败，不能把原教师状态上的低拟合误差当作可执行策略。新增四个隔离模块支持“真实学生前缀→真实教师接管→完整行为通过→只取教师实际执行标签”，没有修改原BC10份签名源、当前模型或既有学生物理worker。本项开发没有执行真实拟合或仿真；23项CPU接口测试通过，其中仅1步合成数据打包优化。

## 读取与标签

`recovery_dataset.read_recovery_episode(run, source_student, balance_preparation=True)`独立识别根流程实现的`r1_executed_student_prefix_then_teacher_v1`。它要求request的`actor_inference=True`、`fixed_command_clock=True`及明确的学生checkpoint/manifest/执行源；旧BC父为12源，约束末层父为16源。不会把request临时改成`actor_inference=False`送入旧教师入口。

物理validator独立复核ordinary零速默认姿态、实际GPU16/4、原质量/执行器、μ3、圆胎40 kg拖车、0.04 m牵引高/0.20 m机器人牵引点偏移和300/16合同，并与来源学生成功教师profile相同。普通源、prefix额外源均查封存SHA；当前编辑中的教师源码不替换历史封存内容。

读取器从原始记录重新执行`real_cart_audit`，必须整回合已跑满声明时窗、无终止，并达到实际1 m、减速、2 s过渡、5 s保持与滚动/健康全部条件。失败回合立即整体拒绝，不能截取局部好帧。原始N帧观测、五帧历史、上一真实执行动作及20N个物理子步先整体核验，再选择标签。

动作源须精确为前K帧学生、从K帧开始永久切换教师；不存在叠加、混合或再次释放。具体核验为：

- `student_prefix_active`与`teacher_executed`互斥且完整覆盖每帧；frame_index连续，takeover标志只出现在K。
- `teacher_time_s`保留reset后的原绝对20 ms时钟，不在接管处归零。
- 前缀`executed_action == student_prefix_student_raw`，后缀`executed_action == student_prefix_teacher_raw`逐项精确一致；教师段学生raw须为零，与实际调用记录一致。
- 用严格来源模型在已记录前缀观测上重算row推理。CPU/GPU差异容差为atol=`3e-5`、rtol=`2e-6`，单独报告实际最大误差；动作选择一致性不使用这个容差。
- `student_prefix_execution.json`与request元数据、前K/后N−K调用数量一致；记录器自己的`complete_success_verified=False`由独立完整行为审查补证，不能自签成功。
- 实际命令起点与所选任务固定时钟一致：当前上坡3.00 s、下坡7.48 s；不读运行时teacher ready决定命令。

前缀学生动作和前缀影子教师动作都不进入训练数组。只输出真实执行的教师后缀，保留整回合ID、原始frame index与时间、完整物理通过和来源数据。`actor_success=False`、`shadow_labels_used=False`、`student_prefix_success_labels=False`始终明确。

`read_recovery_bundle(nominal_runs, recovery_runs, source_student, ...)`保留原完整成功上下坡nominal教师，再合并至少一个完整成功恢复回合；同物理、同基线、同方向固定时钟，重复轨迹SHA也拒绝。没有额外要求恢复回合必须同时覆盖两个方向，但目前双向自主失败都应分别排查。采样组是`来源类型×任务×阶段`；准备阶段按实际记录细分，不把接管时间伪装为t=0。每组列出成员原始帧、时刻、数量，实际训练组概率由拟合档案记录。

## 拟合与来源隔离

`recovery_bc.fit_from_recovery(nominal_runs, recovery_runs, warm_start_student, output, ...)`首先严格读取全部数据，再从明确版本的旧BC或约束末层学生保留base和已训练残差，使用新Adam只继续优化残差。80列历史动作mask、300输入/16输出及非拖载冻结分支保持；不恢复旧优化器或Critic。

只能按完整回合划分。当前学生已经训练过原nominal回合，即使本次将其中一条留出，也不算独立holdout：代码根据祖先训练轨迹SHA拒绝这种错误留出，并记录累计见过的完整回合身份。默认无独立留出仍允许拟合，报告明确`held_out=None`，不新增物理硬门。

产物采用新schema `r1_migration_v1_executed_recovery_bc_candidate_v1`、新artifact_kind。初版尚未生成真实恢复产物；在首次真实拟合前，来源合同已补至19份：原BC10、三个constrained模块、机器人URDF、四个恢复模块及`runtime_checks.py`。保留`teacher_bundle`作为普通双向环境来源，另记完整`recovery_bundle`、warm-start SHA及训练来源。旧`student.load_student`按设计拒绝新schema，不重签旧模型。

`recovery_student.load_recovery_student`重新验证新checkpoint/manifest/19源、原双向教师、每条完整恢复、祖先训练来源、mask/base，再返回原`StudentActionInterface`。此版warm-start/采集源明确支持旧BC及`r1_migration_v1_constrained_head_candidate_v1`，分别调用各自严格加载器；未知schema拒绝，不尝试用旧loader加载新模型。约束末层的拟合目标已使用全部两个普通教师回合，因此它们都记入已训练祖先，即使更早BC曾有留出，也不能再当独立holdout。若后续需要从恢复候选继续采集下一代，仍须另扩版本，不能假冒已支持schema。

约束末层前缀的16源为其原14源（包含递归旧BC来源）加`runtime_checks.py`和`student_prefix.py`，元数据显式记录新schema/loader；旧BC前缀仍维持12源及原字段。历史前缀按自己`source_snapshot`和当时声明SHA验证，当前helper修改不构成重签历史档案的理由。实际执行动作仍逐帧匹配对应学生/教师，失败整回合依然零标签。

本次扩展38项CPU测试通过，包括两类schema分流、12/16源集合、19源递归范围、祖先回合留出拒绝、历史封存helper不同于当前源码时的读取逻辑。合成打包测试每次仅优化1步，本轮两次完整测试运行累计2步；没有真实拟合、前缀执行或仿真。实际旧/新模型只读预检见持续目录`admission_inventory/constrained_prefix_recovery_precheck.json`。

## 无教师执行

薄入口`recovery_student_worker.py`先严格加载新schema，再核对所选任务时钟并冻结实际执行依赖，成功后直接复用原`student_worker.run_simulation(prepared)`。没有修改全局加载器，没有teacher/QP/解析修正。缺成功证据或新artifact时，在AppLauncher之前拒绝，不创建输出。实际回放仍用普通初态和同物理记录器/原动作映射，完整1 m/制动/2+5 s/native-CoP验收保持。

当前真实负例：`student_preparation_up_gpu`、`student_preparation_down_gpu`缺明确恢复合同而拒绝；`recovery_up_prefix004_gpu`物理配置检查通过，但2.26 s终止且没有完整运动/保持，重审后整回合拒绝，未取得任何标签。没有因为其后缀由教师动作产生就采入数据。

可只读查看新入口参数：`python migration_v1/recovery_bc.py --help`、`python migration_v1/recovery_student_worker.py --help`。真实拟合/自主回放由主流程在完整恢复证据成立后调度，正式PPO仍由用户启动；当前不存在由这些CPU接口测试产生的PPO授权。
