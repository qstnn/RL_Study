# R1 exploration_v1 策略接口

本接口不启动训练/仿真，不改变 migration_v1 的旧准入，也不把失败初始化标为行为成功。

`contract.py` 导出 `ACTOR_DIM=302`、`BASE_OBS_DIM=300`、`CRITIC_DIM=252`、`ACTION_DIM=16`、`RESET_TIME_SCALE_S=40.`、`PREPARATION_SCALE_S=8.`。原300列完全保留，附加实际reset后秒数/40、声明准备秒数/8，两列截到[0,1]。默认任务配置由环境负责：拖载上坡3 s、负坡7.48 s、非拖载3 s；函数只收计时/声明值，不读取接触、教师ready、仿真phase/位置。运行端应独立记录未缩放秒数、真实reset步计数，并逐列核对。

一个部署图包含原300输入冻结base与两个302输入可训练残差，由原第299列名义拖载量选择分支。非拖载修正初始为零；它能学习新物理下的空载/背载偏差。两分支都屏蔽205:285原动作历史，base本身不变。附加时钟不是状态估计器，不证明接触过渡可解或策略已成功。

`checkpoint.initialize_policy(*, towing_initialization='zero', base_std, towing_std=None, device='cpu', inference_mode='row', critic_hidden_dims=(512,256,128), baseline_path=BASELINE_PATH, preparation_checkpoint=PREPARATION_PATH)` 返回 `(policy, provenance)`，不创建优化器。只能使用固定SHA的原base；可选拖载初始化为 `zero` 或 `bc_preparation_9f406`。后者通过旧严格loader重新核对完整来源，复制300维残差，新增两列权重为零；302与300 GEMM只声称函数迁移一致，不声称原浮点逐位相同。这个BC候选已在两向自主失败，记录明确保留该事实。

`policy.ExplorationActorCritic` 提供标准RSL非递归接口：`act`、`update_distribution`、`act_inference`、`evaluate`、`get_actions_log_prob`、`action_mean`、`action_std`、`entropy`、`reset`。均值只收302，Critic只收252且全新初始化；不恢复旧Critic/Adam。`inference_mode='row'` 为默认，batch须显式选择并另验证行为。

优化器参数组固定顺序：`towing_residual, non_towing_residual, critic, towing_std, non_towing_std`。调用：

```python
optimizer = policy.make_optimizer(
    actor_lr=1e-5, non_towing_lr=1e-6, critic_lr=1e-4,
    std_lr=1e-5, non_towing_std_lr=1e-6,
)
```

两组Gaussian标准差都可训练；`base_std`缓冲只是初始非拖载std的不可改来源记录，不代表非拖载方差冻结。旧base参数不进入优化器。`assert_invariants(optimizer)`、`project_constraints(optimizer)` 检查/维护原mask及Adam状态。

`begin_update(optimizer, observations302, groups=...)` 复用既有 `PolicyUpdateTransaction`。整个提案包含两残差、两std、Critic、Adam及RNG；整体/分组KL失败恢复全部数值状态。PPO适配器应把事务中的old mean/std替换为采样rollout实际保存值，不能用不同batch重算替代。物理场景不属于数值回滚对象，下轮须收集新的on-policy数据。

`policy.checkpoint(optimizer=None)` / `restore_checkpoint(snapshot, optimizer=None)` 提供新的schema数值保存/恢复。原模型、mask、来源或schema不符拒绝。`checkpoint.save_policy` 新文件导出、`load_policy` 重查固定来源再加载；它们不保存PhysX状态，也不强行声称空rollout边界，边界应由PPO适配器检查。运行端必须另外记录导出文件SHA。

`source_files()` / `source_signatures()` 可从contract或checkpoint导入，当前16份：新策略4文件、旧严格初始化/数据合同及通用事务实际依赖。它们刻意不包括并行开发的environment/adapter/runtime；真实运行者必须额外冻结这些执行文件及实际RSL、Isaac与模型资产。修改旧BC/物理来源会正常触发来源拒绝，不能静默改签原artifact。

验证：17项CPU测试通过，其中仅1次合成回归优化步验证非拖载新分支有梯度；没有机器人拟合/仿真/PPO。真实原base与9f406两种初始化均通过严格只读来源加载，新临时导出再加载后动作逐字节相等；临时文件已删除，未生成可冒充训练成果的常驻模型。记录见 `results/2026-10-23_admission_continuation/admission_inventory/exploration_policy_precheck.json`。运行准入和完整R1行为仍须各自验证。
