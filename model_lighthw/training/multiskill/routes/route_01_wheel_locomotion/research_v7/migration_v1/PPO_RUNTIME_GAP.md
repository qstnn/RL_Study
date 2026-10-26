# 受控 PPO 短验接线：终末观测缺口

2026-10-23。本阶段新增代码和 CPU 合成测试，没有运行 Isaac、RSL 更新或机器人 PPO。本子任务未修改既有 environment_probe/env_adapter/command/reward 模块；已归档 186 源的 `environment_probe64_gpu` 仍只证明当时 64/64 首批与两步接线，不能替代学生行为准入。

`ppo_runtime.prepare(checkpoint, anchor_runs, retention_runs, output, *, base_std, rollout_steps=24)` 首先调用现有 `verify_behavior_admission`，重算原始证据。返回 false 就抛出 `BehaviorAdmissionRejected`，不创建输出目录、不加载 probe 策略、不导入 App。没有传入 saved-pass 或跳过门槛选项。旧 schema 以外的模型仍须明确增加 loader 分派，不能重签伪装。

即使行为通过，目前只生成 64 环境、两次有界 rollout update 的**计划**；`runtime_loop_implemented=false`。`recheck_before_runtime` 再查实际行为与源，随后明确拒绝尚未通过真实 GPU 验证的终末命令预览。预览已实现且通过 CPU 原函数对照，不能据此宣称 GPU 接线通过。没有正式训练 CLI，也没有用改成 terminal 的方式悄悄关闭 timeout bootstrap。

## 已实现的局部捕获

`terminal_capture.PreResetCriticCaptureMixin` 可作为新局部 `MigrationEnv` 子类的第一个基类；本次未实例化真实环境。它仅在原生 `step` 外加 begin/finish，调用原生 `super().step()` 一次；`_reset_idx` 在调用父类之前执行 `compute_group('critic', update_history=False)`，复制本步实际重置槽位的 252 维末态。构造/显式 reset 不作为终末转移。

进入 step 前检查 Critic 的名称、每项宽度、scale/clip、无历史/噪声/modifier、普通函数读数接口。只计算 Critic 组，不计算 policy 组，不推进 Actor 五帧历史，不推进命令或写物理状态。函数体及原生 manager 还必须按源快照核验；这些接口检查不证明任意同名函数都是纯函数。

每步保存 captured mask、原生 terminated/truncated、终止前 episode step，并要求返回信号与重置前完全一致。默认只抓 held-command Critic，此时 `CapturedTransition.for_timeout_bootstrap()` 遇到真 timeout **拒绝**补值。显式启用下文预览后，只对 `truncated & ~terminated` 返回独立的 `post_command_critic`；真终止不补值，held-command 向量始终不冒充下一状态。

## 为什么 clone 还不够

本机原生 `ManagerBasedRLEnv.step` 顺序是：物理步 → episode counter 加一 → termination/reward → `_reset_idx` → `command_manager.compute` → observation compute。通常 Critic 的输入在命令更新之后。`UnifiedEnv._finish_task_at_timeout` 仅补 cycle/hold 完成计数，不推进下表的命令状态。

| Critic 列（左闭右开） | 内容 | 重置前捕获与正常下一观测的差别 |
| --- | --- | --- |
| 9:12 | 实际命令 x/y/yaw | 尚未推进斜率限制器 |
| 242:244 | 相对 hold_origin 的沿坡/横向位移 | hold_origin 可能在新进入保持、非保持或 2 s 边界重锚定 |
| 244:245 | hold_elapsed / 7 | 尚未更新 hold_steps |
| 245:246 | requested_vx / 3.1 | 当前固定命令不会重采样，但一般协议不能忽略 |
| 246:250 | phase one-hot | 起步/达距减速/归零保持的边界可能改变 |
| 250:251 | phase_elapsed / 30 | episode step 已加一，phase_start_step 尚未更新，跨阶段时尤其不同 |
| 251:252 | remaining distance | r1_eval_progress 尚未据刚完成的物理末态更新 |

因此捕获的是“真实物理末态 + 本步保持命令”，不是经过原生命令更新的完整 `s_(t+1)`。直接 `V(captured)` 补 timeout 会混用时序；原 RSL 对 pre-action value 的补值也不能代替修复。现有 adapter 已要求调用者提供真实 terminal next-Critic，接口本身不能证明其来源正确。

## 已实现的纯副本预览

`preview_terminal_critic(base)` 返回 `TerminalPreview`，含完整 `held_command_critic`、`post_command_critic`、副本 `command_state`/`task_state` 与来源/不变性 review。

1. 仅支持当前固定 slew 任务、唯一 `base_velocity`、无 heading/debug 回调、无 interval event，且检查实际 float timer 减 dt 后仍严格大于零；任一不满足即拒绝。原命令类及 Critic 函数按名称、布局和源码文件白名单核对。
2. 用同类 `object.__new__` 创建 command 独立对象，逐张量复制所有命令变量、metrics/计时器、全部 `r1_` 任务张量与 episode counter。`_env`、robot/scene、action 均换成无物理 API 的快照代理。命令和 manager 的原生析构 subscription 句柄先设为 None，绝不共享生产回调。
3. 因已排除到期重采样，副本上直接复用完整原 `CommandTerm.compute(dt)`，包括 `_update_metrics`、timer 减 dt 和 Migration `_update_command`，不手工复写状态机；副本 `_resample` 额外绑定抛错函数，防止未来意外分支。原命令对象不执行 compute。
4. 从实际终末 robot 读取根位姿/速度、q/qd/default、投影重力，读取 height scanner rays 和实际 action，逐张量复制。建立同类 ObservationManager 局部对象，仅含无历史 Critic 配置；分别在推进前后调用原 `compute_group` 重算完整 252 维，不只手工修几个索引。该 Critic 本身没有直接接触力通道，不虚构接触输入。
5. 校验除命令相关列外全都逐值不变，原任务/命令/读数/所有 observation history 不变，所有签名源不变。Torch CPU/当前 CUDA、Python、NumPy RNG 前后比对；异常也恢复 RNG。若发现曾抽样，即使恢复成功仍拒绝预览。原生 sensor/articulation getter 可能物化只读缓存；投影对象无原生 scene 或写物理状态的接口。

真实无 PPO 诊断接线：新建局部 `class CaptureEnv(PreResetCriticCaptureMixin, env_adapter._runtime_types()[1])`，构造后且第一个 step 之前设置 `base.r1_terminal_preview_enabled=True`。这只开启终末诊断，不绕过 PPO 行为 gate；原 environment_probe/env_adapter/command/reward 文件不改。单独 CPU/诊断也可调用 `PreResetCriticCapture(preview_post_command=True)`。预览结果的 `real_gpu_timeout_validation` 在本代码中保持 false，由独立真实运行审查提供 GPU 证据。

## 仍待完成

先做真实 GPU、无 PPO 的终末诊断，核对 pre-reset 采样、投影与返回 reset 态的区别，以及 Actor 历史/原命令无额外推进。行为 gate 仍须独立通过。

之后才完成两次真实 rollout update：实际首批 64 类、原命令/奖励、实际动作映射、标准 RSL ratio/clip/GAE、全组 KL 回滚；第一轮空 rollout 边界保存，在新 adapter 恢复 policy/Adam/RNG 并数值比较，再进行第二轮。环境物理状态未保存，须明确是在仍存活的同一环境上恢复数值状态。没有真实 timeout 的短验不能宣传 timeout 路径已验证。

## 独立真实终末诊断入口（本轮未启动）

新 `terminal_preview_probe.py` 仅运行 64 槽两步、Actor 均值，无 Critic 网络/优化器/PPO。两次独立 App 使用同一候选、物理、普通 reset、原命令/奖励和 seed。`short_timeout` 只将 episode horizon 从 65 s 改为 .04 s，由原生 timeout 在第 2 步触发；`uninterrupted` 保持原 65 s。没有修改 episode counter，也不把强制短超时解释为任务成功。

两 run 保存原始初态、pre-action 观测/动作、真正 pre-reset 物理末态、held/projected/returned-reset Critic 及原生 terminated/truncated 分离信号。`compare` 将投影与另一 App 真正未重置的 next-Critic 对比，先要求首态、动作、实际物理末态在预先固定容差内一致。跨 App 物理分歧或自然失败都会拒绝，不强行把不同轨迹当语义通过。任何结果均 `pre_ppo_behavior_ready=false`、PPO=0。

```bash
V7=/home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/research_v7
RUNS="$V7/results/2026-10-23_admission_continuation"
python "$V7/migration_v1/terminal_preview_probe.py" run --mode short_timeout \
  --checkpoint "$RUNS/bc_preparation_pilot_cpu/student.pt" \
  --output "$RUNS/terminal_preview_short_timeout_gpu"
python "$V7/migration_v1/terminal_preview_probe.py" run --mode uninterrupted \
  --checkpoint "$RUNS/bc_preparation_pilot_cpu/student.pt" \
  --output "$RUNS/terminal_preview_uninterrupted_gpu"
python "$V7/migration_v1/terminal_preview_probe.py" compare \
  --timeout-run "$RUNS/terminal_preview_short_timeout_gpu" \
  --reference-run "$RUNS/terminal_preview_uninterrupted_gpu" \
  --output "$RUNS/controller_audit/terminal_preview_gpu_pair_review.json"
```

本段是诊断命令，不是正式训练启动命令。未来探索路线可以在独立版本采用“运行准入/最终行为验收分开”的准入规则；本文件旧 migration 的行为 gate 未放松，新入口须明确版本化，不能隐式跳过旧门。

## 本次验证

- `tests/test_ppo_runtime.py` 23 项 + `tests/test_terminal_preview.py` 20 项，共 **43 个 CPU 测试通过**。新测试从当前源码 AST 加载原方法体，没有重新实现命令公式；与独立 deepcopy 的未 reset 对象执行原完整 compute/next-Critic 逐值比较。覆盖准备结束前后（上 3 s/下 7.48 s）、正常运动、达距减速、归零、保持、2 s 重锚定、65 s timeout 共 8 组边界；另检查终末 q/qd 当前读数、clone 独立性、history、RNG、错误恢复和不支持路径拒绝。
- `tests/test_terminal_preview_probe.py` 新增 18 项；合计 **61 项 CPU 测试通过**。包括独立 run 来源/原始 SHA、1e-8 数值差记录、超容差拒绝、真实终止不能冒充 timeout、未改计时器及无 App 的 prepare；不是实际 GPU 证据。修正了 runtime manager scale 已是 CUDA Tensor 时应在同设备构造比较标量的兼容问题。
- 对实际 `bc_preparation_pilot_cpu/student.pt` 和现有两向学生、单项 retention 再调用新 gate：以“必须两向 + 原 12 项保留完整覆盖”拒绝；指定输出目录不存在，未加载 App。证据在 `results/2026-10-23_admission_continuation/controller_audit/ppo_runtime_gate_review.json`。该拒绝测试没有补造缺失 11 项，也没有将其解释为新行为测试。
- 本次 RSL 合成 update 调用 **0**、真实环境步 **0**、机器人 PPO **0**。既有 7 次 CPU 合成 RSL adapter 更新不计入本次。
- 交付时对照 probe 的 186 个当前源，`student_worker.py` 已因其他并行任务变化（本子任务未修改）；后续诊断须独立归档当前执行源，不能宣称当前全部 186 源仍与早先 probe 一致。
