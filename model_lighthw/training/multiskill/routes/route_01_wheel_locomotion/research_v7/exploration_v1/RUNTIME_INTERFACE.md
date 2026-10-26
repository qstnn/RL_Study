# R1 探索 PPO 执行接口

本入口区分运行条件与行为验收。模型无需先完成拖载才能探索；完整 R1 行为仍须独立验收。旧 migration 自主门未修改。正式训练由用户手动启动。

- Actor 302 / Critic 252 / action 16；冻结原 base，拖载与非拖载残差均可训练。新增两列仅普通 reset 经过时间与预先声明准备时间。
- 默认初始化是行为已失败的 `bc_preparation_9f406`，不能称成功热启动。`zero` 属于不同配置，必须另做实际短验。
- 原 μ3、GPU 16/4、原伺服/限矩、真实被动车；没有环境噪声和物理随机化。Gaussian 动作探索仍存在。
- 初始非拖载 raw std：12 腿 `.02`、4 轮 `.10`；拖载 `.10/.30`。这是明确设计值，不是恢复旧 checkpoint std。
- 标准本机 RSL PPO ratio/clip/GAE；固定五组 LR `1e-6/1e-6/1e-4/1e-5/1e-6`。每 rollout 24 步，2 epochs × 4 mini-batches，γ .995 / λ .95，整体 KL .01、实际有样本组 KL .02。拒绝时整个模型、Adam 和 RNG 回滚，已执行物理不回滚。
- 超时采用真实 reset 前物理状态、原命令纯副本推进后的 Critic。仅 `truncated & ~terminated` 补终末值；末步非结束状态单独 bootstrap。真实失败不补值。

`train.py prepare` 只做 CPU 来源、terminal 两次真实 GPU 证据复算和初始化验证，不建输出目录或 App。`short` 固定最多 4 次提案取得 2 次接受；首个接受后保存并在新实例中严格恢复模型/Adam/RNG，在当前实际状态逐值比 Gaussian/Critic，再继续真实采样和更新。短验完整保存动作、观测、映射、奖励、终止与源归档；不能把两个更新当拖载成功。

`train` 必须指定 `--validated-short-run` 和 `--iterations`；后者明确为**追加 PPO 提案次数，包括被拒绝的提案**。默认从短验 final checkpoint 延续，仅显式转换 execution_kind。`--resume-checkpoint` 只允许正式运行，仍核同一短验、来源、初始化和配置，恢复模型/Adam/RNG后创建新的普通物理环境，不称物理轨迹无缝续接。

`--num-envs 512 --inference-mode batch` 已通过自身完整实际短验，是当前正式入口验证配置。64/row 是早期数值对照；不同并行数量、推理模式或初始化仍须各自短验，不能借用其他配置的证据。除这些显式选项外，本入口固定已审算法/物理配置。

正式每次更新只写聚合 JSONL（分类奖励、26项实际奖励、终止、命令阶段计数、KL和实际 Adam 前/提案/保留计数），按 `--save-interval` 与结束保存 checkpoint，不连续积累大 raw 轨迹。命令阶段/时钟完成不计作行为成功。没有 200/600 次行为不达标早停；NaN、来源变化、保护失效等真实错误明确失败。外层进程重审必需成功文件、真实数值 final checkpoint、计数差和源归档，不能把 Kit 的退出码 0 当成功。

2026-10-23 最新验证：`exploration_short512_batch_gpu` 实际 GPU 首批覆盖全部 64 许可工况；512 环境 × 48 控制步 = 24,576 条 transition，2 次提案均接受、0 次拒绝，26 个 Adam 参数状态的 step 均为 0→8→16。两次 global KL 为 .001058955/.000683739，最高实际组 KL 为 .003315142/.002557582，原 .01/.02 门未改。冻结 base 逐 tensor 不变，两残差、Critic、两组 std 实际更新。

首个接受更新后，新实例模型/Adam/CPU 与 CUDA RNG、当前实际状态的 Gaussian/Critic 均逐值恢复。204 份当前与归档执行源匹配，动作映射误差 0，26 项加权奖励率乘 .02 后重算误差 7.12e-7；父进程记录 returncode=0 并完成严格原始/数值重审。`training_runtime_ready=true`。正式 512/batch 入口 CPU prepare 也通过，App=0、未建正式输出、没有启动正式训练。

**能力边界：** 此次只推进 0.96 秒；动作前最大 reset elapsed 为 .94 秒，全部 24,576 样本仍处准备阶段，97 次实际物理终止、timeout 0，真实失败正常作为 RL transition。终末超时原语另由 v3 两次 GPU 诊断验证：64 超时，投影对未 reset 原生 next Critic 最大误差 0。运行准入不能代表 1 m 拖载、下坡制动、完整 R1 或硬件验收；`final_behavior_pass=false`。

代码验证包括 15 项 adapter 检查、30 项 runtime/CLI 检查。adapter 最近一次全套执行本机 RSL.update 7 次 CPU 合成调用，6 次完成、1 次注入 Adam 后异常验证回滚；这些与实际机器人更新分别记账。GPU 无 App 数值测试另确认 26 个 Adam step 保持 CPU、moments 保持 CUDA，整个 checkpoint 保存恢复逐值一致，新增 PPO=0。

独立报告：`results/2026-10-23_admission_continuation/controller_audit/exploration_short512_independent_review.json`；正式预检：`results/2026-10-23_admission_continuation/formal_prepare512.json`。

命令模板（路径均需指向实际已验证目录）：

```bash
python exploration_v1/train.py prepare --output /new/short \
  --terminal-timeout-run /actual/terminal_timeout_v3 \
  --terminal-reference-run /actual/terminal_uninterrupted_v3
python exploration_v1/train.py short --output /new/short \
  --terminal-timeout-run /actual/terminal_timeout_v3 \
  --terminal-reference-run /actual/terminal_uninterrupted_v3
# 仅匹配实际 short 通过后，由用户启动：
python exploration_v1/train.py train --validated-short-run /actual/exploration_short512_batch_gpu \
  --num-envs 512 --inference-mode batch \
  --output /new/formal --iterations 10000 --save-interval 100
# 若正式训练曾中断：
python exploration_v1/train.py train --validated-short-run /actual/exploration_short512_batch_gpu \
  --num-envs 512 --inference-mode batch \
  --resume-checkpoint /prior/formal/proposal_000102.pt \
  --output /new/formal_resume --iterations 10000 --save-interval 100
```

首轮实际 64/row 短验 `exploration_short64_row_gpu`：4 次真实提案均因 KL 超限拒绝，各提案实际 Adam 每参数 0→8→0，接受更新 0；保留完整失败源/轨迹。下一候选只将拖载残差 LR 从 1e-5 降为 1e-6，其余四组、KL 门、物理、64/row 均不变。随后保存恢复设备与退出检查问题分别修正；最终 512/batch 已独立通过，旧失败档案保留且不作为正式来源。
