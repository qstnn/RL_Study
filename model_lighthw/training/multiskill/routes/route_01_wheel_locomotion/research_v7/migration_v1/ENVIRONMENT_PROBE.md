# 64 工况环境短探针

`environment_probe.py` 只核对独立迁移环境接线：64 个许可工况从首个动作前实际铺满，普通零速 reset 后执行 1 或 2 个控制步（0.02/0.04 秒）。不采样动作，不创建 Critic/优化器，不进行 PPO，不作为拖行、旧能力或准入验收。

CPU 准备首先调用现有 `load_student(..., inference_mode='row')`，重新核对模型、冻结旧 Actor、80 列历史动作 mask、BC10 来源和完整双向教师。输出必须为新目录。模型或来源缺失时不会创建输出或启动 App；执行时另外冻结 186 个具体环境/命令/奖励/原生依赖并保存快照，前后复核 SHA。新增探针不进入旧学生 176 个执行依赖或 BC10 签名。

App 后复用 `env_adapter.configure_environment/make_environment/verify_pre_action`，同时安装 `command_adapter` 和 `reward_adapter`。64 工况名义条件来源于真实许可 catalog；拖载使用来源教师各方向的固定准备时钟，其余工况的准备仍为 3 秒。其他坡度/质量是候选推广，不是新增成功教师。全槽速度计划仍为空载 3.1、背载/拖载 0.3 m/s；本探针仅检查起始两个零命令准备步。

保存的证据包括：

- `migration_pre_action_physics.npz` / `migration_pre_action_review.json`：实际 64 条件、槽位地形、普通初态、质量/质心/惯量、GPU 16/4、μ3、牵引几何和 Actor 名义上下文。
- `environment_config.yaml` / `agent_config.yaml` / `probe_configuration.json`：已经安装命令/奖励适配后的配置，原基础配置另有其独立说明。
- `environment_probe_trace.npz`：300 维部署观测、252 维环境 Critic 观测（没有 Critic 网络）、原始均值动作、原动作 manager 输入/腿轮目标、全部奖励项的加权未乘 dt 速率、总奖励、done 和终止前物理指标。
- `environment_probe_review.json`：有限性、执行步数、终止槽位、动作映射和 Actor 调用次数；行为、旧能力和 PPO 标志始终为 false。若任一槽位终止即停止追加步骤，终止不被自动 reset 后的观测隐藏。

没有新增 native CoP 验收，也没有收集成功标签。短探针通过仍需另行完成学生双向 1 m / 滚动 / 减速 / 2+5 秒和旧能力验证，以及真正受控 PPO 短验。

2026-10-23 CPU 检查：9 项接口/拒绝门测试通过；当前 `bc_preparation_pilot_cpu/student.pt`（SHA256 `9f406bf2f10b41b83f980a3adf4b4ba660fb4e7949dcd26d3b70db3bef83225b`）严格来源预检通过，未创建仿真输出、未启动 App。旧 BC10 未变化。`_runtime_types()` 原本就已通过 `lru_cache` 缓存；本轮只增加类 identity 回归检查，没有修改不存在的缓存缺陷。证据见持续目录 `admission_inventory/environment_probe_cpu_precheck.json`。

待主线程调度的独立诊断命令（不是正式训练命令）：

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
source /home/user/RL_lab/isaac-sim/setup_conda_env.sh
R1_RESEARCH=model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/research_v7
R1_CURRENT="$R1_RESEARCH/results/2026-10-23_admission_continuation"
python "$R1_RESEARCH/migration_v1/environment_probe.py" \
  --checkpoint "$R1_CURRENT/bc_preparation_pilot_cpu/student.pt" \
  --output "$R1_CURRENT/environment_probe64_gpu" \
  --steps 2
```

所有配置/运行门保留。代码和 CPU 预检完成不代表该 GPU 命令已执行。
