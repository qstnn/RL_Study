# V4 诊断与离线迁移

这些入口用于本轮有界诊断、教师数据采集和离线拟合。正式训练使用 `../scripts/train.py`；部署 Actor 中没有这里的教师控制器。历史运行以各自 `source_snapshot/`、`request.json`、模型/资产 SHA 为准，当前诊断脚本可能包含后续修正，不能冒充早期运行的原始代码。

| 文件 | 用途 |
|---|---|
| `support_probe.py`、`contact_probe.py` | 初始承载、接触与静态补偿的失败对照 |
| `feedback_probe.py`、`integral_probe.py` | 诊断反馈教师和编码器积分控制对照 |
| `support_teacher.py`、`support_ik.py` | 特权状态生成动作标签；仅输出原有 16 个电机动作，不给机器人施加辅助外力 |
| `teacher_probe.py` | 固定 16 项的教师验证与标签采集 |
| `collect_mixed.py` | 全 64 工况同时采样的标签收集；允许 reset，因此不作为首次失败截断的任务验收 |
| `fit_adapter.py` | 用本体历史估计旧教师的速度输入并折叠进首层，作为迁移初始化 |
| `distill.py` | 标准单个 MLP 的离线动作监督拟合；可用后续学生偏离轨迹的教师标签修正 |
| `student_probe.py` | 单 Actor 独立闭环与动作重放；教师只生成旁路标签，不决定仿真动作 |
| `audit.py` | 从 50 Hz CSV 重算验收，并检查 1 kHz 有限性、限幅及快照 SHA |

选用 `candidate_04`，其 Actor 均值权重与 `selected/model_init.pt` 逐张量相同；selected 仅更新合同元数据及 PPO 初始噪声。`candidate_05` 只有离线拟合，没有 GPU 行为验证，未选用。`candidate_03` 是本轮中断的数据加载尝试，保留诊断记录。交错样本的拟合误差只用于离线选点，不是独立泛化证据。

第一次 PPO 短验 `train_smoke_01` 暴露了动作历史输入重入问题：40 kg 上坡零指令由 20 s 变为 0.96 s 失败。只清零这些首层权重的 `ppo2_history_projection` 在 8 s 短测中恢复了存活，但它不是完整任务验收。正式修复在 `spec.install_policy_input_mask` 中永久屏蔽梯度，并检查模型与 Adam 动量均为零；同时采用更小的固定学习率和初始噪声。新设置验证见 `train_smoke_02` 与 `resume_smoke_01`，不要用旧短验模型启动正式任务。

接触探针里的 `net_forces_w` 来自接触传感器的法向接触力。其切向投影接近零不能当作摩擦力实测或缺少摩擦的证据。`--short8` 是固定 8 秒诊断预算，预算结束本身不能称任务通过；所有通过数必须来自完整协议的 `evaluation.json`。

离线重新核验已选 Actor 的完整行为数据：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v4/diagnostics/audit.py \
  model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v4/validation/student_04
```

教师与学生检查只支持本轮固定起点、固定参数的仿真筛选；没有实机、随机初态鲁棒性或完整速度—力矩包络验收。
