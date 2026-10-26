> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A 本机教师动作搜索

本目录与已交付的 `r5a_auto_v6` 隔离。目标是在原模型和电机约束下，找到从固定站姿开始、首次腾空完成后空翻、落地并站稳的教师，然后再迁移到离线电脑做模仿初始化和 PPO。当前搜索记录中的高分、完整转角或 `qualified_landing` 单项不等于完整成功。

## 物理与验收

- 使用 v6 原生 Isaac Sim / PhysX 环境，1 ms 物理、20 ms 控制、原始 420 维策略观测和 16 维动作。
- 保留 48 Nm 腿电机、24 Nm 轮电机、原动作上下界、质量/惯量、摩擦、自碰撞和速度限制；无外力辅助、根状态/速度注入、空中重置或约束修改。
- 固定初态，无观测噪声和物理随机化。CEM 随机采样的是控制器参数。
- 原验收要求：首次腾空 350–385°，轴偏差和首次触地倾角均不超过 20°；无非轮接触等物理失败；6 秒回合终末连续 1 秒符合原站稳条件。
- 空中全身角动量使用原 5% / 0.02 阈值独立复算。原环境及成功判定源码不改。

## 搜索和复核的区别

短时批量搜索保留所有失败标记，但为了比较参数，不在失败时自动重置。`best.json` 明确标为 `search_candidate_not_validated`。它不能作为 PPO 准入凭证。

`probe.py` 使用未经修改的生产环境，从头执行完整回合，遇到原生失败即停止。输出 1 kHz 原始动力学、50 Hz 观测/动作及独立 NumPy 复算；最终只有 `result.json` 中 `success: true` 且来源核验一致才算该次教师通过。重复固定初态不等于鲁棒性检验；CPU 通过后，迁移时仍需在目标 GPU 上复核。

## 本机运行

在项目根目录 `rl_training_r5a` 中运行；输出目录必须是新目录：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash r6.sh python model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_teacher_search/probe_bound.py --device cpu --teacher model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_teacher_search/results/bounded_356_snapshot.json --output model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_teacher_search/results/replay_new
```

上例重现已完整复核的355.225°教师。文件名中的356是搜索阶段标签，实际结果以独立复算为准。搜索保留原 v5 的左右腿交替蹬伸，再优化预备姿态、20 ms 动作时序、空中收腿/展开和落地反馈；参考值与搜索源码随各次运行归档。

## 离线迁移约定

成功后只打包通过复核的教师参数、参考动作、必要代码和小型证据，不搬运全部搜索历史。学生模仿数据使用原策略观测与实际执行动作；教师可读取额外物理量，但学生不能新增特权输入。行为克隆误差下降不能代替学生闭环回放。PPO 保持在线采样，示范单独作为初始化/衰减辅助损失，不冒充 on-policy 轨迹。目标电脑先完成教师复核与学生闭环检查，再启动连续 PPO，末尾统一评估并生成精简 `usb_review`。

已完成51,808个候选搜索，并将完整复核通过的限幅教师交付至独立的 [r5a_guided_v1](../r5a_guided_v1/README.md)。教师355.225°、连续站稳4.527秒、原动力学与角动量筛查通过。修复首帧模仿边界后的学生约353.279°，仍有轴向超限与落地接触失败；学生没有被标为成功。

最终补丁为 `exports/lighthw-r5a-teacher-guided-v1-patch-20261022.zip`，只带一个教师和必要证据。全部搜索数据留本机；本轮未进行机器人PPO训练，目标GPU复核与正式学习在另一台电脑进行。搜索汇总见 [review/REVIEW.md](review/REVIEW.md)。
