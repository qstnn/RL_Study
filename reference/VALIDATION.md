# reference 快照验证

日期：2026-10-26，Asia/Shanghai。以下检查在独立的 `rl_training_reference` 工作目录完成；现有研究目录和训练输出保留。

## 本地 CPU 验证

解释器：工作区已有 `lab` 环境（Python 3.11）。资产检查时在本地临时提供了 canonical URDF 对应的 17 个 mesh；它们已从上传清单排除。

```bash
python -m pytest -q model_lighthw/training/multiskill/tests/test_framework.py model_lighthw/training/multiskill/tests/test_payload.py model_lighthw/training/multiskill/tests/test_payload_runtime.py -k 'not real_rsl'
# 58 passed, 1 deselected（需要真实 RSL 环境的集成用例未纳入这组检查）

python model_lighthw/training/multiskill/run.py check --route 01
python model_lighthw/training/multiskill/run.py check --route 04a
# R4A 入口包含 39 项 CPU 测试，全部通过
python model_lighthw/training/multiskill/run.py check --route 05a
python model_lighthw/training/multiskill/run.py check --route 06 --payload-kg 65

python -m pytest -q model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/kinematics/test_estimator.py
# 3 passed
```

合计 100 项 CPU 测试。R5A 静态检查输出 22 kg 与新的 URDF 哈希；R6 检查输出本体 22 kg、额外载荷 65 kg、总质量 87 kg。R6 `inertia_frame_regression.json` 是保留的历史数值回归 fixture，仅验证惯量坐标表达逻辑。

## 上传检查

`python reference/check_snapshot.py` 检查 Git 清单与来源清单一致、每个文件的 SHA256、UTF-8 文本与文件类型、Python/JSON/TOML/XML/Shell 语法，以及 URDF 的质量、连接和模型元数据。清单自身不参与自哈希；Git 对整个提交提供完整性标识。

本轮没有 GPU 仿真、机器人 PPO/BC 训练或行为评估。已有历史报告的结果仅对应其记录的版本与质量；22 kg 的动力学与策略表现需要后续验证。

补充：Git 空白格式检查报告了 55 个原始源文件中的行尾空白或文件末尾空行；逐文件哈希确认这些文件与来源完全相同。本参考快照保留原文，没有批量格式化。文本/语法/质量/来源检查通过。
