# R1 冻结校准入口

本目录是独立诊断，PPO 更新数始终为 0。旧 V3 生产配置和既有 checkpoint 保留，校准效果不等于完成迁移训练。完整结论见 [2026-10-16 校准审查](../../results/2026-10-16_calibration_review/README.md)。

`probe.py` 使用 V3 原始 16 项协议，固定来源 checkpoint/SHA、seed43、同一套验收门槛。Route53/38 的原 60 维 Actor 使用真实机身线速度作为教师诊断输入；这不是可部署的 290 维观测适配。当前 V3 作为对照直接接收原 290 维观测。

| 组合 | 四轮刚体上限 | 轮速度控制增益 | 用途 |
|---|---:|---:|---|
| baseline | 1000 deg/s | 3 | 原始 R1 物理对照 |
| wheelcap | 3000 deg/s | 3 | 仅修正源路线遗漏的刚体角速度覆盖 |
| wheelcap_d1 | 3000 deg/s | 1 | 在修正上限后，仅恢复旧路线轮增益 |

其余 13 个刚体上限均为 1000 deg/s，真实腿/轮执行器上限仍为 96/48 N·m；摩擦 2.3、滚阻 .02、牵引点名义净高 .10 m。源路线生成器在克隆和物理启动前仅修改四轮；每个环境的实际 USD 属性与执行器张量均落盘复查。不能将刚体角速度限制当作关节速度硬上限，也不能把该修正称为增大电机力矩。

不带 `--full` 仅运行默认最多 8 秒短测，用于定位早期失败，不能通过完整行为门槛。`--full` 包含完整巡航、制动及 5 秒停车窗；负载移动要求前进至少 1 m，无最低速度。两种模式都在首次终止处截断，保留 50 Hz 行为、策略输入和 1 kHz 实际/请求力矩。

建议每个候选独立启动进程。跨候选环境复用虽重置根部状态和策略历史，仍可能有物理求解器数值差异；不宣称固定 seed 等于逐帧确定性或随机鲁棒性验证。

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
export TERM=xterm
R1_CALIBRATION_ROOT=model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v3
R1_CALIBRATION_OUT="$R1_CALIBRATION_ROOT/validation/manual_calibration_$(date +%Y%m%d_%H%M%S_%N)"
python "$R1_CALIBRATION_ROOT/diagnostics/calibration/probe.py" \
  --variant wheelcap_d1 --full --candidate route53 --output "$R1_CALIBRATION_OUT"
python "$R1_CALIBRATION_ROOT/diagnostics/calibration/audit.py" "$R1_CALIBRATION_OUT"
```

`audit.py` 从原始 CSV 重算所有行为指标，重放 Actor 输出，核对输入映射、有效长度、实际力矩上限、物理参数和源模型 SHA。短测与完整评估分开；`status=passed` 只表示数据一致性审计通过，行为是否通过看 `summary.json` 的 `teacher_pre_screen_passed`。

早期 D1 探针继承的 V3 `runtime_check.json` 将轮增益写死为 3；其 `physical.json` 的真实执行器值为 1，审计单独报告这一元数据差异。修订后的入口保留旧报告原件，并按实际执行器张量纠正字段，不回写历史轨迹。

即使教师预筛通过，也仍需要可部署观测适配的同门槛验证。此目录没有正式训练入口，也不会自动放宽门槛或启动 PPO。
