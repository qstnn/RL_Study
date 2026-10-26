> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A Guided v5：候选接触复核、降低冲击与落点恢复

这是供已完成 **Guided v4** 的离线项目使用的增量补丁，仅新增 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5/` 与 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh`。无需 Git 或联网，Isaac/机器人资产继续使用原项目。默认从已成功的 v3 Actor 显式迁移，冻结起跳网络，训练空中与落地网络，重建 Critic/Adam；不直接续训 v4。

## 安装和启动

把 ZIP 与 `.zip.sha256` 放到另一台电脑的**原项目根目录**（能看到 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v4.sh`、`r6.sh` 的目录），在那里执行：

```bash
sha256sum -c lighthw-r5a-guided-v5-contact-patch-20261023.zip.sha256
python3 -m zipfile -e lighthw-r5a-guided-v5-contact-patch-20261023.zip .
python3 lighthw-r5a-guided-v5-contact-patch-20261023/install.py .
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh train --v4-session offline_sessions/route_05a/guided_v4_20261023_102235_566885
```

**最后一条就是完整启动命令。** 项目整体搬迁后只需进入新根目录；session 改名时修改 `--v4-session`。完整 v4 目录应保留，不能用以前精简的 USB review 代替。

安装器先检查全部基础文件和补丁散列，再写入；基础不匹配、补丁损坏、同名文件被修改均拒绝并列出路径。同包重复安装可安全跳过。旧路线、模型和完整训练目录保留。

命令依次执行：

1. CPU 只读核对 v4 基线和最终候选的模型、1 kHz 物理轨迹、50 Hz 策略轨迹及源码，独立核对接触和原成功/动量判定，复算 v4/v5 奖励。中间候选有完整资料时一并统计接触。
2. 在训练电脑上用内置成功 Actor 做两次完整自主回放；失败即回收诊断。
3. 默认 512 环境、连续 3000 次更新尝试，前 20 次包含在预算内用于 Critic 预热，每 200 次仅保存。
4. 结束后一次性评估起点与最后最多 3 个保存候选，选优并生成精简 USB 包。

训练中途不分段冻结评估。第一步缺文件、散列/判定不一致会在启动仿真和 PPO 前停止并打印 `[USB CONTACT REVIEW ZIP]`；请带回该 ZIP 和 `.sha256`。核查通过表示证据与计算可信，不代表奖励已经改善行为。

只核查 v4、不启动仿真或训练：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh audit --v4-session offline_sessions/route_05a/guided_v4_20261023_102235_566885
```

单独 `audit` 后再 `train` 会重新检查。准备缓存仅在代码、模型、输入文件及设备都匹配时重用。

## 观察与回收

`[RECOVERY WINDOW]` 显示最近最多 2048 个结束回合的完整成功率 `success`、合格首触率 `qualified` 和示范权重。`Landing/terminal_drift_m`、`Landing/terminal_xy_speed`、`Landing/terminal_tilt_deg` 和 `Landing/gate_miss_drift_s` 帮助观察恢复。每项奖励输出及全程更新日志用于检查冲击代价。窗口成功率不是冻结模型验收，不能只看总奖励上涨。

训练结束带回打印的 **`[USB REVIEW ZIP]` ZIP 和同名 `.sha256`**，通常在 `usb_review/route_05a/`。包括全程更新日志、训练窗口统计、候选对照、审计结果、两模型各一条原始轨迹，以及选中模型录像：

- 基线胜出：`selected.pt` 是基线，`secondary.pt` 是本轮最佳新候选。
- 新候选胜出：`selected.pt` 是新候选，`secondary.pt` 是起点基线。

角色见 `review_manifest.json` 和 `secondary_selection.json`。`selected_is_baseline=true` 表示新模型未超过起点；`best_available_not_passed` 不表示成功。其他模型/全量回合/缓存留在训练电脑，不用拷整个项目。

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh play
```

保持自由相机。按一次 Ctrl+C 后等待回收，避免立刻再次强制退出；中断包保存最近可读的 `latest_saved.pt`，明确尚未评估。同版本同代码续训需保留原示范和完整 v4 session：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh train --checkpoint /实际路径/latest_saved.pt --v4-session offline_sessions/route_05a/guided_v4_20261023_102235_566885 --updates 3000
```

`--updates` 为追加尝试数。导出失败但完整工作目录仍在时重打包：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh collect --folder /实际路径/offline_sessions/route_05a/guided_v5_时间戳
```

## 验证边界

本机已做 CPU 检查、实际基线奖励重放、真实 RSL 库的合成环境更新和离线安装检查；没有新增机器人仿真或正式 PPO。旧 USB 缺少 v4 新候选模型/原始轨迹，第一步必须在另一台电脑完成。技术参数见安装后的 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5/TECHNICAL_PLAN.md`，测试证据见补丁根目录 `VALIDATION.md`。
