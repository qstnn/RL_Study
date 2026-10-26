# R5A Guided v4 离线补丁：触地前缓冲和落地恢复

本包在已成功的 Guided v3/2601 Actor 上继续优化：冻结起跳网络，训练空中末段和落地恢复，增强冲击/姿态反馈，降低阶段探索。原物理、原成功门及无环境随机化保持。成功初值已内置，无需联网或重新寻找教师。新奖励使用新Critic和Adam，因此不是直接恢复旧v3优化器。

## 安装到另一台电脑

适用目标：**已经成功运行 Guided v3 的原项目**，例如 `~/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019`。这是增量补丁，不是完整Isaac/机器人资产包；不用重新解压旧整包，不需要Git。补丁仅新增 `r5a_guided_v4/` 和 `r5a_guided_v4.sh`，旧训练、模型和其他路线保留。

将 ZIP 与同名 `.sha256` 用U盘拷过去，在二者所在目录校验、解压：

```bash
sha256sum -c lighthw-r5a-guided-v4-landing-patch-20261023.zip.sha256
unzip lighthw-r5a-guided-v4-landing-patch-20261023.zip
```

进入原项目根目录（应能看到 `r5a_guided_v3.sh` 和 `r6.sh`）。下方 `/实际解压位置` 换成上一步解压位置；安装器是标准库Python，检查全部文件后才写入，重复安装同一包可安全跳过。

```bash
cd ~/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
python /实际解压位置/lighthw-r5a-guided-v4-landing-patch-20261023/install.py .
bash r5a_guided_v4.sh check
bash r5a_guided_v4.sh train
```

**最后一条就是完整启动命令，不需要再启动其他训练命令。** 它先在当前GPU复核内置成功Actor两次，然后连续训练3000次更新尝试；每200次只保存，训练中途不做分段冻结评估。结束后统一比较本轮起点和最后最多3个候选，导出选中模型与录像。默认512环境，可显式覆盖：

```bash
bash r5a_guided_v4.sh train --num-envs 512 --updates 3000
```

前20次为Critic预热，包含在3000次中。若两次启动复核没有通过，命令停止并打印诊断ZIP；不要绕过复核或反复原样长训。不要给v4传入v3的 `--checkpoint`，默认命令已经完成正确的显式迁移。

## 回放与回收

```bash
bash r5a_guided_v4.sh play
```

默认打开最新选中模型。相机保持可自由操作，不新增跟随或持续强制固定视角。

训练结束后只需拷回终端打印的 **`[USB REVIEW ZIP]` ZIP及其 `.sha256`**，一般位于 `usb_review/route_05a/`。它包括训练窗口统计、每个站稳门的失败统计、候选比较、选中模型/录像/1kHz轨迹及来源。其他模型、全部回合和缓存留在训练电脑；无需拷整个项目。

若结果 `selected_is_baseline=true`，说明最终比较仍选择了本轮起点，不能把新训练完成当作已经改善。原v3选中模型也继续保留。

按一次Ctrl+C后等待回收完成；中断包有最近可读的 `latest_saved.pt`（可能早于中断时刻），明确尚未评估。可随后执行：

```bash
bash r5a_guided_v4.sh evaluate --checkpoint /实际审查目录/latest_saved.pt
```

同版本、同源码续训才使用下列命令，`--updates`为追加尝试数，旁边需要原始 `training/input/demonstrations.npz`：

```bash
bash r5a_guided_v4.sh train --checkpoint /实际路径/selected.pt --updates 3000
```

如果训练工作目录完整但ZIP导出失败：

```bash
bash r5a_guided_v4.sh collect --folder /实际路径/offline_sessions/route_05a/guided_v4_时间戳
```

## 验证说明

已完成CPU检查、原始轨迹奖励重放、RSL更新/冻结/恢复和离线安装检查。当前制作补丁的工具环境不能连接NVIDIA驱动，**没有新增原生仿真或证明v4行为已改善**。训练命令会在目标机先复核基线；真正改善情况仍需回收训练结果判断。详见补丁根目录 `VALIDATION.md`，技术参数见安装后 `r5a_guided_v4/TECHNICAL_PLAN.md`。
