> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A 自主后空翻 v4：离线安装与训练

本补丁用于已有的、能运行 R5A/v3 的完整离线工程。只新增 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v4/` 与 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v4.sh`，不需要 Git、网络或另拷训练模型。机器人网格、本地天空/地面、Isaac Sim 和离线 Python 依赖继续使用现有工程。

## 在另一台电脑安装

把补丁 ZIP 和同名 `.sha256` 用 U 盘复制过去。**先解压补丁到独立文件夹，再用安装器指定现有项目根目录；不要把补丁文件夹当成完整训练项目。** 项目根目录应能看到 `r6.sh`、`model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh`、`source/`、`Isaaclab/`。

在 ZIP 所在目录验证：

```bash
sha256sum -c lighthw-r5a-autoflip-v4-joint-patch-20261020.zip.sha256
unzip lighthw-r5a-autoflip-v4-joint-patch-20261020.zip
```

把下面第二个参数替换成那台电脑现有工程的绝对路径。例如，v3 返回日志中的项目路径是：

```bash
python3 lighthw-r5a-autoflip-v4-joint-patch-20261020/install.py /home/huawei/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
```

安装器只用 Python 标准库，先校验基础文件和所有补丁文件；遇到同名不同内容会在写入前停止，重复安装相同内容可直接跳过。不要手工覆盖报错文件，应保留报错用于核查。

## 启动

进入刚才指定的完整项目根目录，在现有 `lab` 环境运行：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v4.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v4.sh train
```

默认 **512 环境、3000 次 PPO 更新尝试**，每 200 次只在本机保存检查点；不分段评估、不暂停等待人工检查。`train` 自行完成训练、末尾一次统一评估最后最多 3 个候选、录像、选模与精简打包，**不需要再启动别的训练命令**。连续 KL 超限、非有限值或运行故障会停下并回收诊断，这些保护仍保留。

新建 v4 训练自动使用补丁内已审查的正式 v3/2000 Actor 初始化。空中/落地 Actor 保留，地面 Actor 从其特征复制后缩小输出层并清零输出偏置，Critic、Adam、v4 计数重新开始。早期地面动作重新学习，可能暂时低于 v3 的表现；安装完成不代表已经学会后空翻。

如确需指定预算，可用 `bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v4.sh train --updates 2000`；不要把本机短验模型或旧 v3 的完整检查点作为 `--checkpoint` 传入。只有同版本、同源码的 v4 检查点能显式续训；物理现场会重新开始。

## 完成后只拷这些

终端最后会打印：

- `[LOCAL FULL DATA]`：本机完整资料目录，保留在训练电脑。
- `[USB REVIEW ZIP]`：本次生成的精简 ZIP。

**只把该 ZIP 和旁边的 `.sha256` 拷回。** 不需要拷整个工程、所有旧 `usb_review`、所有模型或 USD 缓存。

精简包保留 1 个选中训练模型、该模型的 1 kHz 原始物理轨迹/50 Hz 策略输入/录像、最多 3 个候选摘要、全部更新记录、全量回合的分窗口统计/最多 20 条示例，以及单份源码/配置/依赖。预蹲、起跳 H/机身转速、联合进度与各轮末段承载也有记录。其余模型、完整训练回合 JSONL 和大日志仍留在训练电脑。

选模先看物理筛查和完整任务等级，再看“转角×真实跳高/滞空”联合进度；同等级且联合进度相差不超过 0.02 时优先恢复质量。都未完成时仍标记 `best_available_not_passed`。无有效腾空时 H 无法检查会单独记录，不能当作物理通过。只比较最后少数候选，不证明全训练历史最佳。

## 回放与回收故障

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v4.sh play
```

默认播放本机最新选中 v4 模型，使用确定性动作。保留 LightHW 原配色、本地天空/地面和 World 自由视角；可在 Isaac Sim 窗口自行调整相机。

若训练已经结束但回收打包失败，将实际 `[LOCAL FULL DATA]` 路径填入：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v4.sh collect --folder /实际完整运行目录
```

若 `r6.sh` 提示找不到 Python/Isaac Sim，沿用原项目已有的 `R5A_PYTHON` / `R5A_ISAAC_SIM` 或 `R6_PYTHON` / `R6_ISAAC_SIM` 配置。本补丁不安装 Isaac Sim，也不联网准备资源。

技术路线见 [DESIGN.md](DESIGN.md)。交付时的 CPU/GPU/离线安装验证见补丁根目录 `VALIDATION.md`；正式 3000 次训练与完整后空翻验收由后续返回结果决定。
