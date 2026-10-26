> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A 自主后空翻 v2：先修复动力学，再重新训练

2026-10-20。新入口 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh`，独立源码 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v2/`。旧 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh` 对应已发现物理漏洞的 v1，请勿用于下一轮正式训练。v2 不加载 v1、J1、highjump 或 R6 模型；第一次直接从新策略训练。

## 为什么重新开始

旧刚体角速度保护阈值 `max_angular_velocity=1000` 的单位为 **deg/s**，即 17.45 rad/s，低于轮关节允许的 37.7 rad/s。轮体世界角速度被截断，破坏了无外力阶段的角动量收支。相同冻结策略，仅把这个保护阈值改为 100000 deg/s，腾空段后旋角动量峰峰变化约 8.073→0.0139 kg·m²/s，后旋约 365.2°→66.4°。旧模型的一圈不能再认定为可信动力学下的空翻。

**腿/轮扭矩仍为 48/24 Nm，关节速度仍为 12.57/37.7 rad/s。** 修的是刚体数值截断，并未增强电机、改变重力或注入根速度。保留零负载、自碰撞、原质量、惯量、地面摩擦及陀螺项。

## U盘安装

将 `lighthw-r5a-autoflip-v2-physics-patch-20261020.zip` 和同名 `.sha256` 拷到另一台电脑，解压。该包是增量补丁，安装目标是已有的完整 R5A 离线工程（之前能运行自主空翻的目录），不是空目录。

在补丁解压目录执行，替换下面的工程路径：

```bash
python install.py "/你的完整R5A工程路径" --check
python install.py "/你的完整R5A工程路径"
cd "/你的完整R5A工程路径"
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh train
```

安装器先核验基线与全部补丁SHA，再新增文件；遇到缺失/不同的基础文件、同名v2文件被改动会停止。无需Git或网络，不覆盖旧任务。项目原有 `lab`/Isaac Sim 路径配置继续有效；必要时沿用已有的 `R5A_PYTHON`、`R5A_ISAAC_SIM` 设置。

## 一条训练命令完成整条流程

`bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh train` 默认512环境，**总计2000次PPO更新尝试，每200次冻结复评后自动接着训练**，不是只训练200次。最后自动选择模型、12例最终评估、3段录像、曲线和ZIP回收。一次回合最长6秒，即300个策略步；回合长度不代表成功率。

可显式设置预算：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh train --num-envs 512 --updates 2000 --interval 200
```

PPO采用tanh有界动作及匹配的对数概率/熵；初始腿/轮噪声0.25/0.10，学习率0.0001。整批KL超过0.03或参数非有限时恢复Actor、Critic和Adam并减小学习率；连续5次拒绝则停止并保留资料。`completed_updates`是更新尝试数，`accepted_updates`是真正保留的更新数。

新奖励分别反馈落地直立、站高、四轮载荷、低速度、低漂移，并惩罚落地轮空转；这些恢复信号只在有效首次腾空后着地时启用。落地冲击惩罚不再在旧上界处饱和。**硬成功判据没有放宽**：首次连续腾空内单圈350–385°、轴向误差和触地倾角均≤20°，无非轮碰撞，最后连续1秒满足四轮支撑、站高、倾角、速度与漂移要求。

冻结复评额外筛查全机刚体角动量：去掉无接触区间前后2ms，至少25样本，最大矢量偏移≤max(0.1 kg·m²/s，起始模长10%)。不通过会停止继续训练并回收资料；无腾空标记为未检查，不视为通过。该阈值是数值异常筛查，不是精确零残差证明，也未把执行器虚拟armature等价为真实电机转子。

模型筛选先排除物理异常，再比较完整成功、合格触地、站稳时长和恢复质量，之后才比较未完成单圈的进展与高度。旧版毫米级高度优势不再压过站稳质量。

## 回放、续训和回收

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh play
```

默认加载本版最近完成的选中模型，World自由视角；不自动跟随或锁死相机。要跟随时自行在Isaac Sim窗口设置。

仅支持**同一份v2源码**的检查点续训（新增预算，不含之前已训练更新）：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh train --checkpoint "/上一轮目录/last.pt" --updates 2000
```

`selected.pt`是冻结评估筛选模型；`last.pt`是最后一段模型。若中途中断，可选择最后一个完整 `train_XXXXXX/final.pt`，不要使用未完成的写入文件。不要直接拿旧v1模型续训来绕过版本检查。

训练结束打印：

```text
[USB REVIEW] .../usb_review/route_05a/autoflip_v2_日期时间
[USB REVIEW ZIP] .../usb_review/route_05a/autoflip_v2_日期时间.zip
```

**带回这个ZIP和同名.sha256即可**。ZIP包括模型/优化器、真实源码、运行配置、1kHz原始轨迹、50Hz策略输入输出、训练日志、冻结复评、录像和逐文件SHA；不含可重建的大型USD缓存。源目录仍保留。异常退出也尽量打包；若打包中断，确认训练已停止后执行：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh collect --folder "/这次审查目录"
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v2.sh verify --folder "/这次审查目录"
```

`session.json`仍标记running时collect会拒绝；先保留整个原目录供检查，不要在训练仍运行时封包。

本次仅交付环境与学习流程修正，以及本机短验依据；没有宣称v2已经学会后空翻。另一台电脑完成正式训练后再审查是否得到足够的真实起跳角动量、空中转动和落地恢复。
