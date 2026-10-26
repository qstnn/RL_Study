# R5A Teacher Guided v3 离线训练

这是 **Guided v3**，独立于旧 AutoFlip v3 和 Guided v2。补丁只新增 `r5a_guided_v3/` 与 `r5a_guided_v3.sh`，以已经能运行 Guided v2 的离线项目为基础。无需联网、Git 或重新下载机器人资产。

本版针对 Guided v2/1401 的真实回放：首次腾空310.88°、首次触地倾角38.77°、末段漂移17.58cm。目标是先保持教师的完整翻转，再优化落地，而不是通过提高跳高补偿缺角。原来的接触、角动量、单圈、站稳判据和48/24Nm限矩保持不变。

## 安装与启动

1. 将 ZIP 及 `.sha256` 拷到另一台电脑，解压到项目之外。
2. 进入原有训练项目根目录，确认这里同时有 `r6.sh`、`r5a_guided_v2.sh`。
3. 使用系统 Python 安装；安装器检查原项目与补丁哈希，拒绝覆盖不同内容的文件，可重复安装相同补丁。

下面以之前的训练目录为例。将 `/你的U盘路径/` 替换为实际目录：

```bash
cd ~/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
python3 -m zipfile -e /你的U盘路径/lighthw-r5a-teacher-guided-v3-patch-20261022.zip /tmp/r5a_guided_v3_patch
python3 /tmp/r5a_guided_v3_patch/lighthw-r5a-teacher-guided-v3-patch-20261022/install.py "$PWD"
bash r5a_guided_v3.sh check
bash r5a_guided_v3.sh train
```

`train` 一条命令会完成：教师在目标电脑两次完整复核 → 分阶段模仿 → 必要时收集纠偏数据并再拟合 → 学生两次自主完整回放通过 → **连续3000次PPO更新尝试** → 末尾统一评估最后最多3个存档 → 导出一个选中模型与精简USB ZIP。无需再执行第二条训练启动命令。

默认512环境。显存不足时可明确改为：

```bash
bash r5a_guided_v3.sh train --num-envs 256 --updates 3000
```

准备阶段日志 `[GUIDED BC] step=5000` 是模仿学习步数，不是PPO更新数；前20次PPO更新是Critic预热。正式训练中没有分段冻结评估，训练回合的在线统计仍正常记录。

如果环境路径不同，沿用原来的覆盖变量，例如：

```bash
R5A_PYTHON=/实际路径/lab/bin/python R5A_ISAAC_SIM=/实际路径/isaac-sim bash r5a_guided_v3.sh train
```

## 本版变化

- 起跳、空中、落地恢复分别使用网络分支，减少恢复学习对空中动作的干扰。输入仍为420维、动作16维，但参数结构和奖励契约已经改变，**不能把v2 PPO检查点直接续训到v3**。
- 学生可以精确输出教师的有界极值；模仿损失使用限幅前输出，保留纠正错误饱和输出的梯度。
- 第一次只在教师数据上拟合；若学生未自主通过，自动尝试教师/学生混合和学生前缀后教师接管。教师标签来自这些实际访问状态，只有完整回合通过物理复核的数据才被加入。混合成功不会被当成自主成功。
- 默认最多6轮纠偏；超出预算或找不到合格纠偏轨迹，会停止并输出诊断ZIP，PPO不会在已知不合格的初值上长时间运行。两次固定初态重复只用于启动检查，不代表扰动鲁棒性。
- 旋转、触地姿态和恢复奖励不再乘跳高因子；保留线性早期引导，并加强接近一整圈的增量收益。冲击代价保持有界，超过26倍体重后仍可区分大小。
- 示范约束读取正常训练中最近2048个已结束回合。至少512个回合，且完整成功率≥80%、合格落地率≥90%时，权重才从0.20减为0.08；退化时恢复0.20。不是额外评估，也不会按轮次直接降到零。
- 环境随机化和观测噪声继续关闭；PPO仍保留学习所需的动作探索。回放关闭动作探索，摄像机可由Isaac Sim窗口自由调整。

## 观察进度、停止与回收

终端会输出准备阶段、PPO更新数、最终评估和 `[USB REVIEW ZIP]`。完整日志留在 `offline_sessions/route_05a/guided_v3_*`。

正式训练中可按一次 `Ctrl+C`，等待子进程结束与压缩完成。中断包包含**一个最近成功保存且可读的 `latest_saved.pt`**，明确标记为未评估；它可能早于中断时刻。不要直接强杀进程或关机后假定ZIP已经完成。

正常结束的包只包含一个选中PPO模型、对应录像/轨迹、训练统计、代码及依赖来源，以及有限的准备阶段证据。完整回合JSONL、其他模型、所有纠偏原始轨迹和USD缓存留在训练电脑。`best_available_not_passed` 表示候选尚未通过，不能把“选中”理解为训练成功。

只需将打印出的 ZIP 和同名 `.sha256` 拷回。中断后若想补评估：

```bash
bash r5a_guided_v3.sh evaluate --checkpoint /实际USB审查目录/latest_saved.pt
```

若压缩失败，但训练工作目录仍在：

```bash
bash r5a_guided_v3.sh collect --folder /实际路径/offline_sessions/route_05a/guided_v3_时间戳
```

同一版本续训时保留原始 `training/input/demonstrations.npz`：

```bash
bash r5a_guided_v3.sh train --checkpoint /实际路径/model_XXXX.pt --updates 3000
```

这里的3000是**追加**更新尝试数。初始化成功证据、示范来源、模型结构和实现哈希必须匹配；奖励变更不能静默恢复旧Adam。新一轮训练会重置物理状态，检查点不包含物理场景快照。

## 验证范围

见补丁中的 `VALIDATION.md`：CPU测试、旧轨迹奖励重放、实际CPU教师/学生完整物理回放、训练更新/恢复/中断回收及离线安装验证分别列出。目标电脑的GPU启动复核由命令自动完成；本机CPU成功不自动当作目标GPU或实机验收。正式训练的改进程度仍以训练后回收包为准。
