> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A 自主后空翻 v1：U盘离线 PPO 训练

这次直接建立新的强化学习任务，策略自行选择预蹲、蹬伸、收腿和展开时机。固定关节轨迹、助跳外力和根速度注入均未使用。补丁是可训练的软件起点，不代表已经学会空翻；本机短验仅用于验证接口，正式能力以另一台电脑训练后的冻结回放为准。

## 安装和开始训练

适用于已经能运行的 R5A J1 v2 或 v3 离线工程。原工程、Isaac Sim、lab 环境和离线资源仍需保留；这个补丁无需联网，也不是完整项目包。不要解压到旧R6工程或另建一个只有补丁的工程。

1. U盘复制 `lighthw-r5a-autoflip-v1-patch-20261019.zip`，解压。
2. 在解压后的补丁目录运行（替换为实际原R5A工程绝对路径）：

```bash
python3 install.py /你的路径/rl_training_r5a
```

安装器先核验基线代码、模型和天空资源以及补丁 SHA，然后只新增 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto/` 和 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh`。文件不符时列出差异并停止，不覆盖旧训练配置；完整重复安装可直接执行。

3. 回到原R5A工程目录，依次运行：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh train
```

`check` 是CPU判据和资源检查，不需要GPU仿真。`train` 会自行完成训练、定期冻结评估、选择检查点、最终评估、3个种子的录像和统一回收。无需再执行 `refine`、`highjump` 或其他训练启动命令。首次运行一个新迁移工程，若提示依赖未准备，先运行原命令 `bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh prepare`，它使用包内 wheels 离线准备。

默认 512 环境、2000 次 PPO 更新，每200次按固定seed43/44/45各2回合冻结评估。最终选出的模型再做每种子4回合评估及每种子1段录像。第一次建议完成这个预算后把资料带回审查，不因 mean episode length 或总奖励高就无限续训。

显存不足可改为：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh train --num-envs 256
```

每回合最长6秒，策略50Hz，因此 mean episode length 上限300步；3秒内未真实起跳会失败。这个步数不是空翻成功率。初期失败、短回合是新动作搜索的一部分，但是否有学习进展必须看有效腾空、空中转角、合格落地和成功率。

## 新任务如何控制和评分

- Actor 输入84维状态帧、5帧历史，共420维；Critic104维；16维动作。12个腿关节为绝对位置目标，4轮为轮速目标。0动作是站姿，仅出生后0.4秒用于安放，之后没有固定蹲起时刻。
- 髋目标±0.38rad；大腿±1.45rad；前膝−2.40至−0.12rad，后膝+0.12至+2.40rad。目标留有余量，避免跨过膝伸直分支；这些是任务动作范围，不修改URDF硬限位。目标每50Hz更新，由原PD和48/24Nm、12.57/37.7rad/s执行器执行。
- 真实起跳要求所有连杆接触力低于2N、轮净空为正、起始COM向上速度>0.25m/s，并连续7个1ms采样确认。连续第一次无接触时间单独计算，不累加多跳。
- 奖励包括有界的起跳/真实高度/后旋进度、落地时的一圈后旋评分、姿态和停稳、终末成功，以及碰撞、侧轴偏离、过大冲击和控制代价。高度和速度是策略要选择的结果，没有逐级增高课程，也没有精确转速指令。辅助量饱和仅用于限制奖励规模，不是已经达到的高度。
- 验收要求一次真实腾空≥0.10秒、离地后COM继续上升≥2cm、空中后旋350–385度、侧轴偏离≤20度、落地倾角≤20度、无非轮接触，并在末段四轮支撑、低速、直立连续保持≥1秒。不是只看最终四元数；0度与360度会区分。
- 全程倾角60度终止及全程直立奖励已经从新任务移除。身体撞地仍失败。空中允许倒立，地面翻滚不算空翻，前翻或多圈不算单次后空翻。
- 整机角动量包含各连杆自转与轨道项，供Critic和审查使用。Actor的COM/接触等来自仿真真值，尚未替换为实机估计器；不据此宣称可直接上实机。

新网络从中性站姿输出初始化，探索标准差前12腿动作0.35、轮0.10。没有沿用J1小残差网络、没有行为克隆参考、没有脚本接管起跳。PPO学习率3e−4、自适应KL调度，实际噪声与每次更新的KL均记录。初始权重是待训练设置，不保证第一轮就有整圈成功。

## 回放、恢复和审查资料

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh play --play-episodes 10
```

默认使用最近一次选中的自主任务模型，World自由相机，可在Isaac窗口自行移动或启用跟随；程序不持续改写相机。沿用原LightHW配色、地面和本地天空。

要恢复本版模型并增加训练预算，显式指定自主任务检查点：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh train --checkpoint /实际路径/last.pt --updates 2000
```

恢复Actor/Critic/Adam、学习率、更新计数和随机状态；物理环境重新从站姿开始，不声称恢复了中断瞬间的仿真状态。旧J1、高跳或R6模型会被拒绝，不能混用。`selected.pt`按固定冻结指标选出，`last.pt`是最终训练模型，二者都保留，模型来源和版本SHA绑定。

任务结束（含正常失败或Ctrl+C）打印：

```text
[USB REVIEW] .../usb_review/route_05a/autoflip_时间戳
```

**把整个这一文件夹拷回**，不要只拷 `selected.pt`。其中包括：会话配置、执行源码快照和SHA、每段初始/最终/中间模型、优化器、TensorBoard及每回合分项奖励、KL/噪声、定期和最终评估、1kHz多体状态/接触/实际力矩、50Hz模型输入和动作、独立NumPy判定、图表、视频、故障日志。未完成的worker也保留原始错误；不要只根据缺少runtime_check.json推断故障。

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto.sh verify --folder /实际路径/autoflip_时间戳
```

常规结束自动写 `SHA256.json`；强制断电时可能没有清单，保留整个原目录带回，不丢弃原始日志。仿真USD缓存和Python缓存不计为必须回收证据。没有GitHub上传或在线依赖安装步骤。
