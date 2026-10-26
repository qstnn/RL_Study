# R6 / Route06 重载静态持载训练

**键盘变载、终端反馈与单腿实时力矩图（v4.3，只用于回放）**

已有 v4、v4.1 或已装 v4.2 键盘变载补丁的电脑：先结束正在运行的项目进程，将 `lighthw-r6-usb-live-torque-20261019.zip` 解压到包含 `rl_training_r6_usb_20261016` 的父目录，覆盖同名文件；旧模型、训练记录和离线依赖保留。本补丁累计包含此前自由视角与键盘变载修改，使用已有离线依赖。进入项目后执行 `bash r6.sh prepare` 校验，然后：

```bash
bash r6.sh play --checkpoint "/实际路径/model_999.pt" --payload-kg 65 --keyboard-payload --play-seconds 120
```

也可用原训练会话自动选择模型与初始载荷：

```bash
bash r6.sh play --session "/实际路径/offline_sessions/route_06/原训练时间" --keyboard-payload --play-seconds 120
```

不要加 `--headless`。窗口打开后切换英文输入法，单击仿真视口，再使用：

| 按键 | 操作 |
|---|---|
| `[` / `]` | 载荷减少 / 增加 5 kg |
| `-` / `=` | 载荷减少 / 增加 1 kg |
| `0` | 切到空载 |
| `6` | 切到 65 kg |
| `1` / `2` / `3` / `4` | 力矩图切到左前 FL / 右前 FR / 左后 RL / 右后 RR |
| `R` | 保留当前载荷和所选腿，保存上一段后重新开始，清空图中历史 |
| `F10` | 保存并结束回放、自动收集 U 盘资料 |

每次按下载荷键，终端打印一次当前实际载荷与总质量；已经到 0 kg / 65 kg 边界时再次按键，也会反馈当前值。只处理按下事件，按住不连续刷屏。变载成功后显示回读校验过的质量，不把按键请求当成已生效的质量。质量限制在 0～65 kg。

窗口 `R6 Payload / Actual Joint Torque` 继续显示载荷、总质量、roll/pitch、高度和 16 个关节实际力矩。新增 `R6 Live Leg Torque` 窗口默认显示左前腿 FL，按 `1` / `2` / `3` / `4` 切换 FL / FR / RL / RR；每次只画所选腿的 hip、thigh、calf、wheel 四个电机，分别放在四幅小图中。横轴为仿真时间，纵轴为 N·m，各图自动调整对称范围并保留正负方向；图上显示完整关节名和当前力矩。曲线使用实际施加力矩 `applied_torque`，不是策略动作或未限幅的计算力矩。

实时图显示最近 10 秒，按 50 Hz 逐策略帧加入历史、每秒约刷新 10 次；切换腿可立即查看它已有的同段历史。按 `R` 保存上一段并重新开始时，清空图中旧历史，保留所选腿，避免把两段首尾连在一起。原始 50 Hz 全关节轨迹和 1 kHz 物理数据仍完整记录，不因只画一条腿而减少。失稳等待重置时不添加重复点。鼠标仍可自由移动视角，需要跟随时自行在 Isaac Sim 设置。

质量切换发生在两个策略步之间：从规范空载 URDF 重新计算质量、合成质心、完整惯量，写入真实物理对象并回读校验，再同步策略与 Critic 的载荷输入。机器人当前位姿和速度不被主动重置；`R` 才请求重新开始。它表示理想刚性载荷参数的瞬时变化，不模拟真实搬放重物的碰撞或动量交换。

`--play-seconds` 是累计仿真时间预算，默认 120 秒，可设 1～600 秒；等待按键和失败后等待 `R` 不计入预算。失稳时保留自动reset前的终止帧，停止推进该段，等待你调整载荷后按 `R`；重置不是策略恢复平衡。请用 `F10` 正常结束，关闭窗口或 Ctrl+C 可能只能保留部分视频/资料；原始会话仍可补收集。

回放结束打印 `[USB REVIEW] ...`，把这个完整文件夹用 U 盘带回即可。新增 `keyboard_payload_events.jsonl`、各段 `*_payload_events.jsonl` / `*_payload.json`、`payload_after_step.jsonl`，保留每次变载的请求和实际质量/质心/惯量核验；每帧记录对应载荷，另生成 `*_payload_response.png`，对照载荷、姿态、高度、支撑和各关节力矩。原 `payload.json` / `params/env.yaml` 表示初始载荷，后续变化以事件与轨迹为准。

交互记录标为 `interactive_payload_change_v1`，不计算固定载荷筛查成功率。当前模型只在 65 kg 下训练过，能否在较轻载荷或突然切换时稳定，要看实际回放；本补丁不会重新训练模型或修改训练奖励。普通 `play` 不加 `--keyboard-payload` 时仍沿用原 20 秒流程。


v4.2 键盘变载基线已有 76 项 CPU 检查、实际 GPU 变载和 GUI 输入/录制链路验证，见[键盘变载记录](records/usb_keyboard_payload_validation.json)。v4.3 终端反馈与实时图的本轮检查见[实时力矩图验证记录](records/usb_live_torque_validation.json)；这些实现短验不代替全载荷稳定性评估。

**v4.1离线训练工作站：** `bash station.sh train --route 06 ...`自动训练、评估、录屏和收集，`bash r6.sh train ...`为同一流程的快捷入口。结束后只拷控制台 `[USB REVIEW]` 对应的完整目录。旧run可用 `station.sh review --route 06 --run ...` 补齐评估；见[完整说明](../../../../../offline/STATION.md)。

**累计包含v3离线播放：** `bash r6.sh play --checkpoint /实际路径/model_999.pt --payload-kg 0` 默认使用本机LightHW深红/金色配色、当前R1 V2蓝灰地面、本地4K天空和自由视角；鼠标可自由旋转、平移和缩放，需要跟随时在IsaacLab → Viewer Settings → Follow Mode中选择Robot。普通 `play` 未加 `--keyboard-payload` 时只调整外观与初始机位，原 R6 平地碰撞/观测/奖励/载荷保持；无需重新训练。完整v4.1包含此前USB修复；自由视角小补丁适用于已安装的v4，覆盖和校验见[U盘启动说明](../../../../../U盘启动说明.md)。

**2026-10-16：本次使用U盘完整包。** 启动入口统一为项目根目录 `bash r6.sh prepare`、`bash r6.sh smoke`、`bash r6.sh train-smoke`，正式训练用 `bash r6.sh train --payload-kg 0 --num_envs 1024 --max_iterations 1000 --seed 42`。详见[U盘启动说明](../../../../../U盘启动说明.md)。环境固定官方RSL-RL2.3.3，修正旧环境METADATA误报3.0.1。

**任务：`MultiSkill-LightHW-HeavyPayload-v0`。GPU 启动和训练链路已验证；本次新增键盘变载回放，跨载荷稳定性须按实际轨迹判断。**
目标对应1-11：静止承载分级增加到65kg。当前实现固定载荷的阶段训练，首轮空载；不自动跳到65kg。
本机正在训练R1，本路线在另一台电脑运行。先按[同步与环境说明](../../README.md)获取
`lighthw-multiskill-heavy-payload-r6`。支持 Git 拉取，也支持网页下载 ZIP 后运行
`python3 model_lighthw/training/multiskill/prepare_zip.py` 补齐网格；再在项目根目录执行以下命令。

## 直接启动

```bash
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/run.py check --route 06 --payload-kg 0
python model_lighthw/training/multiskill/run.py smoke --route 06 --payload-kg 0 --headless --num_envs 32
```

训练/保存链路短验，预算2次更新，生成的模型只作实现检查：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 0 --headless --num_envs 32 --max_iterations 2
```

正式空载持载基线，从零启动，不读取R1模型：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 0 --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

每100次更新保存一次，预算结束退出；不会自动回放、升阶段或延长预算。显存较小时可降低`--num_envs`，
同时会减少每次更新的样本量（默认每环境24步），比较实验时需记录。

## 固定载荷模型与控制契约

| 项目 | 设置 |
|---|---|
| Actor / Critic / Action | 58 / 237 / 16 |
| Actor | R1的57维可部署基础观测，加最后1维已知装载质量/65 |
| Critic | R1的236维特权观测，加同一载荷通道 |
| 动作 | 12腿位置 + 4轮速度；按名称保持FR/FL/RR/RL顺序 |
| 姿态/控制 | 原始0.416m站高；腿PD 80/3；轮阻尼1；1kHz物理、50Hz策略 |
| 命令/地形 | 三轴恒零，平地，20s episode |
| 载荷 | 每个run固定`--payload-kg`，允许0～65kg；无跨run自动晋级 |
| 载荷位置与形状假设 | Base_link坐标(0,0,0.20)m；惯量按0.30×0.25×0.20m均匀箱体 |
| 状态随机化 | reset roll/pitch±0.015rad、yaw±0.05rad；没有外力push |

载荷与Base_link刚性固定，采用平行轴定理合成质量、质心与完整惯量。
每次只写入本run的`input/robot_payload.urdf`，网格指向当前checkout；规范URDF、原始网格与R1配置不变。
运行时会比对PhysX的总质量、Base_link质心与惯量，转换不一致即报错。
载荷通道是已知装载设定值，不是已实现的真实载荷估计器。

载荷不增加碰撞外形；本阶段不模拟放置箱子的接触冲击、单episode动态缓加载、热/结构持续承载。
65kg是目标工况，不是已达标范围；模型48Nm限幅不代表实机连续额定力矩。

奖励分别约束正立/高度、机身静止、四轮支撑、轮速、力分配、腿姿/腿速、模型力矩余量、关节限位和动作变化。
倾角超过30°、高度低于0.25m或非轮碰撞触发终止；episode时限单独计为timeout。

## 查找模型、回放与评估

控制台打印`[R6] <run目录>`。正式训练输出在：

```text
model_lighthw/training/multiskill/routes/route_06_heavy_payload/runs/<上海时间>_payload0kg_seed42/
```

`training_result.json`的`checkpoint`字段是本次最终模型路径。
以下`/absolute/path/to/r6_model.pt`必须替换为这个路径，并明确传入待测载荷：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/play.py --payload-kg 0 --checkpoint /absolute/path/to/r6_model.pt
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/evaluate.py --payload-kg 0 --headless --checkpoint /absolute/path/to/r6_model.pt --eval_seeds 43 44 45 --eval_episodes 4
```

GUI回放单次20s；评估默认3个seed×4次reset，每次20s。开头2s保留作诊断，2～20s为持载统计窗。
提前终止、缺样本、非有限值保留为失败；自动reset后的出生帧不计入通过样本。
评估按50Hz记录姿态、高度、速度、四轮/逐轮力、每关节力矩、饱和与位移；不是1kHz冲击峰值或热验收。
固定筛查门槛见[evaluation](evaluation/README.md)。

## 分级继续训练

建议阶段为0→5→10→15→25→40→65kg。检查当前阶段固定评估，再显式提高载荷，例如从空载模型继续5kg：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 5 --checkpoint /absolute/path/to/r6_model.pt --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

`--max_iterations`表示本次追加更新数。恢复同契约R6的Actor、Critic、探索参数、Adam张量和实际学习率，
从下一个迭代号开始；新run重置物理状态与随机数，不声称恢复同一个仿真帧。
不会覆盖源checkpoint；R1与旧60维模型被拒绝。可将5换为其它已决定的阶段，但没有自动能力背书。
不同负载逐段继续训练可能遗忘轻载能力，晋级前还应对已完成负载复测；不把单阶段奖励上升当作全载荷通过。

## 运行记录

- `run.json`、`request.json`、`contract.json`：时间、状态、参数、契约与输入checkpoint哈希。
- `payload.json`、`input/robot_payload.urdf`：实际载荷的质量/质心/惯量和生成URDF哈希。
- `asset_manifest.json`、`source_manifest.json`、`source_snapshot/`：规范URDF/17网格哈希及源码快照。
- `params/env.yaml`、`params/agent.yaml`、`runtime_check.json`：有效配置、实际维度和物理资产检查。
- `console.log`、`training_metrics.csv`、`training/`：完整控制台、逐更新指标、TensorBoard和checkpoint。
- `training_result.json`、`result.json`：完成更新数、最终模型路径与哈希；缺失时启动器判失败。

这些产物只保存在当前电脑。详见[本轮验证](records/validation.json)；GPU短验未执行，正式R6机器人训练尚未开始。

## 启动提前退出或缺少 runtime_check.json

`runtime_check.json` 是环境创建并通过维度、质量、质心和惯量检查后生成的结果，不能手工补一个空文件。
缺少它说明运行时检查未完成，需要查看同一个run目录的 `console.log` 查找此前异常；进程返回0也可能提前退出。
新版本会在 `worker_status.json` 保存最后阶段（如 `app_start`、`config_load`、`environment_creation`、
`runtime_validation`），并在关闭Isaac Sim之前把Python异常写入 `worker_error.json`。
`run.json` 的 `worker_returncode` 保留子进程实际退出码，`returncode` 是整个启动器最终结果。

请先保留失败目录，读取 `console.log` 末尾约180行和 `run.json`；新版本再查看上述两个worker诊断文件。
早期版本缺少worker诊断文件时，已有 `console.log` 仍可用于分析。
只报告缺少JSON不能判断是依赖、GPU、资产创建还是运行时校验失败，需要前面的日志证据。
