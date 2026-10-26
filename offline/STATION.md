# 离线训练电脑与U盘审查流程

另一台电脑只负责训练、评估和回放。联网下载、代码调整与正式审查在本机进行。训练电脑不需要Git、GitHub、网络证书或在线下载资源；运行期间不要覆盖项目源码，启动和结束均校验源码与快照是否一致；继续使用已安装的Isaac Sim 5.0、NVIDIA驱动、lab Python和随包提供的离线依赖。

## 一条命令完成训练、检查和收集

在项目根目录执行：

```bash
bash station.sh train --route 06 --payload-kg 0 --num_envs 1024 --max_iterations 1000 --seed 42
```

`bash r6.sh train ...`现在是相同流程的R6快捷入口。训练成功后自动：

1. 冻结本次最终模型，使用实际训练请求中的载荷；默认seed43/44/45、每seed4次reset，每次最多20秒仿真，记录50Hz策略轨迹和1kHz物理轨迹。
2. 用同一模型和载荷、训练seed录制一段最多20秒的MP4。后台录屏无需打开GUI；FFmpeg来自已有离线wheel。
3. 生成训练概览图，汇集全部模型、日志、配置、代码快照、数据、视频和实际资产，逐文件SHA256校验。

控制台先打印`[SESSION] ...`，结束时打印`[USB REVIEW] ...`。**只把最后这个[USB REVIEW]文件夹整体拷到U盘带回，不需要人工找日志。** 训练后评估/录屏可能仍需数分钟；以最后的资料目录和session状态为准，不以出现最后一个model文件判断整个流程结束。

短验命令（32环境2次更新、仅1个评估seed×1次episode，不是正式训练）：

```bash
bash station.sh train --route 06 --payload-kg 0 --num_envs 32 --max_iterations 2 --eval_seeds 43 --eval_episodes 1
```

## 目录内容

- `offline_sessions/route_06/<时间>/`：训练电脑的原始工作目录。包含train、evaluate、video等子运行及session.json、阶段日志、当次环境/硬件报告。
- `usb_review/route_06_<收集时间>/`：可直接带回的完整审查目录。每次收集新建，旧审查目录不覆盖；只拷最后打印的那一份即可。

审查目录中保留：

| 文件 | 用途 |
| --- | --- |
| README.md、review.json | 路线、运行状态、完整性提示、各资料的索引 |
| runs/*/training/*.pt | 全部已保存checkpoint，保留Actor/Critic和Adam，不只带最后一个模型 |
| training_metrics.csv、ppo_updates.jsonl、TensorBoard | 姿态/支撑/力矩均值、每次PPO损失、学习率、噪声、已结束episode长度/奖励、各奖励/终止项 |
| *_overview.png | 在不启动TensorBoard的情况下查看概览 |
| params、request、contract、payload、runtime_check | 实际训练参数、路线契约、载荷与运行时物理校验 |
| source_snapshot、source_manifest | 当次运行的真实源码快照和哈希，不用当前新代码冒充旧训练来源 |
| *_trace.npz / *_trace.csv / *_trace.json | 50Hz状态、观测、动作、目标、奖励、接触、终止原因；包含自动reset前失败帧 |
| *_physics.npz | 评估的1kHz位置/速度/接触、实际力矩与计算力矩；顺序与频率见trace.json |
| *.mp4、*_preview.png、*_video_frames.csv | 回放视频、预览以及帧与仿真时间的对应关系 |
| console.log、worker_error/status、run.json、session/ | 原始失败信息、真实返回码和阶段状态 |
| assets、input_checkpoints | 按SHA保存的实际STL/URDF和续训/回放输入模型 |
| environment | 当次模块来源、版本、GPU/驱动、交付版本与依赖清单 |
| manifest.json、verify_review.py | 拷贝完整性核验；只用Python标准库，不启动仿真 |

v4审查统计的力矩饱和按显式执行器effort_limit与PhysX上限的较小值计算，并另外记录计算力矩与实际力矩的裁剪比例。旧统计只用了PhysX上限，不能直接当作执行器余量；原奖励、执行器参数和物理限幅未修改。trace schema 2同时提供actuator_effort_limits与sim_effort_limits。

不复制Isaac Sim安装、wheelhouse、依赖安装目录或可重新生成的USD缓存。审查资料包含真实网格和所有已保存模型，通常为几百MB起；长训练保存更多checkpoint时会增大。原始训练目录、模型及已有审查目录都保留。

## 训练后人工回放

用训练开头打印的SESSION路径执行；自动选用该会话的最终模型和载荷：

```bash
bash station.sh play --route 06 --session "/实际路径/offline_sessions/route_06/本次时间"
```

会打开GUI、启用本机LightHW配色、蓝灰地面、本地天空、默认自由视角，并保存视频和轨迹。启动时只设置初始机位，之后可用鼠标自由旋转、平移和缩放；需要跟随时，在IsaacLab → Viewer Settings → Follow Mode中选择Robot，选择World则取消跟随。回放加入同一会话，结束后再打印一个**新的**完整审查目录，包含训练、原评估及新回放。带回新的那份即可。

也保留直接指定模型的方式：

```bash
bash r6.sh play --checkpoint "/实际路径/model_999.pt" --payload-kg 0
```

若模型仍在原运行的training目录里，会自动关联该训练目录；若只是单独模型文件，只能带回模型和本次回放，原训练曲线不会凭空恢复。

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

## 已经训练完成的旧模型

不需要重训。传入原来的**训练run目录**（含request.json、training_result.json、training等），自动评估、录屏、收集：

```bash
bash station.sh review --route 06 --run "/实际路径/旧训练run目录"
```

若训练中断而没有最终结果，明确选定已有checkpoint：

```bash
bash station.sh review --route 06 --run "/实际路径/旧训练run目录" --checkpoint "/实际路径/旧训练run目录/training/model_500.pt"
```

旧训练没有记录的指标不会补造；旧原始数据保留，新增评估使用本次代码，并分别保存来源快照。只有模型而没有旧run资料时，可使用play入口进行回放记录。

## 失败、中断与补收集

Python异常、子进程失败或正常Ctrl+C后仍尝试收集已有资料，状态明确为失败/中断。评估失败不会删除训练模型，也不会伪装成训练失败的原始原因。新进程不要同时操作同一个会话。

断电、kill -9或拷贝时空间不足可能来不及自动汇总。保留SESSION目录，进程停止后运行：

```bash
bash station.sh collect --session "/实际路径/offline_sessions/route_06/本次时间"
```

其他路线已有的训练目录也可单独收集：

```bash
bash station.sh collect --route 01 --run "/实际路径/R1训练目录"
```

通用收集保留这份目录的现有内容，缺少的配置/轨迹会记录为缺失；不会替其他路线生成尚未实现的评估指标。运行中的进程不收集，避免得到一半旧一半新的文件。

## 拷回后的核验

在返回的审查文件夹中执行：

```bash
python3 verify_review.py .
```

看到`PASSED`证明拷贝内容与清单一致。它不代表策略性能达标；正式审查仍看姿态、速度误差、支撑、航向、碰撞、失败工况和力矩轨迹。训练中只记录统计和已保存模型，不全量记录1024环境的每个物理步；1kHz文件来自训练后的固定模型评估，不用于声称已测得全训练过程的冲击峰值或实机连续额定力矩。

## 后续其他路线

通用工作站入口是station.sh，收集器不绑定R6。当前自动训练→评估→录像适配器在`offline/routes.json`中登记了06。查看：

```bash
bash station.sh routes
```

以后补充其他路线时，将该路线自己的训练/评估代码、离线资产和适配器一起放进新累计补丁。适配器入口需要接受mode、--route、--run-dir、--checkpoint等参数，并写request/run/result/runtime_check及源码/资产清单；评估与录像参数可在适配器中声明。各路线的run与返回目录分开，未登记路线不会误用R6代码或模型。现有R1 Stage1与本机当前R1 V3不是同一实验，不能只修改路线编号就当作已完成迁移。
