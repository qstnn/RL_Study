# R4a短段教师接管与释放

**本目录已归档。** 四条实际采集完成后，报告文件名匹配缺陷使原 session 标为失败；原始来源与轨迹保留。当前使用 [student_segments_v2_1](../student_segments_v2_1/README.md)，修复及独立复核见 [归档说明](ARCHIVE.md)。以下仅保留首次采集时的方案，勿使用本目录命令启动新实验。

独立于已冻结的`sustain_v7`与`student_v1`。使用学生02作为明确起点，固定原305维观测、MLP、教师参数、物理/奖励、无观测噪声和随机化。目标是补充学生实际偏离状态中的反馈动作，再关闭教师判断自主长稳与PPO准入。

## 采集时序

时刻使用环境已有的起身任务时钟；只在采集器选择动作来源，不作为Actor的新输入。教师从reset开始读取实际状态，交接不重置机器人、历史观测、物理时钟、捕获触发器或高频滤波器。区间左闭右开。

| 组 | 短段教师 | 学生释放 | 后续教师恢复与验收 |
|---|---|---|---|
| early | 2.5–2.9、3.2–3.6秒 | 2.9–3.2、3.6–3.8秒 | 3.8秒后至捕获后额外10秒 |
| late | 2.8–3.2、3.4–3.8秒 | 3.2–3.4、3.8–4.0秒 | 4.0秒后至捕获后额外10秒 |
| guarded，预留 | 2.5–3.2、3.3–3.8秒 | 3.2–3.3、3.8–3.9秒 | 3.9秒后至捕获后额外10秒 |

默认先early/late各两槽。预留组不自动运行。所有混合控制轨迹只算采集/恢复证据，不能计为自主成功。失败或仅单槽通过的组不加入示范。

## 数据与有限学习

保留冻结的名义3715帧完整成功轨迹。此前学生01固定接管成功数据仅保留教师实际执行1325帧，排除其320帧学生前缀。新采集也仅选择完整双槽通过组里的教师执行帧，学生前缀和释放段保留在原始轨迹中作覆盖证据，训练权重为0。每帧保存原始控制索引、来源类别、回合、阶段和teacher_executed；不把未执行影子教师动作当成功标签。

BC继续采用四阶段各128帧、batch512、Adam 3e−4余弦降至3e−5、6000次优化器更新，显式学生02初始化、Adam重建。不同时增加后轮权重、曲率项或改变网络，有限预算后关闭教师，从四轮初态完成两槽×seed43/44/45纯学生回放。

验收使用原1kHz门：捕获2秒后额外10秒全程不中断，允许≤0.12m/s慢漂移、保持真实高频检测。只有六例自主全部通过，再完成真实PPO更新/保存/续训短验，才能给正式PPO命令。最终仍以自主60秒重复和300秒耐久为目标。当前入口不含正式PPO，正式训练不自动每100冻结或末尾评估，R4b等待。

## 可复现入口

在本目录运行，所有输出目录必须是新目录：

```bash
bash segments.sh check
bash segments.sh collect --program early late --output runs/new_collection
bash segments.sh prepare --collect runs/new_collection --output datasets/new_dataset
bash segments.sh train --dataset datasets/new_dataset --output runs/new_bc --device cuda:0
bash segments.sh evaluate --actor runs/new_bc/actor.pt --output runs/new_autonomous
bash segments.sh verify runs/new_autonomous
```

这些是有限采集/BC/验收命令，不是正式PPO启动命令。通过记录均需来源、动作、物理和数据角色复核，退出码或低MSE不能代替自主成功。
