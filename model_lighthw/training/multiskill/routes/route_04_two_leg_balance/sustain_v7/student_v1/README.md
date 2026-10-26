# R4a持续站立学生首版

**已运行结果：** 两轮各6000次BC更新及自主六例均完成，自主均0/6、保持0；一次实际状态教师接管2/2额外10秒通过。当前不满足PPO准入，不原样追加第三轮。[完整证据与图表](VALIDATION.md) · [下一步方向](../NEXT_DIRECTION.md)。

本目录独立于已冻结的v7教师诊断入口。目标是从固定四轮初态自主起身、捕获，之后在站立指令有效期间持续保持。原机器人/动作/限矩、无噪声随机化、≤0.12m/s慢漂移和1kHz实际抖动门保持。

数据只使用新v7 `policy_observation` 和对应实际执行动作，前0.2秒以环境实际零动作为标签；旧Actor前缀和教师执行后缀分别标记。不使用旧7112帧、不使用影子教师未执行的前缀、不加入高冲击返回。30秒主轨迹只有固定两槽，短/长公共前缀不重复扩充为独立证据。

首版Actor为全新初始化的305→256→256→128→16 ELU网络。BC每批512帧，入场、3.2秒前起身、捕获、捕获后保持各128帧；6000次Adam更新，学习率3e−4余弦降至3e−5，无后轮额外加权或曲率辅助项。训练误差仅表示当前示范的拟合；没有独立泛化验证集。

自主入口整段只执行新学生，不构造教师，不使用旧观测适配器；两槽×seed43/44/45完整记录50Hz观测/动作和1kHz物理/状态。独立复算支撑、姿态、速度、高频、捕获和首次中断，并用CPU复算Actor/实际执行动作。若指定`--switch`，运行明确标记为接管诊断，不能计为自主通过。

初始准入仍要求六例捕获后额外10秒不中断。通过后还需真实PPO更新/保存/续训短验；首版没有PPO入口。最终方向为自主60秒重复和300秒耐久，教师30秒不代替学生证据。

## 入口

在本目录执行，输出目录必须不存在，防止覆盖：

```bash
bash student.sh prepare --input ../runs/2026-10-22_sustain_30s_blend800 --output datasets/new_dataset
bash student.sh train --dataset datasets/new_dataset --output runs/new_bc --steps 6000 --device cuda:0
bash student.sh evaluate --actor runs/new_bc/actor.pt --output runs/new_autonomous
bash student.sh verify runs/new_autonomous
```

这组命令用于有限学生工作流，不是正式PPO启动命令。模型、数据、源文件和资产均有来源合同，源码改变需要独立版本，不能回写历史结果。
