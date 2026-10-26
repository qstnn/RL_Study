# Route 05A：J1 v2 起跳、落地缓冲与末段停稳

任务 `MultiSkill-LightHW-R5A-JumpLanding-v1`，契约 `lighthw_r5a_jump_landing_v2`。上一轮 v1 的 1000 更新已审查：真实起跳存在，但 50 Hz 判定会漏掉短起跳窗口，历史成功标记也不能代表回合末停稳。当前修复并继续 J1，不进入完整后空翻。

在已经安装本次补丁的原 R5A 工程根目录执行：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh prepare
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh refine
```

`refine` 按 SHA 寻找已审查 v1/model999，先显式迁移并冻结复评，再默认 512 环境进行最多 300 更新。保留 Actor，重建 Critic 和 Adam，探索标准差 0.05、初始学习率 1e-4；v2 从第 0 更新计数，原先 1000 更新保存在来源记录中。冻结复评若未完整起跳落地，会停止在新训练之前并回收诊断。

**固定物理与控制。** 零负载、整机名义质量 11.6681 kg、自碰撞开启，腿 48 Nm/12.57 rad/s、轮 24 Nm/37.7 rad/s；沿用原参考动作及残差上限，不注入根速度或助跳外力。物理 1 kHz、策略 50 Hz、回合 4 秒，Actor/Critic/Action 维度 285/86/16。视角保持 World 自由移动。

**修复与目标。** 起跳、腾空、触地、碰撞与保持状态只按每个已完成的物理步推进一次，与是否录像无关；处理策略边界重复调用和并行环境的独立 reset。起跳门槛保持净空 >15 mm、四轮接触力绝对值 <2 N、COM 向上速度 ≥0.25 m/s。质心升高 ≥5 cm、有效腾空 ≥0.10 秒；回合结束时须连续稳定 ≥1 秒。稳定条件包含四轮支撑、倾角 <15°、水平速度 <0.05 m/s、垂直速度 <0.15 m/s、漂移 <5 cm，以及基座高度 0.416±0.05 m。success 为当前保持，ever_success 仅表示曾经达到条件，成功奖励在正常回合末结算。

落地保持奖励同时关注速度和位置；触地后且回合时刻 ≥1.5 s 时，加强对轮部运动和水平速度的约束。完整物理接触峰值仍参与冲击惩罚；3 倍机重为软奖励阈值，不是硬件承载认证。

**离线回收。** 默认训练前后都做 seed43/44/45 各 4 回合冻结评估，最后生成 seed42 的 3 回合 MP4。原 v1 模型、迁移模型、训练源快照、所有新模型/Adam、1 kHz 轨迹和在线状态、图表、视频及错误记录统一回收。baseline 与 final 分开统计，当前 summary 首页成功率取最终评估。按终端 `[USB REVIEW]` 复制整个目录带回。

详细补丁安装、模型路径、仅冻结评估、回放和中断回收命令见根目录 `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/docs/R5A_START_HERE.md`。代码/短验状态见 `offline/R5A_VALIDATION.json`；正式 v2 训练和行为验收需要后续返回数据，当前未验收完整后空翻或硬件。
