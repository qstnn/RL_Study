# R1 V6：统一 Actor 的重拖强化学习实验

**2026-10-20后续方向：** 转入[R1 V7物理与教师研究](../research_v7/README.md)，用户启动前序研究；新研究μ=3.0。本页V6的μ=4及训练命令保留为历史，不代表新的推荐启动方案。

**2026-10-20训练结果更新：** 首轮在600更新触发旧能力保护停止，重拖仍约2cm，最佳保留Actor仍为初始化。下列命令保留用于复现实验，当前不建议原配置直接续训。[600更新完整审查](results/2026-10-20_600_update_review/README.md)。

本版同时训练空载高速、机身背载和真实被动拖车。部署仍为一个300维输入、16维输出的MLP；64种许可工况从训练开始同时存在。正式训练由用户在终端启动。

## 为什么调整

旧V5的两次200更新实验中，40kg/+25°移动仍停留在约2cm，且部分静止能力退化。不能靠增加训练轮数认定问题会自然解决。本版针对运动探索太小、拖载姿态奖励过紧、机器人移动与拖车进度不一致进行独立实验。完整证据与假设见[历史审查](results/review/README.md)。

| 项目 | V6设置 |
|---|---|
| 真实环境 | 40kg被动四轮拖车、1ms双向牵引力；没有机身推进力或教师动作 |
| 工况混合 | 空载/背载/拖载25%/25%/50%；保留-25°不拖车、载荷与拖车互斥 |
| 探索 | 拖载行进时腿/轮raw std最高.04/.08；零指令与其他任务.005/.01 |
| 拖行奖励 | 机器人与拖车沿线速度较小者计有符号进度；后退扣分，拖车不走不得前进分 |
| 调姿空间 | 拖载机身高度.36–.44m自由带，软轮载目标为名义单轮载荷25% |
| 失败与里程碑 | 拖载真实非超时失败合计-10；1m单次+10，仍必须完成停车验收 |
| 旧能力保护 | 成功缓存软约束、整rollout KL≤.01、每100更新冻结16边界工况检查 |
| 优化 | 20轮Critic预热，Actor/Critic学习率3e-5/1e-4，64步rollout |

力矩上限、摩擦、车体/拖车质量、牵引几何沿用V5：腿/轮96/48Nm，μ4，滚阻.02，初始牵引净高.07m。这里的μ4是当前仿真假设；本实验结论不直接代表实机重拖能力。

已通过37项CPU测试和两次GPU集成短验（合计6更新、其中3轮更新Actor）；冻结初始化复测15/16通过。40kg/+25°移动仍是待学习项，基线进度2.13cm。正式PPO更新0，详见[验证记录](results/review/verification.json)。

## 启动正式训练

```bash
cd /home/user/RL_lab/rl_training
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/unified_v6/scripts/train.py   --headless --num_envs 1024 --max_iterations 1000 --eval_interval 100
```

默认从V5已选Actor初始化，**不从本轮短验模型续训**。新目录自动建在`unified_v6/runs/<时间>/`，不复用之前已中断的目录。先冻结评估初始化Actor，再做20轮Critic预热，之后更新Actor。每100更新暂停训练做同一套16工况复评；同一受保护工况连续两次退化则提前停止，不一定跑满1000更新。

按Ctrl+C会回收本轮worker及冻结评估子进程。只改本路线设置；R5A、R6和原V5文件不参与本次改动。

## 怎样判断训练结果

- `training_result.json`：实际Actor更新数、是否因退化停止、当前/最佳checkpoint。
- `periodic_evaluations/update_*/evaluation.json`与各case轨迹：40kg/+25°是否真正完成1m、减速、2秒过渡、5秒稳定保持，以及旧工况是否保持。
- `training/condition_metrics.csv`和`optimizer_metrics.csv`：分工况进度/终止、Actor梯度、KL、接受步幅及Critic解释方差。
- `quality/best_eligible.pt`只表示保持了本轮受保护能力；若仅15/16通过，不能叫完整R1成功。
- `exported/actor.pt`只在当前检查点通过能力保持检查时导出。仍须阅读同轮验收，不能根据文件存在判断重拖学成。之后还需三种扰动、多次初始状态与全64工况验证，特别是-20°/40kg下坡制动。

完整重拖行为尚未训练验收。当前交付的是经代码与GPU集成验证的RL训练方案，不能保证一次训练收敛。
