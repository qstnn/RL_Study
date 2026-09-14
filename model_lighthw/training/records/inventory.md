# LightHW 文件清单与清理记录

## 当前有效链

| 类别 | 唯一路径 | 说明 |
|---|---|---|
| 机器人资产 | `model_lighthw/lighthw_urdf/urdf/lighthw.urdf` | 17 links、16 joints、总质量约 11.668 kg；所有 mesh 可解析，惯量正定 |
| 资产绑定 | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_lighthw/asset_cfg.py` | `LIGHTHW_ROUTE_CFG` 指向上面的 route-owned URDF |
| 基础环境 | `.../deeprobotics_lighthw/rough_env_cfg.py`、`flat_env_cfg.py` | 统一调用 `m20_reward_profile.py` |
| 奖励 profile | `.../deeprobotics_lighthw/m20_reward_profile.py` | M20 权重和奖励角色的唯一来源，按 hip/thigh/calf 映射 LightHW |
| 当前训练路线 | `model_lighthw/training/routes/route_04_balanced_highspeed/` | `Rough-Deeprobotics-LightHW-BalancedHighSpeed-v0` |
| 历史对照 | `model_lighthw/training/routes/route_00_baseline/` | 仅用于回放/比较，不作为新训练入口 |

## 已删除

- `route_01_hierarchical`、`route_02_static_speed3p5`、`route_03_stage_a_static`：训练结果未达标或阶段 A 未改善，且奖励/命令参数互相覆盖。
- 对应的 hierarchical/static/stage-A 环境配置、速度资产配置、PPO 配置和 Gym 注册项。

## 明确弃用的资产

旧的圆柱轮碰撞 URDF 已删除。它除圆柱轮碰撞外还修改了 hip 质量、关节轴和限位，质量约
11.568 kg，与当前动作和对称奖励不兼容；不要用于训练或回放。标准 URDF 的
`model_lighthw/lighthw_urdf` 现在是唯一训练资产。

## 奖励迁移边界

M20 的主项权重（速度跟踪 2.0/1.0、姿态/高度、关节力矩/加速度/功率、限位、
静止、接触、动作平滑和 upward）已迁移。力矩、功率和加速度项按 M20/LightHW
执行器上限归一化；高度目标改为 LightHW 默认姿态约 0.416 m，接触峰值阈值改为
40 N（M20 的 100 N 对 11.67 kg 机体过高）。镜像项使用 LightHW 默认角和左右
符号映射，避免把正常的前后腿姿态当作误差。

静态检查不能替代 Isaac Sim/PhysX 回放；滚动方向、接触法向和行为仍需 GPU playback 验证。
轮加速度保留 M20 的 `-1e-7` 并按轮速上限归一化；由于 LightHW 单轮质量约为 M20 的
2.7 倍，新增低权重轮力矩/功率抑制，防止高速轮暴冲。
