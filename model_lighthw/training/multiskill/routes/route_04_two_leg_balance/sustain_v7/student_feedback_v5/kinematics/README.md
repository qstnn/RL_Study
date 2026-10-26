# R4a 支撑方向纯本体几何反馈（离线验证）

2026-10-23；本目录只实现/审查状态估计，不改原 `sustain_v7` 环境、奖励、教师、学生、检查点或封存轨迹，也没有新 GPU 仿真、BC 或 PPO。规范 URDF SHA-256 为 `5abb7c07c0ad401f7940018fee739ecc2fb4e0d3e8fc83274e75d1886db19fbb`。

## 可部署输入与公式

`BodyEstimator.estimate(raw305)` 和可导的 `TorchBodyEstimator(raw305)` 只读 305 维政策历史的**最后 61 维当前帧**：IMU 投影重力、机身角速度、12 个腿关节角、16 个关节速度。原站姿偏置和速度缩放按实际观测合同还原；后轮顺序 RR、RL，轮关节顺序 `FR, FL, RR, RL`。不读仿真 COM、根线速度、接触、阶段特权或教师动作。Torch 版可直接作为固定特征层与策略联合求导；输入需是原305维（307维记忆版本先取 `[..., :305]`），使用 `.to(device, dtype=...)` 放置常量。

从 URDF 关节轴/原点、各连杆质量和惯性质心做 FK 得到机身坐标的 `r_COM` 与两后轮轮心。令 `u` 为重力反向的单位向量、`l=r_RL−r_RR`、`f=normalize(l×u)`，则教师使用的有符号前向支撑偏差为 `gap=(r_COM−(r_RR+r_RL)/2)·f`。该公式不需要根位置或航向。实际 1 kHz 教师量只在离线计分中使用。

速度在**两后轮都无滑纯滚动**假设下由每个后轮分别推导再平均。轮绝对角速度为 `ω_wheel=ω_IMU+Σ_ancestor axis_j*qdot_j`，包括髋、大腿、小腿及轮本身；接触径向上方向 `u_rad=normalize(u−(u·wheel_axis)wheel_axis)`。令 `J` 为 URDF FK 雅可比，则每轮给出的 COM 前向速度为

`[R*(ω_wheel×u_rad) + ω_IMU×(r_COM−r_wheel) + (J_COM−J_wheel)qdot]·f`，`R=0.0875 m`。早期只用轮自身转速会漏掉机身/腿的绝对转动，代码保留其输出 `velocity_spin_only_m_s` 作审计基线；另给出垂直半径近似 `velocity_full_vertical_m_s`。默认 `velocity_m_s` 使用完整轮角速度和径向接触方向。URDF STL 轮外半径约 0.0875 m；实际地面接触、胎面变形、打滑和轮离地无法由这些输入直接识别。

## 对齐与结果

只比较有效政策帧 `k>=1`，输入与真实物理轨迹索引 `20k−1` 对齐；初始 `k=0` 无前一段物理索引。三份轨迹的重力逐值相同，腿角最大差 `5.36e−8 rad`、全部关节速度最大差 `9.54e−7 rad/s`。URDF 后轮轴为 `+Y`，轮编码器正号在两后轮承载且平均轮速≥0.5rad/s 的帧上，比反号有约 10 倍更小的速度误差。完整数值、逐分段和来源哈希见 [离线 JSON](results/2026-10-23_offline.json)。

| 来源与过渡时窗 3.2–5.5 s | 有效帧 | gap MAE / P95 | 完整无滑速度 MAE / P95 | 仅轮自身转速 MAE | 速度误差>`0.12 m/s` | `|真速度|≥0.10` 时反号 |
|---|---:|---:|---:|---:|---:|---:|
| 名义教师额外30秒成功 | 230 | `1.08e−7 / 2.69e−7 m` | `0.0205 / 0.131 m/s` | `0.0413 m/s` | 13/230 | 0/129 |
| 最新 memory-exposed 自主失败 | 201 | `1.08e−7 / 2.47e−7 m` | `0.0280 / 0.155 m/s` | `0.0717 m/s` | 13/201 | 0/148 |
| 最新 memory-zero 自主失败 | 190 | `1.29e−7 / 3.09e−7 m` | `0.0287 / 0.139 m/s` | `0.0659 m/s` | 24/190 | 0/143 |

名义教师 5.5 秒后长稳段 3165 帧，速度 MAE 约 `3e−5 m/s`、P95 约 `7e−5 m/s`；失稳时两后轮并非同时承载的样本中，无滑速度 MAE 约 `0.20–0.22 m/s`。因此 **gap 在已测原机器人轨迹上足够精确，可作为几何反馈候选；速度在捕获/失败状态仅是条件近似**，不能据此把 0.12 m/s 严格速度门改成估计门，也不能把教师 COM 速度偷偷作为 Actor 运行输入。两轮同时滑动时，IMU 重力/角速度与关节编码器不足以确定共同平移速度；此可观测性边界不能靠调整符号或更多同分布回归消除。

Python/NumPy 与 Torch 对同一有效帧的输出最大差低于 `4e−16`，Torch 对本体输入的梯度有限。解析 COM 和轮心雅可比已逐关节用中心差分核对。结果是固定两槽、名义成功与最新两组失败的离线分析，未做独立初态/摩擦/滑移变化验证，更不是学生闭环或 PPO 准入。

复算命令（无需 Isaac Sim/GPU）：

```bash
cd /home/user/RL_lab
/home/user/miniconda3/envs/lab/bin/python rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/kinematics/test_estimator.py
/home/user/miniconda3/envs/lab/bin/python rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/kinematics/evaluate.py
```
