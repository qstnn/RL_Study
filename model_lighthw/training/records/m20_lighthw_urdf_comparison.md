# M20 与 LightHW URDF 对比及奖励迁移依据

| 项目 | M20 | LightHW | 奖励影响 |
|---|---:|---:|---|
| links / joints | 17 / 16 | 17 / 16 | 网络动作维度可保持一致，但关节名称和分组不同 |
| 总质量 | 15.000 kg | 11.668 kg | LightHW 底座占比更高，不能直接复制 M20 的高度和接触假设 |
| 底座质量 | 5.4 kg | 6.376 kg | 姿态/角速度项仍需保留，底座惯性主导稳定性 |
| 底座惯量 (Ixx/Iyy/Izz) | 0.057/0.203/0.242 kg·m² | 0.0124/0.0493/0.0561 kg·m² | LightHW 约低 4 倍，角速度扰动更快；若 GPU 回放出现姿态抖动，将 `ang_vel_xy_l2` 从 -0.02 提至 -0.04 |
| 单轮质量 | 0.15 kg | 0.405 kg | LightHW 轮子惯量明显更大，增加低权重轮力矩/功率抑制 |
| 腿部关节轴 | hipx -X；hipy/knee -Y | hip +X；thigh/calf +Y | 镜像奖励按默认角和符号映射，不能直接使用 M20 关节名 |
| 轮关节轴 | -Y | +Y | M20 checkpoint 的轮动作方向迁移时需要显式取反；新 LightHW 训练保持当前动作接口 |
| 腿部限位 | hipx 约 ±0.5，hipy/knee 约 ±2.8 | hip ±0.4，thigh 非对称约 [-2.70,1.56]，calf ±2.5 | 位置限位项保留 M20 权重，但关节选择改为 LightHW leg joints |
| 碰撞几何 | Isaac 从 visual 生成碰撞体 | URDF 自带 collision mesh，禁止从 visual 生成 | 机身/腿误触取决于显式 mesh；`undesired_contacts` 必须匹配 `Base_link`、hip/thigh/calf，而不是沿用 M20 link 名 |
| 运行时腿执行器上限 | 76.4 Nm / 22.4 rad/s | 48 Nm / 12.57 rad/s | 力矩、功率、加速度物理量按上限比归一化 |
| 运行时轮执行器上限 | 21.6 Nm / 79.3 rad/s | 24 Nm / 37.7 rad/s | 保留轮加速度项，并新增很小的轮力矩/功率项 |
| 自然机身高度 | 初始约 0.52 m | 默认姿态约 0.416 m | 高度目标改为 0.416 m，避免持续惩罚正常姿态 |
| 轮静态载荷 | 约 36.8 N/轮 | 约 28.6 N/轮 | 接触峰值阈值设为 40 N；M20 的 100 N 对 LightHW 过高 |
| 底座转动惯量 | Ixx/Iyy/Izz≈0.057/0.203/0.242 | ≈0.012/0.049/0.056 kg·m² | `ang_vel_xy_l2` 从 -0.02 提到 -0.04，抑制更快的姿态角速度 |

## 已落实的 LightHW 针对性项

- `base_height_l2` 使用 LightHW 逐环境射线回退实现，目标高度为 `0.416 m`；
  这样某个环境的无效射线不会让整批环境失去高度约束。
- `action_rate_l2` 使用无隐藏 curriculum 的 LightHW 实现，适配腿部位置和轮速
  混合动作向量。
- 腿部力矩、功率和加速度项按运行时执行器上限归一化；轮子保留加速度项，并以
  更小权重抑制重轮的力矩/功率突发。

## 数据边界

M20 URDF XML 中的 effort limit 为 260/160 Nm，但实际 Isaac 配置使用
76.4/21.6 Nm；奖励归一化以实际 `ArticulationCfg` 为准。M20 同目录 CSV
记录的质量也与 URDF XML 不一致，因此没有用于迁移比例。

LightHW 当前 canonical URDF 为 `model_lighthw/lighthw_urdf/urdf/lighthw.urdf`。
其 17 个 link、16 个 joint、mesh 引用和惯量正定性已通过静态检查；接触法向、
滚动方向和 PhysX 导入仍需 GPU playback 验证。
