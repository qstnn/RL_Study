# R5A 教师示范 → 模仿初始化 → PPO（USB 离线版）

本版教师已在本机 CPU PhysX 中从固定站姿完成后空翻与站稳。正式 PPO 在另一台电脑运行，保持原机器人、动力学、奖励、420/104/16 维接口和成功判定。原 v6 和其他路线不改。

## 已验证到哪里

- 交付教师：首次腾空 **355.225°**，离地后 COM 上升 **0.1939 m**，滞空 **0.458 s**，首次触地倾角 **2.466°**，轴偏差 **18.821°**。
- 6 秒完整回合终末连续站稳 **4.527 s**，无原协议物理失败；末 1 秒四轮支撑 100%，最大落地漂移约 **3.84 cm**。
- 全身角动量矢量漂移 **0.01743**，原筛查上限 **0.21121**；实际最大电机力矩 **48 Nm**。动作限定到当前策略能达到的 `±tanh(3)`。
- 固定初态三次重复均通过，另在交付模块中重放一致。不同 seed 未引入随机初态，不能当作鲁棒性验证；本机无可用 CUDA，本版未在目标 GPU 上验收。
- 本机 3000 步监督模仿后的学生自主回放约 **353.28°**，轴偏差约 20.43°，落地失败，尚未学会完整后空翻。监督误差小不能代替闭环通过。
- 教师仍有很高的刚性触地冲击，峰值约 79 倍体重；本版是仿真学习起点，不是实机动作交付。

## 在另一台电脑安装

将 ZIP 和 `.sha256` 拷入原 R5A 项目根目录，即能看到 `r6.sh` 的目录。在该目录运行：

```bash
sha256sum -c lighthw-r5a-teacher-guided-v1-patch-20261022.zip.sha256
python3 -m zipfile -e lighthw-r5a-teacher-guided-v1-patch-20261022.zip .
python3 lighthw-r5a-teacher-guided-v1-patch-20261022/install.py . --check
python3 lighthw-r5a-teacher-guided-v1-patch-20261022/install.py .
bash r5a_guided_v1.sh check
bash r5a_guided_v1.sh train
```

无需 Git、联网或重新下载资源。安装器先核验现有 v5 基础文件；包内同时包含未修改的 v6 依赖，因此不要求事先安装 v6。已有文件内容不同会在写入前拒绝，不会覆盖旧训练结果或修改模型。目标机器仍需原来能运行 R5A 的 Isaac Sim / Python / 驱动环境。

若 Python 或 Isaac Sim 路径不同，沿用原配置，例如：

```bash
R5A_PYTHON=/绝对路径/lab/bin/python R5A_ISAAC_SIM=/绝对路径/isaac-sim bash r5a_guided_v1.sh train
```

## `train` 自动做什么

1. 校验教师文件，在目标设备上重放两次完整教师动作。任一次未通过则停止，导出诊断 ZIP，**不会进入 PPO**。
2. 从实际执行动作收集示范，按起跳、空中、落地恢复、站稳四阶段均衡采样，执行 3000 步监督模仿。学生输入仍为原 420 维。
3. 回放一次学生，记录学习起点。学生不通过会明确记录；这正是后续 PPO 需要优化的部分，不会冒充已学会。
4. 连续进行默认 **3000 次更新尝试**：前 20 次只预热 Critic；随后 PPO 更新与衰减示范辅助损失共同接受 KL/有限值检查。初始学习率 `1e-5`，地面探索标准差下限为腿 `0.03`、轮 `0.02`。保留 PPO 的动作探索，没有新增环境/观测随机化。
5. 中途每 200 次只保存，不分段评估；结束后统一比较最后最多 3 个保存模型，保留一个选中模型和必要证据。

PPO 的 rollout 来自当前学生在线采样。示范仅用于监督初始化和独立辅助损失，不写入 PPO on-policy 缓冲区；正式执行动作由学生网络输出，没有教师强制接管。

后续在同一源码和设备上再次执行 `train` 会复用已准备的初始化，开始新训练。继续某个检查点必须显式指定：

```bash
bash r5a_guided_v1.sh train --checkpoint /完整路径/final.pt --updates 1000
```

续训时保留该训练目录的 `input/demonstrations.npz`；如果使用返回的 `selected.pt`，保留 USB 包内的 `training/input/demonstrations.npz`。检查点和示范均校验来源。

## 查看进度、回放与回收

终端会输出 `[GUIDED BC]`、`[GUIDED PREPARED]`、`[GUIDED CONTINUOUS]`。只有教师复核和模仿准备结束后才开始计算 3000 次 PPO 更新，20 次 Critic 预热包含在其中。日志与 TensorBoard 文件位于打印的 `offline_sessions/route_05a/guided_v1_*/training/`。

```bash
bash r5a_guided_v1.sh play --checkpoint /完整路径/selected.pt
```

回放沿用本机外观和自由视角；不增加自动视角跟随。

结束后只拷回终端打印的 **`[USB REVIEW ZIP]`** 文件及同名 `.sha256`。包含选中模型、训练曲线、全程更新记录、回合汇总、示范/模仿起点、来源、选中模型的录像和 1 kHz 轨迹；不带回全部搜索候选和其他检查点。按 Ctrl+C 中断也尝试导出诊断 ZIP；没有完成末尾评估时不会把旧模型标成选中版本。

仅准备但暂不开始 PPO：`bash r5a_guided_v1.sh prepare`。本机审查与交付验证见补丁根目录 `VALIDATION.md`。
