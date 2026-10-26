# R4a PPO 接口候选（未接入机器人训练）

这是独立的 CPU 可检验驱动。它调用本机 `rsl_rl` 的真实 `PPO` 更新、存储和 Adam，并沿用 `sustain_v7/continuing_ppo.py` 的重置前 Critic 末值补偿。它不绑定现有失败学生，也没有启动 Isaac Sim、机器人 PPO 或正式训练。

## 输入合同

- `actor: nn.Module` 接收 `[N, D]`，返回 `[N, 16]`；`D` 由 `DriverConfig.actor_obs_dim` 指定，驱动不写死维度。早期309维草案为原始305＋捕获记忆2＋本体FK间隙/滚动速度2；后续结构化候选可能为325维，再增加触发时16维起身动作。新增维度应以最终冻结合同和实测实现为准，不能拿零填充当正式 Actor 输入。
- `feature_builder(raw, done)` 仅用可部署信号；每次控制步返回 `[N, D]`。构造驱动时所有槽按 reset 调用一次，以后每次 `env.step` 后调用一次，且只有 `done` 槽清记忆。不得在一次策略输入中重复推进计数。
- `env` 使用 Isaac Lab RSL wrapper 形式：`get_observations()` 返回原 305 维 policy/88 维 critic，`step()` 返回新的原 policy、reward、done、infos。`infos` 须带 `observations.critic`、`time_outs`、`physical_terminated`、`terminal_observation_valid` 和自动 reset **之前** 的 `terminal_critic_observation`。
- 经验观测归一化保持关闭。本候选的 timeout 价值函数直接读原始末状态 88 维 Critic；若未来使用归一化，需使末状态和普通 Critic 经过同一归一化器并单独重验。
- `env.device` 与驱动的 `device` 必须一致，例如均为 `cuda:0`；原始 Actor、普通 Critic、reward/done 与重置前末状态 Critic 必须处于同一设备。驱动把传入的 Actor 注册进 RSL 策略，随后才创建 Adam，因此 Actor 参数进入优化器。

`ContinuingDriver.collect_rollout()` 使用真实 RSL 存储；`update_from_rollout()` 使用 GAE 和 PPO optimizer。iteration 边界只清 rollout 存储，不调用环境 reset，也不重建记忆。`stop_on_observation=False` 时捕获后仍持续采样；技术时限触发的真实 `truncated` 仍会自动 reset 对应槽。

## CPU 合成接口检查

```bash
cd /home/user/RL_lab/rl_training
/home/user/miniconda3/envs/lab/bin/python -m unittest discover \
  -s model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/ppo_design/tests -v
```

下面是**合成 env** 的接口调用示例，`SyntheticVecEnv` 和 `CausalFeatures` 只为验证维度、时钟和 reset 顺序，并非机器人动力学或真实 FK：

```python
from pathlib import Path
import sys
from torch import nn

base = Path('/home/user/RL_lab/rl_training/model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5/ppo_design')
sys.path.insert(0, str(base))
sys.path.insert(0, str(base / 'tests'))
from continuing_driver import ContinuingDriver, DriverConfig
from test_continuing_driver import SyntheticVecEnv, CausalFeatures

env = SyntheticVecEnv(2)
driver = ContinuingDriver(
    env, nn.Linear(309, 16), CausalFeatures(2),
    DriverConfig(actor_obs_dim=309, num_steps_per_env=2, initial_std=0.05),
    observation_contract='cpu_synthetic_309_v1',
    policy_contract='linear_309_v1',
)
print(driver.run_iteration())  # 一次真实 RSL PPO 更新，数据仍为合成数据
driver.save_checkpoint('/tmp/r4a_cpu_ppo_000001.pt')
```

续训需要新建同合同的 `env/actor/builder/driver`，再在首次采样前调用 `load_checkpoint(path)`。它恢复 Actor、Critic、log-std、Adam、当前学习率、完成次数、控制步/样本计数及 Python/NumPy/torch RNG。**物理世界状态不保存在该 checkpoint 中**，跨进程加载从新四轮回合开始；不得把前后两次进程的站立时间拼接。输出路径须不存在，禁止覆盖已归档模型。

检查点中的模型和 Adam 张量以 CPU 副本保存，加载时先在 CPU 读取，再由 PyTorch 映射到当前策略参数设备；实机 GPU 保存/加载仍需真实短验。驱动在采样前切换策略到 `train()`。若未来 Actor 的冻结起身前缀包含 BatchNorm、Dropout 等具有训练/评估模式差异的层，Actor 自身须明确保持该前缀的 `eval()` 语义并单独验证；单靠 `requires_grad=False` 不会冻结这些层的运行状态。

## 接入前的短验门

先冻结新 Actor 的具体 309 维语义和来源，完成教师全关的两槽×seed43/44/45捕获后额外10秒自主通过，再接一份独立机器人入口。机器人 GPU 短验至少检查 2/16/64 槽的一轮更新与有限性、动作裁剪/探索振动、技术 timeout 的末状态价值只续接一次、物理失败不续接、iteration 跨越时不 reset、保存后新进程加载模型/Adam/计数/RNG并继续更新。自动 reset 前后的 Critic 值应分开记录；进程续训只证明优化状态恢复，不证明物理站立连续。以上通过后才考虑正式启动命令；正式入口只保存 checkpoint，不自动冻结评估，待用户通知后再审查。
