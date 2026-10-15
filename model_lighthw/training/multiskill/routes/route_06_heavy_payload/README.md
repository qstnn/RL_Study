# R6 / Route06 重载静态持载训练

**任务：`MultiSkill-LightHW-HeavyPayload-v0`。代码与 CPU 集成检查已完成，GPU 实例化及行为验收待验证。**
目标对应1-11：静止承载分级增加到65kg。当前实现固定载荷的阶段训练，首轮空载；不自动跳到65kg。
本机正在训练R1，本路线在另一台电脑运行。先按[同步与环境说明](../../README.md)获取
`lighthw-multiskill-heavy-payload-r6`。支持 Git 拉取，也支持网页下载 ZIP 后运行
`python3 model_lighthw/training/multiskill/prepare_zip.py` 补齐网格；再在项目根目录执行以下命令。

## 直接启动

```bash
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/run.py check --route 06 --payload-kg 0
python model_lighthw/training/multiskill/run.py smoke --route 06 --payload-kg 0 --headless --num_envs 32
```

训练/保存链路短验，预算2次更新，生成的模型只作实现检查：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 0 --headless --num_envs 32 --max_iterations 2
```

正式空载持载基线，从零启动，不读取R1模型：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 0 --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

每100次更新保存一次，预算结束退出；不会自动回放、升阶段或延长预算。显存较小时可降低`--num_envs`，
同时会减少每次更新的样本量（默认每环境24步），比较实验时需记录。

## 固定载荷模型与控制契约

| 项目 | 设置 |
|---|---|
| Actor / Critic / Action | 58 / 237 / 16 |
| Actor | R1的57维可部署基础观测，加最后1维已知装载质量/65 |
| Critic | R1的236维特权观测，加同一载荷通道 |
| 动作 | 12腿位置 + 4轮速度；按名称保持FR/FL/RR/RL顺序 |
| 姿态/控制 | 原始0.416m站高；腿PD 80/3；轮阻尼1；1kHz物理、50Hz策略 |
| 命令/地形 | 三轴恒零，平地，20s episode |
| 载荷 | 每个run固定`--payload-kg`，允许0～65kg；无跨run自动晋级 |
| 载荷位置与形状假设 | Base_link坐标(0,0,0.20)m；惯量按0.30×0.25×0.20m均匀箱体 |
| 状态随机化 | reset roll/pitch±0.015rad、yaw±0.05rad；没有外力push |

载荷与Base_link刚性固定，采用平行轴定理合成质量、质心与完整惯量。
每次只写入本run的`input/robot_payload.urdf`，网格指向当前checkout；规范URDF、原始网格与R1配置不变。
运行时会比对PhysX的总质量、Base_link质心与惯量，转换不一致即报错。
载荷通道是已知装载设定值，不是已实现的真实载荷估计器。

载荷不增加碰撞外形；本阶段不模拟放置箱子的接触冲击、单episode动态缓加载、热/结构持续承载。
65kg是目标工况，不是已达标范围；模型48Nm限幅不代表实机连续额定力矩。

奖励分别约束正立/高度、机身静止、四轮支撑、轮速、力分配、腿姿/腿速、模型力矩余量、关节限位和动作变化。
倾角超过30°、高度低于0.25m或非轮碰撞触发终止；episode时限单独计为timeout。

## 查找模型、回放与评估

控制台打印`[R6] <run目录>`。正式训练输出在：

```text
model_lighthw/training/multiskill/routes/route_06_heavy_payload/runs/<上海时间>_payload0kg_seed42/
```

`training_result.json`的`checkpoint`字段是本次最终模型路径。
以下`/absolute/path/to/r6_model.pt`必须替换为这个路径，并明确传入待测载荷：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/play.py --payload-kg 0 --checkpoint /absolute/path/to/r6_model.pt
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/evaluate.py --payload-kg 0 --headless --checkpoint /absolute/path/to/r6_model.pt --eval_seeds 43 44 45 --eval_episodes 4
```

GUI回放单次20s；评估默认3个seed×4次reset，每次20s。开头2s保留作诊断，2～20s为持载统计窗。
提前终止、缺样本、非有限值保留为失败；自动reset后的出生帧不计入通过样本。
评估按50Hz记录姿态、高度、速度、四轮/逐轮力、每关节力矩、饱和与位移；不是1kHz冲击峰值或热验收。
固定筛查门槛见[evaluation](evaluation/README.md)。

## 分级继续训练

建议阶段为0→5→10→15→25→40→65kg。检查当前阶段固定评估，再显式提高载荷，例如从空载模型继续5kg：

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 5 --checkpoint /absolute/path/to/r6_model.pt --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

`--max_iterations`表示本次追加更新数。恢复同契约R6的Actor、Critic、探索参数、Adam张量和实际学习率，
从下一个迭代号开始；新run重置物理状态与随机数，不声称恢复同一个仿真帧。
不会覆盖源checkpoint；R1与旧60维模型被拒绝。可将5换为其它已决定的阶段，但没有自动能力背书。
不同负载逐段继续训练可能遗忘轻载能力，晋级前还应对已完成负载复测；不把单阶段奖励上升当作全载荷通过。

## 运行记录

- `run.json`、`request.json`、`contract.json`：时间、状态、参数、契约与输入checkpoint哈希。
- `payload.json`、`input/robot_payload.urdf`：实际载荷的质量/质心/惯量和生成URDF哈希。
- `asset_manifest.json`、`source_manifest.json`、`source_snapshot/`：规范URDF/17网格哈希及源码快照。
- `params/env.yaml`、`params/agent.yaml`、`runtime_check.json`：有效配置、实际维度和物理资产检查。
- `console.log`、`training_metrics.csv`、`training/`：完整控制台、逐更新指标、TensorBoard和checkpoint。
- `training_result.json`、`result.json`：完成更新数、最终模型路径与哈希；缺失时启动器判失败。

这些产物只保存在当前电脑。详见[本轮验证](records/validation.json)；GPU短验未执行，正式R6机器人训练尚未开始。
