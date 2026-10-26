# LightHW 多技能 PPO 工作区

**2026-10-26 本体已统一更新为 22 kg（质量与惯量同比例）。** 新运行使用新资产合同；历史快照/模型保持原物理。静态和 CPU 验证通过，未做新 GPU/训练验收。[完整模型说明](MODEL_22KG.md)。

**2026-10-20 R4已按用户命名拆分：R4a后腿支撑优先，R4b前腿支撑后续独立训练。** `split_v4`关闭联合模式，调整真实重心转移/卸载及任务完成后的恢复奖励，保留固定条件和完成后用户通知再审查。公共04/04a进入R4a，04b进入R4b；旧联合训练保留追溯。[当前命令、奖励与证据](routes/route_04_two_leg_balance/split_v4/README.md)。

**2026-10-20 R4 v3正式1000轮审查完成，完整双腿目标未通过。** 固定初态/无观测噪声实际生效，997次更新保留；最终前后各0/3成功、严格保持0秒，300/600对照也以四轮支撑为主。下一步保留固定条件，先调整真实转移/卸载与辅助奖励，不原样加预算；训练期间及末尾仍不自动评估，完成后等用户通知审查。[报告与曲线](routes/route_04_two_leg_balance/results/2026-10-20_v3_1000_update_review/README.md)。

**2026-10-16 16:25任务范围修订：−25°下坡排除20/40 kg拖车，保留空载≥3 m/s与15 kg机身负载下坡（无速度要求）；当前64训练组合、16项固定验收。此前18项行为结果为修订前历史，启动命令保持。**

**2026-10-16 16:16：R1当前入口为[V3路线跟踪/坡面调姿/稳停诊断](routes/route_01_wheel_locomotion/unified_v3/README.md)。** 已做72项能力探针、18项CPU回归、真实GPU168环境短验、1024环境2+1次PPO、精确恢复/导出/18工况周期评估及回放。40 kg/±25°仍未达标；下一步由用户启动500次诊断，每100次冻结评估。一个Actor、64种工况从首批同时训练，15 kg/40 kg互斥、≥3 m/s仅空载；μ2.3、滚阻.02、牵引.10 m、腿/轮96/48 N·m为实验条件，未确定电机额定需求。

**V2已由用户停止。** 2026-10-16最后日志4238、最新保存model4200，模型和来源保留；[停止核对](routes/route_01_wheel_locomotion/unified_v2/records/2026-10-16_v2_stopped_next_plan.json)。V1/Stage1历史保留；通用`run.py --route 01`仍指原Stage1，V3使用其独立`scripts/train.py`。

**2026-10-15：Route01 Stage1首轮1000次正式PPO已完成，实际57/236/16与训练保存链路通过，固定行为验收待完成。**
训练15:43:42～16:38:20，最终model_999；[训练审查、曲线与精确回放命令](routes/route_01_wheel_locomotion/results/2026-10-15_stage1_training_review/README.md)。
这里的逻辑 Route 编号与旧 `training/routes/route_00…53` 分属两个命名空间。

先看 [完整审查与规划](PLAN.md)、[验证记录](records/validation.json) 和 [本轮交付清单](records/DELIVERY.md)。
运行源码：[locomotion/multiskill/lighthw](../../../source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/)。

| 逻辑路线 | 状态 | 任务 |
|---|---|---|
| [00 基线索引](routes/route_00_baseline_reference/README.md) | 只读引用 | 原 Rough、近期 route53、独立 stand_route |
| [01 轮式/坡地](routes/route_01_wheel_locomotion/README.md) | V3集成短验完成；500次诊断待启动，全任务未达标，历史保留 | 1-1、1-2、1-5；载荷/全速度/坡面/有轮牵引 |
| [02A 跑步](routes/route_02_legged_stairs/running/README.md) | 待开发 | 1-3 |
| [02B 台阶](routes/route_02_legged_stairs/stairs/README.md) | 待开发 | 1-6 |
| [03 高障碍](routes/route_03_high_obstacle/README.md) | 待可行性分析 | 1-7 |
| [04A 后腿平衡](routes/route_04_two_leg_balance/split_v4/R4a/README.md) | 当前优先；独立固定条件训练，完整目标待验收 | 1-10 |
| [04B 前腿平衡](routes/route_04_two_leg_balance/split_v4/R4b/README.md) | 独立入口，正式训练后续进行 | 1-10 |
| [05A 后空翻](routes/route_05_acrobatics/backflip/README.md) | 待动力学分析 | 1-8 |
| [05B 侧空翻](routes/route_05_acrobatics/sideflip/README.md) | 待动力学分析 | 1-9 |
| [06 重载支撑](routes/route_06_heavy_payload/README.md) | 待结构/静力学分析 | 1-11 |

规划项不注册 Gym task，也没有伪训练脚本。共同入口 `run.py` 拒绝运行规划项。
独立站立线 `../stand_route/` 保留，不迁移、不改变用途；它可为01/06提供后续经验。

仓库根目录、完整 `lab` 环境中：

```bash
python model_lighthw/training/multiskill/run.py list
python model_lighthw/training/multiskill/run.py check --route 01
python model_lighthw/training/multiskill/run.py smoke --route 01 --headless --num_envs 32
```

Route01的`check`只做CPU/注册/资产检查；其`smoke`做100个控制步、0次PPO，约2秒仿真。R4的check和完整回合smoke见其独立README。
smoke通过后再用 Route01 README 的有界PPO命令检验训练/保存链路。正式训练由用户启动。

每个运行目录独立保存：`run.json`、`request.json`、`contract.json`、完整`console.log`、
源码快照/哈希、URDF/mesh哈希、实际`params/env.yaml`/`agent.yaml`、`runtime_check.json`、`asset_cache/`。
PPO原生TensorBoard及checkpoint写入该run的`training/`；评估保存逐case CSV/JSON与`evaluation.json`。
上述通用Stage1入口不自动恢复/晋级；R1 V3独立入口只迁移指定Actor，并按其README每100次自动冻结评估，完成500次后不自动延长。

未来接入技能管理器前，每个Actor必须声明进入/退出条件和经过验证的适用范围，详见各route的
`skill.json`。`deployment_ready=false` 表示这些声明尚未经过实机或全任务验收，不是可用技能列表。
