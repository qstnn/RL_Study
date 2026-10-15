# 本轮交付与验证

更新时间：2026-10-15 15:38（Asia/Shanghai）。本轮没有启动新机器人PPO或GPU rollout，没有修改旧checkpoint。

## 结论

- 保留旧routes的53个目录、独立stand_route与canonical资产；新增并列multiskill。
- 原始Rough/Blind与近期route53不是同一配置。route53有部分平地/楼梯case证据，但完整Rough、多seed及最终负载指标未验收。
- 新Route01 Stage1实现独立环境、奖励、PPO、训练/回放/评估入口，57/236/16声明；其他路线只有规划与待实现结构。
- Route02按跑步/台阶分02A/02B；Route05按后空翻/侧空翻分05A/05B；03/05/06保持硬件可行性待定。
- 10项CPU内容测试、15个新Python文件编译、显式Gym注册、16关节/17mesh与动作限位检查通过。
- 真实AppLauncher导入失败：缺isaacsim；真实环境配置导入失败：缺omni。短验入口在依赖检查处退出，未实例化环境。
- 552个既有源码、URDF、关键model599/999哈希保持；原始证据见protected_files.json。旧权重未写入，未启动训练。
- 纯URDF/FK/理想静载分析见MODEL_AUDIT.md：65kg示例静矩36.423Nm不是持续承载/结构通过。

## 下一步与准确命令

优先Route01 Stage1。完整lab环境、仓库根目录，先100步无PPO短验：

```bash
python model_lighthw/training/multiskill/run.py smoke --route 01 --headless --num_envs 32
```

短验通过后验证2次PPO更新：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py --headless --num_envs 32 --max_iterations 2
```

确认保存和日志后，再由用户运行Stage1预算：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

每次创建新的独立run。从零开始，不自动加载旧Actor；新57维不兼容旧60维。
回放、评估准确参数及checkpoint路径获取方式见Route01 README。
其余新逻辑Route不提供训练命令，因为未实现，不生成空注册。

## 修改范围与实际风险

既有文件仅同步model_lighthw/TRAINING_PROGRESS.md和training/README.md说明。
公共硬件、旧注册、Rough核心、旧runner、旧checkpoint保持；新包import无注册副作用。
新框架复用既有Rough/资产/观测数据模块，因此未来修改共享源码仍可能影响它；每次run保存来源快照与哈希。
真实配置/GPU启动未验证是当前主要限制；不能将代码完整、CPU通过或以后checkpoint生成等同于行为达标。
未提交、未推送Git。

## 新增文件

- `model_lighthw/training/multiskill/.gitignore`
- `model_lighthw/training/multiskill/PLAN.md`
- `model_lighthw/training/multiskill/README.md`
- `model_lighthw/training/multiskill/audit_model.py`
- `model_lighthw/training/multiskill/common.py`
- `model_lighthw/training/multiskill/protocol.py`
- `model_lighthw/training/multiskill/records/MODEL_AUDIT.md`
- `model_lighthw/training/multiskill/records/app_import.log`
- `model_lighthw/training/multiskill/records/cpu_tests.log`
- `model_lighthw/training/multiskill/records/entrypoint_help.log`
- `model_lighthw/training/multiskill/records/environment_config_import.log`
- `model_lighthw/training/multiskill/records/model_audit.json`
- `model_lighthw/training/multiskill/records/protected_files.json`
- `model_lighthw/training/multiskill/records/registration_assets.log`
- `model_lighthw/training/multiskill/records/smoke_attempt.log`
- `model_lighthw/training/multiskill/records/validation.json`
- `model_lighthw/training/multiskill/routes/route_00_baseline_reference/README.md`
- `model_lighthw/training/multiskill/routes/route_00_baseline_reference/skill.json`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/README.md`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/evaluate.py`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/play.py`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py`
- `model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/skill.json`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/running/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/running/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/running/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/running/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/running/skill.json`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/stairs/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/stairs/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/stairs/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/stairs/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_02_legged_stairs/stairs/skill.json`
- `model_lighthw/training/multiskill/routes/route_03_high_obstacle/README.md`
- `model_lighthw/training/multiskill/routes/route_03_high_obstacle/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_03_high_obstacle/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_03_high_obstacle/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_03_high_obstacle/skill.json`
- `model_lighthw/training/multiskill/routes/route_04_two_leg_balance/README.md`
- `model_lighthw/training/multiskill/routes/route_04_two_leg_balance/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_04_two_leg_balance/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_04_two_leg_balance/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_04_two_leg_balance/skill.json`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/skill.json`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/sideflip/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/sideflip/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/sideflip/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/sideflip/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_05_acrobatics/sideflip/skill.json`
- `model_lighthw/training/multiskill/routes/route_06_heavy_payload/README.md`
- `model_lighthw/training/multiskill/routes/route_06_heavy_payload/configs/README.md`
- `model_lighthw/training/multiskill/routes/route_06_heavy_payload/evaluation/README.md`
- `model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/README.md`
- `model_lighthw/training/multiskill/routes/route_06_heavy_payload/skill.json`
- `model_lighthw/training/multiskill/run.py`
- `model_lighthw/training/multiskill/tests/test_framework.py`
- `model_lighthw/training/multiskill/worker.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/__init__.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/__init__.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/catalog.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/ppo_cfg.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/rewards.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_env_cfg.py`
- `model_lighthw/training/multiskill/records/DELIVERY.md`
