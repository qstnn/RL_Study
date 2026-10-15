# Route01：轮式运动与坡地

对应1-1（站立/15kg）、1-2（≥3m/s/15kg/牵引40kg）、1-5（25°同工况）。
**目前只实现Stage1：平地、空载、无牵引，.2～.6m/s前进和25%零指令。** 后续阶段没有注册。
代码/CPU检查不等于学会站立/行走。当前缺isaacsim，实际配置加载和GPU短验待完成。

task：`MultiSkill-LightHW-Wheel-Stage1-v0`；Actor/Critic/action声明为 **57/236/16**。
Actor仅角速度、重力、指令、关节角/速度与上一动作；Critic额外有线速度和地形高度。
旧60维Actor不兼容；训练只从零，`--checkpoint`仅用于本路线的回放和评估。
位置腿12+轮速4不变；线性腿scale .08/.14、URDF限位，轮scale5/clip37。
静止专用接触/姿态/速度约束在运动时关闭；平地正立与高度奖励仅适用于Stage1。

## 直接命令

先按[跨电脑同步说明](../../README.md#两台电脑同步与启动)切换技能分支并下载 Git LFS 网格。
工作目录为本机实际克隆的仓库根，使用完整 Isaac Sim 对应的 `lab` 环境：

```bash
conda activate lab
export TERM=xterm
python model_lighthw/training/multiskill/run.py check --route 01
python model_lighthw/training/multiskill/run.py smoke --route 01 --headless --num_envs 32
```

环境短验通过后，2次PPO更新检查日志与checkpoint保存，独立输出：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py --headless --num_envs 32 --max_iterations 2
```

正式Stage1命令（等待短验结果后由用户启动，本轮没有执行）：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/train.py --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

训练结束会在`runs/<上海时间戳>/training_result.json`记录本次最终模型的准确路径。
将下面`/absolute/path/to/model.pt`替换为该文件中的checkpoint字段，不自动搜索旧模型：

```bash
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/play.py --checkpoint /absolute/path/to/model.pt
python model_lighthw/training/multiskill/routes/route_01_wheel_locomotion/scripts/evaluate.py --headless --checkpoint /absolute/path/to/model.pt --eval_seeds 43 44 45
```

播放是固定18秒起停GUI序列；评估是3seed×3case、共162秒仿真，无PPO更新。
每case失败立即记录并结束该case；不会在重置后继续拼成成功轨迹。
相同确定性初态下不同seed可能完全相同，因此不把多seed运行冒称域随机化鲁棒性验收。

## 文件与输出

- [配置来源](configs/README.md)：环境、独立奖励、PPO及控制契约，不复制公共资产。
- [评估协议](evaluation/README.md)：固定零/.3/.6起停、完整性与分项指标。
- `runs/`：每次独立PPO，`training/`含TensorBoard/checkpoint。
- `validation/`：100步无PPO短验。
- `playbacks/`、`evaluations/`：独立冻结回放和评估。
- [技能声明](skill.json)：候选进入/退出条件；未准备实机部署。

路线课程：低速起停→高速起停→5°→10/15/20/25°→载荷/牵引→组合。
升阶依据固定评估的成功率/稳定性；Stage1不自动升阶，后续需要各自实现并验证。
零指令四轮支撑要求与运动中的动态接触变化分开，移动低载只记录，不直接判失败。
