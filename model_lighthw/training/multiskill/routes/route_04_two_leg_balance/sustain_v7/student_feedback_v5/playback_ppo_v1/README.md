# 最终R4a PPO模型窗口回放

独立入口，默认加载[最终审查](../results/2026-10-23_formal_1000_review/review.json)绑定的300秒自主评估及最终`model_001034.pt`。先通过`common_eval.require`核对冻结源、原机器人、模型和证据，再严格加载完整PPO中的`actor.*`；保留305维原观测、20维因果状态和结构化反馈，执行均值动作，教师全关。不会训练或恢复优化器。

在有桌面显示的本机终端执行：

```bash
cd /home/user/RL_lab/rl_training
R4_BASE=model_lighthw/training/multiskill/routes/route_04_two_leg_balance/sustain_v7/student_feedback_v5
bash "$R4_BASE/playback_ppo_v1/play.sh" --hold-seconds 300
```

默认打开Isaac Sim窗口，镜头相对槽0出生点观察；机器人从原四轮初态起身，完成2秒连续合格捕获后再持续观察300秒，最后退出。没有自动返回四轮命令，也不会不断重播；环境内部在结束步的自动reset仍按原环境发生，程序不把reset后的状态接到前一段站立时长。前腿离地、后腿支撑；R4b不在此入口内。

可用`--hold-seconds 10`较快观察起身与保持，或`--hold-seconds 60`。`--view-env 1`将镜头切换到第二槽。模拟速度取决于机器；终端每1秒**模拟时间**报告实际位置、俯仰、捕获后保持和首次失稳，同时写入新run的`progress.json`。初始化会打印来源检查、仿真启动、模型导入和物理核验阶段，等待时可据此判断进度。没有键盘运动指令，点击窗口只影响普通视角操作；按终端Ctrl+C或关闭窗口结束回放。

CPU预检和短验命令：

```bash
bash "$R4_BASE/playback_ppo_v1/play.sh" --check
bash "$R4_BASE/playback_ppo_v1/play.sh" --headless --max-steps 50 --hold-seconds 10
```

`--check`不启动仿真或创建运行目录。`--max-steps`按50Hz控制步提前结束，用于集成检查，不能算站立验收。`--output /新的绝对目录`可指定保存位置；已有目录直接拒绝覆盖。默认结果写入本目录`runs/时间戳_play/`，包含请求、实际物理核验、每秒进度和末状态。资产从已审查档案复制，USD/cache写入新的回放目录，原训练和评估目录保持不变；不留存新的全1kHz大轨迹。

`play.sh`自动使用`lab` Python并加载本机`isaac-sim/setup_conda_env.sh`；其他安装位置可通过`R4_PYTHON`、`R4_ISAAC_SIM`、`R4_ISAACLAB`设置。旧`split_v4/r4a.sh play`和通用RSL-RL `play.py`不能直接加载该325维结构化模型。

该独立入口已通过实际CPU模型/来源预检、Python编译及shell语法检查；按用户要求直接提供命令，未启动GPU/GUI短验，不能将已有headless300秒证据当作新窗口入口已经实测。窗口回放只作可视化，原正式独立300秒验收继续以原始1kHz轨迹和[审查结果](../results/2026-10-23_formal_1000_review/README.md)为准。
