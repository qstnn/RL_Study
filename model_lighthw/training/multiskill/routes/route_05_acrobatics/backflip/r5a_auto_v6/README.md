> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A 自主后空翻 v6：U盘离线补丁

安装到已有、能够运行v5的完整R5A工程。无需Git、下载资源或重装依赖。本包只新增`model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v6/`和`model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh`，保留v5模型、日志及其他路线。

## 安装

把以下两个文件放到另一台电脑的同一目录：

- `lighthw-r5a-autoflip-v6-timing-patch-20261022.zip`
- `lighthw-r5a-autoflip-v6-timing-patch-20261022.zip.sha256`

在放置压缩包的目录运行：

```bash
sha256sum -c lighthw-r5a-autoflip-v6-timing-patch-20261022.zip.sha256
python3 -m zipfile -e lighthw-r5a-autoflip-v6-timing-patch-20261022.zip .
python3 lighthw-r5a-autoflip-v6-timing-patch-20261022/install.py /home/huawei/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
```

安装器参数是**原完整工程目录**，不是刚解压的补丁目录。上面的路径来自你返回的v5运行记录；如已移动，替换为实际目录。安装器先校验全部基础文件和补丁内容，已有不同内容时停止，同内容重复安装可跳过。

## 开始训练

```bash
conda activate lab
cd /home/huawei/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh check
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh train
```

**首次v6训练不要加`--checkpoint`。** 包内已带回并校验你这次v5的两个Actor权重：源文件`model_1200.pt`，实际完成1201次更新。v6自动做显式迁移，空中/恢复Actor权重原样继承；地面Actor隐藏层保留，输出层权重和偏置缩至35%，给准备动作探索留下空间。Critic、Adam和v6计数重新开始；不是旧v5优化器的原样续训，也不是所有网络从随机开始。

默认**512环境、连续3000次PPO更新尝试，每200次只存档**；末尾一个评估进程比较最后最多3个保存模型，录像并导出一个选中模型的精简USB包。无需再输入第二条训练启动命令。正常情况下训练进度会显示`…/3000`，更新结束后还要等待评估和打包完成。

如需指定Python/Isaac Sim路径，沿用已有`R5A_PYTHON`、`R5A_ISAAC_SIM`或`R6_PYTHON`、`R6_ISAAC_SIM`配置。`check`是CPU和资产检查，不是GPU训练或后空翻行为通过证明。

本机本次无法访问CUDA，已完成CPU反例、历史轨迹奖励重放、模型迁移和离线安装测试，**尚未进行v6 GPU短训**。可选的独立管线短验是：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh train --num-envs 32 --updates 2 --candidates 1 --no-video
```

短验不验证收敛；正式训练仍运行无参数的`bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh train`，不加载短验模型。若末尾是`physics_rejected`/`physics_unchecked`，收回诊断包，不把它标为成功版本。

## 实时查看和拷回

另开终端，进入同一工程目录：

```bash
bash r6.sh python -m tensorboard.main --logdir offline_sessions/route_05a --host 127.0.0.1 --port 6006 --reload_interval 5
```

在训练电脑浏览器打开`http://127.0.0.1:6006`，选择`autoflip_v6_…`。重点看`Ended/angle_deg`（结束回合角度，度）、`Ended/preparation_cm`、`Ended/compact_credit`、`Ended/flight_time`和`Ended/success`。原`AutoFlip/angle_deg`是混合各阶段环境的均值，不能当最终空中转角。

正常结束后终端打印`[USB REVIEW ZIP]`。**只拷回这个ZIP及旁边的`.sha256`**；里面有一个选中模型、物理/策略轨迹、录像、候选摘要、训练窗口统计与源码。完整原始日志及其他模型留在训练电脑。`best_available_not_passed`表示部分动作，不能当成功。

GUI回放继续使用自由相机：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh play
```

## 手动停止与后续评估

`Ctrl+C`按一次后等待`[USB REVIEW ZIP]`；中断包是训练诊断包，不会自动完成最终评估，不包含训练后选中模型/录像。v6会从完整更新日志恢复中断计数，避免显示已训练0次。不要连续打断收尾。

如提前停止，按实际保存路径补一次评估：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v6.sh evaluate --checkpoint /本次v6运行目录/training/training/model_1200.pt
```

上面只是路径示例，需替换成本次真实存在的检查点。仅v6同实现检查点可以用于v6的`evaluate`或`train --checkpoint`；v5来源只通过包内迁移使用。未来要续训同版v6时，`--updates`表示**额外更新次数**。

实现依据和验证边界见`DESIGN.md`及补丁根目录`VALIDATION.md`。不承诺v6已能完成空翻，原动力学、电机限幅和单圈站稳门保持。
