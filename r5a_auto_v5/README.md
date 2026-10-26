# R5A 自主后空翻 v5：U盘离线训练

本补丁安装到已有、可以运行 v4 的完整 R5A 工程。新增 `r5a_auto_v5/` 和 `r5a_auto_v5.sh`，使用现有机器人网格、LightHW 配色、本地地面/天空、Isaac Sim、lab 环境和离线依赖。补丁内含已审查的 v3/2000 Actor，不需要另找模型、Git 或联网。

## 安装

把 ZIP 和 `.sha256` 放在同一目录，执行：

```bash
sha256sum -c lighthw-r5a-autoflip-v5-credit-patch-20261022.zip.sha256
unzip lighthw-r5a-autoflip-v5-credit-patch-20261022.zip
```

安装器的参数是**原完整工程目录**，不是解压的补丁目录。下面路径来自上次训练日志，如已迁移则换成当前路径：

```bash
python3 lighthw-r5a-autoflip-v5-credit-patch-20261022/install.py /home/huawei/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
```

安装器先校验基础文件和全部补丁内容，再添加文件。已有同名不同内容时停止，重复安装同内容可跳过。原 v3/v4 和其他路线保留。

## 开始训练

进入安装时指定的完整工程目录，使用已有的 lab 环境：

```bash
conda activate lab
cd /home/huawei/RL_Lab/RL_Study/model_lighthw/rl_training_r5a_usb_20261019
bash r5a_auto_v5.sh check
bash r5a_auto_v5.sh train
```

`check` 是 CPU/资产检查。默认 `train` 为 **512 环境、3000 次 PPO 更新尝试**，每 200 次仅保存，连续训练到结束，末尾一次统一评估最后最多 3 个候选、录像和精简回收。不需要另开训练命令。KL/非有限值/运行异常保护仍会停止并回收诊断。

新训练自动复制包内 v3 Actor 到地面和空中分支，重新建立 Critic、Adam 和计数；**不要传入 v4 检查点续训**。两分支初始确定性输出保持 v3，训练后的效果要由新结果判断。

本机本轮无法连接 NVIDIA 驱动，因此交付验证包含 CPU 状态机、历史轨迹奖励重放、安装/归档自检；尚未运行 v5 GPU 短验。若希望先检查那台电脑的完整管线，可选执行 `bash r5a_auto_v5.sh train --num-envs 32 --updates 2 --candidates 1 --no-video`；它是独立短验，不是正式模型。短验结束若提示 `physics_rejected`/`physics_unchecked`，读取回收包即可，不从短验模型续训。正式训练仍使用上面的无参数 `train`。

如需覆盖运行环境路径，沿用 `R5A_PYTHON`、`R5A_ISAAC_SIM` 或原 `R6_PYTHON`、`R6_ISAAC_SIM`。补丁不安装 Isaac Sim。

## 完成后拷回

终端打印 `[LOCAL FULL DATA]` 和 `[USB REVIEW ZIP]`。**仅拷回后者的 ZIP 和旁边 `.sha256`**。包中含一个选中模型及其轨迹、录像、候选摘要、全程更新日志、回合窗口统计和一份源码/依赖。

没有通过物理筛查时，仍导出一个明确标记的诊断模型；它不会更新可用模型标记，也不算成功版本。`best_available_not_passed` 只表示物理筛查通过的部分动作，完整空翻仍未通过。筛查通过也不等于严格守恒或实机认证。

```bash
bash r5a_auto_v5.sh play
bash r5a_auto_v5.sh collect --folder /实际完整运行目录
```

默认 `play` 使用最近一个物理筛查通过的 v5 候选；保留 World 自由相机，可在 Isaac Sim 窗口自行操作。物理未通过的诊断模型需要显式指定其路径，不会被默认播放标记选中。

技术依据见 [DESIGN.md](DESIGN.md)，v3/v4 复核和本次验证见补丁根目录 `evidence/` 与 `VALIDATION.md`。
