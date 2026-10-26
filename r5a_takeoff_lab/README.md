# R5A 起跳与角动量离线实验工具

本包是独立参考控制实验，PPO 更新为 0；它不是已经学会后空翻的模型，也不是新的自主空翻训练补丁。当前 v3 的 highjump/refine/train/play 命令保持原样。完整结果见 [报告](evidence/REVIEW.md)，下一阶段奖励设计见 [自主 RL 方案](evidence/RL_DESIGN.md)。

先在已安装高跳 v3 的 R5A 项目根目录执行：

```bash
bash r5a_takeoff_lab/reproduce.sh --check
```

在另一台离线电脑上复现三个种子的基线、较高竖跳、留余量对照、旋转候选，并录制较高竖跳：

```bash
bash r5a_takeoff_lab/reproduce.sh
```

脚本只运行仿真探针，不学习模型；较高跳跃候选接近膝关节限位，留余量候选有停稳失败，旋转候选没有完成后空翻，均作为研究对照保留。默认继续沿用原 LightHW 配色、地面、本地天空和 World 自由视角设置。录像脚本为 headless，手动回放仍使用原项目 play 入口，不改相机跟随行为。

结束后复制终端 `[USB REVIEW]` 指向的整个目录。该目录包含每个候选的配置、源码快照、输入模型、1 kHz 原始轨迹、整机角动量复算、失败日志、录像、summary.json 与 SHA256.json。路径也写在 `usb_review/route_05a/LATEST_TAKEOFF_REVIEW.txt`。失败时同样保留已有文件夹。

只运行一个种子且不录像：

```bash
bash r5a_takeoff_lab/reproduce.sh --seeds 43 --no-video
```

校验返回目录（把路径替换为实际文件夹）：

```bash
bash r5a.sh python r5a_takeoff_lab/verify.py usb_review/route_05a/takeoff_review_实际时间
```

本工具不联网、不调用 Git、不安装新依赖；使用原项目 lab/Isaac Sim 和已准备好的离线依赖。完整自主后空翻训练环境仍待实现，不能用本工具的参考探针成功率代替 PPO 或实机验收。
