# LightHW 多技能训练：R6 重载持载分支

当前 Git 分支：**`lighthw-multiskill-heavy-payload-r6`**。本机继续运行 R1；本分支用于另一台电脑训练 R6。
R6 任务为 `MultiSkill-LightHW-HeavyPayload-v0`，提供从零训练、同路线 checkpoint 续训、GUI 回放和固定持载评估。
完整参数、分级负载和输出说明见 [R6 README](routes/route_06_heavy_payload/README.md)。

## 另一台电脑获取

### 网页下载 ZIP（无需 Git 拉取）

直接下载 [R6 分支 ZIP](https://github.com/qstnn/RL_Study/archive/refs/heads/lighthw-multiskill-heavy-payload-r6.zip)，
或在 GitHub 切换到 `lighthw-multiskill-heavy-payload-r6` 后点击 **Code → Download ZIP**。
这个分支已经包含所需的 lighthw 基线，不需要下载两个分支再覆盖文件。

解压到独立目录，在含有 `source/` 和 `model_lighthw/` 的项目根目录打开终端：

```bash
python3 model_lighthw/training/multiskill/prepare_zip.py
```

GitHub 默认 ZIP 通常只有 Git LFS 指针，17 个真实网格不会随源码一起下载。
上面的入口仅使用 Python 标准库，通过 HTTPS 补齐缺失/指针/校验失败的网格（全部约269MB），
固定已发布资产提交，逐个验证 SHA256，完整后才替换文件。正确网格自动跳过，失败后可重试同一命令。
需要能访问 `media.githubusercontent.com`；不需要 Git 登录、Git LFS 或 `.git` 目录。
只是校验现有文件、不联网时使用 `python3 model_lighthw/training/multiskill/prepare_zip.py --check`。

若希望网页 ZIP 本身直接包含真实网格，仓库管理员也可在 **Settings → General → Archives** 勾选
**Include Git LFS objects in archives** 后重新下载。此项是 GitHub 仓库设置，修改 `.gitattributes`
不能代替开启该设置；本次未改变远端设置。参见 [GitHub 官方说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-git-lfs-objects-in-archives-of-your-repository)。

准备完成后，按下方“启动 R6”激活完整 `lab` 环境，执行 check、smoke 和2次更新短验，再启动正式训练。
不需要修改用户名、项目目录、URDF/mesh 路径，也不要在解压目录运行 `git pull` 或 `git lfs pull`。
ZIP 不包含 Isaac Sim、驱动和 Conda 环境，也不自动更新；更新代码时重新下载到独立目录并保留旧运行数据。

### 使用 Git 获取

在 Ubuntu 安装 Git LFS：`sudo apt-get install git-lfs`，再执行 `git lfs install`。
已有本项目仓库时，在其根目录执行：

```bash
git remote get-url study >/dev/null 2>&1 || git remote add study git@github.com:qstnn/RL_Study.git
git fetch study
git switch lighthw
git pull --ff-only study lighthw
git switch --track study/lighthw-multiskill-heavy-payload-r6
git lfs pull study --include="model_lighthw/lighthw_urdf/meshes/*.STL" --exclude=""
```

`git switch --track` 用于首次建立本地技能分支。后续更新改用：

```bash
git switch lighthw-multiskill-heavy-payload-r6
git pull --ff-only study lighthw-multiskill-heavy-payload-r6
git lfs pull study --include="model_lighthw/lighthw_urdf/meshes/*.STL" --exclude=""
```

如需与另一条正在运行的路线隔离，在 `git fetch study` 后使用独立工作区：

```bash
git worktree add -b lighthw-multiskill-heavy-payload-r6 ../rl_training_r6 study/lighthw-multiskill-heavy-payload-r6
cd ../rl_training_r6
git lfs pull study --include="model_lighthw/lighthw_urdf/meshes/*.STL" --exclude=""
```

以上 worktree 命令适用于本地尚无 R6 分支；已有时去掉 `-b`，末尾改为本地分支名。
全新克隆可直接获取技能分支，无需先合并 lighthw：

```bash
git clone --origin study --branch lighthw-multiskill-heavy-payload-r6 git@github.com:qstnn/RL_Study.git rl_training_r6
cd rl_training_r6
git lfs pull study --include="model_lighthw/lighthw_urdf/meshes/*.STL" --exclude=""
```

分支从本地 `lighthw` 提交 `62e4614` 建立，包含完整已提交基线。2026-10-15 获取远端时，
`study/lighthw` 为 `8b63cf3`，落后本地 1 个提交，这个提交已包含在 R6 分支中。
R6 分支补齐了共享依赖、基础入口引用的 `locomotion/skills/__init__.py`、multiskill 源码和 17 个规范 STL（Git LFS，总计约269MB），
无需手工复制源码或旧模型。只推送 R6 分支，原 `lighthw` 上大量未提交的历史实验不属于此次同步范围。

## 启动 R6

在实际克隆或解压目录的根目录执行。使用已经能运行 Isaac Sim 的完整 `lab` 环境：

```bash
conda activate lab
export TERM=xterm
python -c "from isaaclab.app import AppLauncher; print('Isaac Sim / Isaac Lab import OK')"
python model_lighthw/training/multiskill/run.py check --route 06 --payload-kg 0
```

本机版本：Isaac Lab 2.2.0、Isaac Sim 5.0.0-rc.45、Python 3.11.15、
PyTorch 2.7.0+cu128、rsl-rl-lib 3.0.1、Gymnasium 1.2.0。
`conda activate lab` 必须加载该电脑的 Isaac Sim 环境；直接调用 `envs/lab/bin/python` 不等价。
二进制安装若未设置激活钩子，需要先 `source /你的/isaac-sim/安装目录/setup_conda_env.sh`。
Git 不安装驱动或仿真环境。启动器优先使用当前 checkout 源码，不依赖另一份源码的可编辑安装。

首次在另一台电脑运行，建议先执行两条有界短验：

```bash
python model_lighthw/training/multiskill/run.py smoke --route 06 --payload-kg 0 --headless --num_envs 32
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 0 --headless --num_envs 32 --max_iterations 2
```

**正式启动命令（独立空载持载基线，1000 次 PPO 更新）：**

```bash
python model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts/train.py --payload-kg 0 --headless --num_envs 1024 --max_iterations 1000 --seed 42
```

首轮无需 R1 checkpoint。R6 的 58/237/16 观测/动作契约与 R1 的 57/236/16 分开，不能直接加载 R1 模型。
负载从 0、5、10、15、25、40 到 65kg 分级，**每次运行固定质量，人工检查后升阶段**；续训命令见 R6 README。
它实现刚性固定载荷的分级持载，单个 episode 内动态缓加载、载荷碰撞体、热模型和结构模型尚未实现。

## 验证边界与目录

[R6 验证记录](routes/route_06_heavy_payload/records/validation.json) 区分 CPU 内容检查、真实 RSL-RL CPU 合成数据集成和 GPU 验证。
本轮不占用本机 R1 的 GPU；R6 实际环境实例化、GPU 短验和承载行为待另一台电脑验证。
进程返回0不算成功，启动器必须读取运行时维度/质量/质心/惯量检查及训练结果。

- [R1 轮式行走](routes/route_01_wheel_locomotion/README.md)：保留已实现入口，当前训练由本机负责。
- [R6 重载持载](routes/route_06_heavy_payload/README.md)：本分支主任务，独立数据目录。
- [原规划](PLAN.md)：其余技能的分组与设计；03、05等规划项仍不可训练。
- [初版模型审查](records/MODEL_AUDIT.md)：静力学估计不代表65kg承载或实机持续力矩通过。

Route00/原规划中的 route53、stand_route 是历史设计引用，不是运行 R6 所需源码。
运行数据 `runs/`、`validation/`、`evaluations/`、`playbacks/` 和其中 checkpoint/USD 都不随 Git 同步。
跨电脑回放模型时需另行复制选定的 R6 checkpoint；每个新训练从零启动时不需要模型。
