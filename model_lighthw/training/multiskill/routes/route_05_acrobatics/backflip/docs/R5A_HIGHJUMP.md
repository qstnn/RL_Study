> reference 文本快照：当前目录和入口见 [R5 路线索引](../../README.md)。下文保留原版本研究与离线交付说明；文中模型、数据及旧 ZIP 不随本分支上传，旧补丁安装步骤对应其原始目录布局。

# R5A J1 v3 固定高目标离线训练

适用于已经安装 J1 v2 补丁、能运行 `bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh refine` 的原 R5A 工程。直接用 U 盘传本补丁，不需要 Git 或联网。新阶段仍是零负载起跳、收腿、落地和停稳，为后续空翻准备；未加入完整空翻。

## 安装和开始

先结束此 R5A 工程的训练或回放，把 ZIP 放入原工程根目录（有 model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh 的目录），在那里逐行执行：

```bash
unzip lighthw-r5a-j1-v3-highjump-patch-20261019.zip
python lighthw-r5a-j1-v3-highjump-patch-20261019/apply_patch.py --project .
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh prepare
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh highjump
```

只需最后这一条启动命令，不再运行 refine 或额外 train。安装器核对原版本和补丁哈希、自动备份并在失败时回滚；重复安装会跳过相同文件。有本地改动会列出冲突而停止，不强制覆盖。必须已有 v2 基础包及本地 Isaac Sim/lab 环境；补丁没有重复携带依赖和网格。prepare 只用原包内锁定的 wheels，无需新下载。项目可继续放在原目录，也可复制整个安装后的工程。

## 本轮固定目标与可达实测

不采用逐级升高。从第一轮开始固定优化：四轮同时离地的净空峰值 25 cm，连续无接触时间 0.34 s，相对站姿的整机质心升高 20 cm，离地后质心继续升高 12 cm。四个量分别记录，不能用收腿净空代替质心起跳。配置中的 target_range_m 是净空参考范围 22–28 cm，未随机变更目标。

2026-10-19 的参考动作参数搜索中，能够重复落地停稳的最高四轮净空约 24.15 cm（接近关节边界组 4/4 次末段停稳）、相对站姿质心升高约 17.6 cm、连续无接触约 0.31–0.32 s。它是该参考动作族的仿真实测，不是全局理论极限或新 PPO 模型已收敛。生产参考选用留有收腿角度余量、六例均能停稳的约 22.5 cm 净空方案。保持质量、重力、摩擦、自碰撞和腿/轮 48/24 Nm、12.57/37.7 rad/s 上限，搜索中腿力矩达到限幅。

全成功门槛是最低合格线：真实离地后质心升高 ≥10 cm、相对站姿首段腾空质心升高 ≥16 cm、首段净空 ≥22 cm、连续无接触 ≥0.30 s，同时保持旧严格腾空 ≥0.10 s、完整 4 秒、末段连续稳定 ≥1 秒、漂移 <5 cm、水平速度 <0.05 m/s，并限制总轮法向力瞬时峰值 ≤2300 N。2300 N 是本轮仿真筛选阈值，未作为硬件许可载荷；达最低合格线不代表已达到全部优化目标。只计算第一次连续腾空，落地反弹不累加高度/时间。

## 一条命令完成什么

1. 从包内保存的已审查原 model999 校验 SHA 并迁移。Actor 历史从 285 维扩至 400 维：旧列精确映射，新反馈列为零；Critic 改为 109 维并重建，Adam 重置。
2. 冻结比较两个起点：保留原 Actor 输出的候选，以及保留特征、显式清零残差输出层的参考候选。新参考/观测已变，前者只保证旧输入映射的原始网络输出一致，不声称仿真行为相同。先评估 seed43/44/45 各一回合；不具备有效起跳和末段停稳则停止并回收日志。
3. 先做 20 次 Critic 预热，Actor 与探索噪声不变；随后最多 300 次 Actor 更新尝试，默认 512 环境、seed42。Actor LR 1e-5，Critic LR 1e-4，固定 std 0.05。每次 PPO 更新复查整个 rollout 的 KL，超过 .01 或出现非有限数会回退模型和 Adam，并记录拒绝。
4. 每 50 次更新做固定三例复评，逐例保护起跳、净空、滞空、末段稳定和冲击。连续两轮退化就提前结束，保留此前最佳模型；末次模型也保留，绝不默认选择它。最佳模型可能仍是零残差参考，这会如实记录，不能称作学习进步。
5. 最佳模型再评估 seed43/44/45 各 4 回合，并录制 seed42 三回合，保存全部 1 kHz 原始轨迹、曲线、模型、优化器、配置、源码快照及错误记录，统一输出一个 `[USB REVIEW]` 文件夹。

每轮上限是更新尝试数，拒绝更新也计数；物理状态不跨进程恢复，分块只续接模型、优化器和随机状态。每次重新执行 highjump 创建新的独立训练 session，不自动续旧 session。不要在未经审查时重复堆叠长训。episode length 最大仍是 200，即 4 s / 0.02 s，不能当作成功率。

显存不足可执行 `bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh highjump --num-envs 256`。只检查冻结起点可执行 `bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh highjump --baseline-only`。静态配置检查：`bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh highjump-check`。

## 回放与回收

训练结束后回放最近选中的高跳模型：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh highjump-play
```

默认 World 自由视角，沿用 LightHW 颜色、地面和本地天空，可在 Isaac Sim 窗口自由移动/自行设置跟随。指定任意本阶段模型：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh highjump-play --checkpoint "offline_sessions/route_05a/J1_highjump_v3_实际时间/selected_best.pt" --play-episodes 10
```

旧 `bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh play` 仍服务旧阶段。新模型不要交给旧入口。自动录制用于检查，不取代人工操控窗口检查。

结束后复制终端 `[USB REVIEW]` 指向的整个目录；路径同时记录在 `usb_review/route_05a/LATEST.txt`。回收包包含原模型副本、两个初始化候选、所有中间模型、最佳/末次模型、质量保护记录、视频与失败日志。summary.json 按阶段和评估角色分别统计。可以先在项目目录验证：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh verify --folder "usb_review/route_05a/实际时间"
```

异常中断后可整理已完成资料：

```bash
bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh collect --session "offline_sessions/route_05a/J1_highjump_v3_实际时间"
```

若失败就带回同一个回收文件夹，不要删除失败样本。验证记录见 `offline/R5A_HIGHJUMP_VALIDATION.json` 和补丁目录 `VALIDATION.md`。新阶段目标是否稳定学会，要以另一台电脑的完整训练和回放结果判断。
