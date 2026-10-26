# R5 特技路线索引

R5 的实验、版本参数、奖励实现、启动脚本和说明集中在此目录。R5A 为后空翻研究，R5B 为侧空翻规划；它们共享 22 kg 的 canonical URDF。

```text
route_05_acrobatics/
├── backflip/
│   ├── scripts/             # J1 和各 Auto/Guided 版本的启动入口
│   ├── docs/                # R5A 总说明、高跳说明
│   ├── offline/             # R5 专用迁移、回收与打包工具
│   ├── highjump/            # J1 v3 高跳
│   ├── r5a_auto/            # Auto 初版
│   ├── r5a_auto_v2…v6/      # Auto 各版本，独立保留
│   ├── r5a_guided_v1…v5/    # Guided 各版本，独立保留
│   ├── r5a_teacher_search/  # 教师搜索与控制参数
│   └── r5a_takeoff_lab/     # 起跳参考控制实验
└── sideflip/                # R5B 现有规划
```

## 从哪里看

| 内容 | 入口 |
|---|---|
| 当前保留的 Guided v5 | [版本说明](backflip/r5a_guided_v5/README.md)、[参数与奖励权重](backflip/r5a_guided_v5/spec.py)、[奖励计算](backflip/r5a_guided_v5/shaping.py)、[落地逻辑](backflip/r5a_guided_v5/landing.py)、[环境](backflip/r5a_guided_v5/env.py) |
| Guided 先前版本 | [v1](backflip/r5a_guided_v1/README.md)、[v2](backflip/r5a_guided_v2/README.md)、[v3](backflip/r5a_guided_v3/README.md)、[v4](backflip/r5a_guided_v4/README.md) |
| Auto 版本 | [初版](backflip/r5a_auto/README.md)、[v2](backflip/r5a_auto_v2/README.md)、[v3](backflip/r5a_auto_v3/README.md)、[v4](backflip/r5a_auto_v4/README.md)、[v5](backflip/r5a_auto_v5/README.md)、[v6](backflip/r5a_auto_v6/README.md) |
| 跳跃/高跳前置阶段 | [J1 说明](backflip/docs/R5A_START_HERE.md)、[高跳说明](backflip/docs/R5A_HIGHJUMP.md)、[高跳脚本](backflip/highjump/scripts) |
| 参考控制实验 | [教师搜索](backflip/r5a_teacher_search/README.md)、[起跳实验](backflip/r5a_takeoff_lab/README.md) |
| R5B | [侧空翻规划](sideflip/README.md) |

各 Auto/Guided 目录内的 `spec.py` 保存对应版本参数，`env.py`、`protocol.py` 及相关奖励模块保存实现，版本之间没有合并或覆盖。基础 J1 环境仍位于公共框架源码 `source/rl_training/.../multiskill/lighthw/r5a/` 和 `r5a_highjump/`，与其他路线的框架布局一致。

## 启动入口

在仓库根目录查看帮助，无需启动仿真：

```bash
R5_DIR=model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip
bash "$R5_DIR/scripts/r5a.sh" help
bash "$R5_DIR/scripts/r5a_auto_v6.sh" help
bash "$R5_DIR/scripts/r5a_guided_v5.sh" help
```

R5 专用入口均在 `backflip/scripts/`。根目录的 `r6.sh` 和 `offline/` 是这些入口复用的环境/依赖工具，R5 专用离线逻辑已收进 `backflip/offline/`。

本分支只保存文本，不包含旧版本说明中提到的 checkpoint、教师数据、mesh 或 ZIP。旧文档保留其研究背景；本次目录调整没有新增训练或行为验收。目录路径变化也会改变源码指纹，历史 checkpoint 的来源校验继续生效，不应直接跨目录版本续训。
