# 资产来源记录

| 资产 | 来源 | 去向 | 处理 |
|---|---|---|---|
| `lighthw_urdf` | 原 `deep_robotics_model/lighthw/lighthw_urdf` | `model_lighthw/lighthw_urdf` | 已整理为唯一 canonical 副本 |
| `LIGHTHW_ROUTE_CFG` | `source/rl_training/rl_training/assets/deeprobotics.py` | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_lighthw/asset_cfg.py` | 指向唯一 canonical URDF，保留质量、惯量和执行器参数 |
