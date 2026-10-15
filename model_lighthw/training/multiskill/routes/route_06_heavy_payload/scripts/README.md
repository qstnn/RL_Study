# R6运行脚本

`train.py`、`play.py`、`evaluate.py`调用本目录独立`launch.py`；`worker.py`在AppLauncher后创建环境。
`training.py`管理RSL-RL保存/恢复与CSV；`metrics.py`管理固定持载筛查。
统一入口`multiskill/run.py ... --route 06`会分发到同一R6启动器。
可复制命令和参数见[路线README](../README.md)。
