# R4入口

`r4.sh`绑定本机lab/Isaac安装后运行`run.py`；支持check/smoke/train/evaluate/play。
`worker.py`只执行请求中的一个有界段；`geometry.py`计算独立静态候选；`verify_trace.py`复算1kHz冻结轨迹。
每个训练段均有来源、运行时核查、模型/Adam和日志；无其他路线模型自动搜索。
完整训练命令及恢复规则见[路线说明](../README.md)。
