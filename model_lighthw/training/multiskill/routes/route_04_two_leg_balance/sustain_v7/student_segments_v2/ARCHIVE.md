# 2026-10-23 原始短段采集归档

此版本执行的 early/late 四条物理轨迹完整保留；报告阶段的轨迹文件匹配仍使用旧 `trace_seed*.npz`，没有匹配到新 `trace_early_seed43.npz` / `trace_late_seed43.npz`，导致独立验证查找哈希失败。原始 session 因而保留 `failed`，不能只依据 worker 退出码 0 宣称流水线完成。

当前入口为独立的 [student_segments_v2_1](../student_segments_v2_1/README.md)，修正为 `trace_*.npz`。本目录七份执行源码、原始轨迹、失败报告和源快照未修改。新版本只接受精确锁定的本次旧源码/合同；复核副本填充由原始字节计算的轨迹哈希后，重新通过物理、动作、来源和交接审查。

[复核过程](../student_segments_v2_1/runs/2026-10-23_short_segments_4_reverified/RECOVERY.md) · [本轮报告](../student_segments_v2_1/VALIDATION.md)。复核没有重新运行 GPU，不将原记录与副本计作两份独立实验。不要用本归档入口继续新的采集或训练。
