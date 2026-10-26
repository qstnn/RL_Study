# R4冻结评估

每100更新单独进程冻结均值Actor，默认seed43/44/45、前后两种模式，共6例。
输出 `evaluation.json` 和1kHz原始物理轨迹、50Hz输入/动作；前后模式分开报告，reward不作为成功判据。

严格判据见[路线说明](../README.md)。复算：`lab/bin/python scripts/verify_trace.py <eval目录>`。
首次筛查后至少前后各20例、≥3seed、各≥95%完整成功再讨论长保持；没有自动晋级。
