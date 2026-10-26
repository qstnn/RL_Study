# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""条件回合指标的完整日志适配；不改变 PPO 学习和环境采样。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from rsl_rl.runners import OnPolicyRunner


def complete_episode_metrics(
    ep_infos: Sequence[Mapping[str, object]], device: str | torch.device
) -> dict[str, torch.Tensor]:
    """按首次出现顺序合并全部指标键，缺失值不补零，保留原始已有样本的均值口径。

    RSL 的原日志仅遍历 ep_infos[0] 的键，导致后续回合才出现的静止条件指标遗漏。
    这里把每个键的实际已有值拼接为独立向量；设备和类型提升遵循原日志规则。
    输出张量重新分配存储，因此后续日志格式转换不会修改环境传入的数据。
    """
    grouped: dict[str, list[torch.Tensor]] = {}
    for ep_info in ep_infos:
        for key, value in ep_info.items():
            if isinstance(value, torch.Tensor):
                values = value.to(device).reshape(-1)
            else:
                # 原 runner 使用 torch.Tensor([value])，标量按照默认浮点类型记录。
                values = torch.tensor([value], device=device, dtype=torch.get_default_dtype()).reshape(-1)
            grouped.setdefault(key, []).append(values)
    return {
        # 保留原 runner 从空浮点张量开始 cat 的类型提升规则；cat 同时隔离输入存储。
        key: torch.cat([torch.tensor([], device=device), *values])
        for key, values in grouped.items()
    }


class CompleteEpisodeMetricsRunner(OnPolicyRunner):
    """仅修正日志键覆盖范围；网络、优化器、数据收集和检查点接口全部继承原 runner。"""

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # 不原地改写 learn() 的 locals 或 ep_infos，防止影响其清空及后续迭代。
        logging_locs = locs.copy()
        ep_infos = locs["ep_infos"]
        logging_locs["ep_infos"] = [complete_episode_metrics(ep_infos, self.device)] if ep_infos else []
        return super().log(logging_locs, width=width, pad=pad)
