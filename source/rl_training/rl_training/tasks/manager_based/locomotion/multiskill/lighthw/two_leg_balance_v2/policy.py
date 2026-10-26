"""初始/保持/恢复三个静止阶段降低采样标准差；PPO概率/熵/KL都使用同一条件分布。"""
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


class PhaseActorCritic(ActorCritic):
    def __init__(self, *args, entry_noise_scale=.03, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_noise_scale = float(entry_noise_scale)
        if not 0. < self.entry_noise_scale <= 1.:
            raise ValueError("entry_noise_scale must be in (0, 1]")

    def update_distribution(self, observations):
        mean = self.actor(observations)
        # 最后61维帧的55/57/59号为站稳/保持/恢复，运动阶段1/3保留完整探索。
        # 均值不变、不缩放已采样动作；行为分布与PPO重算密度完全一致。
        quiet_phase = observations[..., [-6, -4, -2]].sum(dim=-1, keepdim=True) > .5
        scale = torch.where(quiet_phase, self.entry_noise_scale, 1.)
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean) * scale)
