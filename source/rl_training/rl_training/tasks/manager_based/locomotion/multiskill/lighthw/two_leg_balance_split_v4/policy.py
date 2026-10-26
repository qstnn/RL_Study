"""初始/恢复低探索，保持段保留更强探索；PPO概率/熵/KL都使用同一条件分布。"""
import math
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


class PhaseActorCritic(ActorCritic):
    def __init__(self, *args, entry_noise_scale=.03, hold_noise_scale=.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_noise_scale = float(entry_noise_scale)
        self.hold_noise_scale = float(hold_noise_scale)
        if not 0. < self.hold_noise_scale <= 1.:
            raise ValueError("hold_noise_scale must be in (0, 1]")
        if not 0. < self.entry_noise_scale <= 1.:
            raise ValueError("entry_noise_scale must be in (0, 1]")

    def update_distribution(self, observations):
        mean = self.actor(observations)
        # 最后61维帧的55/57/59号为站稳/保持/恢复；保持段不再过早冻结探索。
        # 均值不变、不缩放已采样动作；行为分布与PPO重算密度完全一致。
        quiet_phase = observations[..., [-6, -2]].sum(dim=-1, keepdim=True) > .5
        scale = torch.where(quiet_phase, self.entry_noise_scale, 1.)
        frame = observations[..., -61:]
        angle = torch.acos((frame[..., 3:6] * frame[..., 52:55]).sum(-1, keepdim=True).clamp(-1., 1.))
        angular_speed = (frame[..., :3] / .2).norm(dim=-1, keepdim=True)
        # 只使用Actor自身的IMU和任务目标；接近目标且角速度小时减小探索，便于连续保持。
        # 不读接触/COM/成功标签，不改均值；PPO重算时条件由保存的同一观测确定。
        near_target = (angle <= math.radians(12.)) & (angular_speed <= .25)
        hold_scale = torch.where(near_target, self.entry_noise_scale, self.hold_noise_scale)
        scale = torch.where(frame[..., 57:58] > .5, hold_scale, scale)
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean) * scale)
