"""同一个部署MLP的条件探索；没有额外Actor、示范控制或特权观测。"""
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


class MotionExplorationActorCritic(ActorCritic):
    def std_for_observations(self, observations):
        """只用当前速度指令和名义拖车任务信息调整探索。

        零指令/非拖载使用基础std的1/8；拖载指令达到.05m/s时用完整std。
        当前速度指令列42的观测scale在GPU启动时验证为1，拖车列299无噪声。
        概率、熵、PPO ratio和整rollout KL均使用同一标准差，部署只导出均值。
        """
        moving=(observations[...,42].abs()/.05).clamp(0.,1.)
        towing=(observations[...,299]>.01).to(observations.dtype)
        factor=.125+.875*moving*towing
        return self.log_std.exp()*factor.unsqueeze(-1)

    def update_distribution(self, observations):
        self.distribution=Normal(self.actor(observations),self.std_for_observations(observations))
