"""持续任务的末状态价值适配；不修改全局RSL，尚不提供正式训练入口。"""
import torch
from rsl_rl.algorithms import PPO


def timeout_rewards(rewards, dones, infos, value_fn, gamma):
    """目的：只对时间/场地截断续接价值，物理失败不续接，返回新奖励及清理后的infos。

    激活：env提供time_outs时；使用自动reset之前保存的critic观测。
    返回：奖励副本及去掉time_outs的字典，避免RSL自身再次进行旧值补偿。
    """
    info = dict(infos)
    truncated = info.pop('time_outs', torch.zeros_like(dones)).bool()
    terminated = infos.get('physical_terminated', torch.zeros_like(dones)).bool()
    mask = truncated & ~terminated
    result = rewards.clone()
    if mask.any():
        if not dones[mask].bool().all():
            raise ValueError('Timeout must end the current segment')
        valid = infos.get('terminal_observation_valid')
        if valid is None or not valid[mask].all():
            raise ValueError('Missing pre-reset terminal observation')
        with torch.no_grad():
            values = value_fn(infos['terminal_critic_observation'][mask]).reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError('Non-finite terminal value')
        result[mask] += gamma * values
    return result, info


class ContinuingPPO(PPO):
    def process_env_step(self, rewards, dones, infos):
        rewards, infos = timeout_rewards(rewards, dones, infos, self.policy.evaluate, self.gamma)
        return super().process_env_step(rewards, dones, infos)
