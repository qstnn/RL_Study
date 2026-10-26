"""Route17唯一训练变量：按前进命令门控三项接触奖励，原函数和权重不变。"""
import torch

VERSION = 'lighthw_forward_contact_gate_v1'
THRESHOLD = 0.05
TERMS = ('wheel_contact_support', 'weakest_wheel_support', 'wheel_contact_uniformity')


def gate_forward_command(env, original):
    """目的：允许前进时载荷转移；仅命令vx>0.05激活；返回原项或逐环境零值。

    判据不读取实际速度；零、负向、恰好0.05命令均保留原函数值。
    原始函数仍执行，权重及dt由原RewardManager统一应用。
    """
    forward = env.command_manager.get_command('base_velocity')[:, 0] > THRESHOLD
    return torch.where(forward, torch.zeros_like(original), original)


def gated_wheel_contact_support(env, sensor_cfg, min_contact_force=5.0):
    """前进命令>0.05时返回0，否则返回原四轮承载比例；不含权重。"""
    from .adapted_rewards import lighthw_wheel_contact_support
    return gate_forward_command(env, lighthw_wheel_contact_support(env, sensor_cfg, min_contact_force))


def gated_weakest_wheel_support(env, sensor_cfg, threshold=5.0):
    """前进命令>0.05时返回0，否则返回原最弱轮支撑值；不含权重。"""
    from .ppo_action_curriculum import weakest_wheel_support
    return gate_forward_command(env, weakest_wheel_support(env, sensor_cfg, threshold))


def gated_wheel_contact_uniformity(env, sensor_cfg, min_contact_force=5.0, force_scale=8.0):
    """前进命令>0.05时返回0，否则返回原载荷方差/低载代价；不含权重。"""
    from .adapted_rewards import lighthw_wheel_contact_uniformity
    return gate_forward_command(env, lighthw_wheel_contact_uniformity(
        env, sensor_cfg, min_contact_force, force_scale))


def apply_variant(cfg, arm):
    """只替换三个函数引用；所有参数、权重、其他term与终止保持原配置。"""
    from .adapted_rewards import lighthw_wheel_contact_support, lighthw_wheel_contact_uniformity
    from .ppo_action_curriculum import weakest_wheel_support
    if arm not in ('control', 'gated'):
        raise ValueError('Unknown contact-gate arm')
    originals = (lighthw_wheel_contact_support, weakest_wheel_support, lighthw_wheel_contact_uniformity)
    wrappers = (gated_wheel_contact_support, gated_weakest_wheel_support, gated_wheel_contact_uniformity)
    for name, original, wrapper in zip(TERMS, originals, wrappers):
        term = getattr(cfg.rewards, name)
        if term.func is not original:
            raise ValueError('Expected original reference reward before intervention: ' + name)
        if arm == 'gated':
            term.func = wrapper
