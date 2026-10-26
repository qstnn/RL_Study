"""R6 载荷条件与持载代价；纯张量函数，不导入仿真。"""


def payload_observation(env):
    """向策略提供已知装载质量；全时激活；返回(num_envs,1)，以65kg归一化。

    这是装载设定值，不是宣称已有真实载荷估计器，也不向Actor提供速度真值。
    """
    import torch
    return torch.full((env.num_envs, 1), float(env.cfg.payload_mass_kg)/65.0, device=env.device)


def torque_margin_cost(env, asset_cfg, margin=0.8):
    """抑制接近模型力矩限幅；持载全程激活；返回逐环境归一化超额平方和。

    模型effort限制不是电机持续热额定值；仅作为PPO饱和代价。
    """
    data = env.scene[asset_cfg.name].data
    ratio = data.applied_torque[:, asset_cfg.joint_ids].abs()/data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1e-6)
    return (ratio-margin).clamp_min(0.).square().mean(-1)


def load_balance_cost(env, sensor_cfg):
    """减少单轮卸载和过度集中承载；持载全程激活；返回四轮竖直力份额方差。

    先按总竖直力归一化，避免增加载荷时奖励量级随质量平方变化。
    """
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp_min(0.)
    share = force / force.sum(-1, keepdim=True).clamp_min(1.)
    return (share-.25).square().sum(-1)
