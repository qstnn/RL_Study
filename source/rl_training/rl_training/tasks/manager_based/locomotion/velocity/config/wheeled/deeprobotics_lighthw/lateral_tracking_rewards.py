"""独立侧向奖励宽度实验，全部其它奖励继续继承原任务。"""
import torch
from .spin_xy_rewards import track_lin_vel_xy_spin_exp
from .lateral_tracking_core import lateral_command_mask, lateral_tracking_value


def track_lin_vel_xy_lateral_exp(env, std, command_name, spin_std,
                                linear_command_threshold=0.05, yaw_command_threshold=0.05,
                                lateral_std=0.5, lateral_command_threshold=0.05):
    """目的：加强侧移与组合运动的vy跟踪，不新增奖励项或动作。

    激活：仅|命令vy|>0.05，与实际车速无关。此时x宽度仍为std=1.5，
    y宽度为lateral_std=0.5；其余命令逐值返回旧XY奖励（含旧spin规则）。
    返回：未乘weight/dt的标量；原upright因子、weight=5由管理器保留。
    """
    original = track_lin_vel_xy_spin_exp(env, std, command_name, spin_std,
                                        linear_command_threshold, yaw_command_threshold)
    if lateral_std == std:
        return original
    command = env.command_manager.get_command(command_name)
    data = env.scene['robot'].data
    lateral = lateral_tracking_value(command, data.root_lin_vel_b, data.projected_gravity_b,
                                     std, lateral_std)
    return torch.where(lateral_command_mask(command, lateral_command_threshold), lateral, original)
