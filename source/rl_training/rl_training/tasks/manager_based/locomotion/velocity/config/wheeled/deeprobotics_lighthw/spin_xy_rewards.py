"""原地转向奖励宽度对照，复用原 XY 公式，不增加奖励项或改变控制。"""
import torch
from .stable_roll_v3_rewards import track_lin_vel_xy_upright_exp


def spin_command_mask(command, linear_command_threshold=0.05, yaw_command_threshold=0.05):
    """仅按命令判定原地转向：平面范数≤阈值且偏航绝对值>阈值，返回布尔掩码。"""
    return ((command[:, :2].norm(dim=1) <= linear_command_threshold)
            & (command[:, 2].abs() > yaw_command_threshold))


def track_lin_vel_xy_spin_exp(env, std, command_name, spin_std,
                              linear_command_threshold=0.05, yaw_command_threshold=0.05):
    """抑制原地转向的伴随平移；返回未加权、未乘 dt 的原 XY 跟踪奖励。

    对照组 spin_std=std，逐位复用原奖励。实验组只对原地转向命令收紧核宽度；
    零命令、平移与行进转弯保持原公式。判定不读实际速度，upright 因子不变。
    奖励管理器负责原 weight=5 和物理控制周期缩放。
    """
    wide = track_lin_vel_xy_upright_exp(env, std, command_name)
    if spin_std == std:
        return wide
    command = env.command_manager.get_command(command_name)
    mask = spin_command_mask(command, linear_command_threshold, yaw_command_threshold)
    narrow = track_lin_vel_xy_upright_exp(env, spin_std, command_name)
    return torch.where(mask, narrow, wide)
