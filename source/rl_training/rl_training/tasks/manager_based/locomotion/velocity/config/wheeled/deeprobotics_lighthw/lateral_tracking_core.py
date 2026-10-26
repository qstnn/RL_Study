"""侧向跟踪核的纯张量定义；不读取实际速度来选择命令门控。"""
import torch
VERSION = 'lighthw_commanded_lateral_width_v1'


def lateral_command_mask(command, threshold=0.05):
    """目的：选择左右侧移/含侧移的组合命令；|vy_cmd|>阈值时返回True。"""
    return command[:, 1].abs() > threshold


def lateral_tracking_value(command, actual, gravity, std, lateral_std):
    """目的：保持x误差尺度，只调整y；返回未乘weight/dt的姿态加权指数奖励。

    调用者仅在侧移命令激活时选此值。无离地/步态/足端示范或目标控制。
    """
    error = command[:, :2] - actual[:, :2]
    exponent = error[:, 0].square() / std**2 + error[:, 1].square() / lateral_std**2
    upright = (-gravity[:, 2]).clamp(0., 1.)
    return torch.exp(-exponent) * upright
