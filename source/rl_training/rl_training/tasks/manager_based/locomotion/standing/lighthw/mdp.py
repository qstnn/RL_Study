"""站立专用观测与奖励；不导入 rough 课程或已有实验路线。"""

import torch

from .spec import JOINT_NAMES, WHEEL_JOINT_NAMES


def joint_positions(env, asset_cfg):
    """所有站立状态生效；返回按名称排列的16维相对角，连续轮角通道置零。

    轮速反馈仍在独立 joint_vel 观测中保留。索引先取实际关节，再在输出
    的局部通道屏蔽轮角，避免将物理索引误用于已重排的张量。
    """
    asset = env.scene[asset_cfg.name]
    result = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    for channel, name in enumerate(JOINT_NAMES):
        if name in WHEEL_JOINT_NAMES:
            result[:, channel] = 0.0
    return result


def linear_speed_l2(env):
    """抑制站立时平面漂移；全程生效，返回各环境 vx²+vy²，配负权重。"""
    return env.scene["robot"].data.root_lin_vel_b[:, :2].square().sum(dim=1)


def angular_speed_l2(env):
    """抑制机身旋转和摆动；全程生效，返回各环境三轴角速度平方和，配负权重。"""
    return env.scene["robot"].data.root_ang_vel_b.square().sum(dim=1)


def upright_error(env):
    """维持正立；全程生效，返回 (1+gravity_z)²，正立为0、倒置为4。

    使用竖直分量区分正立和倒置，避免只惩罚水平重力时倒置同样得零。
    """
    return (1.0 + env.scene["robot"].data.projected_gravity_b[:, 2]).square()


def wheel_support(env, sensor_cfg, threshold=5.0):
    """鼓励四轮承载；全程生效，返回竖直载荷大于阈值的轮数比例 [0,1]。"""
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces > threshold).to(torch.float32).mean(dim=1)
