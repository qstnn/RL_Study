"""Route01 站立专用奖励；纯张量函数方便验证门控，不导入仿真。"""


def standing_mask(env, command_name="base_velocity", threshold=0.05):
    """零指令门控：三轴指令范数小于阈值时返回逐环境布尔值。"""
    return env.command_manager.get_command(command_name).norm(dim=-1) < threshold


def stand_motion(env):
    """抑制停车后的机身漂移；仅零指令激活；返回平速平方加角速平方。"""
    data = env.scene["robot"].data
    return standing_mask(env) * (data.root_lin_vel_b[:, :2].square().sum(-1)
                                 + data.root_ang_vel_b.square().sum(-1))


def stand_wheel_speed(env, asset_cfg):
    """停车时抑制轮空转；仅零指令激活；返回四轮速度平方和。"""
    return standing_mask(env) * env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].square().sum(-1)


def stand_leg_pose(env, asset_cfg):
    """保持默认腿姿；仅零指令激活；返回腿关节相对默认角的平方和。"""
    data = env.scene[asset_cfg.name].data
    delta = data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]
    return standing_mask(env) * delta.square().sum(-1)


def stand_support(env, sensor_cfg, threshold=5.0):
    """鼓励停车时四轮承载；仅零指令激活；返回承载轮比例[0,1]。

    使用世界竖直力只适用于本阶段平地；运动时不惩罚动态载荷转移。
    """
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return standing_mask(env) * (forces > threshold).float().mean(-1)
