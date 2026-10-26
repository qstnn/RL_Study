"""完整rough的必要命令门控适配：保持原函数、数值权重与加速度采样。"""
from .omni_rough_core import gated_contact, acceleration_factor
from .adapted_rewards import (
    lighthw_wheel_contact_support, lighthw_wheel_contact_uniformity, lighthw_joint_acc_l2,
)
from .ppo_action_curriculum import weakest_wheel_support


def moving_contact_support(env, sensor_cfg, min_contact_force=5.):
    """目的：允许任意方向运动时卸载；命令范数>0.05返回零，否则返回原支撑奖励。"""
    original=lighthw_wheel_contact_support(env,sensor_cfg,min_contact_force)
    return gated_contact(original,env.command_manager.get_command('base_velocity'))


def moving_weakest_support(env, sensor_cfg, threshold=5.):
    """目的：停车保持最弱轮支撑；任意运动命令关闭，返回不含权重的原值或零。"""
    return gated_contact(weakest_wheel_support(env,sensor_cfg,threshold),
                         env.command_manager.get_command('base_velocity'))


def moving_contact_uniformity(env,sensor_cfg,min_contact_force=5.,force_scale=8.):
    """目的：运动允许载荷转移；任意运动命令关闭均载项，零命令保持原代价。"""
    original=lighthw_wheel_contact_uniformity(env,sensor_cfg,min_contact_force,force_scale)
    return gated_contact(original,env.command_manager.get_command('base_velocity'))


def moving_joint_acc_l2(env,asset_cfg,normalization=1.,command_name='base_velocity',
                        command_threshold=.05,moving_multiplier=.05):
    """目的：倒退/侧移/转向共用旧前进系数；返回原末端1ms加速度代价乘门控系数。

    normalization只由原函数乘一次。保留零命令系数1、运动系数0.05；
    不滤波、不用20ms速度差分或窗口均方替换训练项，不含reward weight。
    """
    original=lighthw_joint_acc_l2(env,asset_cfg,normalization)
    return original*acceleration_factor(env.command_manager.get_command(command_name),
                                        command_threshold,moving_multiplier)
