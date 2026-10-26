"""Route26：运动腿/轮加速度代价共用系数，保留原末端1ms加速度语义。"""
import torch
from .adapted_rewards import lighthw_joint_acc_l2


def moving_joint_acc_l2(env, asset_cfg, normalization=1.0, command_name="base_velocity",
                        command_threshold=0.05, moving_multiplier=0.05):
    """降低前进时的加速度代价；零命令及非前进条件保留原代价。

    激活条件只读取命令vx>0.05，不读取实速，不对vx取绝对值。
    返回每环境非负代价；RewardManager另乘原weight和step_dt。
    原函数按asset_cfg选腿或轮，normalization仅乘一次；不改变加速度
    采样/滤波，不将20ms窗口均方M替换训练使用的末端1ms样本E。
    """
    original = lighthw_joint_acc_l2(env, asset_cfg, normalization)
    forward = env.command_manager.get_command(command_name)[:, 0] > command_threshold
    return torch.where(forward, original * moving_multiplier, original)
