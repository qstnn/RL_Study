"""LightHW low-speed experiment with the repeated-touchdown reward removed."""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_straight_stable_physics_precision_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsPrecisionEnvCfg,
)


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsContactEnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsPrecisionEnvCfg
):
    """Isolate removal of the reward that encourages repeated wheel contact.

    Train and play both use this configuration. The precision task retains its
    original reward for comparison; all other environment settings are inherited.
    """

    def __post_init__(self):
        super().__post_init__()

        # 原奖励统计最近一个控制步内的首次触地，且在 |command| < 0.5
        # 时激活，覆盖本路线全部 ±0.4 m/s 命令。1 kHz 回放显示策略会
        # 通过短暂卸载、重新触地反复得分。仅在新任务中移除该项；持续
        # 支撑仍由 wheel_contact_support 提供，保留原速度核和其他奖励。
        self.rewards.feet_contact_without_cmd = None

        print(
            "[LightHW Flat Straight Stable Physics Contact] "
            "policy=60, critic=236, action=16; "
            f"speed_reward_std={self.rewards.track_lin_vel_xy_exp.params['std']:.2f}, "
            "feet_contact_without_cmd=disabled, "
            f"wheel_contact_support={self.rewards.wheel_contact_support.weight:g}; "
            "vx=(-0.4, 0.4), vy=0, wz=0, zero=25%"
        )
