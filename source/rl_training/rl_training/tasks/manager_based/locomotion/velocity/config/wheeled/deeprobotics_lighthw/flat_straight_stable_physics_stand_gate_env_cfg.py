"""Separate zero-command holding from low-speed motion for the Contact policy."""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_straight_stable_physics_contact_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsContactEnvCfg,
)


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandGateEnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsContactEnvCfg
):
    """Change only the six standing reward gates; use the same config in play."""

    def __post_init__(self):
        super().__post_init__()

        # 本路线最小非零运动命令为 0.08 m/s。旧阈值 0.15 会在小速度
        # 命令下同时要求跟踪和静止，轮速惩罚尤其会处罚正常滚动。
        # 统一收窄至 0.05，保留零命令保持；不改变各项函数、权重和
        # 速度奖励 std=0.50。stand_still 的阈值 0.06 已低于 0.08。
        for name in (
            "stand_lin_vel_xy_l2",
            "stand_ang_vel_xy_l2",
            "stand_flat_orientation_l2",
            "stand_contact_force_uniformity",
            "stand_wheel_velocity_l2",
            "stand_leg_velocity_l2",
        ):
            getattr(self.rewards, name).params["command_threshold"] = 0.05

        print(
            "[LightHW Flat Straight Stable Physics StandGate] "
            "policy=60, critic=236, action=16; six_stand_gates=0.05, "
            f"minimum_abs_speed={self.commands.base_velocity.minimum_abs_speed:g}, "
            f"speed_reward_std={self.rewards.track_lin_vel_xy_exp.params['std']:.2f}; "
            "vx=(-0.4, 0.4), vy=0, wz=0, zero=25%; "
            "warm_start_source=Contact/model_499"
        )
