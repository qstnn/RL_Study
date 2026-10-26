"""Isolate the next straight-line speed step from StandHoldPrecision."""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_straight_stable_physics_stand_hold_precision_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionEnvCfg,
)


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionVx06EnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionEnvCfg
):
    """StandHoldPrecision with only the straight ``vx`` envelope raised to 0.6 m/s.

    The 0.35 tracking-reward kernel, -10 zero-command body-velocity penalty,
    0.05 standing gates, zero-command ratio, pose, robot, observations,
    actions, physics and PPO settings are inherited unchanged.  This task is
    intentionally separate so the validated +/-0.4 task remains available for
    comparison.
    """

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.stage_start_steps = (0,)
        command.stage_max_speeds = (0.60,)
        command.ranges.lin_vel_x = (-0.60, 0.60)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)

        print(
            "[LightHW Flat Straight Stable Physics StandHoldPrecisionVx06] "
            "policy=60, critic=236, action=16; six_stand_gates=0.05, "
            f"stand_lin_vel_xy_l2_weight={self.rewards.stand_lin_vel_xy_l2.weight:g}, "
            f"speed_reward_std={self.rewards.track_lin_vel_xy_exp.params['std']:.2f}; "
            "vx=(-0.6, 0.6), vy=0, wz=0, zero=25%; "
            "warm_start_source=StandHoldPrecision/model_300"
        )
