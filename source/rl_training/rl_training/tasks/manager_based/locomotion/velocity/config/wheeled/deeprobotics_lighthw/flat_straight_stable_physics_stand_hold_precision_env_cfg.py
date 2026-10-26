"""Isolate low-speed velocity precision from the StandHold policy."""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_straight_stable_physics_stand_hold_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldEnvCfg,
)


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandHoldPrecisionEnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandHoldEnvCfg
):
    """StandHold with a narrower upright linear-velocity tracking kernel.

    The -10 zero-command body-velocity penalty, 0.05 standing gates, command
    distribution, robot, observations, actions, and simulation are inherited
    unchanged.  Only the tracking reward ``std`` changes from 0.50 to 0.35.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35

        print(
            "[LightHW Flat Straight Stable Physics StandHoldPrecision] "
            "policy=60, critic=236, action=16; six_stand_gates=0.05, "
            f"stand_lin_vel_xy_l2_weight={self.rewards.stand_lin_vel_xy_l2.weight:g}, "
            f"speed_reward_std={self.rewards.track_lin_vel_xy_exp.params['std']:.2f}; "
            "vx=(-0.4, 0.4), vy=0, wz=0, zero=25%; "
            "warm_start_source=StandHold/model_499"
        )
