"""Isolate stronger whole-body zero-command holding from the StandGate route."""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_straight_stable_physics_stand_gate_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsStandGateEnvCfg,
)


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsStandHoldEnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsStandGateEnvCfg
):
    """StandGate with a stronger zero-command body-velocity penalty.

    This task keeps the six 0.05 standing gates and every other environment
    setting unchanged.  Only ``stand_lin_vel_xy_l2`` is strengthened so that
    motion after a stop is tested without changing the speed command range,
    default pose, observations, actions, or PPO algorithm.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.stand_lin_vel_xy_l2.weight = -10.0

        print(
            "[LightHW Flat Straight Stable Physics StandHold] "
            "policy=60, critic=236, action=16; six_stand_gates=0.05, "
            f"stand_lin_vel_xy_l2_weight={self.rewards.stand_lin_vel_xy_l2.weight:g}, "
            f"minimum_abs_speed={self.commands.base_velocity.minimum_abs_speed:g}, "
            f"speed_reward_std={self.rewards.track_lin_vel_xy_exp.params['std']:.2f}; "
            "vx=(-0.4, 0.4), vy=0, wz=0, zero=25%; "
            "warm_start_source=StandGate/model_499"
        )
