"""Low-speed precision fine-tuning configuration for the LightHW flat route.

This task keeps the validated 60/236/16 observation/action interface and all
physics, reset, command, posture, contact, and termination settings from the
stable-physics route.  The only environment change is a narrower linear-speed
tracking kernel, so the experiment isolates velocity-tracking precision while
preserving the learned stance and wheel behavior from the continuation model.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_straight_stable_physics_env_cfg import (
    DeeproboticsLightHWFlatStraightStablePhysicsEnvCfg,
)


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsPrecisionEnvCfg(
    DeeproboticsLightHWFlatStraightStablePhysicsEnvCfg
):
    """Precision-only variant of the validated flat straight route."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the route-09 reward terms and upright gate unchanged.  A
        # narrower kernel makes errors around the +/-0.4 m/s target visible to
        # PPO without changing commands, contacts, actions, or observations.
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.50

        print(
            "[LightHW Flat Straight Stable Physics Precision] "
            "policy=60, critic=236, action=16; speed_reward_std=0.50, "
            "vx=(-0.4, 0.4), vy=0, wz=0, warm_start_from=model_1499"
        )
