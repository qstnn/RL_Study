"""LightHW high-stance baseline with validated physics timing and fall handling."""
from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .stable_roll_env_cfg import DeeproboticsLightHWStableRollEnvCfg
from .default_pose_0545 import LIGHTHW_0545_BASE_HEIGHT, LIGHTHW_0545_JOINT_POS, LIGHTHW_0545_SPAWN_HEIGHT
from .stable_roll_v3_rewards import (
    stand_upright_orientation_l2,
    track_ang_vel_z_upright_exp,
    track_lin_vel_xy_upright_exp,
)


@configclass
class DeeproboticsLightHWStableRollV3EnvCfg(DeeproboticsLightHWStableRollEnvCfg):
    """Keep the StableRoll policy interface, asset, gains and command envelope."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the already-trained V3 task reproducible after the shared
        # LightHW route is restored to the original 0.416 m low-speed pose.
        self.scene.robot.init_state.pos = (0.0, 0.0, LIGHTHW_0545_SPAWN_HEIGHT)
        self.scene.robot.init_state.joint_pos = dict(LIGHTHW_0545_JOINT_POS)
        self.rewards.base_height_l2.params["target_height"] = LIGHTHW_0545_BASE_HEIGHT

        # Explicit PD was unstable at 5 ms. Preserve the 50 Hz policy rate and
        # the previous 15 ms contact-history window when using 1 ms physics.
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.history_length = 15
        for sensor in (self.scene.height_scanner, self.scene.height_scanner_base):
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt

        # Keep native gains/limits and the validated 0--1 physics-step target
        # delay. At 1 kHz this now represents 0--1 ms, rather than 0--5 ms.
        self.terminations.bad_orientation_2 = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.pi / 3, "asset_cfg": SceneEntityCfg("robot")},
            time_out=False,
        )
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Base_link"]), "threshold": 1.0},
            time_out=False,
        )
        # Leave every weight and selector intact; these V3 functions close the
        # orientation loopholes without changing M20 or legacy LightHW tasks.
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_upright_exp
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_upright_exp
        self.rewards.stand_flat_orientation_l2.func = stand_upright_orientation_l2

        print("[LightHW StableRoll V3] policy=60, critic=236, action=16; physics=1000 Hz, policy=50 Hz")
        print("[LightHW StableRoll V3] target_base_height=0.545, fail on tilt > 60 deg or Base_link contact > 1 N")
