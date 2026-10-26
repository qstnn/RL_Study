"""Fresh flat LightHW baseline with a stable 1 ms physics step.

This route is intentionally independent from route 08.  It keeps the
policy-blind / critic-privileged 60/236 observation layout and 16 actions, but
starts PPO from the default 0.416 m pose instead of inheriting the saturated
route-08 actor.  The first lesson is straight rolling at +/-0.4 m/s with a
large zero-command slice.  Physics runs at 1 kHz while policy actions remain
at 50 Hz.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

from .adapted_rewards import (
    lighthw_joint_action_saturation_fraction,
    lighthw_joint_action_saturation_l1,
    lighthw_joint_pair_symmetry_l2,
    lighthw_straight_yaw_rate_l2,
    lighthw_upright_orientation_l2,
    lighthw_wheel_contact_support,
    lighthw_wheel_contact_uniformity,
)
from .default_pose import (
    LIGHTHW_DEFAULT_BASE_HEIGHT,
    LIGHTHW_DEFAULT_JOINT_POS,
    LIGHTHW_DEFAULT_SPAWN_HEIGHT,
)
from .flat_straight_recovery_env_cfg import (
    DeeproboticsLightHWFlatStraightRecoveryEnvCfg,
)
from .progressive_commands import ProgressiveLightHWVelocityCommandCfg
from .rough_env_cfg import (
    DeeproboticsLightHWRewardsCfg,
    HIP_JOINT_NAMES,
    LIGHTHW_SYMMETRY_PAIRS,
)
from .stable_roll_v3_rewards import (
    stand_upright_orientation_l2,
    track_ang_vel_z_upright_exp,
    track_lin_vel_xy_upright_exp,
)


@configclass
class LightHWFlatStraightStablePhysicsCommandsCfg(CommandsCfg):
    """The first fresh-start lesson: straight +/-0.4 m/s and 25% standing."""

    base_velocity = ProgressiveLightHWVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        lateral_range=(0.0, 0.0),
        yaw_range=(0.0, 0.0),
        stage_start_steps=(0,),
        stage_max_speeds=(0.40,),
        zero_command_probability=0.25,
        minimum_abs_speed=0.08,
        anchor_probability=0.50,
        ranges=ProgressiveLightHWVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.40, 0.40),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class DeeproboticsLightHWStablePhysicsRewardsCfg(DeeproboticsLightHWRewardsCfg):
    """Reward fields owned only by the fresh stable-physics route."""

    # Non-zero defaults keep these terms alive through the inherited
    # ``disable_zero_weight_rewards`` pass.  The route sets the final weights
    # and selectors in its post-init method.
    joint_action_saturation_l1 = RewTerm(
        func=lighthw_joint_action_saturation_l1,
        weight=-0.01,
        params={
            "action_term_name": "joint_pos",
            "joint_names": HIP_JOINT_NAMES,
            "raw_limit": 3.0,
        },
    )
    joint_action_saturation_fraction = RewTerm(
        func=lighthw_joint_action_saturation_fraction,
        weight=-0.10,
        params={
            "action_term_name": "joint_pos",
            "joint_names": HIP_JOINT_NAMES,
            "raw_limit": 3.0,
        },
    )


@configclass
class DeeproboticsLightHWFlatStraightStablePhysicsEnvCfg(
    DeeproboticsLightHWFlatStraightRecoveryEnvCfg
):
    """Fresh-start, plane-only, 1 kHz-physics LightHW baseline."""

    commands: LightHWFlatStraightStablePhysicsCommandsCfg = LightHWFlatStraightStablePhysicsCommandsCfg()
    rewards: DeeproboticsLightHWStablePhysicsRewardsCfg = DeeproboticsLightHWStablePhysicsRewardsCfg()

    def __post_init__(self):
        # The recovery parent supplies the LightHW asset, M20-compatible
        # reward schema, wheel support terms, and the 60/236/16 action layout.
        super().__post_init__()

        # ------------------------- Pose and simulation --------------------
        self.scene.robot.init_state.pos = (0.0, 0.0, LIGHTHW_DEFAULT_SPAWN_HEIGHT)
        self.scene.robot.init_state.joint_pos = dict(LIGHTHW_DEFAULT_JOINT_POS)
        self.rewards.base_height_l2.params["target_height"] = LIGHTHW_DEFAULT_BASE_HEIGHT

        # Keep policy/control timing at 50 Hz, but integrate the explicit PD
        # at 1 kHz.  This is the only simulation time-step change in this
        # route; legacy tasks retain their existing 5 ms settings.
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.history_length = 15
        for sensor in (self.scene.height_scanner, self.scene.height_scanner_base):
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt

        # Do not mix the first dynamics baseline with random delay or domain
        # randomization.  These are reintroduced only after fixed-command
        # playback passes posture/contact gates.  The two reset events are
        # deliberately handled below: a zero-range reset is still required
        # to write the default root/joint state back to PhysX at timeout.
        for event_name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_inertia",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "randomize_actuator_gains",
            "randomize_push_robot",
        ):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)
        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {},
                "velocity_range": {},
            },
        )
        self.events.randomize_reset_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.scene.robot.actuators["joint"].min_delay = 0
        self.scene.robot.actuators["joint"].max_delay = 0
        self.scene.robot.actuators["wheel"].min_delay = 0
        self.scene.robot.actuators["wheel"].max_delay = 0

        # A pure plane keeps the first run's contact and height measurements
        # interpretable.  The critic height scan stays present, so its input
        # remains 236 dimensions and train/play use the same task.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.gait_level = None
        self.observations.policy.height_scan = None
        if self.observations.critic.height_scan is None:
            raise ValueError("Stable physics route requires the critic height_scan term.")

        # ----------------------------- Commands ---------------------------
        command = self.commands.base_velocity
        command.stage_start_steps = (0,)
        command.stage_max_speeds = (0.40,)
        command.zero_command_probability = 0.25
        command.minimum_abs_speed = 0.08
        command.anchor_probability = 0.50
        command.resampling_time_range = (8.0, 8.0)
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.heading_command = False
        command.lateral_range = (0.0, 0.0)
        command.yaw_range = (0.0, 0.0)
        command.ranges.lin_vel_x = (-0.40, 0.40)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = None

        # ------------------------------ Actions ---------------------------
        # A +/-0.36 rad physical hip target leaves a margin below the URDF
        # hard limit.  The separate raw-action terms start penalizing at 3.0
        # (0.24 rad with scale .08), before this physical clip is reached.
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.08,
            ".*_(thigh|calf)_joint": 0.14,
        }
        self.actions.joint_pos.clip[".*_hip_joint"] = (-0.36, 0.36)
        self.actions.joint_vel.scale = 30.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-30.0, 30.0)}
        self.scene.robot.actuators["joint"].stiffness = 80.0
        self.scene.robot.actuators["joint"].damping = 3.0
        self.scene.robot.actuators["wheel"].damping = 1.0

        # ------------------------------ Rewards ---------------------------
        # Gate velocity rewards by upright alignment so an inverted policy
        # cannot keep collecting speed reward.  The standard deviation stays
        # broad enough for the first +/-0.4 m/s lesson.
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_upright_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.75
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_upright_exp
        self.rewards.base_height_l2.weight = -3.0
        self.rewards.upward = RewTerm(
            func=lighthw_upright_orientation_l2,
            weight=-0.10,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["Base_link"])},
        )
        self.rewards.stand_flat_orientation_l2.func = stand_upright_orientation_l2
        self.rewards.stand_flat_orientation_l2.weight = -0.50
        self.rewards.joint_pos_limits.weight = -6.0
        self.rewards.hipx_joint_pos_penalty.weight = -0.50
        self.rewards.joint_pair_symmetry_l2 = RewTerm(
            func=lighthw_joint_pair_symmetry_l2,
            weight=-0.08,
            params={
                "joint_pairs": LIGHTHW_SYMMETRY_PAIRS,
                "asset_cfg": SceneEntityCfg("robot"),
                "use_default_offset": True,
            },
        )
        self.rewards.wheel_contact_support.func = lighthw_wheel_contact_support
        self.rewards.wheel_contact_support.weight = 0.45
        self.rewards.wheel_contact_uniformity.func = lighthw_wheel_contact_uniformity
        self.rewards.wheel_contact_uniformity.weight = -0.03
        self.rewards.wheel_contact_uniformity.params["force_scale"] = 8.0
        self.rewards.stand_contact_force_uniformity.weight = -0.03
        self.rewards.straight_yaw_rate_l2 = RewTerm(
            func=lighthw_straight_yaw_rate_l2,
            weight=-0.12,
            params={
                "command_name": "base_velocity",
                "min_abs_lin_vel_x": 0.08,
                "max_abs_lin_vel_y": 0.03,
                "max_abs_ang_vel_z": 0.03,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.joint_action_saturation_l1.weight = -0.01
        self.rewards.joint_action_saturation_fraction.weight = -0.10

        # Turn on a conservative fall/base-contact termination only for this
        # new route.  The old LightHW tasks, including route 08, remain
        # unchanged and keep their user-requested termination settings.
        self.terminations.bad_orientation_2 = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.pi / 3.0, "asset_cfg": SceneEntityCfg("robot")},
            time_out=False,
        )
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Base_link"]),
                "threshold": 1.0,
            },
            time_out=False,
        )

        print(
            "[LightHW Flat Straight Stable Physics] policy=60, critic=236, action=16; "
            "physics=1000 Hz, policy=50 Hz, vx=(-0.4, 0.4), vy=0, wz=0, zero=25%, fresh_start"
        )
