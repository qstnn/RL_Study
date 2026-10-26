"""Route 15：从通用环境直接定义的 LightHW 全 PPO 平地起停基线。"""

import math
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .asset_cfg import LIGHTHW_ROUTE_CFG
from .rough_env_cfg import ARTICULATION_JOINT_NAMES, LEG_JOINT_NAMES, WHEEL_JOINT_NAMES
from . import roll_stop_mdp as local


@configclass
class RollStopActionsCfg:
    joint_pos = local.RollStopLegActionCfg(asset_name="robot", joint_names=LEG_JOINT_NAMES,
        preserve_order=True, use_default_offset=True, scale=1.0)
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEEL_JOINT_NAMES,
        preserve_order=True, use_default_offset=False, scale=5.0, clip={".*": (-37.0, 37.0)})


@configclass
class RollStopCommandsCfg:
    base_velocity = local.RollStopCommandCfg(asset_name="robot", resampling_time_range=(1e9, 1e9),
        heading_command=False, rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
        ranges=local.RollStopCommandCfg.Ranges(lin_vel_x=(0.0, 0.4), lin_vel_y=(0.0, 0.0),
                                              ang_vel_z=(0.0, 0.0), heading=None))


@configclass
class RollStopRewardsCfg:
    # 明确列出全部11项；不继承历史35项奖励，没有高度/镜像/前后均载叠加。
    track_velocity = RewTerm(func=local.velocity_tracking, weight=3.0)
    body_motion = RewTerm(func=local.body_motion, weight=-1.0)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    nonwheel_contact = RewTerm(func=mdp.undesired_contacts, weight=-3.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["Base_link", ".*_(hip|thigh|calf)_link"])})
    wheel_support = RewTerm(func=local.wheel_support, weight=0.30,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link")})
    leg_posture = RewTerm(func=local.leg_regularization, weight=-0.02)
    action_smoothness = RewTerm(func=local.action_smoothness, weight=-0.01)
    leg_effort = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)})
    stop_motion = RewTerm(func=local.stop_motion, weight=-2.0)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)})
    roll_stop_metrics = RewTerm(func=local.metrics, weight=1.0)


@configclass
class RollStopEventsCfg:
    randomize_reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}})
    randomize_reset_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})


@configclass
class RollStopTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Base_link"), "threshold": 1.0})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 3})


@configclass
class RollStopCurriculumCfg:
    terrain_levels = None
    command_levels = None
    gait_level = None


@configclass
class LightHWRollStopEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: RollStopActionsCfg = RollStopActionsCfg()
    commands: RollStopCommandsCfg = RollStopCommandsCfg()
    rewards: RollStopRewardsCfg = RollStopRewardsCfg()
    events: RollStopEventsCfg = RollStopEventsCfg()
    terminations: RollStopTerminationsCfg = RollStopTerminationsCfg()
    curriculum: RollStopCurriculumCfg = RollStopCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = LIGHTHW_ROUTE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["joint"].damping = 3.0
        self.scene.robot.actuators["wheel"].damping = 1.0
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
        self.sim.dt, self.decimation = 0.001, 20
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0
        self.scene.num_envs = 1024
        self.scene.env_spacing = 5.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.visual_material = None
        self.scene.contact_forces.update_period = 0.001
        self.scene.contact_forces.history_length = 20
        for scanner in (self.scene.height_scanner, self.scene.height_scanner_base):
            scanner.prim_path = "{ENV_REGEX_NS}/Robot/Base_link"
            scanner.update_period = 0.02
        self.scene.height_scanner.pattern_cfg.resolution = 0.08
        self.scene.height_scanner.pattern_cfg.size = [1.2, 0.8]
        self.scene.height_scanner_base.pattern_cfg.resolution = 0.04
        self.scene.height_scanner_base.pattern_cfg.size = (0.12, 0.12)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = mdp.joint_pos_rel_without_wheel
            group.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot",
                joint_names=ARTICULATION_JOINT_NAMES, preserve_order=True)
            group.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot",
                joint_names=WHEEL_JOINT_NAMES, preserve_order=True)
            group.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot",
                joint_names=ARTICULATION_JOINT_NAMES, preserve_order=True)
            group.base_lin_vel.scale = 2.0
            group.base_ang_vel.scale = 0.5
            group.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None
        print("[LightHW RollStop] policy=60, critic=236, action=16; PPO from scratch; "
              "original pose; flat; 25% stop per full cycle; no height target; "
              "linear bounded legs; native PPO controls all 16 actions.")
