"""独立 Route01 Stage1：复用 Rough 场景/本体信息，平地空载从零训练。"""

from copy import deepcopy
import math

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.rough_env_cfg import (
    DeeproboticsLightHWRoughEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.linear_leg_spec import LIMITS, SCALE
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.reference_observations import joint_positions_v1
from .catalog import JOINT_NAMES, LEG_NAMES, WHEEL_NAMES, WHEEL_BODIES
from . import rewards


@configclass
class WheelRewardsCfg:
    track_linear = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0,
                          params={"command_name": "base_velocity", "std": 0.35})
    track_yaw = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0,
                       params={"command_name": "base_velocity", "std": 0.5})
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.416})
    vertical_speed = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    roll_pitch_speed = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    leg_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5,
                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    stand_motion = RewTerm(func=rewards.stand_motion, weight=-5.0)
    stand_wheels = RewTerm(func=rewards.stand_wheel_speed, weight=-0.02,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(WHEEL_NAMES))})
    stand_pose = RewTerm(func=rewards.stand_leg_pose, weight=-0.5,
                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    stand_support = RewTerm(func=rewards.stand_support, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(WHEEL_BODIES), preserve_order=True)})


@configclass
class WheelEventsCfg:
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                          params={"pose_range": {}, "velocity_range": {}})
    reset_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})


@configclass
class WheelStage1EnvCfg(DeeproboticsLightHWRoughEnvCfg):
    def __post_init__(self):
        # 必须等父类完成 M20 schema 配置再替换奖励，避免父类访问不存在的字段。
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.physics_material.restitution = 0.0
        self.scene.terrain.visual_material = None
        self.scene.robot = deepcopy(self.scene.robot)
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
        self.scene.robot.actuators["joint"].stiffness = 80.0
        self.scene.robot.actuators["joint"].damping = 3.0
        self.scene.robot.actuators["wheel"].damping = 1.0
        self.sim.dt, self.decimation = 0.001, 20
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.history_length = 20
        for sensor in (self.scene.height_scanner, self.scene.height_scanner_base):
            sensor.update_period = self.sim.dt * self.decimation
        self.actions.joint_pos.scale = deepcopy(SCALE)
        self.actions.joint_pos.clip = deepcopy(LIMITS)
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_vel.clip = {".*_wheel_joint": (-37.0, 37.0)}
        self.actions.joint_vel.use_default_offset = False
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = joint_positions_v1
            group.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
                                      "wheel_joint_names": WHEEL_NAMES}
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(8.0, 8.0), rel_standing_envs=0.25,
            rel_heading_envs=0.0, heading_command=False, debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.2, 0.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        )
        self.rewards = WheelRewardsCfg()
        self.events = WheelEventsCfg()
        self.curriculum = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.bad_orientation_2 = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(45)})
        self.terminations.illegal_contact = DoneTerm(func=mdp.illegal_contact, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Base_link", ".*_(hip|thigh|calf)_link"]),
            "threshold": 1.0,
        })
        print("[MultiSkill Route01 S1] 57/236/16; plane, payload0, towing0; fresh PPO; no automatic promotion")
