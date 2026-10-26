"""LightHW 平地零命令站立环境；直接使用 ManagerBasedRLEnvCfg。"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp as stand_mdp
from .asset_cfg import LIGHTHW_STAND_CFG
from .spec import BASE_HEIGHT, JOINT_NAMES, LEG_JOINT_NAMES, LEG_TARGET_LIMITS, WHEEL_BODY_NAMES, WHEEL_JOINT_NAMES


@configclass
class StandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
    )
    robot: ArticulationCfg = LIGHTHW_STAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.001, history_length=20, track_air_time=True,
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class StandCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0, rel_heading_envs=0.0, heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class StandActionsCfg:
    # 原生动作的 clip 在默认角偏置之后执行，因此这里给绝对关节角边界。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(LEG_JOINT_NAMES), preserve_order=True,
        scale={".*_hip_joint": 0.08, ".*_(thigh|calf)_joint": 0.14},
        use_default_offset=True, clip=dict(LEG_TARGET_LIMITS),
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=list(WHEEL_JOINT_NAMES), preserve_order=True,
        scale=5.0, use_default_offset=False, clip={".*_wheel_joint": (-37.0, 37.0)},
    )


@configclass
class StandObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=stand_mdp.joint_positions, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
        })
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
        })
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class StandEventsCfg:
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {}, "velocity_range": {},
    })
    reset_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset", params={
        "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
    })


@configclass
class StandRewardsCfg:
    # 初版站立基线：固定默认姿态附近保持；权重尚未经过训练行为验收。
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    upright = RewTerm(func=stand_mdp.upright_error, weight=-5.0)
    height = RewTerm(func=mdp.base_height_l2, weight=-20.0, params={"target_height": BASE_HEIGHT})
    linear_speed = RewTerm(func=stand_mdp.linear_speed_l2, weight=-10.0)
    angular_speed = RewTerm(func=stand_mdp.angular_speed_l2, weight=-1.0)
    wheel_support = RewTerm(func=stand_mdp.wheel_support, weight=1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(WHEEL_BODY_NAMES), preserve_order=True),
        "threshold": 5.0,
    })
    leg_pose = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES)),
    })
    leg_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.02, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES)),
    })
    wheel_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.02, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=list(WHEEL_JOINT_NAMES)),
    })
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES)),
    })
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class StandTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(45.0)})
    low_base = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    body_contact = DoneTerm(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Base_link", ".*_(hip|thigh|calf)_link"]),
        "threshold": 1.0,
    })


@configclass
class LightHWStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: StandSceneCfg = StandSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: StandObservationsCfg = StandObservationsCfg()
    actions: StandActionsCfg = StandActionsCfg()
    commands: StandCommandsCfg = StandCommandsCfg()
    events: StandEventsCfg = StandEventsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()
    terminations: StandTerminationsCfg = StandTerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (2.5, 2.5, 1.5)
