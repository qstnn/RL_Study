"""单个R1任务：载荷、拖车、全速度与上下坡同时采样，共享一个Actor。"""
from copy import deepcopy
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp as native
from isaaclab.managers import EventTermCfg as EventTerm,ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm,TerminationTermCfg as DoneTerm,SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from ..wheel_env_cfg import WheelStage1EnvCfg
from ..catalog import LEG_NAMES,WHEEL_NAMES,WHEEL_BODIES
from .. import rewards as stand_rewards
from . import mdp,cart,spec
from .robot import TorqueObservedArticulation
from .commands import UnifiedVelocityCommandCfg
from .terrain import UnifiedTerrainCfg


@configclass
class UnifiedRewards:
    track_broad=RewTerm(func=mdp.tracking,weight=4.)
    track_fine=RewTerm(func=mdp.tracking,weight=2.,params={"fine":True})
    tracking_error=RewTerm(func=mdp.tracking_l1,weight=-.5)
    track_yaw=RewTerm(func=mdp.yaw_tracking,weight=1.)
    heading=RewTerm(func=mdp.heading_cost,weight=-.5)
    surface_tilt=RewTerm(func=mdp.surface_tilt,weight=-2.)
    # 旧策略载荷case持续低于高度门槛；8cm误差的瞬时成本从0.032提高到0.64。
    surface_height=RewTerm(func=mdp.surface_height,weight=-100.)
    surface_normal_speed=RewTerm(func=mdp.normal_motion,weight=-1.)
    surface_tilt_rate=RewTerm(func=mdp.tilt_rate,weight=-.05)
    effort=RewTerm(func=mdp.effort_usage,weight=-.05)
    hitch_clearance=RewTerm(func=mdp.hitch_clearance_cost,weight=-2.)
    # reward manager乘20ms，非超时失败扣2；正常达到episode时限不受此项惩罚。
    failure=RewTerm(func=native.is_terminated,weight=-100.)
    joint_limits=RewTerm(func=native.joint_pos_limits,weight=-5.,params={"asset_cfg":SceneEntityCfg("robot",joint_names=list(LEG_NAMES))})
    action_rate=RewTerm(func=native.action_rate_l2,weight=-.005)
    target_overflow=RewTerm(func=mdp.target_overflow,weight=-.1)
    stand_motion=RewTerm(func=mdp.stand_motion,weight=-3.)
    stand_wheels=RewTerm(func=stand_rewards.stand_wheel_speed,weight=-.01,
                        params={"asset_cfg":SceneEntityCfg("robot",joint_names=list(WHEEL_NAMES))})
    stand_pose=RewTerm(func=stand_rewards.stand_leg_pose,weight=-.05,
                      params={"asset_cfg":SceneEntityCfg("robot",joint_names=list(LEG_NAMES))})
    stand_support=RewTerm(func=mdp.stand_support,weight=1.,
                         params={"sensor_cfg":SceneEntityCfg("contact_forces",body_names=list(WHEEL_BODIES),preserve_order=True)})


@configclass
class UnifiedEvents:
    payload=EventTerm(func=mdp.set_payload_properties,mode="startup")
    reset_base=EventTerm(func=mdp.reset_ramp,mode="reset")
    reset_joints=EventTerm(func=native.reset_joints_by_offset,mode="reset",
                          params={"position_range":(0.,0.),"velocity_range":(0.,0.)})


@configclass
class UnifiedEnvCfg(WheelStage1EnvCfg):
    ground_friction: float=2.3
    rolling_coefficient: float=.02
    leg_torque_limit: float=96.
    wheel_torque_limit: float=48.
    hitch_height_m: float=.10
    loaded_speed_min: float=.2
    loaded_speed_max: float=.6
    fixed_slope_deg: float | None=None
    fixed_payload_kg: float | None=None
    fixed_towed_kg: float | None=None
    fixed_speed_mps: float | None=None
    eval_distance_m: float=1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain=UnifiedTerrainCfg(prim_path="/World/ground",terrain_type="generator",
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=self.ground_friction,dynamic_friction=self.ground_friction,
                                                            friction_combine_mode="min",restitution=0.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25,.28,.32)))
        self.sim.physics_material=deepcopy(self.scene.terrain.physics_material)
        self.scene.robot.class_type=TorqueObservedArticulation
        self.scene.robot.actuators["wheel"].damping=3.
        self.scene.robot.actuators["joint"].effort_limit=self.leg_torque_limit
        self.scene.robot.actuators["wheel"].effort_limit=self.wheel_torque_limit
        self.actions.joint_vel.class_type=cart.TowingWheelAction
        self.scene.trailer=ArticulationCfg(prim_path="{ENV_REGEX_NS}/Trailer",
            spawn=sim_utils.UrdfFileCfg(asset_path="worker_must_set_cart_urdf",fix_base=False,root_link_name="cart_base",
                joint_drive=None,make_instanceable=True,force_usd_conversion=True,activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.,
                                                            max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                    solver_position_iteration_count=4,solver_velocity_iteration_count=1)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(-1.73,0.,.30),joint_pos={".*":0.},joint_vel={".*":0.}),
            actuators={"passive":ImplicitActuatorCfg(joint_names_expr=[".*"],stiffness=0.,damping=0.,
                                                    effort_limit_sim=0.,velocity_limit_sim=200.)})
        self.scene.trailer_contacts=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Trailer/.*",update_period=.001,history_length=1)
        self.observations.policy.heading_error=ObsTerm(func=mdp.heading_error)
        self.observations.policy.history_length=spec.HISTORY
        self.observations.policy.flatten_history_dim=True
        self.observations.critic.context=ObsTerm(func=mdp.privileged_context)
        self.commands.base_velocity=UnifiedVelocityCommandCfg(asset_name="robot",resampling_time_range=(6.,8.),
            rel_standing_envs=.25,rel_heading_envs=0.,heading_command=False,debug_vis=False,
            ranges=UnifiedVelocityCommandCfg.Ranges(lin_vel_x=(.2,3.1),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.)))
        self.events=UnifiedEvents()
        self.rewards=UnifiedRewards()
        self.curriculum=None
        self.terminations.bad_orientation_2=DoneTerm(func=mdp.bad_surface_orientation)
        self.terminations.outside_lane=DoneTerm(func=mdp.outside_lane)
        self.terminations.cart_failure=DoneTerm(func=mdp.cart_failure)
        self.terminations.drawbar_ground=DoneTerm(func=mdp.drawbar_ground)
        self.episode_length_s=20.
