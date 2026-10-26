"""R6 平地固定载荷持载；独立运行时URDF、奖励和载荷观测。"""
import math
import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from .catalog import LEG_NAMES, WHEEL_NAMES, WHEEL_BODIES
from .wheel_env_cfg import WheelStage1EnvCfg
from . import payload_mdp, rewards
from .payload_spawn import spawn_payload
from .local_ground_plane import LocalTerrainImporter


@configclass
class PayloadRewardsCfg:
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    height = RewTerm(func=mdp.base_height_l2, weight=-30.0, params={"target_height": .416})
    still = RewTerm(func=rewards.stand_motion, weight=-10.0)
    vertical_speed = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    support = RewTerm(func=rewards.stand_support, weight=1.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(WHEEL_BODIES), preserve_order=True)})
    balance = RewTerm(func=payload_mdp.load_balance_cost, weight=-.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(WHEEL_BODIES), preserve_order=True)})
    wheel_speed = RewTerm(func=rewards.stand_wheel_speed, weight=-.03, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(WHEEL_NAMES))})
    leg_pose = RewTerm(func=rewards.stand_leg_pose, weight=-.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    leg_speed = RewTerm(func=mdp.joint_vel_l2, weight=-.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    torque_margin = RewTerm(func=payload_mdp.torque_margin_cost, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_NAMES))})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-.015)


@configclass
class HeavyPayloadEnvCfg(WheelStage1EnvCfg):
    payload_mass_kg: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.terrain.class_type = LocalTerrainImporter
        # 离线纯色穹顶灯不请求默认 HDR 贴图。
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0))
        self.scene.robot.spawn.func = spawn_payload
        self.commands.base_velocity.ranges.lin_vel_x = (0., 0.)
        self.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.observations.policy.payload = ObsTerm(func=payload_mdp.payload_observation)
        self.observations.critic.payload = ObsTerm(func=payload_mdp.payload_observation)
        self.events.reset_base.params['pose_range'] = {'roll': (-.015, .015), 'pitch': (-.015, .015), 'yaw': (-.05, .05)}
        self.rewards = PayloadRewardsCfg()
        self.terminations.bad_orientation_2 = DoneTerm(func=mdp.bad_orientation, params={'limit_angle': math.radians(30)})
        self.terminations.low_height = DoneTerm(func=mdp.root_height_below_minimum, params={'minimum_height': .25})
