"""独立 R5A J1：有限时域的一次起跳、落地和恢复。"""
from copy import deepcopy
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.asset_cfg import LIGHTHW_ROUTE_CFG
from r5a_auto import spec
from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a.appearance import spawn_robot, terrain_class



def robot_config():
    cfg=deepcopy(LIGHTHW_ROUTE_CFG).replace(prim_path='{ENV_REGEX_NS}/Robot')
    cfg.spawn.func=spawn_robot
    cfg.spawn.articulation_props.enabled_self_collisions=True
    cfg.spawn.articulation_props.solver_position_iteration_count=8
    cfg.spawn.articulation_props.solver_velocity_iteration_count=2
    for key,effort,velocity in (('joint',48.,12.57),('wheel',24.,37.7)):
        motor=cfg.actuators[key]
        motor.effort_limit=motor.effort_limit_sim=effort
        motor.velocity_limit=motor.velocity_limit_sim=velocity
        motor.min_delay=motor.max_delay=0
    cfg.actuators['joint'].stiffness=80.; cfg.actuators['joint'].damping=3.
    cfg.actuators['wheel'].stiffness=0.; cfg.actuators['wheel'].damping=1.
    return cfg


@configclass
class JumpSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = robot_config()
    terrain = TerrainImporterCfg(prim_path='/World/ground',terrain_type='plane',
        class_type=terrain_class(),collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    contact_forces = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=.001,history_length=20)
    sky_light = AssetBaseCfg(prim_path='/World/sky',spawn=sim_utils.DomeLightCfg(intensity=750.,color=(1.,1.,1.)))


