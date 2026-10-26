"""Route48平地侧移专项：只集中地形/命令任务，沿用route46奖励和PPO物理接口。"""
from copy import deepcopy
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from .lateral_tracking_env_cfg import LightHWLateralTrackingEnvCfg
from .omni_rough_env_cfg import OmniLegAction
from .flat_lateral_commands import FlatLateralVelocityCommandCfg
from .flat_lateral_core import COMMAND_CONTRACT

class FlatLateralLegAction(OmniLegAction):
    def _signature_payload(self):
        value=super()._signature_payload()
        value['command_distribution']=deepcopy(COMMAND_CONTRACT)
        return value

@configclass
class LightHWFlatLateralEnvCfg(LightHWLateralTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 保留网格尺寸、列数和原材料；每个tile实际mesh均为z=0平面。
        # 旧子地形键仅为奖励对象的几何配置兼容容器，不代表仍生成台阶。
        # 原StairCompletionReward的centerZ>0激活条件在本任务恒false，不改权重/函数。
        terrain=self.scene.terrain
        for sub in terrain.terrain_generator.sub_terrains.values():sub.function=flat_terrain
        terrain.terrain_generator.curriculum=False
        terrain.max_init_terrain_level=0
        self.curriculum.terrain_levels=None
        self.actions.joint_pos.class_type=FlatLateralLegAction
        old=self.commands.base_velocity
        self.commands.base_velocity=FlatLateralVelocityCommandCfg(**{k:v for k,v in vars(old).items() if k!='class_type'})
        command=self.commands.base_velocity
        command.heading_command=False;command.rel_heading_envs=0.
        command.ranges.lin_vel_x=(0.,0.);command.ranges.ang_vel_z=(0.,0.)
        command.ranges.lin_vel_y=(-1.,1.)
        print('[LightHW route48] policy=60 critic=236 action=16; all16 PPO; FLAT lateral only; '
              'vx=wz=0; 25%zero, 75%left/right abs(vy)=0.2..1.0; original leg noise; no terrain progression',flush=True)
