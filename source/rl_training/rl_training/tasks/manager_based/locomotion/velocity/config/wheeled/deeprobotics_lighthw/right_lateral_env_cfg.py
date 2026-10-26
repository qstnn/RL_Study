"""Route51独立平地右移优先任务；原route49/50配置保持不变。"""
from copy import deepcopy
import torch
from isaaclab.utils import configclass
from .flat_lateral_env_cfg import LightHWFlatLateralEnvCfg,FlatLateralLegAction
from .flat_lateral_commands import FlatLateralVelocityCommand,FlatLateralVelocityCommandCfg
from .right_lateral_core import COMMAND_CONTRACT,lateral_commands

class RightLateralVelocityCommand(FlatLateralVelocityCommand):
    def _resample_command(self,env_ids):
        ids=self._ids(env_ids)
        if not ids.numel():return
        if self.cfg.right_probability!=.75:raise ValueError('Route51 requires75%right among moving commands')
        stop=torch.rand(len(ids),device=self.device)<self.cfg.zero_command_probability
        signed=torch.rand(len(ids),device=self.device)*2-1
        self.sampled_command[ids]=lateral_commands(signed,stop,self.cfg.right_probability)
        self.is_standing_env[ids]=stop;self.is_heading_env[ids]=False
        self.mode[ids]=torch.where(stop,0,3)
        self._update_command()

@configclass
class RightLateralVelocityCommandCfg(FlatLateralVelocityCommandCfg):
    class_type:type=RightLateralVelocityCommand
    right_probability:float=.75

class RightLateralLegAction(FlatLateralLegAction):
    def _signature_payload(self):
        value=super()._signature_payload()
        value['command_distribution']=deepcopy(COMMAND_CONTRACT)
        return value

@configclass
class LightHWRightLateralEnvCfg(LightHWFlatLateralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old=self.commands.base_velocity
        self.commands.base_velocity=RightLateralVelocityCommandCfg(**{k:v for k,v in vars(old).items() if k!='class_type'})
        self.actions.joint_pos.class_type=RightLateralLegAction
        print('[LightHW route51] policy=60 critic=236 action=16;25%zero, moving:right75%/left25%; '
              'abs(vy) original0.2..1.0; all16 PPO; rewards/actions/PD/noise unchanged',flush=True)
