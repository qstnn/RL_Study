"""只改变命令任务采样；沿用8秒重采样与25%零命令，不控制腿轮动作。"""
import torch
from isaaclab.utils import configclass
from .omni_rough_commands import OmniRoughVelocityCommand,OmniRoughVelocityCommandCfg
from .flat_lateral_core import lateral_commands

class FlatLateralVelocityCommand(OmniRoughVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.retention.fill_(False)
        self._flat=list(range(env.scene.terrain.cfg.terrain_generator.num_cols))
        self._obstacles=[]

    def _resample_command(self,env_ids):
        ids=self._ids(env_ids)
        if not ids.numel():return
        stop=torch.rand(len(ids),device=self.device)<self.cfg.zero_command_probability
        signed=torch.rand(len(ids),device=self.device)*2-1
        self.sampled_command[ids]=lateral_commands(signed,stop)
        self.is_standing_env[ids]=stop;self.is_heading_env[ids]=False
        self.mode[ids]=torch.where(stop,0,3)
        self._update_command()

    def _update_command(self):
        self.vel_command_b.copy_(self.sampled_command)
        self.vel_command_b[self.is_standing_env]=0.

@configclass
class FlatLateralVelocityCommandCfg(OmniRoughVelocityCommandCfg):
    class_type:type=FlatLateralVelocityCommand
