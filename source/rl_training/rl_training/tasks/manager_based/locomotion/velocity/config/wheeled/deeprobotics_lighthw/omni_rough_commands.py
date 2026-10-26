"""一个Actor、两类on-policy环境：保留直线能力与学习M20多方向命令。"""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import (
    UniformThresholdVelocityCommand,UniformThresholdVelocityCommandCfg,
)
from .safe_terrain_spec import columns_for_surface
from .terrain_speed_core import terrain_command
from .omni_rough_core import retention_mask,limit_obstacle_command


class OmniRoughVelocityCommand(UniformThresholdVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        with torch.inference_mode(False):
            self.retention=retention_mask(torch.arange(self.num_envs,device=self.device))
            self.sampled_command=torch.zeros_like(self.vel_command_b)
            self.mode=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self._flat=columns_for_surface('flat')
        self._obstacles=sum((columns_for_surface(n) for n in ('pyramid_stairs','pyramid_stairs_inv','boxes')),[])

    def _ids(self,ids):
        return torch.arange(self.num_envs,device=self.device)[ids] if isinstance(ids,slice) else torch.as_tensor(ids,device=self.device,dtype=torch.long)

    def terrain_columns(self):
        t=self._env.scene.terrain;p=self.robot.data.root_pos_w
        size=p.new_tensor(t.cfg.terrain_generator.size)
        cell=torch.floor((p[:,:2]-(t.terrain_origins[0,0,:2]-size/2))/size).long()
        inside=(cell>=0).all(1)&(cell[:,0]<10)&(cell[:,1]<20)
        return torch.where(inside,cell[:,1],t.terrain_types)

    def _resample_command(self,env_ids):
        ids=self._ids(env_ids)
        if not ids.numel():return
        n=ids.numel();device=self.device
        stop=torch.rand(n,device=device)<self.cfg.zero_command_probability
        retained=self.retention[ids]
        mode=torch.randint(2,7,(n,),device=device)
        mode=torch.where(retained,1,mode)
        mode=torch.where(stop,0,mode)
        self.mode[ids]=mode
        cmd=torch.zeros(n,3,device=device)
        # 旧前进分位/锚点保留在固定的一半环境；源物理状态/RNG不冒称连续恢复。
        speed=torch.empty(n,device=device).uniform_(self.cfg.minimum_abs_speed,3.)
        anchor=torch.rand(n,device=device)<self.cfg.anchor_probability
        cmd[:,0]=torch.where(mode==1,torch.where(anchor,3.,speed),0.)
        random=torch.rand(n,3,device=device)*2-1
        minimum=random.sign()*torch.maximum(random.abs(),random.new_tensor(.2))
        cmd[:,0]=torch.where(mode==2,minimum[:,0]*2.,cmd[:,0])
        cmd[:,1]=torch.where(mode==3,minimum[:,1],cmd[:,1])
        cmd[:,2]=torch.where(mode==4,minimum[:,2],cmd[:,2])
        combined=(mode==5)|(mode==6)
        cmd[combined]=random[combined]*cmd.new_tensor([2.,1.,1.])
        small=cmd[:,:2].norm(dim=1)<.2
        cmd[small&~retained,:2]=0.
        self.heading_target[ids]=torch.empty(n,device=device).uniform_(-math.pi,math.pi)
        self.is_heading_env[ids]=mode==6
        self.is_standing_env[ids]=mode==0
        self.sampled_command[ids]=cmd
        self._update_command()

    def _update_command(self):
        self.vel_command_b.copy_(self.sampled_command)
        columns=self.terrain_columns()
        flat=torch.isin(columns,torch.as_tensor(self._flat,device=self.device))
        self.vel_command_b[:,0]=torch.where(self.retention,
            terrain_command(self.sampled_command[:,0],flat),self.vel_command_b[:,0])
        heading=self.is_heading_env
        self.vel_command_b[heading,2]=(.5*wrap_to_pi(self.heading_target[heading]-self.robot.data.heading_w[heading])).clamp(-1.,1.)
        obstacle=torch.isin(columns,torch.as_tensor(self._obstacles,device=self.device))&~self.retention
        self.vel_command_b.copy_(limit_obstacle_command(self.vel_command_b,obstacle))
        self.vel_command_b[self.is_standing_env]=0.


@configclass
class OmniRoughVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    class_type:type=OmniRoughVelocityCommand
    zero_command_probability:float=.25
    minimum_abs_speed:float=.08
    anchor_probability:float=.5
