"""全速度段同时训练的命令采样；误差日志按实际采样步数归一化。"""
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand,UniformVelocityCommandCfg
from isaaclab.utils import configclass
from . import mdp


class UnifiedVelocityCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.metric_steps=torch.zeros(self.num_envs,device=self.device)

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        target=(self._env.r1_payload_kg[env_ids]>=15)&(self._env.r1_towed_kg[env_ids]>=40)&(self._env.r1_slope_rad[env_ids].abs()>.43)
        probability=torch.where(target,.7,.4)
        anchor=torch.rand(len(env_ids),device=self.device)<probability
        self.vel_command_b[env_ids,0]=torch.where(anchor,3.1,self.vel_command_b[env_ids,0])
        if self._env.cfg.fixed_speed_mps is not None:
            self.vel_command_b[env_ids]=0.
            self.is_standing_env[env_ids]=False
            self.is_heading_env[env_ids]=False

    def _update_command(self):
        super()._update_command()
        if self._env.cfg.fixed_speed_mps is not None:
            t=self._env.episode_length_buf*self._env.step_dt
            self.vel_command_b[:]=0.
            self.vel_command_b[:,0]=torch.where((t>=3.)&(t<11.),self._env.cfg.fixed_speed_mps,0.)

    def _update_metrics(self):
        _,v,_=mdp.state(self._env)
        yaw=(self.robot.data.root_ang_vel_w*self._env.r1_normal).sum(-1)
        self.metrics["error_vel_xy"]+=(v-self.vel_command_b[:,:2]).norm(dim=-1)
        self.metrics["error_vel_yaw"]+=(yaw-self.vel_command_b[:,2]).abs()
        self.metric_steps+=1

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        means={name:float((value[ids]/self.metric_steps[ids].clamp_min(1.)).mean()) for name,value in self.metrics.items()}
        extras=super().reset(env_ids)
        extras.update(means)
        self.metric_steps[ids]=0
        return extras


@configclass
class UnifiedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type=UnifiedVelocityCommand
