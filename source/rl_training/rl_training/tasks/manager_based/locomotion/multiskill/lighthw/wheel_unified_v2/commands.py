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
        unloaded=(self._env.r1_payload_kg[env_ids]==0)&(self._env.r1_towed_kg[env_ids]==0)
        anchor=torch.rand(len(env_ids),device=self.device)<.4
        high=torch.where(anchor,3.1,self.vel_command_b[env_ids,0])
        low=torch.empty(len(env_ids),device=self.device).uniform_(self._env.cfg.loaded_speed_min,self._env.cfg.loaded_speed_max)
        sampled=torch.where(unloaded,high,low)
        # reset后的第一帧也必须尊重停车标记，不能等下一次compute才清零。
        self.vel_command_b[env_ids,0]=torch.where(self.is_standing_env[env_ids],0.,sampled)
        if self._env.cfg.fixed_speed_mps is not None:
            self.vel_command_b[env_ids]=0.
            self.is_standing_env[env_ids]=False
            self.is_heading_env[env_ids]=False

    def _update_command(self):
        super()._update_command()
        if self._env.cfg.fixed_speed_mps is not None:
            t=self._env.episode_length_buf*self._env.step_dt
            self.vel_command_b[:]=0.
            env=self._env
            speed=torch.as_tensor(env.cfg.fixed_speed_mps,device=self.device).expand(self.num_envs)
            loaded=(env.r1_payload_kg>0)|(env.r1_towed_kg>0)
            distance_case=loaded&(speed>0.)
            starting=distance_case&(env.r1_eval_phase==0)&(t>=3.)
            env.r1_eval_origin[starting]=self.robot.data.root_pos_w[starting]
            env.r1_eval_phase[starting]=1
            delta=self.robot.data.root_pos_w-env.r1_eval_origin
            progress=(delta*env.r1_tangent).sum(-1)
            env.r1_eval_progress[distance_case]=progress[distance_case]
            reached=distance_case&(env.r1_eval_phase==1)&(progress>=env.cfg.eval_distance_m)
            env.r1_eval_phase[reached]=2
            env.r1_eval_stop_time[reached]=t[reached]
            fixed_moving=(t>=3.)&(t<11.)
            moving=torch.where(distance_case,env.r1_eval_phase==1,fixed_moving)
            self.vel_command_b[:,0]=torch.where(moving,speed,0.)

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
