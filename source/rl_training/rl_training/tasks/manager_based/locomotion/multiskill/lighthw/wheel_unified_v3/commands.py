"""V3单Actor指令：原始目标与实际限加速度指令分离，起停窗口以实际归零计时。"""
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand,UniformVelocityCommandCfg
from isaaclab.utils import configclass
from . import mdp,core


class UnifiedVelocityCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.metric_steps=torch.zeros(self.num_envs,device=self.device)
        self.target_vx=torch.zeros_like(self.metric_steps)
        self.requested_vx=torch.zeros_like(self.metric_steps)
        self.hold_elapsed=torch.zeros_like(self.metric_steps)
        self.hold_origin=torch.zeros(self.num_envs,3,device=self.device)

    def _resample_command(self,env_ids):
        # Native采样会写vel_command；保留旧实际值，避免绕过斜率限制造成跳变。
        current=self.vel_command_b[env_ids,0].clone()
        super()._resample_command(env_ids)
        unloaded=(self._env.r1_payload_kg[env_ids]==0)&(self._env.r1_towed_kg[env_ids]==0)
        anchor=torch.rand(len(env_ids),device=self.device)<.4
        high=torch.where(anchor,3.1,self.vel_command_b[env_ids,0])
        low=torch.empty(len(env_ids),device=self.device).uniform_(self._env.cfg.loaded_speed_min,self._env.cfg.loaded_speed_max)
        target=torch.where(unloaded,high,low)
        self.requested_vx[env_ids]=torch.where(self.is_standing_env[env_ids],0.,target)
        self.vel_command_b[env_ids]=0.
        self.vel_command_b[env_ids,0]=current
        if self._env.cfg.fixed_speed_mps is not None:
            self.requested_vx[env_ids]=0.
            self.is_heading_env[env_ids]=False

    def _update_command(self):
        env=self._env
        t=env.episode_length_buf*env.step_dt
        loaded=(env.r1_payload_kg>0)|(env.r1_towed_kg>0)
        fixed=env.cfg.fixed_speed_mps is not None
        if fixed:
            speed=torch.as_tensor(env.cfg.fixed_speed_mps,device=self.device).expand(self.num_envs)
            distance=loaded&(speed>0.)
            start=(speed>0.)&(env.r1_eval_phase==0)&(t>=3.)
            env.r1_eval_origin[start]=self.robot.data.root_pos_w[start]
            env.r1_eval_phase[start]=1
            delta=self.robot.data.root_pos_w-env.r1_eval_origin
            progress=(delta*env.r1_tangent).sum(-1)
            env.r1_eval_progress[distance]=progress[distance]
            reached=distance&(env.r1_eval_phase==1)&(progress>=env.cfg.eval_distance_m)
            timed=(~distance)&(speed>0.)&(env.r1_eval_phase==1)&(t>=11.)
            env.r1_eval_phase[reached|timed]=2
            target=torch.where(env.r1_eval_phase==1,speed,0.)
            self.requested_vx[:]=target
        else:
            # 每次重置真实重力和牵引立即生效，只将速度命令保留1秒零值用于落地稳定。
            target=torch.where(t<1.,0.,self.requested_vx)
        self.target_vx[:]=target
        rate=torch.where(loaded,env.cfg.loaded_acceleration,env.cfg.unloaded_acceleration)
        if env.cfg.command_profile=="slew":
            self.vel_command_b[:,0]=core.slew(self.vel_command_b[:,0],target,rate,env.step_dt)
        elif env.cfg.command_profile=="step":
            self.vel_command_b[:,0]=target
        else:
            raise ValueError("Unknown command profile")
        self.vel_command_b[:,1:]=0.
        if fixed:
            stopped=(env.r1_eval_phase==2)&(self.vel_command_b[:,0].abs()<1e-6)
            env.r1_eval_phase[stopped]=3
            env.r1_eval_stop_time[stopped]=t[stopped]
        holding=(self.vel_command_b[:,0].abs()<1e-4)&(target.abs()<1e-4)
        previous=self.hold_elapsed.clone()
        self.hold_elapsed[:]=torch.where(holding,previous+env.step_dt,0.)
        # 刹停2秒过渡期结束时重新锚定；奖励/验收都在完整稳停窗口看漂移。
        anchor=(~holding)|(previous==0)|((previous<2.)&(self.hold_elapsed>=2.))
        self.hold_origin[anchor]=self.robot.data.root_pos_w[anchor]

    def _update_metrics(self):
        _,velocity,_=mdp.state(self._env)
        yaw=(self.robot.data.root_ang_vel_w*self._env.r1_normal).sum(-1)
        self.metrics["error_vel_xy"]+=(velocity-self.vel_command_b[:,:2]).norm(dim=-1)
        self.metrics["error_vel_yaw"]+=(yaw-self.vel_command_b[:,2]).abs()
        self.metric_steps+=1

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        means={name:float((value[ids]/self.metric_steps[ids].clamp_min(1.)).mean()) for name,value in self.metrics.items()}
        self.vel_command_b[ids]=0.;self.target_vx[ids]=0.;self.requested_vx[ids]=0.;self.hold_elapsed[ids]=0.
        self.hold_origin[ids]=self.robot.data.root_pos_w[ids]
        extras=super().reset(env_ids);extras.update(means);self.metric_steps[ids]=0
        return extras


@configclass
class UnifiedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type=UnifiedVelocityCommand
