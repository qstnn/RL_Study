"""V6完整任务循环：3秒落地、行进、实际归零、2秒过渡和5秒稳停。"""
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand,UniformVelocityCommandCfg
from isaaclab.utils import configclass
from . import mdp,core


class UnifiedVelocityCommand(UniformVelocityCommand):
    """全部训练工况共用状态机；只在完整循环结束后采下一指令。

    phase 0=起步前等待/静止任务，1=行进，2=减速，3=实际归零后的停车。
    requested_vx 是本轮名义任务速度，target_vx 是当前阶段给斜率限制器的目标。
    不把初始漂移算作行进距离，也不通过重采样中断5秒验收保持窗。
    """
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.metric_steps=torch.zeros(self.num_envs,device=self.device)
        self.target_vx=torch.zeros_like(self.metric_steps)
        self.requested_vx=torch.zeros_like(self.metric_steps)
        self.hold_elapsed=torch.zeros_like(self.metric_steps)
        self.hold_origin=torch.zeros(self.num_envs,3,device=self.device)
        self.hold_steps=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.holding_active=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.cycle_start_step=torch.zeros_like(self.hold_steps)
        self.phase_start_step=torch.zeros_like(self.hold_steps)
        self.cycle_completed=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.cycles_completed=torch.zeros_like(self.hold_steps)
        self.hold_windows_completed=torch.zeros_like(self.hold_steps)
        self.static_cycles_completed=torch.zeros_like(self.hold_steps)
        self.completed_cycle_this_step=torch.zeros_like(self.cycle_completed)
        self.completed_hold_this_step=torch.zeros_like(self.cycle_completed)
        self.distance_reached_this_step=torch.zeros_like(self.cycle_completed)

    def _resample_command(self,env_ids):
        # Native采样会写vel_command；保留当前值，绝不绕过斜率限制。
        current=self.vel_command_b[env_ids,0].clone()
        super()._resample_command(env_ids)
        unloaded=(self._env.r1_payload_kg[env_ids]==0)&(self._env.r1_towed_kg[env_ids]==0)
        anchor=torch.rand(len(env_ids),device=self.device)<.4
        high=torch.where(anchor,3.1,self.vel_command_b[env_ids,0])
        low=torch.empty(len(env_ids),device=self.device).uniform_(self._env.cfg.loaded_speed_min,self._env.cfg.loaded_speed_max)
        target=torch.where(unloaded,high,low)
        self.requested_vx[env_ids]=torch.where(self.is_standing_env[env_ids],0.,target)
        if self._env.cfg.fixed_speed_mps is not None:
            speed=torch.as_tensor(self._env.cfg.fixed_speed_mps,device=self.device).expand(self.num_envs)
            self.requested_vx[env_ids]=speed[env_ids]
            self.is_heading_env[env_ids]=False
        self.vel_command_b[env_ids]=0.
        self.vel_command_b[env_ids,0]=current
        self.target_vx[env_ids]=0.
        self.hold_steps[env_ids]=0
        self.holding_active[env_ids]=False
        self.hold_elapsed[env_ids]=0.
        self.hold_origin[env_ids]=self.robot.data.root_pos_w[env_ids]
        self.cycle_start_step[env_ids]=self._env.episode_length_buf[env_ids]
        self.phase_start_step[env_ids]=self.cycle_start_step[env_ids]
        self.cycle_completed[env_ids]=False
        self._env.r1_eval_phase[env_ids]=0
        self._env.r1_eval_stop_time[env_ids]=-1.
        self._env.r1_eval_origin[env_ids]=self.robot.data.root_pos_w[env_ids]
        self._env.r1_eval_progress[env_ids]=0.

    @property
    def phase_elapsed(self):
        return (self._env.episode_length_buf-self.phase_start_step)*self._env.step_dt

    @property
    def cycle_elapsed(self):
        return (self._env.episode_length_buf-self.cycle_start_step)*self._env.step_dt

    def _update_command(self):
        env=self._env
        steps=env.episode_length_buf
        t=steps*env.step_dt
        fixed=env.cfg.fixed_speed_mps is not None
        self.completed_cycle_this_step.zero_()
        self.completed_hold_this_step.zero_()
        self.distance_reached_this_step.zero_()
        # 在完成帧的下一步才采新任务，日志保留完整最后一帧，不覆盖完成事件。
        if not fixed:
            ids=self.cycle_completed.nonzero().flatten()
            if len(ids):
                self._resample(ids)
        loaded=(env.r1_payload_kg>0)|(env.r1_towed_kg>0)
        moving=self.requested_vx>0.
        distance=loaded&moving
        elapsed_steps=steps-self.cycle_start_step
        start=moving&(env.r1_eval_phase==0)&(elapsed_steps>=round(3./env.step_dt))
        env.r1_eval_origin[start]=self.robot.data.root_pos_w[start]
        env.r1_eval_phase[start]=1
        self.phase_start_step[start]=steps[start]
        delta=self.robot.data.root_pos_w-env.r1_eval_origin
        progress=(delta*env.r1_tangent).sum(-1)
        # 起步帧锚定后从0累计净位移；后退必须真正补回，不能累加正速度骗距离。
        active=distance&(env.r1_eval_phase>=1)
        env.r1_eval_progress[:]=torch.where(active,progress,0.)
        reached=distance&(env.r1_eval_phase==1)&(env.r1_eval_progress>=env.cfg.eval_distance_m)
        timed=(~distance)&moving&(env.r1_eval_phase==1)&(elapsed_steps>=round(11./env.step_dt))
        self.distance_reached_this_step[:]=reached
        braking=reached|timed
        env.r1_eval_phase[braking]=2
        self.phase_start_step[braking]=steps[braking]
        target=torch.where(env.r1_eval_phase==1,self.requested_vx,0.)
        self.target_vx[:]=target
        rate=torch.where(loaded,env.cfg.loaded_acceleration,env.cfg.unloaded_acceleration)
        if env.cfg.command_profile=="slew":
            self.vel_command_b[:,0]=core.slew(self.vel_command_b[:,0],target,rate,env.step_dt)
        elif env.cfg.command_profile=="step":
            self.vel_command_b[:,0]=target
        else:
            raise ValueError("Unknown command profile")
        self.vel_command_b[:,1:]=0.
        stopped=(env.r1_eval_phase==2)&(self.vel_command_b[:,0].abs()<1e-6)
        env.r1_eval_phase[stopped]=3
        env.r1_eval_stop_time[stopped]=t[stopped]
        self.phase_start_step[stopped]=steps[stopped]
        holding=(self.vel_command_b[:,0].abs()<1e-6)&(target.abs()<1e-6)
        previous=self.hold_steps.clone()
        entered=holding&(~self.holding_active)
        # 归零这一帧为elapsed=0，之后才累计已真实经过的时间；与验收stop_time完全对齐。
        self.hold_steps[:]=torch.where(holding&self.holding_active,previous+1,0)
        self.hold_elapsed[:]=self.hold_steps*env.step_dt
        # 刹停2秒过渡结束时只锚定一次，此后完整5秒的漂移不被逐步清零。
        settle_steps=round(2./env.step_dt)
        anchor=(~holding)|entered|((previous<settle_steps)&(self.hold_steps>=settle_steps))
        self.hold_origin[anchor]=self.robot.data.root_pos_w[anchor]
        self.holding_active[:]=holding
        completed_hold=moving&(env.r1_eval_phase==3)&((steps-self.phase_start_step)>=round(7./env.step_dt))
        completed_static=(~moving)&(elapsed_steps>=round(20./env.step_dt))
        completed=(completed_hold|completed_static)&(~self.cycle_completed)
        self.completed_cycle_this_step[:]=completed
        self.completed_hold_this_step[:]=completed&completed_hold
        self.cycles_completed+=completed.long()
        self.hold_windows_completed+=(completed&completed_hold).long()
        self.static_cycles_completed+=(completed&completed_static).long()
        self.cycle_completed|=completed

    def _update_metrics(self):
        _,velocity,_=mdp.state(self._env)
        yaw=(self.robot.data.root_ang_vel_w*self._env.r1_normal).sum(-1)
        self.metrics["error_vel_xy"]+=(velocity-self.vel_command_b[:,:2]).norm(dim=-1)
        self.metrics["error_vel_yaw"]+=(yaw-self.vel_command_b[:,2]).abs()
        self.metric_steps+=1

    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        means={name:float((value[ids]/self.metric_steps[ids].clamp_min(1.)).mean()) for name,value in self.metrics.items()}
        means.update(task_cycles_completed=float(self.cycles_completed[ids].float().mean()),
                     task_hold_windows_completed=float(self.hold_windows_completed[ids].float().mean()),
                     task_static_cycles_completed=float(self.static_cycles_completed[ids].float().mean()))
        self.vel_command_b[ids]=0.;self.target_vx[ids]=0.;self.requested_vx[ids]=0.;self.hold_elapsed[ids]=0.
        self.hold_steps[ids]=0;self.holding_active[ids]=False
        self.cycles_completed[ids]=0;self.hold_windows_completed[ids]=0;self.static_cycles_completed[ids]=0
        self.completed_cycle_this_step[ids]=False;self.completed_hold_this_step[ids]=False
        self.distance_reached_this_step[ids]=False
        self.hold_origin[ids]=self.robot.data.root_pos_w[ids]
        extras=super().reset(ids);extras.update(means);self.metric_steps[ids]=0
        # Isaac在command.reset之后才清零episode步数；父类重采样此刻读到的是旧回合。
        # 新回合必须从0计时，否则timeout后起步时间为负，下一回合也无法起步。
        # 只在episode reset覆盖锚点；回合内_resample_command仍以当前步数开始新任务。
        self.cycle_start_step[ids]=0
        self.phase_start_step[ids]=0
        return extras


@configclass
class UnifiedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type=UnifiedVelocityCommand
