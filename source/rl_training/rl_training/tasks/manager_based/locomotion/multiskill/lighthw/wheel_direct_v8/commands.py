"""V8完整任务循环：3秒落地、行进、实际归零、2秒过渡和5秒稳停。"""
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
        self.successful_hold_this_step=torch.zeros_like(self.cycle_completed)
        self.cart_hold_origin=torch.zeros(self.num_envs,3,device=self.device)
        self.hold_count=torch.zeros_like(self.hold_steps)
        self.hold_support_sum=torch.zeros_like(self.metric_steps)
        self.hold_speed_sum=torch.zeros_like(self.metric_steps)
        self.hold_yaw_sum=torch.zeros_like(self.metric_steps)
        self.hold_max_drift=torch.zeros_like(self.metric_steps)
        self.hold_posture_ok=torch.ones_like(self.cycle_completed)
        self.task_sequence=torch.full_like(self.hold_steps,-1)

    def _resample_command(self,env_ids):
        # Native采样会写vel_command；保留当前值，绝不绕过斜率限制。
        current=self.vel_command_b[env_ids,0].clone()
        # 第一阶段固定任务顺序和速度，取消观测/物理扰动之外的命令随机采样。
        # task_sequence跨episode保留；静止任务不能一直占住同一工况，失败也推进序列。
        self.task_sequence[env_ids]+=1
        unloaded=(self._env.r1_payload_kg[env_ids]==0)&(self._env.r1_towed_kg[env_ids]==0)
        target=torch.where(unloaded,3.1,.3)
        hard=(self._env.r1_towed_kg[env_ids]==40)&((self._env.r1_slope_rad[env_ids]>.43)|
             torch.isclose(self._env.r1_slope_rad[env_ids],torch.full_like(target,-20.*torch.pi/180.),atol=1.e-6))
        period=torch.where(hard,10,4)
        standing=(env_ids+self.task_sequence[env_ids])%period==0
        self.is_standing_env[env_ids]=standing
        self.is_heading_env[env_ids]=False
        self.requested_vx[env_ids]=torch.where(standing,0.,target)
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
        self._env.r1_cart_origin[env_ids]=self._env.scene["trailer"].data.root_pos_w[env_ids]
        self.cart_hold_origin[env_ids]=self._env.scene["trailer"].data.root_pos_w[env_ids]
        self.hold_count[env_ids]=0;self.hold_support_sum[env_ids]=0.
        self.hold_speed_sum[env_ids]=0.;self.hold_yaw_sum[env_ids]=0.
        self.hold_max_drift[env_ids]=0.;self.hold_posture_ok[env_ids]=True
        self.successful_hold_this_step[env_ids]=False

    def robot_progress(self):
        return ((self.robot.data.root_pos_w-self._env.r1_eval_origin)*self._env.r1_tangent).sum(-1)

    def cart_progress(self):
        return ((self._env.scene["trailer"].data.root_pos_w-self._env.r1_cart_origin)*self._env.r1_tangent).sum(-1)

    def joint_progress(self):
        robot=self.robot_progress()
        return torch.where(self._env.r1_towed_kg>0,torch.minimum(robot,self.cart_progress()),robot)

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
        self.successful_hold_this_step.zero_()
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
        env.r1_cart_origin[start]=env.scene["trailer"].data.root_pos_w[start]
        env.r1_eval_phase[start]=1
        self.phase_start_step[start]=steps[start]
        delta=self.robot.data.root_pos_w-env.r1_eval_origin
        progress=(delta*env.r1_tangent).sum(-1)
        # 起步帧锚定后从0累计净位移；后退必须真正补回，不能累加正速度骗距离。
        active=distance&(env.r1_eval_phase>=1)
        env.r1_eval_progress[:]=torch.where(active,progress,0.)
        reached=distance&(env.r1_eval_phase==1)&(self.joint_progress()>=env.cfg.eval_distance_m)
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
        self.cart_hold_origin[anchor]=env.scene["trailer"].data.root_pos_w[anchor]
        self.holding_active[:]=holding
        completed_hold=moving&(env.r1_eval_phase==3)&((steps-self.phase_start_step)>=round(7./env.step_dt))
        completed_static=(~moving)&(elapsed_steps>=round(20./env.step_dt))
        completed=(completed_hold|completed_static)&(~self.cycle_completed)
        self.completed_cycle_this_step[:]=completed
        self.completed_hold_this_step[:]=completed&completed_hold
        self.cycles_completed+=completed.long()
        self.hold_windows_completed+=(completed&completed_hold).long()
        self.static_cycles_completed+=(completed&completed_static).long()
        # 单次完整稳停奖励单独累积真实窗口，不能只看计时器或一帧低速度。
        window=moving&(env.r1_eval_phase==3)&(self.hold_elapsed>=2.)&(self.hold_elapsed<7.)
        _,v,up=mdp.state(env)
        force=env.scene["contact_forces"].data.net_forces_w
        wn=(force[:,env.r1_wheel_sensor_ids]*env.r1_normal[:,None]).sum(-1)
        threshold=(mdp.spec.CONTRACT["robot_mass_kg"]+env.r1_payload_kg)*9.81*env.r1_normal[:,2]/4.*.1
        support=(wn>threshold[:,None]).all(-1)
        yaw=(self.robot.data.root_ang_vel_w*env.r1_normal).sum(-1).abs()
        def drift(position,origin):
            delta=position-origin
            return (delta-env.r1_normal*(delta*env.r1_normal).sum(-1,keepdim=True)).norm(dim=-1)
        drift_value=torch.maximum(drift(self.robot.data.root_pos_w,self.hold_origin),
                                  drift(env.scene["trailer"].data.root_pos_w,self.cart_hold_origin))
        posture=((up*env.r1_normal).sum(-1)>=torch.cos(torch.tensor(15.*torch.pi/180.,device=self.device)))
        posture &= mdp.heading_error(env).squeeze(-1).abs()<=10.*torch.pi/180.
        posture &= ((force[:,env.r1_body_sensor_ids].norm(dim=-1)<=1.).all(-1))
        posture &= mdp.hitch_clearance(env)>=.005
        cart_v=core.route_velocity(env.scene["trailer"].data.root_lin_vel_w,env.r1_tangent,env.r1_normal).norm(dim=-1)
        self.hold_count+=window.long()
        self.hold_support_sum+=window*support
        self.hold_speed_sum+=window*torch.maximum(v.norm(dim=-1),cart_v)
        self.hold_yaw_sum+=window*yaw
        self.hold_max_drift=torch.where(window,torch.maximum(self.hold_max_drift,drift_value),self.hold_max_drift)
        self.hold_posture_ok &= (~window)|posture
        count=self.hold_count.clamp_min(1)
        good=(self.hold_count>=250)&(self.hold_support_sum/count>=.95)&(self.hold_speed_sum/count<=.05)
        good &= (self.hold_yaw_sum/count<=.05)&(self.hold_max_drift<=.05)&self.hold_posture_ok
        good &= self.joint_progress()>=env.cfg.eval_distance_m
        self.successful_hold_this_step[:]=completed&completed_hold&good&(env.r1_towed_kg>0)
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
