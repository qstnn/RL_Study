"""普通初态调姿：当前几何的支撑分配与伺服前馈，仅输出原16轴动作。

QP的浮基残差是诊断量，不能充作实际接触力；不写根状态、不施加外部助力。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from model import Model
from entry_path import DEFAULT_Q
from partial_support_control import SCALE, attitude, solve_foot_pose
from research_common import LEGS
from pose_search import basis
from qp import friction_rows


class EntrySupport:
    def __init__(self, plan, reference, slope, kp, kd, adaptive=False, grounded=False,wheel_torque=False,planned=False,force_filter=0.,target_rate=0.):
        self.plan=plan;self.reference=reference;self.model=Model();self.B=basis(slope)
        self.kp=np.asarray(kp);self.kd=np.asarray(kd);self.adaptive=adaptive
        self.clock=0.;self.previous_force=None;self.ready_time=0.;self.finished=False
        self.qpath=np.asarray(plan['q']);self.vpath=np.asarray(plan['joint_velocity_rad_s'])
        self.ppath=np.asarray(plan['body_position_plan_m']);self.rpath=np.asarray(plan['body_quaternion_plan'])
        self.duration=float(plan['time_s'][-1]);self.ready_at=None
        self.grounded=grounded
        self.wheel_torque=wheel_torque
        self.planned=planned
        self.force_filter=force_filter;self.target_rate=target_rate;self.filtered_external=None;self.last_target=DEFAULT_Q[:12].copy()
        self.height_path=self.ppath@self.B[:,2]
        self.height_velocity=np.gradient(self.height_path,.02)
        self.height_acceleration=np.gradient(self.height_velocity,.02)

    def step(self,time,q,qd,quat,omega,height,velocity,external,hitch_local,normal_forces,nominal_action):
        if self.force_filter:
            if self.filtered_external is None:self.filtered_external=external.copy()
            self.filtered_external+=(-np.expm1(-.02/self.force_filter))*(external-self.filtered_external)
            external=self.filtered_external.copy()
        n=self.B[:,2];t=self.B[:,0];R=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        idx=min(round(self.clock/.02),len(self.qpath)-1)
        desired=self.qpath[idx].copy();desired_v=self.vpath[idx].copy()
        height_error=float(height-self.ppath[idx]@n)
        tracking_error=float(abs(q[:12]-desired[:12]).max())
        rate=1.
        if self.adaptive:
            # 只控制参考路径进度；物理时钟、真实负载及验收时长保持。
            rate=float(np.clip(1.-max(abs(height_error)-.015,0.)/.06,0.,1.))
            rate=min(rate,float(np.clip(1.-max(tracking_error-.12,0.)/.35,0.,1.)))
        desired_v*=rate
        angles,rates=attitude(quat,n,omega)
        target_angles,_=attitude(self.rpath[idx],n,np.zeros(3))
        state=self.model.state(n*height,quat,q)
        if self.planned:state=self.model.state(self.ppath[idx],self.rpath[idx],desired)
        A,points,centers=self.model.contacts(state,n)
        _,g=self.model.dynamics(state)
        if self.grounded:
            planned=self.model.state(self.ppath[idx],self.rpath[idx],desired)
            goal=np.array([planned['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS])
            desired=solve_foot_pose(self.model,q,(goal-n*height)@R,np.zeros((4,3)))
            desired_v[:12]=0.
        point=n*height+R@hitch_local
        if self.planned:point=state['trans']['Base_link'][:3,3]+state['trans']['Base_link'][:3,:3]@hitch_local
        bias=g-state['jac']('Base_link',point)[:3].T@external
        E=A.T[:6];rhs=bias[:6].copy()
        if self.grounded:
            acceleration=np.clip(self.height_acceleration[idx]*rate**2-80.*height_error
                +16.*(self.height_velocity[idx]*rate-float(np.asarray(velocity)@n)),-4.,4.)
            rhs[:3]+=sum(self.model.mass.values())*acceleration*n
        # 俯仰/横滚阻尼通过地面力矩请求产生，不施加浮基外力。
        correction=np.array([-18.*(angles[1]-target_angles[1])-2.*rates[1],
                             18.*(angles[0]-target_angles[0])+2.*rates[0],0.])
        rhs[3:6]+=R@correction
        F,b=friction_rows(self.B,mu=3.,minimum_normal=0.)
        # 法向离地轮不能提供计划支撑；很小接触间隙容忍原碰撞接触偏置。
        gaps=points@n
        upper=np.where(gaps>.02,0.,350.)
        if self.planned:upper[:]=350.
        N=np.kron(np.eye(4),n[None,:])
        limits=np.r_[np.full(12,96.),np.full(4,48.)]
        G=np.vstack((F,N,-A.T[6:],A.T[6:]));bound=np.r_[b,upper,limits-bias[6:],limits+bias[6:]]
        weights=np.array([1.,1.,1.,10.,10.,10.])
        H=E.T@(weights[:,None]*E)+np.eye(12)*.002
        linear=-E.T@(weights*rhs)
        seed=np.zeros(12) if self.previous_force is None else self.previous_force.copy()
        result=minimize(lambda f:.5*f@H@f+linear@f,seed,jac=lambda f:H@f+linear,
            constraints=[dict(type='ineq',fun=lambda f:bound-G@f,jac=lambda f:-G)],
            method='SLSQP',options=dict(maxiter=80,ftol=1.e-8))
        valid=bool(np.isfinite(result.x).all() and np.max(G@result.x-bound)<1.e-4)
        force=result.x if valid else np.zeros(12);self.previous_force=force.copy()
        tau=bias[6:]-A.T[6:]@force
        target=desired[:12]+tau[:12]/self.kp+self.kd*desired_v[:12]/self.kp
        # 保持原关节限位及执行器限矩；裁剪量也随轨迹保存。
        unclipped=target.copy();target=np.clip(target,self.model.limits[:12,0],self.model.limits[:12,1])
        if self.target_rate:target=self.last_target+np.clip(target-self.last_target,-.02*self.target_rate,.02*self.target_rate)
        self.last_target=target.copy()
        leg=(target-DEFAULT_Q[:12])/SCALE
        # 腿调姿期滚动运动与力矩合成；路径结束后保持零速直到准备完成。
        wheel=np.asarray(self.plan['wheel_rad_s'][idx])*rate+tau[12:]/2.
        if self.wheel_torque:wheel=qd[12:]+tau[12:]/2.
        final_error=float(abs(q[:12]-self.qpath[-1,:12]).max())
        final_angles,_=attitude(self.reference['quat'],n,np.zeros(3))
        ready=(self.clock>=self.duration and abs(height-self.reference['height_m'])<.04
               and final_error<.22 and abs(angles-final_angles).max()<.12 and abs(omega).max()<.5)
        self.ready_time=self.ready_time+.02 if ready else 0.
        if self.ready_time>=.4 and self.ready_at is None:self.ready_at=time
        blend=0. if self.ready_at is None else float(np.clip((time-self.ready_at)/1.,0.,1.))
        action=np.r_[leg,np.clip(wheel,-50.,50.)/5.]
        action=(1.-blend)*action+blend*nominal_action
        self.finished=blend>=1.
        self.clock=min(self.duration,self.clock+.02*rate)
        trace=dict(entry_support_clock_s=self.clock,entry_support_rate=rate,
            entry_support_height_error_m=height_error,entry_support_joint_error_rad=tracking_error,
            entry_support_force_w_N=force.reshape(4,3),entry_support_wrench_residual=E@force-rhs,
            entry_support_qp_success=float(result.success and valid),entry_support_ready_s=self.ready_time,
            entry_support_blend=blend,entry_support_target_clipping_rad=unclipped-target,
            entry_support_contact_gap_m=gaps,entry_support_tau_Nm=tau)
        return action,trace
