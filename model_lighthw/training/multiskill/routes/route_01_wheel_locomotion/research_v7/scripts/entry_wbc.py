"""普通调姿全身动力学研究：真实M/C/g、部分接触、原伺服与力矩限幅。"""
import numpy as np
from scipy.spatial.transform import Rotation
from model import Model
from pose_search import basis
from qp import solve,servo_bounds,to_action
from entry_path import DEFAULT_Q
from partial_support_control import attitude


class EntryWBC:
    def __init__(self,plan,reference,slope,contact_gap=.006,recoil=False):
        self.model=Model();self.plan=plan;self.reference=reference;self.B=basis(slope)
        self.duration=float(plan['time_s'][-1]);self.finished=False;self.ready_at=None;self.ready_time=0.
        self.previous=None;self.previous_tau=np.zeros(16);self.last=None
        self.qpath=np.array(plan['q']);self.vpath=np.array(plan['joint_velocity_rad_s'])
        self.apath=np.gradient(self.vpath,.02,axis=0)
        self.ppath=np.array(plan['body_position_plan_m']);self.pv=np.gradient(self.ppath,.02,axis=0);self.pa=np.gradient(self.pv,.02,axis=0)
        self.contact_gap=contact_gap;self.recoil=recoil

    def step(self,base,rec,time,nominal_action):
        from isaaclab.utils.math import quat_apply
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.cart_asset import drawbar_force
        import torch
        cpu=lambda x:x.detach().cpu().numpy().astype(float)
        r=base.scene['robot'];c=base.scene['trailer'];view=r.root_physx_view
        ids=np.array(base.r1_joint_ids);order=np.r_[np.arange(6),ids+6]
        q=cpu(r.data.joint_pos[0])[ids];qd=cpu(r.data.joint_vel[0])[ids]
        quat=cpu(r.data.root_quat_w[0]);p=cpu(r.data.root_pos_w[0]-base.scene.env_origins[0]);R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        v=np.r_[cpu(r.data.root_com_lin_vel_w[0]),cpu(r.data.root_ang_vel_w[0]),qd]
        M=cpu(view.get_generalized_mass_matrices()[0])[np.ix_(order,order)]+np.diag(np.r_[np.zeros(6),cpu(view.get_dof_armatures()[0])[ids]])
        h=cpu(view.get_gravity_compensation_forces()[0]+view.get_coriolis_and_centrifugal_compensation_forces()[0])[order]
        rh=quat_apply(r.data.root_quat_w,base.r1_robot_hitch)+r.data.root_pos_w
        ch=quat_apply(c.data.root_quat_w,base.r1_cart_hitch)+c.data.root_pos_w
        rv=r.data.root_com_lin_vel_w+torch.cross(r.data.root_ang_vel_w,rh-r.data.root_com_pos_w,dim=-1)
        cv=c.data.root_com_lin_vel_w+torch.cross(c.data.root_ang_vel_w,ch-c.data.root_com_pos_w,dim=-1)
        drawbar,_,_=drawbar_force(rh,ch,rv,cv,base.r1_towed_kg>0)
        state=self.model.state(p,quat,q);n=self.B[:,2];A,points,centers=self.model.contacts(state,n)
        external=state['jac']('Base_link',cpu(rh[0]-base.scene.env_origins[0]))[:3].T@cpu(-drawbar[0])
        Adotv=self.model.contact_derivative(p,quat,q,v,n)@v
        gaps=points@n;loads=cpu(rec.buffers['native_ground_normal_w'][0]).mean(0)@n
        mask=(gaps<self.contact_gap)&((loads>1.)|(gaps<self.contact_gap/4.)) if time else np.ones(4,dtype=bool)
        idx=min(round(time/.02),len(self.qpath)-1)
        targetR=Rotation.from_quat(np.array(self.plan['body_quaternion_plan'][idx])[[1,2,3,0]]).as_matrix()
        acceleration=self.pa[idx]+80.*(self.ppath[idx]@n-p@n)*n+16.*(self.pv[idx]@n-v[:3]@n)*n
        acceleration+=self.B[:,0]*np.clip(-4.*float(v[:3]@self.B[:,0]),-3.,3.)
        if self.recoil:
            old_ax=np.clip(-4.*float(v[:3]@self.B[:,0]),-3.,3.)
            # 允许真实整车短暂顺坡滚动以降低进入期牵引力；没有拖车驱动或重力缩放。
            start_ax=-.7*9.81*self.B[2,0]
            ax=start_ax if time<self.duration*.75 else np.clip(-2.*float(v[:3]@self.B[:,0]),-1.5,1.5)
            acceleration+=self.B[:,0]*(ax-old_ax)
        acceleration+=self.B[:,1]*(-40.*float(p@self.B[:,1])-10.*float(v[:3]@self.B[:,1]))
        alpha=60.*Rotation.from_matrix(targetR@R.T).as_rotvec()-12.*v[3:6]
        qa=self.apath[idx,:12]+100.*(self.qpath[idx,:12]-q[:12])+14.*(self.vpath[idx,:12]-qd[:12])
        task=np.eye(22)[:18];target=np.r_[acceleration,alpha,qa]
        weights=np.r_[np.full(3,30.),np.full(3,20.),np.full(12,1.)]
        lo=np.r_[np.full(3,-12.),np.full(3,-40.),np.full(16,-500.)];hi=-lo
        lo[6:18]=np.maximum(lo[6:18],2*(self.model.limits[:12,0]-q[:12]-.02*qd[:12])/.02**2)
        hi[6:18]=np.minimum(hi[6:18],2*(self.model.limits[:12,1]-q[:12]-.02*qd[:12])/.02**2)
        tl,tu=servo_bounds(q,qd,self.model.limits)
        result=solve(M,h,A,Adotv,v,external,self.B,target,task,weights,tl,tu,lo,hi,self.previous,
            mu=3.,minimum_normal=0.,contact_mask=mask,servo_horizon_s=.02,
            tau_reference=self.previous_tau,tau_smooth_scale=np.full(16,5.),tau_smooth_weight=.01)
        if result['success']:
            action=to_action(result['initial_servo_torque'],q,qd,DEFAULT_Q)
            self.previous=result['x'];self.previous_tau=result['tau'];self.last=action.copy()
        else:action=nominal_action if self.last is None else self.last.copy()
        angles,rates=attitude(quat,n,R.T@v[3:6]);goal,_=attitude(self.reference['quat'],n,np.zeros(3))
        ready=(time>=self.duration and abs(p@n-self.reference['height_m'])<.03
            and abs(q[:12]-self.qpath[-1,:12]).max()<.22 and abs(angles-goal).max()<.10 and abs(v[3:6]).max()<.5 and result['success'])
        self.ready_time=self.ready_time+.02 if ready else 0.
        if self.ready_time>=.4 and self.ready_at is None:self.ready_at=time
        blend=0. if self.ready_at is None else float(np.clip((time-self.ready_at)/1.,0.,1.))
        self.finished=blend>=1.;action=(1.-blend)*action+blend*nominal_action
        trace=dict(entry_wbc_success=float(result['success']),entry_wbc_mask=mask.astype(float),entry_wbc_blend=blend,
            entry_wbc_ready_s=self.ready_time,entry_wbc_gap_m=gaps,
            entry_wbc_acceleration=result.get('acceleration',np.zeros(22)),
            entry_wbc_force_w_N=result.get('contact_forces',np.zeros((4,3))),
            entry_wbc_tau_Nm=result.get('tau',np.zeros(16)),
            entry_wbc_solve_s=result['solve_s'],entry_wbc_eq_residual=result.get('eq_residual',-1.))
        return action,trace
