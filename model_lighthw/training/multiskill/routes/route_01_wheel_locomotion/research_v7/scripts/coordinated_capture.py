"""隔离的腿轮联合捕获教师：原生动力学、部分支撑和原50Hz伺服。

输出仍为12个腿位置增量与4个轮速，不施加额外机身力。
QP的接触预测必须与下一周期实际1kHz数据比较；可解不是行为通过。
"""
import numpy as np
from scipy.spatial.transform import Rotation
from model import Model,skew
from pose_search import basis
from qp import solve,servo_bounds,to_action
from entry_path import DEFAULT_Q
from entry_capture import quintic


class CaptureHandoff:
    """捕获到原教师的状态机；用真实失稳序列核查，避免仅按时间完成交接。"""
    def __init__(self,disabled=False):
        self.ready_s=0.;self.progress=0.;self.armed=False;self.finished=False;self.disabled=disabled

    def advance(self,ready,time):
        if self.finished:return 1.
        self.ready_s=self.ready_s+.02 if ready else 0.
        if self.disabled:return 0.
        if self.armed:
            # 接触/姿态或速度失去条件时，按有限速率退回捕获控制，再重新资格确认。
            self.progress=float(np.clip(self.progress+(.02 if ready else -.04),0.,1.))
            if not ready and self.progress==0.:self.armed=False
        elif self.ready_s>=.4:self.armed=True
        blend=quintic(self.progress)
        self.finished=blend>=1.
        return blend


class CoordinatedCapture:
    def __init__(self,reference,slope,servo_horizon=.02,contact_model='native',leg_trust_rad=0.,nominal_hips=False,joint_impedance=False,servo_discrete=False,hold_only=False):
        if servo_horizon not in (0.,.02) or contact_model not in ('native','geometric'):
            raise ValueError('Invalid coordinated capture model')
        self.reference=reference;self.B=basis(slope);self.model=Model()
        self.servo_horizon=servo_horizon;self.contact_model=contact_model
        if not np.isfinite(leg_trust_rad) or not 0<=leg_trust_rad<=.2:raise ValueError('Invalid leg target trust region')
        self.leg_trust_rad=leg_trust_rad
        self.nominal_hips=nominal_hips
        self.joint_impedance=joint_impedance
        if servo_discrete and servo_horizon!=.02:raise ValueError('Discrete servo requires a full 20ms horizon')
        self.servo_discrete=servo_discrete
        self.previous=None;self.finished=False;self.handoff=CaptureHandoff(disabled=hold_only)
        self.lateral=None

    def compute(self,z,nominal):
        """纯NumPy接口：同一原生前态可用于离线候选核查，不能生成成功标签。"""
        m=self.model;B=self.B;n=B[:,2];q=z['q'];v=z['v'];qd=v[6:];p=z['pos'];quat=z['quat']
        state=m.state(p,quat,q);R=state['trans']['Base_link'][:3,:3]
        A,points,_=m.contacts(state,n);Adotv=m.contact_derivative(p,quat,q,v,n)@v
        force_A=A.copy();gap=points@n
        if self.contact_model=='native':
            native_available=(z['counts']>0)&np.isfinite(z['gap'])
            native_loaded=native_available&(z['normal']>1.)
            for k,name in enumerate(('FR','FL','RR','RL')):
                if native_loaded[k]:
                    point=z['cop'][k]
                    force_A[3*k:3*k+3]=state['jac'](name+'_wheel_link',point)[:3]
                    gap[k]=z['gap'][k]
            if z['time']>0.:
                # Native contact loss must not reactivate an ideal circular-wheel
                # support. GPU contact counts can remain positive for separated
                # speculative points, so require both recent load and native gap.
                # Missing points keep the recorder's 1000 m sentinel in the trace.
                gap=np.where(native_available,z['gap'],1000.)
                mask=native_loaded&(gap<.002)
            else:
                mask=(gap<.002)&((z['normal']>1.)|(gap<.0005))
        else:
            mask=(gap<.002)&((z['normal']>1.)|(gap<.0005))
        if z['time']==0.:mask[:]=True
        if self.lateral is None:self.lateral=float(p@B[:,1])
        acceleration=n*np.clip(60.*(self.reference['height_m']-p@n)-12.*(v[:3]@n),-4.,4.)
        acceleration+=B[:,0]*np.clip(-1.5*(v[:3]@B[:,0]),-.3,.3)
        acceleration+=B[:,1]*np.clip(20.*(self.lateral-p@B[:,1])-8.*(v[:3]@B[:,1]),-2.,2.)
        targetR=Rotation.from_quat(np.asarray(self.reference['quat'])[[1,2,3,0]]).as_matrix()
        attitude_error=Rotation.from_matrix(targetR@R.T).as_rotvec()
        alpha=40.*attitude_error-10.*v[3:6]
        qa=50.*(np.asarray(self.reference['q'])[:12]-q[:12])-12.*qd[:12]
        if self.joint_impedance:
            # 原任务的统一50/s²关节反馈忽略各关节的接触约束等效惯量。
            # 用已核对的M及当前滚动约束把原80/3恢复力映射为关节加速度。
            inv=np.linalg.inv(z['M']);active=A[np.repeat(mask,3)]
            mobility=inv-inv@active.T@np.linalg.pinv(active@inv@active.T,rcond=1.e-8)@active@inv
            spring=np.r_[np.zeros(6),80.*(np.asarray(self.reference['q'])[:12]-q[:12])-3.*qd[:12],np.zeros(4)]
            qa=np.clip((mobility@spring)[6:18],-500.,500.)
        task=np.eye(22)[:18];target=np.r_[acceleration,alpha,qa]
        weights=np.r_[np.full(3,50.),np.full(3,100.),np.full(12,2.)]
        lo=np.r_[np.full(3,-12.),np.full(3,-40.),np.full(16,-500.)];hi=-lo
        lo[6:18]=np.maximum(lo[6:18],2*(m.limits[:12,0]-q[:12]-.02*qd[:12])/.02**2)
        hi[6:18]=np.minimum(hi[6:18],2*(m.limits[:12,1]-q[:12]-.02*qd[:12])/.02**2)
        tl,tu=servo_bounds(q,qd,m.limits)
        # 用既有成功控制的瞬时请求作软参考，防止没有必要的内力跳变。
        tau_ref=np.r_[80.*(DEFAULT_Q[:12]+nominal[:12]*np.tile([.08,.14,.14],4)-q[:12])-3.*qd[:12],
            2.*(nominal[12:]*5.-qd[12:])]
        if self.leg_trust_rad:
            # 限制QP对原教师腿位置目标的改动，而不是把有限力矩误当成刚性支撑。
            # 对初始PD力矩施加这个界，与位置目标±leg_trust_rad严格等价。
            tl[:12]=np.maximum(tl[:12],tau_ref[:12]-80.*self.leg_trust_rad)
            tu[:12]=np.minimum(tu[:12],tau_ref[:12]+80.*self.leg_trust_rad)
        if self.nominal_hips:
            # 前后髋相反方向漂移会改变轮轴与横向支撑；保留原髋位置伺服，
            # 仅让QP协调矢状面腿/轮，不能用自由髋运动补偿接触模型误差。
            hip=np.array([0,3,6,9]);tl[hip]=np.maximum(tl[hip],tau_ref[hip]);tu[hip]=np.minimum(tu[hip],tau_ref[hip])
        if self.servo_discrete:
            from discrete_servo_qp import solve as discrete_solve
            result=discrete_solve(z['M'],z['h'],A,Adotv,v,z['external'],B,target,weights,tl,tu,q,m.limits,tau_ref,force_A,mask,self.previous)
        else:
            result=solve(z['M'],z['h'],A,Adotv,v,z['external'],B,target,task,weights,tl,tu,lo,hi,self.previous,
                mu=3.,minimum_normal=0.,contact_mask=mask,force_matrix=force_A,servo_horizon_s=self.servo_horizon,
                tau_reference=tau_ref,tau_smooth_scale=np.full(16,5.),tau_smooth_weight=.1)
        if result['success']:
            self.previous=result['x'];action=to_action(result['initial_servo_torque'],q,qd,DEFAULT_Q)
        else:action=np.asarray(nominal).copy()
        return action,result,dict(mask=mask,gap=gap,attitude_error=attitude_error,requested_acceleration=target)

    def step(self,base,rec,time,nominal):
        from momentum_probe import native_dynamics
        from isaaclab.utils.math import quat_apply
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.cart_asset import drawbar_force
        import torch
        cpu=lambda x:x.detach().cpu().numpy().astype(float)
        r=base.scene['robot'];c=base.scene['trailer'];origin=cpu(base.scene.env_origins[0]);dyn=native_dynamics(base)
        v=dyn['dynamics_pre_velocity'];q=cpu(r.data.joint_pos[0])[base.r1_joint_ids];quat=cpu(r.data.root_quat_w[0])
        rh=quat_apply(r.data.root_quat_w,base.r1_robot_hitch)+r.data.root_pos_w
        ch=quat_apply(c.data.root_quat_w,base.r1_cart_hitch)+c.data.root_pos_w
        rv=r.data.root_com_lin_vel_w+torch.cross(r.data.root_ang_vel_w,rh-r.data.root_com_pos_w,dim=-1)
        cv=c.data.root_com_lin_vel_w+torch.cross(c.data.root_ang_vel_w,ch-c.data.root_com_pos_w,dim=-1)
        force=-cpu(drawbar_force(rh,ch,rv,cv,base.r1_towed_kg>0)[0][0])
        arm=cpu(rh[0]-r.data.root_com_pos_w[0]);J=dyn['dynamics_jacobian'][0]
        normal=cpu(rec.buffers['native_ground_normal_w'][0]).mean(0)@self.B[:,2]
        z=dict(time=time,q=q,quat=quat,pos=cpu(r.data.root_pos_w[0])-origin,v=v,M=dyn['dynamics_mass'],h=dyn['dynamics_bias'],
            external=J[:3].T@force+J[3:].T@np.cross(arm,force),normal=normal,
            counts=cpu(rec.buffers['native_contact_count'][0,-1]),gap=cpu(rec.buffers['native_contact_min_separation_m'][0,-1]),
            cop=cpu(rec.buffers['native_contact_cop_w'][0,-1])-origin)
        if self.finished:
            result=dict(success=False);data=dict(mask=np.zeros(4),gap=z['gap'],attitude_error=np.zeros(3),requested_acceleration=np.zeros(18));action=nominal
            ready=False
        else:
            action,result,data=self.compute(z,np.asarray(nominal))
            supported=normal>5.
            ready=(result['success'] and abs(v[:3]@self.B[:,0])<.08 and abs(z['pos']@self.B[:,2]-self.reference['height_m'])<.03
                and abs(data['attitude_error']).max()<.08 and abs(v[3:6]).max()<.3
                and abs(q[:12]-np.asarray(self.reference['q'])[:12]).max()<.18
                and supported.any() and abs(v[18:][supported]*.0875-v[:3]@self.B[:,0]).max()<.1)
        blend=self.handoff.advance(ready,time)
        self.finished=self.handoff.finished;action=(1-blend)*action+blend*np.asarray(nominal)
        trace=dict(coord_active=float(not self.finished),coord_success=float(result['success']),coord_mask=np.asarray(data['mask'],dtype=float),
            coord_gap_m=data['gap'],coord_normal_measured_N=normal,coord_ready_s=self.handoff.ready_s,coord_handoff=blend,
            coord_attitude_error_rad=data['attitude_error'],coord_target_acceleration=data['requested_acceleration'],
            coord_acceleration=result.get('acceleration',np.zeros(22)),coord_force_w_N=result.get('contact_forces',np.zeros((4,3))),
            coord_tau_mean_Nm=result.get('tau',np.zeros(16)),coord_tau_initial_Nm=result.get('initial_servo_torque',np.zeros(16)),
            coord_velocity_pre=v,coord_solve_s=result.get('solve_s',0.),coord_eq_residual=result.get('eq_residual',-1.))
        return action,trace
