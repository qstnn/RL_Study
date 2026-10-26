"""PhysX浮基动力学 WBC 候选；原50Hz腿位置/轮速度接口，不加推进外力。"""
import numpy as np
from scipy.spatial.transform import Rotation
from model import Model,skew,rotation
from settings import TEACHER_ACCELERATION_LIMIT, GROUND_FRICTION
from pose_search import basis
from qp import solve,servo_bounds,to_action
from contact_state import check_contact,recovery_gaps

def tow_velocity_target(cmd,robot_v,cart_v,cart_mass,robot_mass,fraction,forward_speed_kp=3.,forward_acceleration_limit_mps2=TEACHER_ACCELERATION_LIMIT):
    """Bounded teacher-only acceleration demand for a lagging passive cart.

    The ordinary robot speed loop treats spring stretch as robot acceleration.
    The optional extra term accounts for the cart's measured speed error and
    known mass, while the QP retains its native friction/motor/support limits.
    """
    ax=(forward_speed_kp if cmd>0. else 3.)*(cmd-robot_v)
    cap=forward_acceleration_limit_mps2 if cmd>0. else TEACHER_ACCELERATION_LIMIT
    if cmd>0. and ax>0. and cart_mass>0. and fraction>0.:
        ratio=cart_mass/robot_mass
        ax+=3.*fraction*ratio*max(cmd-cart_v,0.)
        cap=min(2.5,cap*(1.+fraction*ratio))
        return float(np.clip(ax,-TEACHER_ACCELERATION_LIMIT,cap))
    return float(np.clip(ax,-TEACHER_ACCELERATION_LIMIT,cap))

def delayed_forward_bias(cmd,ax,elapsed,amplitude,start,ramp,cap):
    """Research-only model-error probe, smoothly enabled after stable towing begins."""
    if cmd<=0. or ax<=0. or amplitude<=0.:return float(ax)
    u=float(np.clip((elapsed-start)/ramp,0.,1.))
    smooth=u*u*(3.-2.*u)
    return float(np.clip(ax+amplitude*smooth,-TEACHER_ACCELERATION_LIMIT,cap))

def attitude_acceleration(error,omega,ground_normal,yaw_kp=30.):
    """Keep roll/pitch gains fixed while probing a smaller terrain-yaw correction."""
    yaw=np.dot(error,ground_normal)
    return 30.*error-(30.-yaw_kp)*yaw*ground_normal-8.*omega

def native_traction_residual(predicted_forces,friction_history,normal_history,point_count_history,B,robot_mass):
    """Compare the previous executed QP force with its following 20ms native trace.

    Contact dropout or near-saturated tire friction invalidates the residual;
    treating a missing contact as an enormous force deficit would worsen lift.
    """
    predicted=np.asarray(predicted_forces);friction=np.asarray(friction_history)
    loads=np.asarray(normal_history);counts=np.asarray(point_count_history)
    if predicted.shape!=(4,3) or friction.shape!=(20,4,3) or loads.shape!=(20,4,3) or counts.shape!=(20,4) or B.shape!=(3,3) or robot_mass<=0.:
        raise ValueError('Invalid native traction feedback trace')
    mean_friction=friction.mean(0);mean_normal=loads.mean(0)@B[:,2]
    coverage=(counts>0).mean(0)
    usage=np.linalg.norm(mean_friction@B[:,:2],axis=1)/np.maximum(mean_normal,1.e-9)
    finite=all(np.isfinite(x).all() for x in (predicted,friction,loads,counts,mean_normal,usage))
    valid=bool(finite and np.all(coverage>=.95) and np.all(mean_normal>=5.) and np.all(usage<=2.7))
    predicted_t=float((predicted@B[:,0]).sum())
    measured_t=float((mean_friction@B[:,0]).sum())
    raw=float(np.clip((predicted_t-measured_t)/robot_mass,0.,.8)) if valid else 0.
    return dict(valid=valid,predicted_t_N=predicted_t,measured_t_N=measured_t,coverage=coverage.tolist(),
        mean_normal_N=mean_normal.tolist(),friction_usage=usage.tolist(),raw_bias_mps2=raw)

def traction_disturbance_external(external,B,filtered_bias_mps2,robot_mass,fraction):
    """Represent measured wheel-force shortfall in the next QP, not in PhysX.

    The previous QP force exceeds the native contact force by ``mass * bias``.
    Model that shortfall as opposing tangential disturbance so the next QP
    asks its real wheels for the missing force, subject to unchanged actuator,
    friction, support and posture bounds. No force is applied to the simulator.
    """
    external=np.asarray(external,dtype=float);B=np.asarray(B,dtype=float)
    if external.shape!=(22,) or B.shape!=(3,3) or not np.isfinite(external).all() or not np.isfinite(B).all() or not np.isfinite(filtered_bias_mps2) or not np.isfinite(robot_mass) or not np.isfinite(fraction) or robot_mass<=0. or not 0.<=fraction<=1.:
        raise ValueError('Invalid native traction disturbance')
    force_N=float(np.clip(filtered_bias_mps2,0.,.8)*robot_mass*fraction)
    corrected=external.copy();corrected[:3]-=force_N*B[:,0]
    return corrected,force_N

class Teacher:
    def __init__(self,base,pose,physical,options=None):
        defaults=dict(servo_horizon_s=0.,load_balance_weight=0.,minimum_normal_N=1.,native_force_points=0.,contact_recovery_s=0.,tow_velocity_feedback_fraction=0.,forward_speed_kp=3.,forward_acceleration_limit_mps2=TEACHER_ACCELERATION_LIMIT,forward_bias_mps2=0.,forward_bias_start_s=9.,forward_bias_ramp_s=.5,yaw_kp=30.,joint_posture_weight=.1,native_traction_feedback_fraction=0.,native_traction_disturbance_fraction=0.)
        self.options=defaults|dict(options or {})
        if set(self.options)!=set(defaults) or any(not np.isfinite(v) or v<0 for v in self.options.values()):
            raise ValueError('Invalid teacher options')
        if self.options['contact_recovery_s'] not in (0.,.1):raise ValueError('Contact recovery is either disabled or limited to 100ms')
        if self.options['native_force_points'] not in (0.,1.):raise ValueError('native_force_points must be 0 or 1')
        if self.options['minimum_normal_N']<1.:raise ValueError('Cannot lower the original predicted support floor')
        if self.options['servo_horizon_s'] not in (0.,.02):raise ValueError('Servo horizon must be 0 or 20ms')
        if self.options['tow_velocity_feedback_fraction']>1.:raise ValueError('Tow velocity feedback fraction must be [0,1]')
        if not 3.<=self.options['forward_speed_kp']<=8.:raise ValueError('Forward speed gain must be [3,8]')
        if not TEACHER_ACCELERATION_LIMIT<=self.options['forward_acceleration_limit_mps2']<=1.2:raise ValueError('Forward acceleration target cap must be [0.6,1.2]m/s2')
        if not 0.<=self.options['forward_bias_mps2']<=.8 or not 3.<=self.options['forward_bias_start_s']<=15. or not .1<=self.options['forward_bias_ramp_s']<=2.:
            raise ValueError('Invalid delayed forward bias magnitude/start/ramp')
        if not 5.<=self.options['yaw_kp']<=30.:raise ValueError('Terrain yaw gain must be [5,30]')
        if not .1<=self.options['joint_posture_weight']<=5.:raise ValueError('Joint posture weight must be [0.1,5]')
        if self.options['native_traction_feedback_fraction']>1.:raise ValueError('Native traction feedback fraction must be [0,1]')
        if self.options['native_traction_disturbance_fraction']>1.:raise ValueError('Native traction disturbance fraction must be [0,1]')
        self.physical=physical
        self.base=base;self.robot=base.scene['robot'];self.model=Model();self.pose=pose
        self.ids=list(base.r1_joint_ids);self.order=np.r_[np.arange(6),np.asarray(self.ids)+6]
        self.wheels=[self.robot.body_names.index(k+'_wheel_link') for k in ('FR','FL','RR','RL')]
        self.loss_started={};self.closed=set();self.previous={};self.last_action={};self.last_success_elapsed={};self.anchors={};self.holds={};self.failures={};self.audit={};self.last={};self.traction_bias={}

    def apply(self,baseline,step):
        import torch
        from isaaclab.utils.math import quat_apply
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.cart_asset import drawbar_force
        b,r=self.base,self.robot;view=r.root_physx_view;c=b.scene['trailer']
        rh=quat_apply(r.data.root_quat_w,b.r1_robot_hitch)+r.data.root_pos_w
        ch=quat_apply(c.data.root_quat_w,b.r1_cart_hitch)+c.data.root_pos_w
        rr=rh-r.data.root_com_pos_w;cr=ch-c.data.root_com_pos_w
        rv=r.data.root_com_lin_vel_w+torch.cross(r.data.root_ang_vel_w,rr,dim=-1)
        cv=c.data.root_com_lin_vel_w+torch.cross(c.data.root_ang_vel_w,cr,dim=-1)
        force,_,_=drawbar_force(rh,ch,rv,cv,b.r1_towed_kg>0)
        cpu=lambda x:x.detach().cpu().numpy().astype(float)
        values=dict(M=cpu(view.get_generalized_mass_matrices()),g=cpu(view.get_gravity_compensation_forces()),
            c=cpu(view.get_coriolis_and_centrifugal_compensation_forces()),J=cpu(view.get_jacobians()),
            arm=cpu(view.get_dof_armatures()),q=cpu(r.data.joint_pos),qd=cpu(r.data.joint_vel),
            pos=cpu(r.data.root_pos_w),quat=cpu(r.data.root_quat_w),v=cpu(r.data.root_com_lin_vel_w),omega=cpu(r.data.root_ang_vel_w),
            body_com=cpu(r.data.body_com_pos_w),hitch=cpu(rh),external=cpu(-force),
            default=cpu(r.data.default_joint_pos),cmd=cpu(b.command_manager.get_command('base_velocity')[:,0]),
            cart_v=cpu((c.data.root_com_lin_vel_w*b.r1_tangent).sum(-1)),towed_kg=cpu(b.r1_towed_kg),
            origin=cpu(b.scene.env_origins),normal=cpu(b.r1_normal),slope=cpu(b.r1_slope_rad),elapsed=cpu(b.episode_length_buf)*.02,
            contact=cpu(b.scene['contact_forces'].data.net_forces_w[:,b.r1_wheel_sensor_ids]),applied_tau=cpu(r.data.applied_torque),
            contact_history=cpu(self.physical.buffers['contact_force_w']),
            friction_history=cpu(self.physical.buffers['friction_force_w']),
            contact_cop=cpu(self.physical.buffers['native_contact_cop_w']),
            contact_count=cpu(self.physical.buffers['native_contact_count'][:,-1]),
            contact_separation=cpu(self.physical.buffers['native_contact_min_separation_m'][:,-1]),
            contact_count_history=cpu(self.physical.buffers['native_contact_count']),
            contact_separation_history=cpu(self.physical.buffers['native_contact_min_separation_m']),
            wheel_center_history=cpu(self.physical.buffers['wheel_pos_w']))
        action=baseline.clone();self.last={}
        for i in range(b.num_envs):
            if i in self.closed or i in self.failures or values['elapsed'][i]<1.5:continue
            try:
                a,info=self.single(i,values)
                self.last[i]=info
                if not info['success']:
                    self.failures[i]=dict(step=step,time_s=float(values['elapsed'][i]),**{k:v for k,v in info.items() if k not in ('x','tau','contact_forces','acceleration','rolling_velocity')})
                    continue
                action[i]=torch.as_tensor(a,device=action.device,dtype=action.dtype)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                self.failures[i]=dict(step=step,time_s=float(values['elapsed'][i]),reason=type(exc).__name__+': '+str(exc))
        return action

    def single(self,i,z):
        m=self.model;q=z['q'][i,self.ids];qd=z['qd'][i,self.ids];p=z['pos'][i];quat=z['quat'][i]
        state=m.state(p,quat,q);normal=z['normal'][i];B=basis(np.rad2deg(z['slope'][i]))
        v=np.r_[z['v'][i],z['omega'][i],qd]
        Aurdf,points,_=m.contacts(state,normal);jac=z['J'][i][:,:,self.order]
        # PhysX雅可比基点和连杆线速度都在各刚体COM，必须加接触力臂。
        A=np.vstack([jac[j,:3]-skew(points[k]-z['body_com'][i,j])@jac[j,3:] for k,j in enumerate(self.wheels)])
        rawM=z['M'][i][np.ix_(self.order,self.order)]
        # 本机API短验：get_generalized_mass_matrices不包含DOF armature；仿真关节惯量需加回。
        arm=z['arm'][i,self.ids];M=rawM+np.diag(np.r_[np.zeros(6),arm])
        g=z['g'][i,self.order];h=g+z['c'][i,self.order]
        checkM,checkg=m.dynamics(state)
        errors=dict(jacobian_max_abs=float(np.max(abs(A-Aurdf))),mass_matrix_max_abs=float(np.max(abs(rawM-checkM))),
            gravity_max_abs=float(np.max(abs(g-checkg))),min_mass_eigenvalue=float(np.linalg.eigvalsh(M).min()))
        if errors['jacobian_max_abs']>2.e-4 or errors['mass_matrix_max_abs']>.005 or errors['gravity_max_abs']>.005 or errors['min_mass_eigenvalue']<=0:
            raise ValueError('Dynamics coordinate/model crosscheck failed: '+str(errors))
        self.audit[i]=errors
        normal_load=z['contact'][i]@normal
        contact_estimate=check_contact(z['contact_history'][i],normal,z['contact_count'][i],z['contact_separation'][i])
        contact_mask=np.ones(4,dtype=bool)
        if not contact_estimate['supported']:
            self.traction_bias[i]=0.
            if not self.options['contact_recovery_s']:
                return None,dict(success=False,reason='four_contact_assumption_failed',contact_estimate=contact_estimate)
            native_gaps=np.asarray(contact_estimate['minimum_separation_m']);loads=np.asarray(contact_estimate['mean_normal_N'])
            gaps=recovery_gaps(z['wheel_center_history'][i],z['contact_separation_history'][i],
                z['contact_count_history'][i],normal,z['origin'][i])
            contact_estimate['recovery_geometric_gap_m']=gaps.tolist()
            contact_mask=(loads>=1.)&(native_gaps<=1.e-4)&(native_gaps>=-1.e-3)&(np.asarray(contact_estimate['point_count'])>0)
            if not np.isfinite(gaps).all() or not np.isfinite(loads).all() or contact_mask.sum()<3 or np.any(gaps>.005) or np.any(gaps<-.001):
                return None,dict(success=False,reason='contact_recovery_outside_envelope',contact_estimate=contact_estimate)
        else:
            gaps=np.asarray(contact_estimate['minimum_separation_m'])
        for k in range(4):
            key=(i,k)
            if contact_mask[k]:self.loss_started.pop(key,None)
            else:
                since=self.loss_started.setdefault(key,z['elapsed'][i])
                if z['elapsed'][i]-since>=self.options['contact_recovery_s']:
                    return None,dict(success=False,reason='contact_recovery_timeout',contact_estimate=contact_estimate)
        if not np.all(contact_mask):
            # A near-ground, one-frame native contact dropout is not a new
            # three-wheel stance. Replanning with only three supports produced
            # saturated torques and ~100 m/s2 root acceleration in the trace.
            # Hold one previously validated actuator target within the same
            # 5 mm / 100 ms envelope, then reobserve; never count this as
            # evidence of continuous four-wheel contact in the evaluator.
            if i not in self.last_action:
                return None,dict(success=False,reason='contact_recovery_no_previous_action',contact_estimate=contact_estimate)
            return self.last_action[i].copy(),dict(success=True,reason='transient_contact_hold',
                contact_estimate=contact_estimate,contact_mask=contact_mask.tolist())

        traction_feedback=None
        if self.options['native_traction_feedback_fraction'] or self.options['native_traction_disturbance_fraction']:
            recent_success=abs(z['elapsed'][i]-self.last_success_elapsed.get(i,-1.)-.02)<.005
            if z['elapsed'][i]>=self.options['forward_bias_start_s'] and i in self.previous and recent_success:
                traction_feedback=native_traction_residual(self.previous[i][22:].reshape(4,3),z['friction_history'][i],
                    z['contact_history'][i],z['contact_count_history'][i],B,sum(m.mass.values()))
                if traction_feedback['valid']:
                    previous_bias=self.traction_bias.get(i,0.)
                    alpha=.02/(.4+.02)
                    self.traction_bias[i]=previous_bias+alpha*(traction_feedback['raw_bias_mps2']-previous_bias)
                else:self.traction_bias[i]=0.
            else:self.traction_bias[i]=0.
        
        force_A=A;force_points=points
        if self.options['native_force_points']:
            weights=np.maximum(z['contact_history'][i]@normal,0.)
            force_points=(z['contact_cop'][i]*weights[:,:,None]).sum(0)/np.maximum(weights.sum(0)[:,None],1.e-12)
            force_A=np.vstack([jac[j,:3]-skew(force_points[k]-z['body_com'][i,j])@jac[j,3:] for k,j in enumerate(self.wheels)])
            independent=np.vstack([state['jac'](leg+'_wheel_link',force_points[k])[:3] for k,leg in enumerate(('FR','FL','RR','RL'))])
            if np.max(abs(force_A-independent))>2.e-4:raise ValueError('Native force point Jacobian mismatch')
        Adotv=m.contact_derivative(p,quat,q,v,normal)@v
        external=state['jac']('Base_link',z['hitch'][i])[:3].T@z['external'][i]
        tau_lo,tau_hi=servo_bounds(q,qd,m.limits)
        # 原1/.5Nm每步硬限额与下坡摩擦约束冲突；实测QP反例已保存。
        # 该尺度改为软平滑归一化，不改变96/48Nm和原生伺服可达硬界。
        last_tau=z['applied_tau'][i,self.ids];delta=np.r_[np.ones(12),np.full(4,.5)]
        # Keep a contact-model discontinuity from requesting an impulse-like
        # floating-base motion even when the optimizer reports a low residual.
        lo=np.r_[np.full(3,-5.),np.full(3,-20.),np.full(16,-500.)];hi=-lo
        dt=.02
        lo[6:18]=np.maximum(lo[6:18],2*(m.limits[:12,0]-q[:12]-dt*qd[:12])/dt**2)
        hi[6:18]=np.minimum(hi[6:18],2*(m.limits[:12,1]-q[:12]-dt*qd[:12])/dt**2)
        lo[18:]=np.maximum(lo[18:],(-60.-qd[12:])/dt);hi[18:]=np.minimum(hi[18:],(60.-qd[12:])/dt)
        com=m.whole_com(state);mass=sum(m.mass.values())
        Jcom=sum(m.mass[n]*state['jac'](n,state['com'][n])[:3] for n in m.links)/mass
        if i not in self.anchors:self.anchors[i]=dict(q=q.copy(),height=float(com@normal),rotation=rotation(quat),lateral=float(com@B[:,1]),x=float(com@B[:,0]))
        anchor=self.anchors[i];ramp=np.clip((z['elapsed'][i]-1.5)/1.,0.,1.)
        targetR=B@Rotation.from_rotvec([0.,-np.deg2rad(self.pose['pitch_deg']),0.]).as_matrix()
        targetq=np.asarray(self.pose['q']);targetstate=m.state(z['origin'][i]+normal*self.pose['height_m'],np.roll(Rotation.from_matrix(targetR).as_quat(),1),targetq)
        targetheight=float(m.whole_com(targetstate)@normal)
        desiredheight=(1-ramp)*anchor['height']+ramp*targetheight
        comv=Jcom@v;cmd=z['cmd'][i]
        if abs(cmd)<1.e-6:
            self.holds.setdefault(i,float(com@B[:,0]));ax=3*(-comv@B[:,0])+4*(self.holds[i]-com@B[:,0])
        else:
            self.holds.pop(i,None)
            ax=tow_velocity_target(cmd,comv@B[:,0],z['cart_v'][i],z['towed_kg'][i],mass,self.options['tow_velocity_feedback_fraction'],
                self.options['forward_speed_kp'],self.options['forward_acceleration_limit_mps2'])
        ax_limit=self.options['forward_acceleration_limit_mps2'] if cmd>0. else TEACHER_ACCELERATION_LIMIT
        if cmd>0. and ax>0. and self.options['tow_velocity_feedback_fraction']:
            ax_limit=min(2.5,ax_limit*(1.+self.options['tow_velocity_feedback_fraction']*z['towed_kg'][i]/mass))
        feedback_bias=self.traction_bias.get(i,0.)*self.options['native_traction_feedback_fraction']
        ax=delayed_forward_bias(cmd,ax,z['elapsed'][i],self.options['forward_bias_mps2']+feedback_bias,
            self.options['forward_bias_start_s'],self.options['forward_bias_ramp_s'],ax_limit)
        disturbance_N=0.
        if cmd>0. and self.options['native_traction_disturbance_fraction']:
            external,disturbance_N=traction_disturbance_external(external,B,self.traction_bias.get(i,0.),mass,
                self.options['native_traction_disturbance_fraction'])
        acceleration=B@np.array([np.clip(ax,-TEACHER_ACCELERATION_LIMIT,ax_limit),-10*(com@B[:,1]-anchor['lateral'])-4*(comv@B[:,1]),
            60*(desiredheight-com@normal)-12*(comv@normal)])
        Rgoal=Rotation.from_rotvec(ramp*Rotation.from_matrix(targetR@anchor['rotation'].T).as_rotvec()).as_matrix()@anchor['rotation']
        alpha=attitude_acceleration(Rotation.from_matrix(Rgoal@rotation(quat).T).as_rotvec(),v[3:6],normal,self.options['yaw_kp'])
        qgoal=anchor['q']*(1-ramp)+targetq*ramp;qa=25*(qgoal[:12]-q[:12])-6*qd[:12]
        angular=np.zeros((3,22));angular[:,3:6]=np.eye(3)
        posture=np.zeros((12,22));posture[:,6:18]=np.eye(12)
        # COM雅可比偏置用同一构型方向差分计算，保持与接触Adot相同的坐标约定。
        jcs=[];epsilon=1.e-5;R=rotation(quat)
        for sign in (-1.,1.):
            ds=sign*epsilon;Rn=Rotation.from_rotvec(v[3:6]*ds).as_matrix()@R
            pn=state['ref']+ds*v[:3]-Rn@m.base_offset
            st=m.state(pn,np.roll(Rotation.from_matrix(Rn).as_quat(),1),q+ds*qd)
            jcs.append(sum(m.mass[n]*st['jac'](n,st['com'][n])[:3] for n in m.links)/mass)
        jdotv=(jcs[1]-jcs[0])/(2*epsilon)@v
        # 一步预测的根高度、牵引净高和姿态边界；软任务不能覆盖这些硬研究约束。
        extra=[];lower=[];upper=[]
        for point,min_height,max_height in ((p,.336,.496),(z['hitch'][i],.01,1.)):
            J=state['jac']('Base_link',point)[:3];lever=point-state['ref']
            bias_point=np.cross(v[3:6],np.cross(v[3:6],lever))
            height=(point-z['origin'][i])@normal;speed=normal@J@v
            extra.append(normal@J)
            lower.append(2*(min_height-height-dt*speed)/dt**2-normal@bias_point)
            upper.append(2*(max_height-height-dt*speed)/dt**2-normal@bias_point)
        relative=Rotation.from_matrix(B.T@rotation(quat)).as_rotvec()
        # 小姿态旋转向量的一步局部线性化，roll/pitch各±10.5°；实际姿态仍按完整协议验证。
        for axis in (0,1):
            row=np.zeros(22);row[3:6]=B[:,axis];speed=B[:,axis]@v[3:6]
            extra.append(row);bound=np.deg2rad(10.5)
            lower.append(2*(-bound-relative[axis]-dt*speed)/dt**2)
            upper.append(2*(bound-relative[axis]-dt*speed)/dt**2)
        task=np.vstack((Jcom,angular,posture));target=np.r_[acceleration-jdotv,alpha,qa]
        task_weight=np.r_[np.full(3,30.),np.full(3,10.),np.full(12,self.options['joint_posture_weight'])]
        for k in np.flatnonzero(~contact_mask):
            row=normal@A[3*k:3*k+3];gap=float(gaps[k])
            desired=np.clip(-400.*max(gap,0.)-20.*(row@v),-2.,2.)-normal@Adotv[3*k:3*k+3]
            task=np.vstack((task,row));target=np.r_[target,desired];task_weight=np.r_[task_weight,30.]
        result=solve(M,h,A,Adotv,v,external,B,target,task,
            task_weight,tau_lo,tau_hi,lo,hi,self.previous.get(i),
            mu=GROUND_FRICTION,extra_matrix=np.array(extra),extra_lower=np.array(lower),extra_upper=np.array(upper),
            tau_reference=last_tau,tau_smooth_scale=delta,tau_smooth_weight=.1,
            servo_horizon_s=self.options['servo_horizon_s'],minimum_normal=self.options['minimum_normal_N'],
            normal_reference=next(scene['normal_N'] for scene in self.pose['scenarios'] if scene['phase']=='static' and abs(scene['slope_deg']-np.rad2deg(z['slope'][i]))<.01),
            load_balance_weight=self.options['load_balance_weight'],force_matrix=force_A,contact_mask=contact_mask)
        result['contact_estimate']=contact_estimate
        if not result['success']:return None,result
        action=to_action(result['initial_servo_torque'],q,qd,z['default'][i,self.ids])
        if not np.isfinite(action).all() or np.max(abs(action))>100.:
            return None,dict(success=False,reason='native_raw_action_range')
        self.previous[i]=result['x'];self.last_action[i]=np.asarray(action).copy();self.last_success_elapsed[i]=z['elapsed'][i];result.update(whole_com_w=com,actual_normal_N=normal_load,model_contact_points_w=points,force_contact_points_w=force_points,
            commanded_com_acceleration_w=acceleration,predicted_com_acceleration_w=Jcom@result['acceleration']+jdotv,
            pre_joint_position=q,pre_joint_velocity=qd,pre_com_velocity_w=comv,
            native_traction_feedback=traction_feedback,filtered_traction_bias_mps2=feedback_bias,
            qp_model_disturbance_N=disturbance_N)
        return action,result
