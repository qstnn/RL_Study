"""34变量直线四轮 WBC 原型；硬动力学/滚动/力矩/摩擦约束，软运动任务。

求解失败返回证据，不放宽物理上限。SLSQP用于离线教师研究，未声称实时硬件可用。
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize
from settings import GROUND_FRICTION
from servo_model import correction,initial_torque


def friction_rows(B,mu=GROUND_FRICTION,minimum_normal=1.):
    rows=[];limits=[]
    for k in range(4):
        for a in np.arange(16)*2*np.pi/16:
            row=np.zeros(12);row[k*3:k*3+3]=B@np.array([np.cos(a),np.sin(a),-mu*np.cos(np.pi/16)])
            rows.append(row);limits.append(0.)
        row=np.zeros(12);row[k*3:k*3+3]=-B[:,2];rows.append(row);limits.append(-minimum_normal)
    return np.array(rows),np.array(limits)


def solve(M,h,A,Adot_v,velocity,external,B,target_acceleration,task_matrix,task_weights,
          tau_lower,tau_upper,accel_lower,accel_upper,previous=None,mu=GROUND_FRICTION,extra_matrix=None,extra_lower=None,extra_upper=None,
          tau_reference=None,tau_smooth_scale=None,tau_smooth_weight=.1,servo_horizon_s=0.,
          minimum_normal=1.,normal_reference=None,load_balance_weight=0.,force_matrix=None,contact_mask=None):
    """输入均为明确坐标/顺序的numpy数组；输出残差和未经裁剪的关节力矩。"""
    started=time.monotonic();force_A=A if force_matrix is None else np.asarray(force_matrix)
    if force_A.shape!=A.shape or not np.isfinite(force_A).all():raise ValueError('Invalid force Jacobian')
    D=np.column_stack((M,-force_A.T));bias=h-external
    mask=np.ones(4,dtype=bool) if contact_mask is None else np.asarray(contact_mask,dtype=bool)
    if mask.shape!=(4,):raise ValueError('Contact mask must describe four wheels')
    active=np.repeat(mask,3)
    E=np.vstack((D[:6],np.column_stack((A[active],np.zeros((int(active.sum()),12))))))
    rhs=np.r_[-bias[:6],(-Adot_v-8.*(A@velocity))[active]]
    if not mask.all():
        # 离地轮不能提供接触力；删除其滚动约束，同时将三维接触力严格置零。
        zero=np.zeros((int((~active).sum()),34));zero[:,22:]=np.eye(12)[~active]
        E=np.vstack((E,zero));rhs=np.r_[rhs,np.zeros(len(zero))]
    T=D[6:];offset=bias[6:]
    F,fb=friction_rows(B,mu,minimum_normal)
    for k in np.flatnonzero(~mask):fb[k*17+16]=0.
    servo_bias,servo_acc=correction(velocity[6:],servo_horizon_s)
    initial_T=T.copy();initial_T[:,6:22]+=np.diag(servo_acc)
    initial_offset=offset+servo_bias
    G=np.vstack((initial_T,-initial_T,np.column_stack((np.zeros((len(F),22)),F))))
    bound=np.r_[tau_upper-initial_offset,-tau_lower+initial_offset,fb]
    if servo_horizon_s:
        # 恒加速度近似下的末端力矩也受原电机限幅约束；中间实际值另行审计。
        hdt=servo_horizon_s
        end_T=T.copy();end_T[:,6:22]-=np.diag(np.r_[np.full(12,80*hdt*hdt/3+3*hdt/2),np.full(4,hdt)])
        end_offset=offset-servo_bias
        limit=np.r_[np.full(12,96.),np.full(4,48.)]
        G=np.vstack((G,end_T,-end_T));bound=np.r_[bound,limit-end_offset,limit+end_offset]
    if extra_matrix is not None:
        extra=np.column_stack((extra_matrix,np.zeros((len(extra_matrix),12))))
        G=np.vstack((G,extra,-extra));bound=np.r_[bound,extra_upper,-extra_lower]
    lower=np.r_[accel_lower,np.full(12,-2000.)];upper=np.r_[accel_upper,np.full(12,2000.)]
    if np.any(lower>upper) or np.any(tau_lower>tau_upper):
        return dict(success=False,reason='empty_action_or_velocity_bounds',solve_s=time.monotonic()-started)
    # 量纲缩放只影响求解条件数；最后用原始N/Nm/m/s²残差独立验收。
    es=np.maximum(np.linalg.norm(E,axis=1),1.);gs=np.maximum(np.linalg.norm(G,axis=1),1.)
    En=E/es[:,None];rn=rhs/es;Gn=G/gs[:,None];bn=bound/gs
    C=np.column_stack((task_matrix,np.zeros((len(task_matrix),12))))
    W=np.asarray(task_weights);H=C.T@(W[:,None]*C)+np.diag(np.r_[np.full(22,1.e-4),np.full(12,2.e-5)])
    linear=-C.T@(W*target_acceleration)
    if tau_reference is not None:
        # 保留真实力矩/摩擦硬界；变化量仅作软代价，不能把可行稳定动作排除。
        reference=np.asarray(tau_reference);scale=np.asarray(tau_smooth_scale)
        if reference.shape!=(16,) or scale.shape!=(16,) or not np.isfinite(reference).all() or not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite(tau_smooth_weight) or tau_smooth_weight<0:
            raise ValueError('Invalid torque smoothing target/scale/weight')
        weights=tau_smooth_weight/scale**2
        H+=T.T@(weights[:,None]*T)
        linear+=T.T@(weights*(offset-reference))
    if normal_reference is not None and load_balance_weight:
        reference=np.asarray(normal_reference,dtype=float)
        if reference.shape!=(4,) or not np.isfinite(reference).all():raise ValueError('Invalid normal load reference')
        N=np.zeros((4,34))
        for k in range(4):N[k,22+3*k:25+3*k]=B[:,2]
        H+=load_balance_weight*N.T@N;linear-=load_balance_weight*N.T@reference
    x0=previous
    if x0 is None or np.max(abs(E@x0-rhs))>.1 or np.max(G@x0-bound)>.1 or np.any(x0<lower) or np.any(x0>upper):
        feasible=linprog(np.zeros(34),A_ub=Gn,b_ub=bn,A_eq=En,b_eq=rn,bounds=list(zip(lower,upper)),method='highs')
        if not feasible.success:
            return dict(success=False,reason='hard_constraints_infeasible',message=feasible.message,solve_s=time.monotonic()-started)
        x0=feasible.x
    result=minimize(lambda x:.5*x@H@x+linear@x,x0,jac=lambda x:H@x+linear,method='SLSQP',
        bounds=list(zip(lower,upper)),constraints=[dict(type='eq',fun=lambda x:En@x-rn,jac=lambda x:En),
        dict(type='ineq',fun=lambda x:bn-Gn@x,jac=lambda x:-Gn)],options=dict(maxiter=100,ftol=1.e-7))
    x=result.x;eq=float(np.max(abs(E@x-rhs)));ineq=float(max(0.,np.max(G@x-bound),np.max(lower-x),np.max(x-upper)))
    success=bool(result.success and np.isfinite(x).all() and eq<1.e-4 and ineq<1.e-4)
    return dict(success=success,reason='ok' if success else 'solver_or_residual_failure',message=result.message,
        eq_residual=eq,inequality_violation=ineq,x=x,tau=T@x+offset,contact_forces=x[22:].reshape(4,3),
        contact_mask=mask,initial_servo_torque=initial_T@x+initial_offset,servo_horizon_s=servo_horizon_s,
        acceleration=x[:22],rolling_velocity=A@velocity,torque_change_Nm=(T@x+offset-np.asarray(tau_reference)) if tau_reference is not None else None,solve_s=time.monotonic()-started,iterations=result.nit)


def servo_bounds(q,qd,position_limits,kp=80.,kd=3.,wheel_damping=2.):
    """交集约束：原96/48Nm、腿位置界与±50rad/s轮速度目标，不靠动作后裁剪。"""
    lo=-np.r_[np.full(12,96.),np.full(4,48.)];hi=-lo
    lo[:12]=np.maximum(lo[:12],kp*(position_limits[:12,0]-q[:12])-kd*qd[:12])
    hi[:12]=np.minimum(hi[:12],kp*(position_limits[:12,1]-q[:12])-kd*qd[:12])
    lo[12:]=np.maximum(lo[12:],wheel_damping*(-50.-qd[12:]))
    hi[12:]=np.minimum(hi[12:],wheel_damping*(50.-qd[12:]))
    return lo,hi


def to_action(tau,q,qd,default):
    """取样瞬时PD反解。20ms持有后的误差必须另用1kHz轨迹衡量。"""
    out=np.empty(16);scale=np.tile([.08,.14,.14],4)
    out[:12]=(q[:12]+(tau[:12]+3.*qd[:12])/80.-default[:12])/scale
    out[12:]=(qd[12:]+tau[12:]/2.)/5.
    return out
