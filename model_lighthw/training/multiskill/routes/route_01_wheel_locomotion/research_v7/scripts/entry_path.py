"""普通站姿到承重姿态的接地几何路径；只生成关节/轮速参考，不写仿真根状态。"""
import numpy as np
from scipy.spatial.transform import Rotation
from model import Model
from partial_support_control import solve_foot_pose
from research_common import LEGS

DEFAULT_Q=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])


def build_path(reference,slope_deg,duration,dt=.02,forward_m=0.,crouch_first=False,start_progress=0.,preserve_wheel_phase=False):
    if duration<.2 or abs(round(duration/dt)*dt-duration)>1.e-8:raise ValueError('Entry duration must be >=.2s on the action grid')
    if not np.isfinite(forward_m) or abs(forward_m)>.3:raise ValueError('Entry displacement must be within0.3m')
    if not np.isfinite(start_progress) or not 0<=start_progress<=1:raise ValueError('Initial pose progress must lie in [0,1]')
    if start_progress and crouch_first:raise ValueError('Pose continuation and crouch-first are separate diagnostics')
    model=Model();slope=np.deg2rad(slope_deg);normal=np.array([-np.sin(slope),0,np.cos(slope)]);tangent=np.array([np.cos(slope),0,np.sin(slope)])
    R0=Rotation.from_euler('y',-slope);R1=Rotation.from_quat(np.asarray(reference['quat'])[[1,2,3,0]])
    rotvec=(R1*R0.inv()).as_rotvec();p0=normal*.416223;p1=normal*reference['height_m']+tangent*forward_m
    source_q=np.asarray(reference['q']);q=DEFAULT_Q.copy()
    if preserve_wheel_phase:q[12:]=source_q[12:]
    states=[model.state(p0,R0.as_quat()[[3,0,1,2]],q),model.state(p1,R1.as_quat()[[3,0,1,2]],source_q)]
    centers=[np.array([s['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS]) for s in states]
    times=np.arange(round(duration/dt)+1)*dt;x=times/duration;fraction=x**3*(10-15*x+6*x*x);rate=30*x*x*(1-x)**2/duration
    if start_progress:
        # 仅插值接地几何；初始化时允许取路径起点，随后根姿态完全由物理积分。
        fraction=start_progress+(1-start_progress)*fraction;rate=(1-start_progress)*rate
        q=source_q.copy()
    qs=[];positions=[];quats=[];max_error=0.
    if crouch_first and duration<.6:raise ValueError('Crouch-first path requires >=0.6s')
    for i,a in enumerate(fraction):
        if crouch_first:
            split=.3;phase=times[i]/split if times[i]<=split else (times[i]-split)/(duration-split)
            a=np.clip(phase,0.,1.)**3*(10.-15.*np.clip(phase,0.,1.)+6.*np.clip(phase,0.,1.)**2)
            if times[i]<=split:R=R0;p=p0+a*(normal*.26-p0);goal=centers[0]
            else:R=Rotation.from_rotvec(a*rotvec)*R0;p=normal*.26+a*(p1-normal*.26);goal=centers[0]+a*(centers[1]-centers[0])
        else:
            R=Rotation.from_rotvec(a*rotvec)*R0;p=p0+a*(p1-p0)
            goal=centers[0]+a*(centers[1]-centers[0])
        local=(goal-p)@R.as_matrix()
        q=source_q.copy() if start_progress==1. else solve_foot_pose(model,q,local,np.zeros((4,3)))
        s=model.state(p,R.as_quat()[[3,0,1,2]],q)
        actual=np.array([s['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS])
        max_error=max(max_error,float(abs(actual-goal).max()))
        qs.append(q.copy());positions.append(p);quats.append(R.as_quat()[[3,0,1,2]])
    qs=np.array(qs);qd=np.gradient(qs,dt,axis=0);qd[[0,-1]]=0.
    path_v=np.gradient(np.array(positions),dt,axis=0);path_v[[0,-1]]=0.
    path_w=np.zeros((len(times),3))
    if crouch_first:
        for i in range(1,len(times)-1):
            before=Rotation.from_quat(quats[i-1][[1,2,3,0]]);after=Rotation.from_quat(quats[i+1][[1,2,3,0]])
            path_w[i]=(after*before.inv()).as_rotvec()/(2*dt)
    wheels=[];residuals=[]
    for i,(q,p,quat) in enumerate(zip(qs,positions,quats)):
        R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();omega=rotvec*rate[i]
        v=np.r_[(p1-p0)*rate[i]+np.cross(omega,R@model.base_offset),omega,qd[i]];v[18:]=0
        if crouch_first:
            omega=path_w[i];v[:3]=path_v[i]+np.cross(omega,R@model.base_offset);v[3:6]=omega
        A,_,_=model.contacts(model.state(p,quat,q),normal)
        for k in range(4):
            row=tangent@A[3*k:3*k+3];v[18+k]=-float(row@v)/row[18+k]
        wheels.append(v[18:].copy());residuals.append(float(abs((A@v).reshape(4,3)@tangent).max()))
    if max_error>1.e-4:raise ValueError('Entry path unreachable within original joint limits')
    return dict(time_s=times.tolist(),q=np.asarray(qs).tolist(),joint_velocity_rad_s=qd.tolist(),wheel_rad_s=np.asarray(wheels).tolist(),
        body_position_plan_m=np.asarray(positions).tolist(),body_quaternion_plan=np.asarray(quats).tolist(),
        maximum_ik_error_m=max_error,maximum_tangent_rolling_residual_mps=max(residuals),
        maximum_planned_leg_speed_rad_s=float(abs(qd[:,:12]).max()),
        planned_root_forward_m=forward_m,crouch_first=crouch_first,start_progress=start_progress,preserve_wheel_phase=preserve_wheel_phase,
        note='Kinematic reference only. Root poses/velocities are never commanded to the simulation after reset.')
