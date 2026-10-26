"""联合筛选COM、根高度、牵引净高、四轮法向载荷；静力候选不代表动态成功。"""
import itertools,json
import numpy as np
from scipy.optimize import least_squares,linprog
from scipy.spatial.transform import Rotation
from research_common import ROOT, ROUTE, NAMES, LEGS, launch
from model import Model,RADIUS
from qp import friction_rows
from settings import GROUND_FRICTION, LOADED_ACCELERATION, TEACHER_ACCELERATION_LIMIT


def basis(angle):
    a=np.deg2rad(angle);t=np.array([np.cos(a),0.,np.sin(a)]);n=np.array([-np.sin(a),0.,np.cos(a)])
    return np.column_stack((t,np.cross(n,t),n))


def force_lp(A,bias,B,mu,minimum_normal=0.):
    """联合6维动力学、16轴力矩和4轮内接摩擦锥，最大化最弱轮载荷。"""
    E=np.column_stack((A.T[:6],np.zeros(6)))
    F,fb=friction_rows(B,mu=mu,minimum_normal=minimum_normal)
    G=np.pad(F,((0,0),(0,1)));b=list(fb)
    limits=np.r_[np.full(12,96.),np.full(4,48.)]
    G=np.vstack((G,np.column_stack((-A.T[6:],np.zeros(16))),np.column_stack((A.T[6:],np.zeros(16)))))
    b.extend(limits-bias[6:]);b.extend(limits+bias[6:])
    for k in range(4):
        row=np.zeros(13);row[k*3:k*3+3]=-B[:,2];row[-1]=1.;G=np.vstack((G,row));b.append(0.)
    objective=np.zeros(13);objective[-1]=-1.
    return linprog(objective,A_ub=G,b_ub=b,A_eq=E,b_eq=bias[:6],bounds=[(None,None)]*12+[(minimum_normal,None)],method='highs')


def search(out):
    m=Model()
    source=ROUTE/'unified_v5/results/2026-10-19_towing_recovery/control_screen_02/seed43_case00_policy_trace.npz'
    q0=np.load(source)['joint_pos'][100:140].mean(0).astype(float)
    nominal=m.state(np.zeros(3),np.array([1.,0.,0.,0.]),q0)
    centers=np.array([nominal['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS]);centers[:,2]=RADIUS
    reports=[];problems={}
    for height,pitch,shift in itertools.product((.38,.40,.42),(-6.,-3.,0.),(0.,.04,.08)):
        key=f'h{height:.2f}_p{pitch:g}_x{shift:.2f}';r=dict(id=key,height_m=height,pitch_deg=pitch,forward_shift_m=shift,feasible=False)
        R=Rotation.from_rotvec([0.,-np.deg2rad(pitch),0.]).as_matrix();quat=np.roll(Rotation.from_matrix(R).as_quat(),1)
        pos=np.array([shift,0.,height]);q=q0.copy();errors=[]
        for k,leg in enumerate(LEGS):
            ix=np.arange(k*3,k*3+3)
            def fun(x):
                trial=q.copy();trial[ix]=x
                return m.state(pos,quat,trial)['trans'][leg+'_wheel_link'][:3,3]-centers[k]
            fit=least_squares(fun,q[ix],bounds=(m.limits[ix,0]+1.e-5,m.limits[ix,1]-1.e-5),max_nfev=100,gtol=1.e-10,ftol=1.e-10,xtol=1.e-10)
            q[ix]=fit.x;errors.append(float(np.linalg.norm(fit.fun)))
        r['ik_error_m']=max(errors)
        if max(errors)>1.e-4:r['reason']='unreachable_wheel_centers';reports.append(r);continue
        scenarios=[]
        for slope,(phase,acceleration,rolling) in itertools.product((25.,-20.),(('static',0.,False),('start',LOADED_ACCELERATION,True),('cruise',0.,True),('brake',-LOADED_ACCELERATION,True),('correction_forward',TEACHER_ACCELERATION_LIMIT,True),('correction_brake',-TEACHER_ACCELERATION_LIMIT,True))):
            B=basis(slope);worldpos=B@pos;worldR=B@R;worldq=np.roll(Rotation.from_matrix(worldR).as_quat(),1)
            st=m.state(worldpos,worldq,q);A,contacts,_=m.contacts(st,B[:,2]);M,g=m.dynamics(st)
            # 当前LIGHTHW_CFG轮轴armature；与GPU原生get_dof_armatures短验一致。
            # 它改变加速轴力矩，不增加浮基平动质量或地面合力。
            M[18:,18:]+=np.eye(4)*.00243216
            hitch=worldpos+worldR@np.array([-.38,0.,.07-.416223])
            clear=float(hitch@B[:,2]);com=m.whole_com(st)
            tension=40*9.81*(np.sin(np.deg2rad(slope))+(.02*np.cos(np.deg2rad(slope)) if rolling else 0.))+42*acceleration
            force=-tension*B[:,0];ext=st['jac']('Base_link',hitch)[:3].T@force
            acc=np.zeros(22);acc[:3]=acceleration*B[:,0]
            # 轮轴符号由实际URDF雅可比确定，不能假设四轴名称方向等同。
            for k in range(4):
                column=A[k*3:k*3+3,18+k]
                acc[18+k]=-acceleration/(B[:,0]@column)
            bias=M@acc+g-ext
            result=force_lp(A,bias,B,GROUND_FRICTION)
            problems.setdefault(key,[]).append((A,bias,B))
            limits=np.r_[np.full(12,96.),np.full(4,48.)]
            entry=dict(slope_deg=slope,phase=phase,acceleration_mps2=acceleration,drawbar_signed_N=tension,hitch_clearance_m=clear,
                com_lane_m=(B.T@com).tolist(),feasible=bool(result.success and clear>=.01))
            if result.success:
                forces=result.x[:12].reshape(4,3);tau=bias[6:]-A.T[6:]@result.x[:12]
                entry.update(minimum_normal_N=float(result.x[-1]),normal_N=(forces@B[:,2]).tolist(),forces_w_N=forces.tolist(),
                    joint_torque_Nm=tau.tolist(),max_torque_fraction=float(np.max(abs(tau)/limits)))
            scenarios.append(entry)
        r.update(q=q.tolist(),scenarios=scenarios,feasible=all(s['feasible'] for s in scenarios))
        if r['feasible']:r['worst_minimum_normal_N']=min(s['minimum_normal_N'] for s in scenarios)
        reports.append(r)
    selected=sorted((r for r in reports if r['feasible']),key=lambda r:r['worst_minimum_normal_N'],reverse=True)[:3]
    for candidate in selected:
        for entry,(A,bias,B) in zip(candidate['scenarios'],problems[candidate['id']]):
            low,high=0.,GROUND_FRICTION
            if not force_lp(A,bias,B,high,minimum_normal=1.).success:
                entry['minimum_mu_with_1N_each_wheel']=None
                continue
            for _ in range(22):
                middle=(low+high)/2
                if force_lp(A,bias,B,middle,minimum_normal=1.).success:high=middle
                else:low=middle
            entry['minimum_mu_with_1N_each_wheel']=high
        minima=[x['minimum_mu_with_1N_each_wheel'] for x in candidate['scenarios']]
        candidate['coupled_minimum_mu']=max(minima) if all(x is not None for x in minima) else None
        candidate['friction_margin_fraction']=GROUND_FRICTION/candidate['coupled_minimum_mu']-1 if candidate['coupled_minimum_mu'] else None
    result=dict(ground_friction=GROUND_FRICTION,stage='pose_screen',dynamic_success=False,pose_source=str(source),pose_source_sha256=launch.sha256(source),
        candidate_count=len(reports),candidates=reports,selected=selected,
        assumptions='Rigid ideal discs, parallel drawbar, fixed stance, no collision dynamics. 1cm hitch screen; native behavior guard remains5mm. Static LP only.')
    launch.write_json(out,result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,required=True);a=p.parse_args();search(a.output)
