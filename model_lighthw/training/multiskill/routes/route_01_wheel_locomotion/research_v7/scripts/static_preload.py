"""按当前坡向及真实牵引几何重算静态支撑；只修改关节伺服目标。"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize,linprog
from model import Model
from pose_search import basis,force_lp
from qp import friction_rows
from partial_support_control import SCALE,attitude


def rebase(reference,slope_deg,cart_mass,hitch_height,robot_offset,stiffness,mu,front_share=None):
    model=Model();B=basis(slope_deg);tangent,normal=B[:,0],B[:,2]
    R=Rotation.from_quat(np.asarray(reference['quat'])[[1,2,3,0]]).as_matrix()
    q=np.asarray(reference['q']);root=normal*reference['height_m'];state=model.state(root,reference['quat'],q)
    robot_point=root+R@np.array([-.38,0.,hitch_height+robot_offset-.416223])
    length=1.+cart_mass*9.81*np.sin(np.deg2rad(slope_deg))/2000.
    normal_gap=float(robot_point@normal-hitch_height)
    if abs(normal_gap)>=length:raise ValueError('Infeasible static hitch geometry')
    direction=(tangent*np.sqrt(length*length-normal_gap*normal_gap)+normal*normal_gap)/length
    external=-2000.*(length-1.)*direction
    A,points,centers=model.contacts(state,normal);_,g=model.dynamics(state)
    bias=g-state['jac']('Base_link',robot_point)[:3].T@external
    lp=force_lp(A,bias,B,mu)
    if not lp.success:raise ValueError('Loaded posture lacks static wrench feasibility at this slope')
    F,b=friction_rows(B,mu=mu,minimum_normal=0.)
    limits=np.r_[np.full(12,96.),np.full(4,48.)]
    G=np.vstack((F,-A.T[6:],A.T[6:]));bounds=np.r_[b,limits-bias[6:],limits+bias[6:]]
    E=A.T[:6];rhs=bias[:6];seed=lp.x[:12]
    if front_share is not None:
        if not 0<front_share<1:raise ValueError('Invalid static traction share')
        tangential=np.kron(np.eye(4),tangent[None,:])
        weights=np.array([front_share,front_share,1-front_share,1-front_share])/2.
        total=float(bias[:3]@tangent)
        E=np.vstack((E,tangential[:3]));rhs=np.r_[rhs,total*weights[:3]]
        matched=linprog(np.zeros(12),A_ub=G,b_ub=bounds,A_eq=E,b_eq=rhs,bounds=[(None,None)]*12,method='highs')
        if not matched.success:raise ValueError('Fixed wheel traction share is statically infeasible')
        seed=matched.x
    def objective(f):
        local=f.reshape(4,3)@B
        return float(np.sum(f*f)+10*np.sum(local[:,1]**2)+10*np.sum((local[::2]-local[1::2])**2))
    result=minimize(objective,seed,method='SLSQP',constraints=[
        {'type':'eq','fun':lambda f:E@f-rhs},
        {'type':'ineq','fun':lambda f:bounds-G@f}],options={'maxiter':200,'ftol':1.e-8})
    force=(result.x if result.success else seed).reshape(4,3)
    tau=bias[6:]-A.T[6:]@force.ravel()
    residual=float(abs(A.T[:6]@force.ravel()-bias[:6]).max())
    if residual>1.e-5:raise ValueError('Static wrench residual too large')
    default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5])
    target=q[:12]+tau[:12]/np.asarray(stiffness)
    if np.any(target<model.limits[:12,0]) or np.any(target>model.limits[:12,1]):raise ValueError('Static servo target exceeds original joint limits')
    output=dict(reference)
    output.update(leg_action=((target-default)/SCALE).tolist(),force_body_N=(force@R).tolist(),
        moment_body_Nm=(np.cross(points-centers,force)@R).tolist(),
        angles_rad=attitude(reference['quat'],normal,np.zeros(3))[0].tolist(),
        static_rebase=dict(slope_deg=slope_deg,front_force_share=front_share,external_force_w_N=external.tolist(),force_world_N=force.tolist(),
            normal_N=(force@normal).tolist(),tangent_N=(force@tangent).tolist(),joint_torque_Nm=tau.tolist(),
            residual_N_Nm=residual,source_leg_action=reference['leg_action'],qp_success=bool(result.success),
            note='Ideal static calculation only; actual passive cart simulation and full brake/hold audit required.'))
    return output


def mirror_posture(reference,slope_deg):
    """下坡前后交换支撑几何；URDF限位内IK，不交换质量/电机或施加额外支撑。"""
    from partial_support_control import solve_foot_pose
    from research_common import LEGS
    model=Model();q=np.asarray(reference['q']);state=model.state(np.zeros(3),[1,0,0,0],q)
    centers=np.array([state['trans'][l+'_wheel_link'][:3,3] for l in LEGS])[[2,3,0,1]];centers[:,0]*=-1
    guess=q.copy();guess[:12]=q[:12].reshape(4,3)[[2,3,0,1]].ravel();guess[:12].reshape(4,3)[:,1:]*=-1
    candidate=solve_foot_pose(model,guess,centers,np.zeros((4,3)))
    solved=model.state(np.zeros(3),[1,0,0,0],candidate)
    error=float(abs(np.array([solved['trans'][l+'_wheel_link'][:3,3] for l in LEGS])-centers).max())
    if error>1.e-4:raise ValueError('Mirrored posture unreachable')
    plane=Rotation.from_euler('y',-slope_deg,degrees=True)
    relative=(plane.inv()*Rotation.from_quat(np.asarray(reference['quat'])[[1,2,3,0]])).as_matrix()
    reflection=np.diag([-1.,1.,1.]);rotation=plane*Rotation.from_matrix(reflection@relative@reflection)
    result=dict(reference,q=candidate.tolist(),quat=rotation.as_quat()[[3,0,1,2]].tolist())
    result['mirrored_posture']=dict(source_q=reference['q'],source_quat=reference['quat'],maximum_ik_error_m=error,
        note='Fore/aft contact geometry reflected for downhill; static servo preload must be recalculated.')
    return result
