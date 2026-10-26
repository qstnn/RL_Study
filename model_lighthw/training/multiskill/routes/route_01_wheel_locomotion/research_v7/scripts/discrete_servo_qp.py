"""固定接触模式内的20个1ms伺服预测；研究用，不保证下一周期接触成立。

决策量是原位置/轮速接口对应的16个初始力矩。逐步保留80/3腿和2轮阻尼，
不以恒加速度近似其响应。M、接触雅可比、外载在一个周期内冻结，实际接触
冲量和构型变化仍需1kHz执行记录核对。只返回动作，不写物理状态或施加辅助力。
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize
from qp import friction_rows


def predict(M,h,A,Adot_v,v,external,force_A,mask,steps=20,dt=.001):
    """返回形如[...,17]的仿射量，最后一列是常量，其余列乘初始力矩。"""
    M,h,A,Adot_v,v,external,force_A=[np.asarray(x,dtype=np.float64) for x in (M,h,A,Adot_v,v,external,force_A)]
    active=np.repeat(mask,3);C=A[active];J=force_A[active]
    inv=np.linalg.inv(M);S=np.eye(22)[:,6:]
    contact_inverse=np.linalg.pinv(C@inv@J.T,rcond=1.e-9)
    dx=np.zeros((22,17));dv=dx.copy();initial=np.column_stack((np.eye(16),np.zeros(16)))
    K=np.r_[np.full(12,80.),np.zeros(4)];D=np.r_[np.full(12,3.),np.full(4,2.)]
    records={key:[] for key in ('tau','force','acc','dx','dv','constraint_residual')}
    for _ in range(steps):
        tau=initial-K[:,None]*dx[6:]-D[:,None]*dv[6:]
        drive=S@tau;drive[:,-1]+=external-h
        rhs=-8.*C@dv;rhs[:,-1]+=-Adot_v[active]-8.*C@v
        f=contact_inverse@(rhs-C@inv@drive)
        acc=inv@(drive+J.T@f)
        force=np.zeros((12,17));force[active]=f
        residual=C@acc-rhs
        dv=dv+dt*acc
        # PhysX semi-implicit position integration; frozen coordinates are an approximation.
        dx=dx+dt*dv;dx[:,-1]+=dt*v
        for key,value in [('tau',tau),('force',force),('acc',acc),('dx',dx),('dv',dv),('constraint_residual',residual)]:
            records[key].append(value.copy())
    return {key:np.asarray(value) for key,value in records.items()}


def solve(M,h,A,Adot_v,v,external,B,target,task_weights,tau_lower,tau_upper,q,limits,
          tau_reference,force_A,mask,previous=None):
    started=time.monotonic()
    if np.any(tau_lower>tau_upper):return dict(success=False,reason='empty_action_bounds',solve_s=time.monotonic()-started)
    pred=predict(M,h,A,Adot_v,v,external,force_A,mask)
    mean_acc=pred['acc'].mean(0);mean_force=pred['force'].mean(0);mean_tau=pred['tau'].mean(0)
    G=[];bound=[]
    def upper(affine,limit):
        a=np.asarray(affine).reshape(-1,17);b=np.broadcast_to(limit,np.asarray(affine).shape[:-1]).reshape(-1)
        G.append(a[:,:16]);bound.append(b-a[:,-1])
    torque_limit=np.r_[np.full(12,96.),np.full(4,48.)]
    upper(pred['tau'],torque_limit);upper(-pred['tau'],torque_limit)
    upper(pred['dx'][:,6:18],limits[:12,1]-q[:12])
    upper(-pred['dx'][:,6:18],q[:12]-limits[:12,0])
    F,fb=friction_rows(B,3.,0.)
    upper(np.einsum('ij,njk->nik',F,pred['force']),fb)
    G=np.vstack(G);bound=np.concatenate(bound)
    scale=np.maximum(np.linalg.norm(G,axis=1),1.);Gn=G/scale[:,None];bn=bound/scale
    C=mean_acc[:18,:16];offset=mean_acc[:18,-1]-target
    H=C.T@(task_weights[:,None]*C)+np.eye(16)*.004
    linear=C.T@(task_weights*offset)-.004*tau_reference
    x0=previous
    if x0 is None or x0.shape!=(16,) or np.max(G@x0-bound)>.001 or np.any(x0<tau_lower) or np.any(x0>tau_upper):
        feasible=linprog(np.zeros(16),A_ub=Gn,b_ub=bn,bounds=list(zip(tau_lower,tau_upper)),method='highs')
        if not feasible.success:return dict(success=False,reason='discrete_constraints_infeasible',solve_s=time.monotonic()-started)
        x0=feasible.x
    result=minimize(lambda x:.5*x@H@x+linear@x,x0,jac=lambda x:H@x+linear,method='SLSQP',
        bounds=list(zip(tau_lower,tau_upper)),constraints=[dict(type='ineq',fun=lambda x:bn-Gn@x,jac=lambda x:-Gn)],
        options=dict(maxiter=100,ftol=1.e-7))
    x=result.x;ax=np.r_[x,1.]
    eq=float(abs(pred['constraint_residual']@ax).max()) if pred['constraint_residual'].size else 0.
    violation=float(max(0.,np.max(G@x-bound),np.max(tau_lower-x),np.max(x-tau_upper)))
    good=bool(result.success and np.isfinite(x).all() and eq<1.e-4 and violation<1.e-4)
    return dict(success=good,reason='ok' if good else 'discrete_solver_or_residual_failure',message=result.message,
        x=x,eq_residual=eq,inequality_violation=violation,acceleration=mean_acc@ax,
        contact_forces=(mean_force@ax).reshape(4,3),tau=mean_tau@ax,initial_servo_torque=x,
        solve_s=time.monotonic()-started,iterations=result.nit)
