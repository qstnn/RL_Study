"""CPU物理/协议检查；不替代Isaac GPU行为验收。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.spatial.transform import Rotation
from model import Model,RADIUS
from qp import solve,servo_bounds,to_action,friction_rows
from pose_search import basis
import protocol_v7
from gates import teacher_gate

class Geometry(unittest.TestCase):
    def setUp(self):
        self.m=Model();self.q=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])
        self.p=np.array([.12,-.2,.42]);self.quat=np.array([1.,0.,0.,0.]);self.n=np.array([0.,0.,1.])
        self.s=self.m.state(self.p,self.quat,self.q)
    def test_mass_and_gravity_virtual_work(self):
        M,g=self.m.dynamics(self.s);mass=sum(self.m.mass.values())
        np.testing.assert_allclose(M[:3,:3],mass*np.eye(3),atol=1.e-12)
        np.testing.assert_allclose(g[:3],[0,0,mass*9.81],atol=1.e-12)
        self.assertGreater(np.linalg.eigvalsh(M).min(),0.)
        eps=1.e-6
        for j in (0,4,9,12):
            trial=self.q.copy();trial[j]+=eps;a=self.m.state(self.p,self.quat,trial)
            trial[j]-=2*eps;b=self.m.state(self.p,self.quat,trial)
            energy=lambda st:sum(self.m.mass[n]*9.81*st['com'][n][2] for n in self.m.links)
            self.assertAlmostEqual(g[j+6],(energy(a)-energy(b))/(2*eps),places=6)
    def test_com_jacobian_against_world_motion(self):
        qdot=np.linspace(-.2,.3,16);name='RL_wheel_link';J=self.s['jac'](name,self.s['com'][name])
        eps=1.e-6;a=self.m.state(self.p,self.quat,self.q+eps*qdot);b=self.m.state(self.p,self.quat,self.q-eps*qdot)
        np.testing.assert_allclose(J[:3,6:]@qdot,(a['com'][name]-b['com'][name])/(2*eps),atol=1.e-8)
    def test_pure_rolling_not_zero_material_acceleration(self):
        A,points,_=self.m.contacts(self.s,self.n);v=np.zeros(22);v[0]=.3
        for k in range(4):v[18+k]=-.3/A[k*3,18+k]
        np.testing.assert_allclose(A@v,0.,atol=1.e-12)
        np.testing.assert_allclose(self.m.contact_derivative(self.p,self.quat,self.q,v,self.n)@v,0.,atol=1.e-8)
        # 同一个轮固连材料点经dt后不再是几何最低点；它具有向心加速度。
        name='FR_wheel_link';local=np.linalg.inv(self.s['trans'][name])@np.r_[points[0],1.]
        eps=1.e-5;velocities=[]
        for sign in (-1,1):
            dt=sign*eps;st=self.m.state(self.p+dt*v[:3],self.quat,self.q+dt*v[6:]);point=(st['trans'][name]@local)[:3]
            velocities.append(st['jac'](name,point)[:3]@v)
        acceleration=(velocities[1]-velocities[0])/(2*eps)
        self.assertAlmostEqual(acceleration@self.n,.3**2/RADIUS,places=5)
    def test_stationary_derivative(self):
        np.testing.assert_equal(self.m.contact_derivative(self.p,self.quat,self.q,np.zeros(22),self.n),np.zeros((12,22)))
    def test_tilted_leg_and_yaw_derivative_convergence(self):
        quat=np.roll(Rotation.from_euler('xyz',[.05,-.35,.12]).as_quat(),1)
        q=self.q.copy();q[0]=.03;q[3]=-.02
        v=np.linspace(-.12,.18,22)
        first=self.m.contact_derivative(self.p,quat,q,v,self.n,epsilon=2.e-5)
        second=self.m.contact_derivative(self.p,quat,q,v,self.n,epsilon=1.e-5)
        np.testing.assert_allclose(first,second,atol=1.e-8)
    def test_servo_inverse_at_sample_only(self):
        qd=np.linspace(-.1,.1,16);tau=np.linspace(-2.,2.,16)
        a=to_action(tau,self.q,qd,self.q)
        recovered=np.r_[80*(self.q[:12]+a[:12]*np.tile([.08,.14,.14],4)-self.q[:12])-3*qd[:12],2*(a[12:]*5-qd[12:])]
        np.testing.assert_allclose(tau,recovered,atol=2.e-14)
        lo,hi=servo_bounds(self.q,qd,self.m.limits);self.assertTrue(np.all(lo<=tau)&np.all(tau<=hi))
    def test_friction_polygon_is_inside_circular_cone(self):
        F,b=friction_rows(np.eye(3),mu=4.)
        for angle in np.linspace(0,2*np.pi,97):
            f=np.tile([4.001*np.cos(angle),4.001*np.sin(angle),1.],4)
            self.assertGreater(np.max(F@f-b),0.)
    def test_qp_dynamics_and_torque_bounds(self):
        M,g=self.m.dynamics(self.s);A,_,_=self.m.contacts(self.s,self.n)
        result=solve(M,g,A,np.zeros(12),np.zeros(22),np.zeros(22),np.eye(3),np.zeros(22),np.eye(22),np.full(22,100.),
            np.r_[np.full(12,-96.),np.full(4,-48.)],np.r_[np.full(12,96.),np.full(4,48.)],np.full(22,-100.),np.full(22,100.))
        self.assertTrue(result['success'],result)
        rhs=np.r_[np.zeros(6),result['tau']]+A.T@result['contact_forces'].ravel()
        np.testing.assert_allclose(M@result['acceleration']+g,rhs,atol=1.e-5)
        self.assertLess(result['eq_residual'],1.e-4)
    def test_infeasible_dynamics_not_fallback_success(self):
        M,g=self.m.dynamics(self.s);A,_,_=self.m.contacts(self.s,self.n)
        ext=np.zeros(22);ext[0]=1.e6
        result=solve(M,g,A,np.zeros(12),np.zeros(22),ext,np.eye(3),np.zeros(22),np.eye(22),np.ones(22),
            np.full(16,-1.),np.ones(16),np.full(22,-.01),np.full(22,.01))
        self.assertFalse(result['success']);self.assertEqual(result['reason'],'hard_constraints_infeasible')

class Friction(unittest.TestCase):
    def test_nominal_and_correction_bounds(self):
        from friction import requirement
        nominal=requirement(25.,0.,40.,.3)
        self.assertAlmostEqual(nominal['ground_longitudinal_signed_N'],237.42319298,places=6)
        self.assertAlmostEqual(nominal['mu_circular_lower_bound'],2.28864194,places=7)
        braking=requirement(-20.,0.,40.,-.3)
        self.assertLess(braking['drawbar_signed_N'],0.)
        self.assertAlmostEqual(braking['mu_circular_lower_bound'],1.69283706,places=7)
    def test_selected_margin_and_sixty_four_conditions(self):
        from friction import calculate
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:r=calculate(Path(d)/'friction.json')
        self.assertEqual(r['selected_static_friction'],3.)
        self.assertGreaterEqual(r['actual_margin_over_16sided_bound'],.20)
        conditions={(x['slope_deg'],x['payload_kg'],x['towed_kg']) for x in r['nominal_cases']}
        self.assertEqual(len(conditions),64)
        self.assertLess(r['teacher_correction_envelope']['mu_16sided_conservative_bound'],2.492)

class Acceptance(unittest.TestCase):
    def test_route_coverage(self):
        self.assertEqual(len(protocol_v7.cases()),18);full=protocol_v7.cases(suite='full');self.assertEqual(len(full),128)
        self.assertEqual(len({(c['slope_deg'],c['payload_kg'],c['towed_kg']) for c in full}),64)
        self.assertTrue(any(c['slope_deg']==-20. and c['towed_kg']==40 and c['speed']>0 for c in protocol_v7.cases()))
        self.assertFalse(any(c['slope_deg']==-25. and c['towed_kg'] for c in full))
        self.assertFalse(any(c['payload_kg'] and c['towed_kg'] for c in full))
    def test_empty_teacher_evidence_rejected(self):
        self.assertFalse(teacher_gate([])['passed'])

class Inference(unittest.TestCase):
    def test_row_inference_preserves_single_case_and_permutation(self):
        import torch
        from inference import infer_actor
        torch.manual_seed(42)
        actor=torch.nn.Sequential(torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,16)).eval()
        obs=torch.randn(4,300)
        with torch.inference_mode():
            expected=torch.cat([actor(row[None]) for row in obs])
            actual=infer_actor(actor,obs)
            self.assertTrue(torch.equal(expected,actual))
            order=torch.tensor([3,0,2,1])
            self.assertTrue(torch.equal(infer_actor(actor,obs[order]),actual[order]))
            self.assertTrue(torch.equal(infer_actor(actor,obs,'batch'),actor(obs)))
        with self.assertRaises(ValueError):infer_actor(actor,obs,'unknown')

class ContactEstimation(unittest.TestCase):
    def test_short_missing_contact_uses_recent_native_radius(self):
        from contact_state import recovery_gaps
        center=np.zeros((20,4,3));center[:,:,2]=.085
        gap=np.zeros((20,4));count=np.full((20,4),4.)
        center[-1,0,2]=.087;gap[-1,0]=1000.;count[-1,0]=0.
        np.testing.assert_allclose(recovery_gaps(center,gap,count,np.array([0.,0.,1.]),np.zeros(3))[0],.002,atol=1.e-12)
        count[:,1]=0.;gap[:,1]=1000.
        self.assertEqual(recovery_gaps(center,gap,count,np.array([0.,0.,1.]),np.zeros(3))[1],1000.)
    def test_recorded_zero_impulse_with_closed_contact_not_flight(self):
        import json
        from contact_state import check_contact
        root=Path(__file__).resolve().parent/'fixtures/contact_guard_cases.json'
        for sample in json.loads(root.read_text()):
            f=np.array(sample['force_history']);normal=np.array(sample['normal'])
            self.assertTrue(np.any(f[-1]@normal<1.))
            self.assertTrue(check_contact(f,normal,sample['point_count'],sample['minimum_separation'])['supported'])
    def test_gap_unloaded_and_deep_penetration_rejected(self):
        from contact_state import check_contact
        f=np.zeros((20,4,3));f[:,:,2]=10.;n=np.array([0.,0.,1.]);counts=np.full(4,4.);gap=np.zeros(4)
        self.assertTrue(check_contact(f,n,counts,gap)['supported'])
        for value in (.001,-.002,float('nan')):
            g=gap.copy();g[0]=value
            self.assertFalse(check_contact(f,n,counts,g)['supported'])
        unloaded=f.copy();unloaded[:,0]=0
        self.assertFalse(check_contact(unloaded,n,counts,gap)['supported'])
        counts[0]=0
        self.assertFalse(check_contact(f,n,counts,gap)['supported'])

class NativeConstraintRegression(unittest.TestCase):
    def test_tow_feedback_uses_cart_speed_and_preserves_braking_limit(self):
        from controller import tow_velocity_target
        ordinary=tow_velocity_target(.2,.02,.03,40.,11.668,0.)
        compensated=tow_velocity_target(.2,.02,.03,40.,11.668,.5)
        self.assertAlmostEqual(ordinary,.54)
        self.assertGreater(compensated,ordinary)
        self.assertLessEqual(compensated,.6*(1.+.5*40./11.668))
        self.assertAlmostEqual(tow_velocity_target(.05,.3,0.,40.,11.668,1.),-.6)
    def test_forward_speed_gain_can_probe_model_error_without_changing_braking(self):
        from controller import tow_velocity_target
        self.assertAlmostEqual(tow_velocity_target(.2,0.,0.,40.,11.668,0.),.6)
        self.assertAlmostEqual(tow_velocity_target(.2,0.,0.,40.,11.668,0.,6.,1.2),1.2)
        self.assertAlmostEqual(tow_velocity_target(.05,.3,0.,40.,11.668,0.,6.,1.2),-.6)
        self.assertAlmostEqual(tow_velocity_target(-.2,0.,0.,40.,11.668,0.,6.,1.2),-.6)
    def test_delayed_forward_bias_preserves_early_motion_and_braking(self):
        from controller import delayed_forward_bias
        self.assertAlmostEqual(delayed_forward_bias(.2,.6,8.9,.6,9.,.5,1.2),.6)
        self.assertAlmostEqual(delayed_forward_bias(.2,.6,9.25,.6,9.,.5,1.2),.9)
        self.assertAlmostEqual(delayed_forward_bias(.2,.6,9.5,.6,9.,.5,1.2),1.2)
        self.assertAlmostEqual(delayed_forward_bias(.2,-.2,10.,.6,9.,.5,1.2),-.2)
    def test_yaw_gain_changes_only_rotation_about_ground_normal(self):
        from controller import attitude_acceleration
        error=np.array([.1,.2,.3]);omega=np.array([.01,.02,.03]);normal=np.array([0.,0.,1.])
        nominal=attitude_acceleration(error,omega,normal)
        reduced=attitude_acceleration(error,omega,normal,10.)
        np.testing.assert_allclose(nominal-reduced,[0.,0.,6.])
    def test_traction_residual_rejects_missing_or_saturated_contact(self):
        from controller import native_traction_residual
        B=np.eye(3);predicted=np.tile([25.,0.,20.],(4,1))
        measured=np.tile([23.,0.,0.],(20,4,1));normal=np.tile([0.,0.,20.],(20,4,1));counts=np.full((20,4),4.)
        valid=native_traction_residual(predicted,measured,normal,counts,B,11.668)
        self.assertTrue(valid['valid']);self.assertAlmostEqual(valid['raw_bias_mps2'],8./11.668)
        counts[:2,0]=0.
        self.assertFalse(native_traction_residual(predicted,measured,normal,counts,B,11.668)['valid'])
        counts[:2,0]=4.;measured[:,:,0]=55.
        self.assertFalse(native_traction_residual(predicted,measured,normal,counts,B,11.668)['valid'])
    def test_measured_traction_shortfall_changes_only_qp_model(self):
        from controller import traction_disturbance_external
        incline=np.deg2rad(25.)
        B=np.column_stack(([np.cos(incline),0.,np.sin(incline)], [0.,1.,0.],
                           [-np.sin(incline),0.,np.cos(incline)]))
        external=np.arange(22,dtype=float)
        corrected,force=traction_disturbance_external(external,B,.7,11.668,1.)
        self.assertAlmostEqual(force,8.1676)
        np.testing.assert_allclose(corrected[:3],external[:3]-force*B[:,0])
        np.testing.assert_array_equal(corrected[3:],external[3:])
        np.testing.assert_array_equal(external,np.arange(22,dtype=float))
        unchanged,zero=traction_disturbance_external(external,B,.7,11.668,0.)
        self.assertEqual(zero,0.);np.testing.assert_array_equal(unchanged,external)
        capped,limit=traction_disturbance_external(external,B,100.,11.668,1.)
        self.assertAlmostEqual(limit,.8*11.668)
        np.testing.assert_allclose(capped[:3],external[:3]-limit*B[:,0])
        with self.assertRaises(ValueError):traction_disturbance_external(external,B,.1,11.668,1.01)
    def test_smoothing_does_not_block_feasible_native_dynamics(self):
        z=dict(np.load(Path(__file__).resolve().parent/'fixtures/native_slew_conflict.npz'))
        lo=z.pop('nominal_tau_lower');hi=z.pop('nominal_tau_upper')
        original=solve(**z)
        self.assertFalse(original['success'])
        reference=(z['tau_lower']+z['tau_upper'])/2
        scale=(z['tau_upper']-z['tau_lower'])/2
        z.update(tau_lower=lo,tau_upper=hi,tau_reference=reference,tau_smooth_scale=scale,tau_smooth_weight=.1)
        result=solve(**z)
        self.assertTrue(result['success'],result)
        self.assertLess(result['eq_residual'],1.e-4)
        self.assertLess(result['inequality_violation'],1.e-4)
        self.assertTrue(np.all(result['tau']>=lo-1.e-5)&np.all(result['tau']<=hi+1.e-5))

class EpisodeLifecycle(unittest.TestCase):
    def test_completed_case_is_closed_without_closing_other_slot(self):
        cases=protocol_v7.cases(suite='teacher')[:2];closed=set()
        finish=protocol_v7.track_completion(protocol_v7.finished,cases,closed)
        self.assertFalse(finish([dict(t=19.98)],cases[0]));self.assertFalse(closed)
        self.assertTrue(finish([dict(t=20.)],cases[0]));self.assertEqual(closed,{0})
        self.assertFalse(finish([dict(t=20.,eval_phase=1)],cases[1]));self.assertEqual(closed,{0})
        self.assertTrue(finish([dict(t=20.,eval_phase=3,eval_stop_time_s=8.)],cases[1]))
        self.assertEqual(closed,{0,1})

if __name__=='__main__':unittest.main()
