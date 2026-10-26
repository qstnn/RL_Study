"""伺服区间积分与边界反例；CPU检查不替代接触行为验证。"""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from servo_model import initial_torque
from qp import to_action

class ServoInterval(unittest.TestCase):
    def test_original_inverse_does_not_execute_mean_torque_under_motion(self):
        q=np.linspace(-.3,.4,16);qd=np.linspace(-.4,.5,16);qdd=np.linspace(-8,9,16);mean=np.linspace(-7,8,16)
        t=np.linspace(0,.02,10001)[:,None];pos=q+t*qd+.5*t*t*qdd;vel=qd+t*qdd
        def integrated(initial):
            a=to_action(initial,q,qd,q)
            target=q[:12]+a[:12]*np.tile([.08,.14,.14],4)
            tau=np.column_stack((80*(target-pos[:,:12])-3*vel[:,:12],2*(a[12:]*5-vel[:,12:])))
            return np.trapz(tau,t[:,0],axis=0)/.02
        self.assertGreater(np.max(abs(integrated(mean)-mean)),.1)
        compensated=initial_torque(mean,qd,qdd,.02)
        np.testing.assert_allclose(integrated(compensated),mean,atol=1.e-8)
        np.testing.assert_equal(initial_torque(mean,qd,qdd,0.),mean)
    def test_invalid_horizon_rejected(self):
        with self.assertRaises(ValueError):initial_torque(np.zeros(16),np.zeros(16),np.zeros(16),.2)

class NativeQP(unittest.TestCase):
    def problem(self):
        z=dict(np.load(Path(__file__).resolve().parent/'fixtures/native_slew_conflict.npz'))
        z['tau_lower']=z.pop('nominal_tau_lower');z['tau_upper']=z.pop('nominal_tau_upper')
        return z
    def test_held_servo_respects_original_native_targets(self):
        from qp import solve
        z=self.problem();r=solve(**z,servo_horizon_s=.02)
        self.assertTrue(r['success'],r)
        np.testing.assert_allclose(r['initial_servo_torque'],initial_torque(r['tau'],z['velocity'][6:],r['acceleration'][6:],.02),atol=1.e-10)
        self.assertTrue(np.all(r['initial_servo_torque']>=z['tau_lower']-1.e-5))
        self.assertTrue(np.all(r['initial_servo_torque']<=z['tau_upper']+1.e-5))
    def test_airborne_wheel_never_supplies_fictitious_support(self):
        from qp import solve
        z=self.problem();r=solve(**z,contact_mask=[True,True,False,True])
        self.assertTrue(r['success'],r)
        np.testing.assert_allclose(r['contact_forces'][2],0.,atol=1.e-8)
        qdd=r['acceleration'];rhs=np.r_[np.zeros(6),r['tau']]+z['A'].T@r['contact_forces'].ravel()+z['external']
        np.testing.assert_allclose(z['M']@qdd+z['h'],rhs,atol=1.e-5)

class ConsistentTeacherGate(unittest.TestCase):
    def test_cannot_merge_different_controllers_into_one_pass(self):
        import json,tempfile
        from gates import teacher_gate
        from protocol_v7 import cases
        with tempfile.TemporaryDirectory() as temp:
            runs=[]
            for seed in (43,44,45):
                for layout in (0,1):
                    path=Path(temp)/f'{seed}_{layout}';path.mkdir();runs.append(path)
                    suite=cases(suite='teacher');suite=suite if not layout else list(reversed(suite))
                    request=dict(seed=seed,noise=True,ground_friction=3.,pose={'id':0},cases=suite,teacher_options={},checkpoint_sha256='fixture',position_iterations=4,velocity_iterations=1,actor_inference='row')
                    report=dict(teacher=True,short_diagnostic=False,cases=[dict(case=c,success=True,complete=True) for c in suite],pass_count=4)
                    (path/'request.json').write_text(json.dumps(request));(path/'evaluation.json').write_text(json.dumps(report))
            self.assertTrue(teacher_gate(runs)['passed'])
            request['teacher_options']={'servo_horizon_s':.02}
            (runs[-1]/'request.json').write_text(json.dumps(request))
            self.assertFalse(teacher_gate(runs)['passed'])
            request['teacher_options']={};(runs[-1]/'request.json').write_text(json.dumps(request))
            report['cases'][0].update(success=False,complete=False,candidate_aborted=True)
            (runs[-1]/'evaluation.json').write_text(json.dumps(report))
            self.assertFalse(teacher_gate(runs)['passed'])

if __name__=='__main__':unittest.main()
