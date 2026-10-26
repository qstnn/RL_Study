"""行为语义回归：真实物理样本、反弹/站立反例、质量退化与迁移。"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'r5a.sh').is_file())
CORE=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw'
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path[:0]=[str(SCRIPTS),str(CORE)]
from r5a_highjump import spec
from r5a_highjump.protocol import JumpState,reference_pose
from quality import compare,initialization_ok
from migrate import migrate,actor

torch.set_num_threads(2)


class HighJumpTests(unittest.TestCase):
    def test_real_trace_reconstructs_physical_release_and_hold(self):
        a=np.load(Path(__file__).parent/'fixtures/reference_85mm.npz',allow_pickle=False)
        state=JumpState(1);origin=a['com'][0,:2].copy()
        scalar=lambda x:torch.tensor([float(x)],dtype=torch.float32)
        vector=lambda x:torch.as_tensor(x,dtype=torch.float32).reshape(1,-1)
        for i,t in enumerate(a['t'][1:],1):
            if t<=.5:origin=a['com'][i,:2].copy()
            state.update(.001,scalar(t),scalar(a['com'][i,2]),scalar(a['com_velocity'][i,2]),
                vector(a['wheel_force_z'][i]),vector(a['wheel_clearance'][i]),scalar(a['tilt'][i]),
                scalar(a['base_height'][i]),scalar(np.linalg.norm(a['com_velocity'][i,:2])),
                scalar(np.linalg.norm(a['com'][i,:2]-origin)),scalar(a['body_force'][i]),scalar(abs(a['applied_torque'][i]).max()))
            self.assertEqual(bool(state.released[0]),bool(a['online_release'][i]))
            self.assertAlmostEqual(float(state.flight_rise[0]),float(a['online_flight_rise'][i]),places=6)
            self.assertAlmostEqual(float(state.free_flight_time[0]),float(a['online_free_flight'][i]),places=6)
            self.assertAlmostEqual(float(state.stable_time[0]),float(a['online_stable_time'][i]),places=5)
        self.assertAlmostEqual(float(state.flight_rise[0]),.0857675,places=5)
        self.assertAlmostEqual(float(state.free_flight_time[0]),.264,places=5)

    def sample(self,s,t,z=.35,vz=1.,contact=False):
        f=lambda x:torch.tensor([x],dtype=torch.float32)
        return s.update(.001,f(t),f(z),f(vz),torch.full((1,4),30. if contact else 0.),
            torch.full((1,4),0. if contact else .02),f(0),f(.416),f(0),f(0),f(0),f(0))

    def test_leg_extension_with_ground_contact_is_not_high_jump(self):
        s=JumpState(1);s.armed.fill_(True)
        for t in np.arange(.8,1.2,.001):self.sample(s,t,z=.5,vz=1.,contact=True)
        self.assertFalse(s.released[0]);self.assertEqual(float(s.flight_rise[0]),0.)

    def test_bounce_cannot_add_height_or_airtime(self):
        s=JumpState(1);s.armed.fill_(True)
        self.sample(s,.9,z=.35);self.sample(s,.901,z=.36)
        self.sample(s,.902,z=.36,contact=True)
        before=s.free_flight_time.clone();height=s.flight_rise.clone()
        self.sample(s,1.0,z=.8)
        torch.testing.assert_close(s.free_flight_time,before);torch.testing.assert_close(s.flight_rise,height)

    def test_reset_does_not_inherit_release_or_success(self):
        s=JumpState(2);s.released[:]=True;s.flight_rise[:]=.2;s.ever_success[:]=True
        s.reset([0],torch.tensor([.32]))
        self.assertFalse(s.released[0]);self.assertFalse(s.ever_success[0]);self.assertTrue(s.released[1])
        self.assertEqual(float(s.flight_rise[0]),0.)

    def test_reference_waits_for_release_event(self):
        s=JumpState(1);t=torch.tensor([1.1]);before=reference_pose(t,s)
        s.released[:]=True;s.release_t[:]=1.0;after=reference_pose(t,s)
        self.assertFalse(torch.equal(before,after))

    def valid_case(self):
        return dict(seed=43,episode=0,uniform_physics_trace=True,reason='timeout',duration_s=4.,
            launched=True,landed=True,failed=False,success=True,height_qualified=True,
            flight_rise=spec.CONTRACT['target_flight_rise_m'],free_flight_time=.3,flight_time=.2,
            stable_time=2.,impact_peak=1400.,wheel_clearance_peak=.25,first_flight_peak_rise=.20)

    def test_quality_rejects_short_jump_lost_hold_and_impact(self):
        base={'cases':[self.valid_case()]}
        self.assertTrue(compare(base,base)['eligible'])
        for key,value in [('flight_time',.099),('height_qualified',False),('success',False),('impact_peak',2000.),('failed',True),('stable_time',.9),('wheel_clearance_peak',.20),('flight_rise',.09)]:
            candidate=copy.deepcopy(base);candidate['cases'][0][key]=value
            self.assertFalse(compare(candidate,base)['eligible'],key)

    def test_initialization_need_not_already_reach_target(self):
        c=self.valid_case();c.update(success=False,height_qualified=False,flight_rise=.09,
            first_flight_peak_rise=.15,wheel_clearance_peak=.20,free_flight_time=.28)
        self.assertTrue(initialization_ok({'cases':[c]}))
        c['stable_time']=.9
        self.assertFalse(initialization_ok({'cases':[c]}))

    def test_reference_tuck_respects_urdf_limits(self):
        from static_audit import audit
        result=audit(spec.URDF,spec)
        self.assertIn('tuck',result['poses'])
        self.assertFalse(spec.CONTRACT['height_curriculum'])
        self.assertEqual(spec.CONTRACT['target_wheel_clearance_m'],.25)

    def test_quality_requires_same_case_keys(self):
        b={'cases':[self.valid_case()]};c=copy.deepcopy(b);c['cases'][0]['seed']=44
        with self.assertRaises(ValueError):compare(c,b)

    def test_migration_preserves_raw_actor_and_zero_head_is_explicit(self):
        source=SCRIPTS.parent/'initialization/source_v1.pt'
        with tempfile.TemporaryDirectory() as d:
            out=migrate(source,Path(d)/'migration')
            new=torch.load(out,weights_only=False,map_location='cpu')
            old=torch.load(source,weights_only=False,map_location='cpu')
            x=torch.randn(32,5,57);extra=torch.randn(32,5,23)
            torch.testing.assert_close(actor(old['model_state_dict'],x.flatten(1)),actor(new['model_state_dict'],torch.cat((x,extra),-1).flatten(1)),atol=2e-6,rtol=1e-5)
            zero=torch.load(out.with_name('reference_start.pt'),weights_only=False,map_location='cpu')
            self.assertTrue(torch.equal(actor(zero['model_state_dict'],torch.randn(32,400)),torch.zeros(32,16)))
            self.assertIsNone(new['optimizer_state_dict'])
            self.assertFalse(zero['infos']['migration']['actor_raw_output_preserved'])

    def test_migration_rejects_unrecognized_file_before_load(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'wrong.pt';source.write_bytes(b'not a model')
            with self.assertRaises(ValueError):migrate(source,Path(d)/'out')


if __name__=='__main__':unittest.main()
