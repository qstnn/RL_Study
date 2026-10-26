"""CPU invariants for explicit migration, phase costs, frozen takeoff and candidate safety."""
import contextlib,io,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from .spec import CONTRACT as C,ROOT

def fixture(path):
 obs=np.zeros((300,420),np.float32);obs[:,-12]=np.arange(300)/300
 obs[30:,-8]=1;obs[55:,-7]=1;obs[45:55,-11]=.9;obs[45:55,-28]=-.25
 action=np.zeros((300,16),np.float32);action[20:55]=.2
 np.savez_compressed(path,observation=obs,action=action)

def model():
 from .policy import SquashedActorCritic
 with contextlib.redirect_stdout(io.StringIO()):
  return SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log')

class Checks(unittest.TestCase):
 def test_physics_observation_and_acceptance_unchanged(self):
  from r5a_guided_v3.spec import CONTRACT as old
  changed={k for k in C if k not in old or C[k]!=old[k]}
  allowed={'version','ground_exploration_leg_floor','ground_exploration_wheel_floor','ground_exploration_multiplier','landing_exploration_multiplier','prelanding_angle_deg','prelanding_exploration_multiplier','moving_exploration_floor','freeze_ground_actor','initialization','demonstration_phase_weights','reward_weights','actor_start_learning_rate','minimum_learning_rate','maximum_learning_rate'}
  self.assertEqual(changed,allowed)
  for name in ['scene.py','dynamics.py','physics_review.py']:
   self.assertEqual((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3'/name).read_text().replace('r5a_guided_v3','r5a_guided_v5'),(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5'/name).read_text())
 def test_baseline_assets_and_exact_inference(self):
  from .station import verify_baseline_manifest
  from .learning import load_initial
  verify_baseline_manifest();p=model();load_initial(p,ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5/baseline/actor.pt')
  d=np.load(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5/baseline/demonstrations.npz')
  with torch.no_grad():a=p.act_inference(torch.tensor(d['observation'])).numpy()
  self.assertLess(float(abs(a[20:]-d['action'][20:]).max()),2e-6)
 def test_phase_balancing_keeps_late_flight_separate(self):
  from .learning import Demonstrations
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'d.npz';fixture(p);d=Demonstrations(p,'cpu')
   self.assertEqual([len(g) for g in d.groups],[10,15,10,45,200]);self.assertEqual(d.sample(7)[0].shape,(35,420))
 def test_actual_baseline_contains_all_phases(self):
  from .learning import Demonstrations
  d=Demonstrations(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5/baseline/demonstrations.npz','cpu');self.assertTrue(all(len(g) for g in d.groups))
 def test_sampling_stays_nonzero_and_landing_is_quieter(self):
  p=model();o=torch.zeros(4,420);o[1:,-8]=1;o[2,-11]=.9;o[2,-28]=-.25;o[3,-7]=1
  with torch.no_grad():p.log_std.fill_(math.log(.03));p.update_distribution(o)
  torch.testing.assert_close(p.action_std[:,0],torch.tensor([.005,.024,.012,.0045]),rtol=1e-5,atol=1e-7)
  self.assertTrue(torch.isfinite(p.get_actions_log_prob(p.act(o))).all())
 def test_prelanding_gate_matches_existing_observation(self):
  from .landing import prelanding_gate,observation_prelanding
  n=5;s=NS(launched=torch.tensor([1,1,1,1,0],dtype=torch.bool),landed=torch.tensor([0,0,0,1,0],dtype=torch.bool),failed=torch.zeros(n,dtype=torch.bool),angle=torch.tensor([5.,5.,3.,5.,0.]))
  v=torch.zeros(n,3);v[:,2]=torch.tensor([-1.,1.,-1.,-1.,-1.]);o=torch.zeros(n,420);o[:,-8]=s.launched;o[:,-7]=s.landed;o[:,-11]=s.angle/(2*math.pi);o[:,-28]=v[:,2]*.25
  self.assertEqual(prelanding_gate(s,{'com_vel':v}).tolist(),[True,False,False,False,False]);self.assertTrue(torch.equal(prelanding_gate(s,{'com_vel':v}),observation_prelanding(o)))
 def state_metrics(self):
  from .protocol import FlipState
  s=FlipState(1);s.launched.fill_(True);s.angle.fill_(math.radians(345));s.landing_xy.zero_()
  m=dict(com=torch.zeros(1,3),com_vel=torch.tensor([[0.,0.,-1.]]),force_z=torch.ones(1,4)*30,tilt=torch.zeros(1),base_z=torch.tensor([.416]),leg_velocity=torch.ones(1,12)*12.57,root_angular_velocity=torch.tensor([[0.,8.,0.]]))
  return s,m
 def test_1khz_costs_and_one_touch_event(self):
  from .landing import advance
  s,m=self.state_metrics()
  for i in range(10):advance(s,m,.001,torch.tensor([.8+i*.001]),torch.tensor([False]))
  self.assertAlmostEqual(float(s.prelanding_speed_cost),.01,places=6)
  s.landed.fill_(True);s.landing_t.fill_(.81)
  advance(s,m,.001,torch.tensor([.81]),torch.tensor([True]));advance(s,m,.001,torch.tensor([.811]),torch.tensor([False]))
  self.assertAlmostEqual(float(s.touchdown_spin_cost),1.,places=6);self.assertAlmostEqual(float(s.landing_speed_cost),.002,places=6)
  self.assertAlmostEqual(float(s.prelanding_speed_cost),.01,places=6)
 def test_recovery_gate_counters_and_reset(self):
  from .landing import advance,FLOATS,BOOLS
  s,m=self.state_metrics();s.landed.fill_(True);s.qualified_landing.fill_(True)
  advance(s,m,.001,torch.tensor([1.]),torch.tensor([False]));self.assertAlmostEqual(float(s.gate_miss_z_speed_s),.001,places=6)
  self.assertEqual(float(s.gate_miss_tilt_s),0.)
  s.reset(torch.tensor([0]),torch.zeros(1),torch.zeros(1,2),torch.tensor([[1.,0.,0.,0.]]))
  self.assertTrue(all(float(getattr(s,k)[0])==0 for k in FLOATS+BOOLS))
 def test_peak_impact_telescopes_and_does_not_flatten(self):
  from .shaping import reward_potentials
  z=torch.zeros(4);s=NS(impact_peak=torch.tensor([6.,26.,77.8,97.2])*10*9.81,push_peak=z,height_peak=z,rotation_potential=z,airtime_peak=z)
  p=reward_potentials(s,torch.ones(4)*10)[:,4]
  self.assertTrue((p[1:]>p[:-1]).all());self.assertGreater(float((p[3]-p[2])*4),.8)
  self.assertAlmostEqual(float(torch.diff(p).sum()),float(p[-1]),places=6)
 def test_terminal_quality_requires_complete_success(self):
  from .landing import terminal_quality
  s,m=self.state_metrics();m['com_vel'].zero_();s.success.fill_(True)
  upright=float(terminal_quality(s,m));m['tilt'].fill_(math.radians(10));self.assertLess(float(terminal_quality(s,m)),upright*.3)
  s.success.fill_(False);self.assertEqual(float(terminal_quality(s,m)),0.)
 def test_full_success_precedes_smaller_impact(self):
  from .cli import select_candidate
  def item(label,success,impact,tilt):
   return dict(label=label,metrics=dict(physics_failure_rate=0.,momentum_unchecked_rate=0.,success_rate=success,qualified_rate=success,mean_raw_recovery_score=.9,mean_impact_bodyweights=impact,mean_end_tilt_deg=tilt,mean_stable_s=2.,mean_recovery_score=.9,valid_flight_rate=1.,failed_rate=0.,mean_joint_progress=1.))
  baseline=item('baseline',1.,78.,9.5);better=item('better',1.,50.,3.);failed=item('failed',0.,5.,0.)
  self.assertEqual(select_candidate([baseline,better,failed])['label'],'better')
  better['metrics']['physics_failure_rate']=1.;self.assertEqual(select_candidate([baseline,better,failed])['label'],'baseline')
 def test_rollback_restores_landing_head_and_optimizer(self):
  from .update_guard import guarded_update
  p=model();o=torch.zeros(2,4,420);o[:,:,-8]=1;o[:,:,-7]=1;p.update_distribution(o.flatten(0,1))
  alg=NS(policy=p,optimizer=torch.optim.Adam(p.parameters(),lr=.001),learning_rate=.001,storage=NS(observations=o,actions=torch.zeros(2,4,16),mu=p.action_mean.detach().reshape(2,4,16),sigma=p.action_std.detach().reshape(2,4,16)))
  before={k:v.clone() for k,v in p.state_dict().items()}
  def change():
   with torch.no_grad():p.landing_actor[-1].bias.add_(10.)
   return {'loss':0.}
  _,row=guarded_update(alg,change,.03);self.assertFalse(row['accepted'])
  self.assertTrue(all(torch.equal(v,p.state_dict()[k]) for k,v in before.items()))
 def test_mixed_teacher_proof_cannot_enable_ppo(self):
  from .station import validate_ready
  from .io import implementation,sha
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'actor';p.write_text('fixture');r=dict(success=True,protocol_success=True,qualified_landing=True,online_agrees=True,physics_consistent=True,failed=False,duration_s=6.,stable_s=2.)
   proof=dict(controller='correction_mixed',actor_sha256=sha(p),report=r)
   ready=dict(device='cpu',implementation_sha256=implementation(),actor=str(p),actor_sha256=sha(p),demonstrations=str(p),demonstrations_sha256=sha(p),autonomous_proof=[proof,proof])
   self.assertFalse(validate_ready(ready,'cpu'));proof['controller']='student_actor';self.assertTrue(validate_ready(ready,'cpu'));self.assertFalse(validate_ready(ready,'cuda:0'))

def run_tests():
 torch.set_num_threads(1)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise RuntimeError('Guided v5 CPU checks failed')
if __name__=='__main__':run_tests()
