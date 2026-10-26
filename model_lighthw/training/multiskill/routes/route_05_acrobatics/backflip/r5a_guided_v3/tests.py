"""CPU checks for unchanged physics, demonstration sampling, and update rollback."""
import math,tempfile,unittest,contextlib,io
from pathlib import Path
import numpy as np
import torch
from .spec import CONTRACT as C,ROOT
from .learning import Demonstrations,auxiliary_update

def fixture(path):
 obs=np.zeros((300,420),np.float32);obs[:,-12]=np.arange(300)/300
 obs[30:,-8]=1;obs[55:,-7]=1
 action=np.zeros((300,16),np.float32);action[20:55]=.2
 np.savez_compressed(path,observation=obs,action=action)

class Checks(unittest.TestCase):
 def test_physics_and_acceptance_unchanged(self):
  from r5a_guided_v2.spec import CONTRACT as old
  protected=('physics_dt','policy_dt','episode_seconds','settle_seconds','launch_deadline_s','action_order','action_lower','action_upper','stand',
   'leg_effort_Nm','wheel_effort_Nm','leg_speed_rad_s','wheel_speed_rad_s','rigid_max_angular_velocity_deg_s','gyroscopic_forces',
   'solver_position_iterations','solver_velocity_iterations','wheel_armature','self_collisions','payload_kg','friction','restitution','leg_pd',
   'airborne_force_N','illegal_contact_N','release_samples','min_takeoff_vz','min_flight_s','min_flight_rise_m','rotation_range_deg',
   'max_axis_error_deg','max_landing_tilt_deg','stable_tilt_deg','stable_speed_m_s','stable_vertical_speed_m_s','stable_landing_drift_m',
   'stable_hold_s','standing_base_z','standing_height_tolerance_m','observation_noise','initial_state_randomization','physics_domain_randomization')
  self.assertEqual({k:C[k] for k in protected},{k:old[k] for k in protected})
  for name in ('scene.py','dynamics.py','physics_review.py'):
   self.assertEqual((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2'/name).read_text().replace('r5a_guided_v2','r5a_guided_v3'),(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3'/name).read_text())
 def test_demonstration_balances_active_phases(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.npz';fixture(p);d=Demonstrations(p,'cpu');obs,act=d.sample(32)
   self.assertEqual(obs.shape,(128,420));self.assertTrue((obs[:32,-8]<.5).all());self.assertTrue((obs[32:64,-7]<.5).all())
   self.assertEqual([len(x) for x in d.groups],[10,25,45,200])
 def test_reject_invalid_demonstrations(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.npz';fixture(p);a=dict(np.load(p));a['action'][0,0]=1.;np.savez(p,**a)
   with self.assertRaises(ValueError):Demonstrations(p,'cpu')
   a['action'][0,0]=float('nan');np.savez(p,**a)
   with self.assertRaises(ValueError):Demonstrations(p,'cpu')
 def test_first_control_tick_survives_float32_clock_rounding(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.npz';fixture(p);a=dict(np.load(p));a['observation'][20,-12]=np.float32(.399999976/6);np.savez(p,**a)
   d=Demonstrations(p,'cpu');self.assertIn(20,d.groups[0].tolist());self.assertNotIn(19,d.groups[0].tolist())
 def test_teacher_and_student_range(self):
  from .teacher.control_split_bound import ACTION_LIMIT
  self.assertAlmostEqual(ACTION_LIMIT,math.tanh(3.));self.assertLess(C['ground_exploration_leg_floor'],.1)
 def test_auxiliary_update_is_covered_by_rollback(self):
  from types import SimpleNamespace
  from .policy import SquashedActorCritic
  from .update_guard import guarded_update
  with contextlib.redirect_stdout(io.StringIO()):model=SquashedActorCritic(420,104,16,actor_hidden_dims=[16],critic_hidden_dims=[16],noise_std_type='log',activation='elu')
  with torch.no_grad():model.log_std.fill_(math.log(.03))
  obs=torch.zeros(2,4,420);model.update_distribution(obs.flatten(0,1))
  alg=SimpleNamespace(policy=model,optimizer=torch.optim.Adam(model.parameters(),lr=.001),learning_rate=.001,
   storage=SimpleNamespace(observations=obs,actions=torch.zeros(2,4,16),mu=model.action_mean.detach().reshape(2,4,16),sigma=model.action_std.detach().reshape(2,4,16)))
  before={k:v.clone() for k,v in model.state_dict().items()}
  def destructive_auxiliary():
   with torch.no_grad():model.ground_actor[-1].bias.add_(10.)
   return dict(surrogate=0.,demonstration_loss=1.)
  _,row=guarded_update(alg,destructive_auxiliary,.03)
  self.assertFalse(row['accepted'])
  for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
 def test_guidance_uses_behavior_and_restores_after_regression(self):
  from .learning import Guidance
  g=Guidance();self.assertEqual(g.summary()['weight'],.2)
  g.observe([dict(success=False,qualified_landing=False)]*3000)
  self.assertEqual(g.summary()['weight'],.2)
  g.observe([dict(success=True,qualified_landing=True)]*2048)
  self.assertEqual(g.summary()['weight'],.08)
  restored=Guidance(list(g.rows));self.assertEqual(g.summary(),restored.summary())
  g.observe([dict(success=False,qualified_landing=False)]*500)
  self.assertEqual(g.summary()['weight'],.2)
 def test_compact_preparation_does_not_recurse_into_itself(self):
  from .station import compact_preparation
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'preparation.json').write_text('{}');(p/'source_snapshot').mkdir();(p/'source_snapshot/huge.json').write_text('{}')
   compact_preparation(p,p/'compact')
   self.assertTrue((p/'compact/preparation.json').exists());self.assertFalse((p/'compact/compact').exists());self.assertFalse((p/'compact/source_snapshot').exists())
 def boundary(self):
  import json
  from types import SimpleNamespace
  from .dynamics import rotation
  from .teacher.control_handoff import Teacher
  class Actor(torch.nn.Module):
   def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
   def act_inference(self,obs):return torch.zeros(len(obs),16)
  data=np.load(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3/teacher/evidence/target_gpu_boundary.npz')
  p=json.loads((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3/teacher/candidates/teacher_rear15_d80.json').read_text())['parameters']
  root=data['root_state'];rot=rotation(root[None,3:7])[0]
  robot=SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor(rot.T@np.array([0,0,-1.]),dtype=torch.float32)[None],root_ang_vel_b=torch.tensor(rot.T@root[10:13],dtype=torch.float32)[None]))
  state=SimpleNamespace(angle=torch.tensor([float(data['angle'])]),landed=torch.tensor([True]),launched=torch.tensor([True]),landing_t=torch.tensor([float(data['landing_t'])]),landing_xy=torch.tensor(data['landing_xy'])[None])
  metrics={k:torch.tensor(data[v])[None] for k,v in [('com','com'),('com_vel','com_velocity'),('clearance','clearance')]}
  env=SimpleNamespace(state=state,robot=robot,current_metrics=metrics,episode_physics_steps=torch.tensor([int(data['physics_tick'])]),physics_dt=.001)
  return Teacher(Actor(),[p]),torch.tensor(data['observation'])[None],env
 def test_actual_target_touchdown_has_no_mode_step(self):
  from .teacher.control_handoff import Overlay
  teacher,obs,env=self.boundary()
  airborne=Overlay(env,state=Overlay(env.state,landed=torch.tensor([False])))
  before=teacher(obs,airborne);after=teacher(obs,env)
  # First actual GPU control tick is just 1ms after touch; the new mode must
  # remain near its airborne opening command instead of making a full step.
  self.assertLess(float((before-after).abs().max()),.005)
  old_step=float((teacher.base(obs,airborne)-teacher.base(obs,env)).abs().max())
  self.assertGreater(old_step,.5)
  self.assertTrue(bool(env.state.landed[0]))
 def test_handoff_ends_at_original_capture_and_stays_bounded(self):
  teacher,obs,env=self.boundary();env.episode_physics_steps.fill_(1200)
  torch.testing.assert_close(teacher(obs,env),teacher.base(obs,env),atol=0,rtol=0)
  self.assertLessEqual(float(teacher(obs,env).abs().max()),math.tanh(3.)+1e-7)
 def test_rear_opening_precedes_v1_on_target_state(self):
  import json
  from .teacher.control_split_bound import Teacher as Old
  teacher,obs,env=self.boundary();env.state.landed.fill_(False)
  # The v1 rear opening offset was 0.352527 rad. Keep everything else equal.
  p=teacher.parameters[:,:39].clone();p[:,37]=.352527
  old=Old(teacher.actor,p)
  self.assertGreater(float(teacher(obs,env)[0,10]-old(obs,env)[0,10]),.1)
 def test_qualified_first_landing_cannot_admit_failed_teacher(self):
  from .station import teacher_passed
  r=dict(success=False,protocol_success=False,qualified_landing=True,failed=True,online_agrees=True,physics_consistent=True,duration_s=1.04,stable_s=0.)
  self.assertFalse(teacher_passed(r))
  r.update(success=True,protocol_success=True,failed=False,duration_s=6.,stable_s=1.2)
  self.assertTrue(teacher_passed(r))
  r['physics_consistent']=False;self.assertFalse(teacher_passed(r))
 def test_last_fifty_degrees_have_more_reward_than_earlier_fifty(self):
  from .shaping import landing_progress
  value=landing_progress(torch.tensor([240.,290.,300.,350.])*math.pi/180)
  self.assertGreater(float(value[3]-value[2]),float(value[1]-value[0]))
  self.assertEqual(float(landing_progress(torch.tensor([-1.]))),0.)
 def test_partial_rotation_recovery_cannot_be_improved_by_height(self):
  from types import SimpleNamespace as NS
  from .recovery import recovery_terms
  s=NS(flight_valid=torch.tensor([True]),landed=torch.tensor([True]),failed=torch.tensor([False]),angle=torch.tensor([5.4]),
   flight_rise=torch.tensor([.10]),flight_time=torch.tensor([.3]),landing_xy=torch.zeros(1,2))
  m=dict(tilt=torch.tensor([.1]),base_z=torch.tensor([.4]),force_z=torch.ones(1,4)*25,com_vel=torch.zeros(1,3),com=torch.zeros(1,3))
  before=recovery_terms(s,m,torch.zeros(1,4),.02)
  s.flight_rise.fill_(.6);s.flight_time.fill_(.8)
  after=recovery_terms(s,m,torch.zeros(1,4),.02)
  for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
 def test_impact_distinguishes_high_impacts_and_stays_bounded(self):
  from types import SimpleNamespace as NS
  from .shaping import reward_potentials
  z=torch.zeros(3);s=NS(impact_peak=torch.tensor([26.,92.,119.])*10*9.81,push_peak=z,height_peak=z,rotation_potential=z,airtime_peak=z)
  impact=reward_potentials(s,torch.ones(3)*10)[:,4]
  self.assertTrue(bool((impact[1:]>impact[:-1]).all()));self.assertTrue(bool((impact<4).all()))
 def test_landing_head_does_not_change_flight_or_ground(self):
  from .policy import SquashedActorCritic
  with contextlib.redirect_stdout(io.StringIO()):p=SquashedActorCritic(420,104,16,actor_hidden_dims=[16],critic_hidden_dims=[16],noise_std_type='log')
  obs=torch.zeros(3,420);obs[1:,-8]=1;obs[2,-7]=1
  before=p.act_inference(obs).detach().clone()
  with torch.no_grad():p.landing_actor[-1].bias.add_(1.)
  after=p.act_inference(obs)
  torch.testing.assert_close(before[:2],after[:2],rtol=0,atol=0)
  self.assertGreater(float((before[2]-after[2]).abs().max()),.1)
 def test_teacher_action_extrema_are_exactly_reachable(self):
  from .policy import SquashedActorCritic
  with contextlib.redirect_stdout(io.StringIO()):p=SquashedActorCritic(420,104,16,actor_hidden_dims=[16],critic_hidden_dims=[16],noise_std_type='log')
  with torch.no_grad():p.ground_actor[-1].weight.zero_();p.ground_actor[-1].bias.fill_(4.)
  torch.testing.assert_close(p.act_inference(torch.zeros(1,420)),torch.full((1,16),math.tanh(3.)),rtol=0,atol=1e-7)
 def test_supervised_correction_can_escape_saturated_wrong_prediction(self):
  from .policy import SquashedActorCritic
  with contextlib.redirect_stdout(io.StringIO()):p=SquashedActorCritic(420,104,16,actor_hidden_dims=[16],critic_hidden_dims=[16],noise_std_type='log')
  with torch.no_grad():p.ground_actor[-1].weight.zero_();p.ground_actor[-1].bias.fill_(4.)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'data.npz';fixture(path);d=Demonstrations(path,'cpu')
   d.loss(p,full=True).backward()
   self.assertTrue(bool((p.ground_actor[-1].bias.grad>0).all()))
 def test_correction_deduplicates_repeated_nominal_rollouts(self):
  from .station import merge_demonstrations
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'a.npz';fixture(p);a=dict(np.load(p));a['critic']=np.zeros((300,104),np.float32);np.savez(p,**a)
   q=root/'merged.npz';info=merge_demonstrations([p,p],q)
   self.assertEqual(info['frames'],300);self.assertEqual(info['before_deduplication'],600)
 def test_ready_cache_cannot_use_mixed_success_as_autonomous_proof(self):
  from .station import validate_ready
  from .io import implementation,sha
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'actor';p.write_text('fixture')
   proof=dict(controller='correction_mixed',actor_sha256=sha(p),report=dict(success=True,protocol_success=True,qualified_landing=True,online_agrees=True,physics_consistent=True,failed=False,duration_s=6.,stable_s=2.))
   ready=dict(device='cpu',implementation_sha256=implementation(),actor=str(p),actor_sha256=sha(p),demonstrations=str(p),demonstrations_sha256=sha(p),autonomous_proof=[proof,proof])
   self.assertFalse(validate_ready(ready,'cpu'))
   proof['controller']='student_actor';self.assertTrue(validate_ready(ready,'cpu'));self.assertFalse(validate_ready(ready,'cuda:0'))

def run_tests():
 torch.set_num_threads(1)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise RuntimeError('Guided CPU checks failed')
if __name__=='__main__':run_tests()
