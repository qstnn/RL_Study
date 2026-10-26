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
  from r5a_auto_v6.spec import CONTRACT as old
  changed={'version','initialization','ground_actor_initial_output_scale','policy_architecture','critic_warmup_updates','demonstration_weight','demonstration_decay_updates',
   'ground_exploration_leg_floor','ground_exploration_wheel_floor','air_exploration_multiplier','exploration_cap'}
  self.assertEqual({k:v for k,v in C.items() if k not in changed},{k:v for k,v in old.items() if k not in changed})
  for name in ('env.py','scene.py','protocol.py','shaping.py','recovery.py','dynamics.py','physics_review.py','record.py'):
   self.assertEqual((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v6'/name).read_text().replace('r5a_auto_v6','r5a_guided_v2'),(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2'/name).read_text())
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
 def test_auxiliary_weight_decays_without_touching_rollout_buffer(self):
  from types import SimpleNamespace
  marker=object();alg=SimpleNamespace(storage=marker)
  self.assertEqual(auxiliary_update(alg,None,2000),(0.,0.));self.assertIs(alg.storage,marker)
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
  data=np.load(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2/teacher/evidence/target_gpu_boundary.npz')
  p=json.loads((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2/teacher/candidates/teacher_rear15_d80.json').read_text())['parameters']
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

def run_tests():
 torch.set_num_threads(1)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise RuntimeError('Guided CPU checks failed')
if __name__=='__main__':run_tests()
