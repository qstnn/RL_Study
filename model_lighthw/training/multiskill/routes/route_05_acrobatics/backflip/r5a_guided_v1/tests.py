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
   self.assertEqual((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v6'/name).read_text().replace('r5a_auto_v6','r5a_guided_v1'),(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v1'/name).read_text())
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

def run_tests():
 torch.set_num_threads(1)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise RuntimeError('Guided CPU checks failed')
if __name__=='__main__':run_tests()
