"""v2概率、优化器回滚、恢复反例、物理异常筛查与回收边界。"""
import copy,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.distributions import Normal,TransformedDistribution,TanhTransform
from .policy import SquashedActorCritic
from .physics_review import momentum_screen
from .recovery import recovery_terms
from .spec import CONTRACT as C

class V2Tests(unittest.TestCase):
 def model(self):
  torch.manual_seed(17)
  return SquashedActorCritic(3,4,2,actor_hidden_dims=[8],critic_hidden_dims=[8],noise_std_type='log',init_noise_std=.25)
 def test_transformed_density_and_gradient(self):
  p=self.model();obs=torch.randn(64,3);actions=p.act(obs)
  self.assertTrue(bool((actions.abs()<1).all()))
  reference=TransformedDistribution(p.distribution,[TanhTransform(cache_size=1)])
  torch.testing.assert_close(p.get_actions_log_prob(actions),reference.log_prob(actions).sum(-1),atol=3e-5,rtol=1e-5)
  loss=-(p.get_actions_log_prob(actions).mean()+.002*p.entropy.mean());loss.backward()
  self.assertTrue(all(bool(torch.isfinite(x.grad).all()) for x in p.actor.parameters()))
  with torch.no_grad():p.actor[-1].bias.fill_(100)
  self.assertTrue(bool((p.act_inference(obs).abs()<1).all()))
  self.assertTrue(bool(torch.isfinite(p.get_actions_log_prob(torch.ones_like(actions))).all()))
 def test_tanh_kl_matches_monte_carlo(self):
  torch.manual_seed(22)
  old=Normal(torch.tensor([.1,-.2]),torch.tensor([.3,.2]));new=Normal(torch.tensor([.15,-.15]),torch.tensor([.32,.22]))
  z=old.sample((100000,));a=z.tanh()
  td=lambda d:TransformedDistribution(d,[TanhTransform()])
  mc=(td(old).log_prob(a)-td(new).log_prob(a)).sum(-1).mean()
  exact=torch.distributions.kl_divergence(old,new).sum()
  self.assertAlmostEqual(float(mc),float(exact),delta=.004)
 def test_kl_rejection_restores_policy_and_adam(self):
  from .update_guard import guarded_update
  p=self.model();opt=torch.optim.Adam(p.parameters(),lr=1e-4)
  x=torch.randn(12,3);p.act_inference(x).sum().backward();opt.step();opt.zero_grad()
  p.update_distribution(x)
  storage=SimpleNamespace(observations=x.reshape(3,4,3),mu=p.action_mean.detach().reshape(3,4,2).clone(),sigma=p.action_std.detach().reshape(3,4,2).clone(),actions=p.act_inference(x).detach())
  alg=SimpleNamespace(policy=p,optimizer=opt,learning_rate=1e-4,storage=storage)
  before=copy.deepcopy(p.state_dict());adam=copy.deepcopy(opt.state_dict())
  def bad():
   p.act_inference(x).sum().backward();opt.step()
   with torch.no_grad():p.actor[-1].bias.add_(2.)
   return {'surrogate':1.}
  _,row=guarded_update(alg,bad)
  self.assertFalse(row['accepted']);self.assertGreater(row['rollout_KL'],.03)
  for k,v in before.items():torch.testing.assert_close(v,p.state_dict()[k],atol=0,rtol=0)
  for k,v in adam['state'].items():
   for name,value in v.items():torch.testing.assert_close(value,opt.state_dict()['state'][k][name],atol=0,rtol=0)
  self.assertEqual(alg.learning_rate,5e-5)
  _,row=guarded_update(alg,lambda:{'surrogate':0.})
  self.assertTrue(row['accepted']);self.assertAlmostEqual(row['rollout_KL'],0.,places=6)
 def test_recovery_dense_and_gated(self):
  n=4;s=SimpleNamespace(flight_valid=torch.tensor([False,True,True,True]),landed=torch.ones(n,dtype=torch.bool),failed=torch.tensor([False,False,False,True]),angle=torch.full((n,),2*math.pi),flight_rise=torch.full((n,),.3),flight_time=torch.full((n,),.4),landing_xy=torch.zeros(n,2))
  m=dict(tilt=torch.tensor([0.,.35,0.,0.]),base_z=torch.tensor([.416,.27,.416,.416]),force_z=torch.tensor([[30.,30.,30.,30.],[30.,0.,30.,30.],[30.,30.,30.,30.],[30.,30.,30.,30.]]),com_vel=torch.zeros(n,3),com=torch.zeros(n,3))
  r=recovery_terms(s,m,torch.zeros(n,4),.02)
  for k,v in r.items():self.assertEqual(float(v[0]),0.);self.assertEqual(float(v[3]),0.)
  for k in ('recovery_upright','recovery_height','recovery_support'):
   self.assertGreater(float(r[k][1]),0.);self.assertGreater(float(r[k][2]),float(r[k][1]))
 def test_momentum_detects_clamp_not_posture(self):
  h=np.zeros((200,3));h[:,1]=-1.
  self.assertTrue(momentum_screen(h)['physics_consistent'])
  h[:,1]-=np.linspace(0,8,200)
  self.assertFalse(momentum_screen(h)['physics_consistent'])
  h[:,1]=-1.;h[:,0]=np.linspace(0,1,200)
  self.assertFalse(momentum_screen(h)['physics_consistent'])
  self.assertIsNone(momentum_screen([])['physics_consistent'])
 def test_rigid_cap_units_do_not_change_joint_limits(self):
  self.assertLess(math.radians(1000),C['wheel_speed_rad_s'])
  self.assertGreater(math.radians(C['rigid_max_angular_velocity_deg_s']),10*C['wheel_speed_rad_s'])
  self.assertEqual((C['leg_effort_Nm'],C['wheel_effort_Nm'],C['leg_speed_rad_s'],C['wheel_speed_rad_s']),(48.,24.,12.57,37.7))
 def test_selection_prefers_recovery_over_millimeters(self):
  from .cli import rank
  base=dict(physics_failure_rate=0.,success_rate=0.,qualified_rate=1.,mean_stable_s=0.,mean_recovery_score=.2,valid_flight_rate=1.,mean_valid_angle_deg=360.,launch_rate=1.,failed_rate=0.,mean_impact_bodyweights=40.,mean_flight_rise_m=.28)
  recovery=dict(base,mean_recovery_score=.8,mean_flight_rise_m=.27)
  self.assertGreater(rank(recovery),rank(base))
  self.assertLess(rank(dict(recovery,physics_failure_rate=1.)),rank(base))
 def test_usb_zip_excludes_machine_cache(self):
  from .io import seal,pack_review,sha
  import zipfile
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'review';(root/'asset_cache').mkdir(parents=True);(root/'asset_cache/cache.usd').write_bytes(b'cache')
   (root/'evidence.txt').write_text('review');seal(root);archive=pack_review(root)
   with zipfile.ZipFile(archive) as z:
    self.assertIsNone(z.testzip());self.assertEqual(set(z.namelist()),{'review/evidence.txt','review/SHA256.json'})
   self.assertTrue(archive.with_name(archive.name+'.sha256').read_text().startswith(sha(archive)))
