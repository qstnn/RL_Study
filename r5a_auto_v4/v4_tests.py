"""v4关键反例：预蹲刷分、低跳换转角、相位概率不一致、迁移污染与选模误标。"""
import copy,json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch.distributions import TransformedDistribution,TanhTransform
from .spec import CONTRACT as C
from .shaping import flight_quality,landing_progress,flight_shaping,preparation_potential,takeoff_score
from .recovery import recovery_terms
from .policy import SquashedActorCritic

class V4Tests(unittest.TestCase):
 def model(self):
  torch.manual_seed(8)
  return SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu')
 def test_low_jump_cannot_hide_behind_larger_angle(self):
  angle=torch.tensor([147.56,147.56,101.46]);height=torch.tensor([.1149,.30,.2369]);time=torch.tensor([.273,.40,.355])
  joint=angle/360*flight_quality(height,time)
  self.assertGreater(float(joint[1]),float(joint[0])*2.5)
  self.assertGreater(float(joint[2]),float(joint[0]))
  self.assertEqual(float(flight_quality(torch.tensor([0.]),torch.tensor([100.]))),0.)
 def test_preparation_discounted_return_no_stalling_profit(self):
  # First down/up/down then wait: no successful flight, terminal Phi=0.
  for potentials in ([0.,.5,0.,1.]+[1.]*20+[0.],[0.,1.,0.,1.,0.], [0.]*20):
   total=sum(C['gamma']**i*(C['gamma']*potentials[i+1]-potentials[i]) for i in range(len(potentials)-1))
   self.assertAlmostEqual(total,0.,places=12)
 def test_preparation_outcome_requires_real_valid_flight(self):
  n=4;s=NS(landing_event=torch.tensor([False,True,True,True]),flight_valid=torch.tensor([False,True,False,True]),failed=torch.tensor([False,False,False,True]),release_inertia=torch.ones(n),release_H=torch.ones(n),release_omega=torch.ones(n),launch_vz=torch.ones(n)*2.,release_compression=torch.ones(n)*100,launched=torch.ones(n,dtype=torch.bool),landed=torch.ones(n,dtype=torch.bool),flight_rise=torch.ones(n)*.3,flight_time=torch.ones(n)*.4)
  r=flight_shaping(s,dict(locked_inertia=torch.ones(n)*.5),torch.ones(n),.02)
  torch.testing.assert_close(r['preparation_credit'],torch.tensor([0.,1.,0.,0.]));self.assertEqual(float(r['compact_rotation'].sum()),0.)
 def test_preparation_excludes_settle_and_unloaded_lean(self):
  from .protocol import FlipState
  s=FlipState(1);q=torch.tensor([[1.,0.,0.,0.]])
  def metric(z,load=30.,tilt=0.):
   force=torch.zeros(1,4,3);force[:,:,2]=load
   return dict(com=torch.tensor([[0.,0.,z]]),com_vel=torch.zeros(1,3),force=force,force_z=force[:,:,2],clearance=torch.zeros(1,4),tilt=torch.tensor([tilt]),base_z=torch.tensor([z+.08]),body_force=torch.zeros(1),torque_peak=torch.zeros(1))
  s.reset(torch.tensor([0]),torch.tensor([.4]),torch.zeros(1,2),q)
  s.update(.001,torch.tensor([.4]),metric(.32),q);self.assertEqual(float(s.compression_peak),0.)
  s.update(.001,torch.tensor([.401]),metric(.25,1.),q);self.assertEqual(float(s.compression_peak),0.)
  s.update(.001,torch.tensor([.402]),metric(.25,30.,1.),q);self.assertEqual(float(s.compression_peak),0.)
  s.update(.001,torch.tensor([.403]),metric(.26),q);self.assertAlmostEqual(float(s.compression_peak),.06,places=5)
 def test_phase_actor_migration_and_probability(self):
  from .bootstrap import initialize
  p=self.model();opt=torch.optim.Adam(p.parameters());r=NS(device='cpu',alg=NS(policy=p,optimizer=opt),completed_updates=0)
  critic=copy.deepcopy(p.critic.state_dict())
  with tempfile.TemporaryDirectory() as td:initialize(r,Path(td));meta=json.loads((Path(td)/'bootstrap.json').read_text());self.assertTrue(meta['ground_actor_reinitialized'])
  for k,v in critic.items():torch.testing.assert_close(v,p.critic.state_dict()[k],atol=0,rtol=0)
  self.assertFalse(opt.state)
  obs=torch.randn(96,420)*.05;obs[:,-8]=0.;obs[32:,-8]=1.;obs[:,-7]=0.;obs[64:,-7]=1.
  with torch.no_grad():
   reference=torch.tanh(3*torch.tanh(p.actor(obs)/3));actual=p.act_inference(obs)
   torch.testing.assert_close(actual[32:],reference[32:],atol=0,rtol=0);self.assertGreater(float((actual[:32]-reference[:32]).abs().max()),.01)
  a=p.act(obs);dist=p.distribution
  self.assertGreaterEqual(float(dist.scale[:32,:12].min()),.34999)
  self.assertTrue(bool((dist.scale[64:]<=dist.scale[32:64]).all()))
  td=TransformedDistribution(dist,[TanhTransform(cache_size=1)])
  torch.testing.assert_close(p.get_actions_log_prob(a),td.log_prob(a).sum(-1),rtol=2e-5,atol=5e-5)
  loss=-(p.get_actions_log_prob(a).mean()+.002*p.entropy.mean());loss.backward()
  for module in (p.actor,p.ground_actor):self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in module.parameters()))
 def test_ground_only_update_is_tracked_and_rolls_back(self):
  from .update_guard import guarded_update
  p=self.model();obs=torch.zeros(12,420);opt=torch.optim.Adam(p.parameters(),lr=1e-4)
  p.act_inference(obs).sum().backward();opt.step();opt.zero_grad();p.update_distribution(obs)
  storage=NS(observations=obs.reshape(3,4,420),mu=p.action_mean.detach().reshape(3,4,16).clone(),sigma=p.action_std.detach().reshape(3,4,16).clone(),actions=p.act_inference(obs).detach())
  alg=NS(policy=p,optimizer=opt,learning_rate=1e-4,storage=storage);before=copy.deepcopy(p.state_dict());adam=copy.deepcopy(opt.state_dict())
  def excessive():
   p.act_inference(obs).sum().backward();opt.step()
   with torch.no_grad():p.ground_actor[-1].bias.add_(3.)
   return {'surrogate':1.}
  _,row=guarded_update(alg,excessive);self.assertFalse(row['accepted'])
  for k,v in before.items():torch.testing.assert_close(v,p.state_dict()[k],atol=0,rtol=0)
  for k,v in adam['state'].items():
   for name,value in v.items():torch.testing.assert_close(value,opt.state_dict()['state'][k][name],atol=0,rtol=0)
 def test_recovery_requires_support_and_quiet_together(self):
  n=3;s=NS(flight_valid=torch.ones(n,dtype=torch.bool),landed=torch.ones(n,dtype=torch.bool),failed=torch.zeros(n,dtype=torch.bool),angle=torch.ones(n)*2*math.pi,flight_rise=torch.ones(n)*.3,flight_time=torch.ones(n)*.4,landing_xy=torch.zeros(n,2))
  m=dict(tilt=torch.zeros(n),base_z=torch.ones(n)*.416,force_z=torch.ones(n,4)*30,com_vel=torch.zeros(n,3),com=torch.zeros(n,3));m['force_z'][0,2]=0.;m['com_vel'][1,0]=.3;m['com'][1,0]=1.
  r=recovery_terms(s,m,torch.zeros(n,4),.02)
  self.assertEqual(float(r['recovery_combined'][0]),0.);self.assertLess(float(r['recovery_combined'][1]),1e-20)
  for k,v in r.items():
   if k!='landing_wheel_spin':self.assertLessEqual(float(v[2])*300,1.00001)
 def test_best_model_does_not_mean_largest_angle(self):
  from .cli import select_candidate,summary
  base=dict(success=False,qualified_landing=False,flight_valid=True,launched=True,physics_consistent=True,stable_s=0.,failed=False,flight_s=.273,peak_support_bodyweights=14.,flight_rise_m=.115)
  rows=[]
  for label,angle,recovery in [('unstable',147.56,.42),('steady',146.96,.65),('bad_physics',360.,1.)]:
   c=dict(base,airborne_angle_deg=angle,recovery_score=recovery,physics_consistent=label!='bad_physics');rows.append(dict(label=label,metrics=summary(dict(cases=[c]))))
  self.assertEqual(select_candidate(rows)['label'],'steady')

 def test_record_preparation_even_when_no_takeoff(self):
  import numpy as np
  from .record import judge
  n=501;t=np.arange(n)*.001;com=np.zeros((n,3));com[:,2]=.33;com[450:,2]=.30
  root=np.zeros((n,13));root[:,3]=1.;force=np.zeros((n,4,3));force[:,:,2]=30.
  a=dict(t=t,com=com,com_velocity=np.zeros((n,3)),root_state=root,contact_force=force,wheel_clearance=np.zeros((n,4)),
   wheel_force_z=force[:,:,2],body_force=np.zeros(n),base_height=np.ones(n)*.416,tilt=np.zeros(n),applied_torque=np.zeros((n,16)),
   online_launched=np.zeros(n,dtype=bool),online_landed=np.zeros(n,dtype=bool),online_failed=np.zeros(n,dtype=bool),online_success=np.zeros(n,dtype=bool),online_angle=np.zeros(n),online_stable_time=np.zeros(n),online_preparation=np.r_[np.zeros(450),np.ones(n-450)*.03])
  r=judge(a,dict(reason='interrupted',masses=[11.668]))
  self.assertFalse(r['launched']);self.assertAlmostEqual(r['active_preparation_COM_m'],.03,places=6)
  self.assertTrue(r['preparation_online_agrees']);self.assertTrue(r['online_agrees'])
