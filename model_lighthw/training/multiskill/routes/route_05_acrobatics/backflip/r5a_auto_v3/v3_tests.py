"""v3真实失败模式反例：消失的反馈、辅助刷分、迁移污染。"""
import math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from .spec import CONTRACT as C
from .shaping import landing_progress,takeoff_score,flight_shaping
from .recovery import recovery_terms
class V3Tests(unittest.TestCase):
 def test_partial_turn_feedback_and_full_flip_preference(self):
  a=torch.tensor([0.,-math.pi,math.radians(101.5),2*math.pi,4*math.pi]);x=landing_progress(a)
  self.assertEqual(float(x[0]),0.);self.assertEqual(float(x[1]),0.)
  self.assertGreater(float(x[2]),.07);self.assertLess(float(x[2]),.09)
  self.assertEqual(float(x[3]),1.);self.assertLess(float(x[4]),1e-9)
  self.assertGreater(16+6+4+40,16*101.5/360+6*.08+8*.08+3+.5+2+2+1+.5)
 def test_takeoff_requires_H_and_direction(self):
  h=torch.tensor([0.,-1.,1.,1.]);omega=torch.tensor([10.,10.,-4.,4.]);score=takeoff_score(h,omega)
  self.assertEqual(float(score[0]),0.);self.assertEqual(float(score[1]),0.)
  self.assertGreater(float(score[3]),float(score[2]));self.assertTrue(bool((score<=1).all()))
 def test_preparation_only_paid_after_first_valid_flight(self):
  n=4;s=SimpleNamespace(landing_event=torch.tensor([False,True,True,False]),flight_valid=torch.tensor([False,True,False,True]),failed=torch.zeros(n,dtype=torch.bool),release_inertia=torch.ones(n),release_H=torch.ones(n),release_omega=torch.ones(n),release_compression=torch.ones(n)*100,launched=torch.ones(n,dtype=torch.bool),landed=torch.ones(n,dtype=torch.bool),flight_rise=torch.ones(n))
  r=flight_shaping(s,dict(locked_inertia=torch.ones(n)*.5),torch.ones(n),.02)
  torch.testing.assert_close(r['preparation_credit'],torch.tensor([0.,1.,0.,0.]))
  self.assertEqual(float(r['compact_rotation'].sum()),0.)
  s.landed.zero_();r=flight_shaping(s,dict(locked_inertia=torch.ones(n)*.5),torch.ones(n)*.1,.02)
  self.assertTrue(bool((r['compact_rotation']<=.1).all()))
 def test_recovery_total_bounded_and_no_standing_farm(self):
  n=3;s=SimpleNamespace(flight_valid=torch.tensor([False,True,True]),landed=torch.ones(n,dtype=torch.bool),failed=torch.zeros(n,dtype=torch.bool),angle=torch.tensor([2*math.pi,math.radians(101.5),2*math.pi]),landing_xy=torch.zeros(n,2))
  m=dict(tilt=torch.zeros(n),base_z=torch.ones(n)*.416,force_z=torch.ones(n,4)*30,com_vel=torch.zeros(n,3),com=torch.zeros(n,3))
  r=recovery_terms(s,m,torch.zeros(n,4),.02)
  for k,v in r.items():
   self.assertEqual(float(v[0]),0.)
   if k!='landing_wheel_spin':
    self.assertGreater(float(v[1]),0.);self.assertLessEqual(float(v[2])*300,1.00001)
 def test_actor_only_migration(self):
  from .policy import SquashedActorCritic
  from .bootstrap import initialize
  torch.manual_seed(4);p=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu')
  opt=torch.optim.Adam(p.parameters());r=SimpleNamespace(device='cpu',alg=SimpleNamespace(policy=p,optimizer=opt),completed_updates=0)
  before={k:v.clone() for k,v in p.state_dict().items() if k.startswith('critic.')}
  with tempfile.TemporaryDirectory() as td:initialize(r,Path(td));self.assertTrue((Path(td)/'bootstrap.json').exists())
  for k,v in before.items():torch.testing.assert_close(v,p.state_dict()[k],rtol=0,atol=0)
  self.assertFalse(opt.state)
