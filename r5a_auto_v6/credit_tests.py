"""v6 credit tests: actual state machine, deferred payment, no hop/oscillation farming."""
import math,unittest
import torch
from .protocol import FlipState
from .shaping import flight_shaping,takeoff_score

class CreditTests(unittest.TestCase):
 def scenario(self,short=False,failed=False,oscillate=False,second=False):
  s=FlipState(1);q0=torch.tensor([[1.,0,0,0.]])
  s.reset(torch.tensor([0]),torch.tensor([.33]),torch.zeros(1,2),q0)
  paid=0.;first=None;early=0.;maximum=0.;before_second=None
  land=480 if short else 750
  for k in range(400,1101):
   old=s.compact_potential.clone();s.valid_air_event.zero_();s.landing_event.zero_()
   flying=430<=k<land or (second and 850<=k<1050)
   force=torch.zeros(1,4,3);force[:,:,2]=0. if flying else 30.
   angle=min(.6,max(0.,(k-430)*.012))
   if oscillate and k>500:angle=.3+.3*math.sin((k-500)*.1)
   if second and k>=850:angle=.6+(k-850)*.01
   q=torch.tensor([[math.cos(angle/2),0,-math.sin(angle/2),0.]])
   # Compact only in first 60ms, long BEFORE valid flight; old v5 lost this entire credit.
   inertia=.8 if 431<=k<490 else 1.
   z=.45+max(0,k-430)*.001 if flying else .33
   m=dict(com=torch.tensor([[0.,0.,z]]),com_vel=torch.tensor([[0.,0.,1. if flying else 0.]]),
    force=force,force_z=force[:,:,2],clearance=torch.ones(1,4)*(.02 if flying else 0.),
    tilt=torch.tensor([angle]),base_z=torch.tensor([.416]),body_force=torch.tensor([10. if failed and k>=520 else 0.]),
    torque_peak=torch.zeros(1),H_backward=torch.tensor([2.6]),omega_backward=torch.tensor([8.]),locked_inertia=torch.tensor([inertia]))
   s.update(.001,torch.tensor([k],dtype=torch.float32)*.001,m,q)
   credit=float(flight_shaping(s,m,s.compact_potential-old,.001)['compact_rotation'])
   if credit>0 and first is None:first=k
   paid+=credit
   if k<530:early+=credit
   maximum=max(maximum,float(s.compact_credit))
   if k==849:before_second=paid
  return s,dict(paid=paid,first=first,early=early,maximum=maximum,before_second=before_second)
 def test_early_tuck_paid_only_after_validation(self):
  s,r=self.scenario()
  self.assertEqual(r['early'],0.);self.assertGreaterEqual(r['first'],530)
  self.assertAlmostEqual(r['paid'],.6/(2*math.pi),places=5)
  self.assertFalse(bool(s.success));self.assertFalse(bool(s.qualified_landing))
 def test_short_hop_and_failure_cannot_claim_stored_credit(self):
  for kwargs in ({'short':True},{'failed':True}):
   s,r=self.scenario(**kwargs);self.assertEqual(r['paid'],0.);self.assertIsNone(r['first'])
 def test_oscillations_do_not_repeat_angle_credit(self):
  _,a=self.scenario();_,b=self.scenario(oscillate=True)
  self.assertAlmostEqual(a['paid'],b['paid'],places=6);self.assertLessEqual(b['maximum'],1.)
 def test_second_jump_does_not_repay_and_reset_clears_credit(self):
  s,r=self.scenario(second=True);self.assertAlmostEqual(r['paid'],r['before_second'],places=7)
  s.reset(torch.tensor([0]),torch.tensor([.33]),torch.zeros(1,2),torch.tensor([[1.,0,0,0.]]))
  for name in ('compact_credit','compact_potential','raw_rotation_peak','candidate_compact_credit'):
   self.assertEqual(float(getattr(s,name)),0.)
 def test_H_feedback_is_bounded_but_not_capped_at_old_H(self):
  r=takeoff_score(torch.tensor([0.,2.6,5.2,26.]),torch.ones(4)*8,torch.ones(4)*2.4)
  self.assertEqual(float(r[0]),0.);self.assertTrue(bool((r[1:]>r[:-1]).all()));self.assertTrue(bool((r<1).all()))
