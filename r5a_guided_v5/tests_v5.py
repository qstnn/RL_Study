"""V5 CPU invariants; synthetic fixtures are not robot acceptance."""
import contextlib, io, math, unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from .tests_v4 import Checks as PreviousChecks, fixture
from .spec import CONTRACT as C, ROOT

class Checks(PreviousChecks):
 def test_physics_observation_and_acceptance_unchanged(self):
  from r5a_guided_v4.spec import CONTRACT as old
  changed={k for k in C if k not in old or C[k]!=old[k]}
  self.assertEqual(changed,{'version','reward_weights','impact_free_bodyweights','impact_scale_bodyweights','impact_huber_knee','recovery_catch_grace_s','recovery_anchor_scale_m','recovery_return_gain_s_inv','recovery_return_speed_m_s','reward_revision','candidate_contact_audit_required'})
  for name in ('scene.py','dynamics.py','physics_review.py','policy.py','protocol.py'):
   self.assertEqual((ROOT/'r5a_guided_v4'/name).read_text().replace('r5a_guided_v4','r5a_guided_v5'),(ROOT/'r5a_guided_v5'/name).read_text())
  self.assertEqual(C['reward_weights']['terminal_success'],80.)
 def test_mixed_teacher_proof_cannot_enable_ppo(self):
  # Test the independent autonomous-proof gate with contact evidence isolated.
  # Real missing/tampered contact evidence is tested by tests_workflow_v5.
  with patch('r5a_guided_v5.candidate_prepare.validate_candidate_audit',return_value=True):
   super().test_mixed_teacher_proof_cannot_enable_ppo()
 def test_counterfactual_reward_properties(self):
  from .reward_audit import formula_checks
  self.assertTrue(formula_checks()['passed'])
 def test_anchor_is_bounded_and_grace_preserves_catching(self):
  from .landing import advance
  s,m=self.state_metrics();s.landed.fill_(True);s.qualified_landing.fill_(True);s.landing_t.fill_(1.)
  m['com'][:,0]=.15;m['com_vel'][:,0]=.2
  advance(s,m,.001,torch.tensor([1.249]),torch.tensor([False]))
  self.assertEqual(float(s.anchor_position_cost),0.);self.assertEqual(float(s.anchor_return_cost),0.)
  advance(s,m,.001,torch.tensor([1.251]),torch.tensor([False]))
  self.assertGreater(float(s.anchor_position_cost),0.);self.assertLess(float(s.anchor_position_cost),.001/6)
  s.failed.fill_(True);before=float(s.anchor_position_cost)
  advance(s,m,.001,torch.tensor([1.252]),torch.tensor([False]));self.assertEqual(before,float(s.anchor_position_cost))
 def test_anchor_integrates_once_at_actual_physics_dt(self):
  from .landing import advance
  values=[]
  for dt in (.001,.002,.02):
   s,m=self.state_metrics();s.landed.fill_(True);s.qualified_landing.fill_(True)
   m['com'][:,0]=.12;m['com_vel'][:,0]=.1
   for i in range(round(1/dt)):advance(s,m,dt,torch.tensor([2.+i*dt]),torch.tensor([False]))
   values.append(float(s.anchor_position_cost))
  self.assertLess(max(values)-min(values),2e-6)
 def test_peak_cost_is_charged_even_when_failed(self):
  from .shaping import reward_potentials
  s,m=self.state_metrics();s.landed.fill_(True)
  before=reward_potentials(s,torch.tensor([10.]))[:,4]
  s.failed.fill_(True);s.impact_peak.fill_(100*10*9.81)
  after=reward_potentials(s,torch.tensor([10.]))[:,4]
  self.assertGreater(float(after-before),0.)
  self.assertEqual(float((reward_potentials(s,torch.tensor([10.]))[:,4]-after)),0.)
 def test_bounce_keeps_first_landing_anchor(self):
  from .protocol import FlipState
  s=FlipState(1);s.launched.fill_(True);s.landed.fill_(True);s.landing_xy[:]=torch.tensor([[.03,.02]])
  s.release_t.fill_(.5);s.landing_t.fill_(1.)
  m=dict(com=torch.tensor([[.2,.1,.4]]),com_vel=torch.zeros(1,3),force=torch.ones(1,4,3)*10,
   force_z=torch.ones(1,4)*30,clearance=torch.ones(1,4)*.01,tilt=torch.zeros(1),base_z=torch.ones(1)*.416,body_force=torch.zeros(1),torque_peak=torch.zeros(1))
  for t in (1.2,1.3):s.update(.001,torch.tensor([t]),m,torch.tensor([[1.,0.,0.,0.]]))
  torch.testing.assert_close(s.landing_xy,torch.tensor([[.03,.02]]))
  self.assertEqual(float(s.landing_t),1.)

def run_tests():
 torch.set_num_threads(1)
 from .tests_contact import ContactTests
 from .tests_workflow_v5 import WorkflowTests
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(k) for k in (Checks,ContactTests,WorkflowTests))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise RuntimeError('V5 CPU checks failed')
if __name__=='__main__':run_tests()
