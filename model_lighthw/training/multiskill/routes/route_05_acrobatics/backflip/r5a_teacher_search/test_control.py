import unittest,math
from types import SimpleNamespace as NS
import torch
from .control import Teacher,interpolate,DIM
from .control_actions import Teacher as ActionTeacher
class Dummy(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
 def latent_mean(self,obs):return obs[:,:16]*.7+self.weight
 def act_inference(self,obs):return torch.tanh(self.latent_mean(obs))
class Tests(unittest.TestCase):
 def environment(self,n,launched=False,landed=False,t=.5):
  return NS(state=NS(launched=torch.full((n,),launched),landed=torch.full((n,),landed),angle=torch.ones(n)),episode_physics_steps=torch.full((n,),round(t/.001)),physics_dt=.001)
 def test_zero_residual_preserves_feedback_actor(self):
  obs=torch.randn(9,420);actor=Dummy()
  for cls in (Teacher,ActionTeacher):
   for flying in (False,True):
    torch.testing.assert_close(cls(actor,torch.zeros(9,DIM))(obs,self.environment(9,flying)),actor.act_inference(obs),atol=0,rtol=0)
 def test_no_action_residual_before_control_or_after_landing(self):
  obs=torch.randn(4,420);actor=Dummy()
  for cls in (Teacher,ActionTeacher):
   for env in (self.environment(4,t=.3),self.environment(4,True,True)):
    torch.testing.assert_close(cls(actor,torch.ones(4,DIM)*2)(obs,env),actor.act_inference(obs),atol=0,rtol=0)
 def test_actions_bounded_even_for_extreme_parameters(self):
  for cls in (Teacher,ActionTeacher):
   out=cls(Dummy(),torch.randn(128,DIM)*100)(torch.randn(128,420),self.environment(128,True))
   self.assertTrue(torch.isfinite(out).all());self.assertLessEqual(float(out.abs().max()),1)
 def test_interpolation_has_no_overshoot_and_exact_knots(self):
  values=torch.tensor([[[0.],[1.],[.5]]]).repeat(5,1,1)
  out=interpolate(values,torch.tensor([-1.,0.,.5,1.,2.]),[0.,.5,1.])
  torch.testing.assert_close(out[:,0],torch.tensor([0.,0.,1.,.5,.5]))
 def test_all_families_obey_action_and_recovery_contract(self):
  from .registry import MODULES,controller
  obs=torch.randn(7,420);actor=Dummy()
  for family in MODULES:
   module=controller(family);p=torch.randn(7,module.DIM)*5;teacher=module.Teacher(actor,p)
   for flying in (False,True):
    result=teacher(obs,self.environment(7,flying))
    self.assertEqual(tuple(result.shape),(7,16));self.assertTrue(torch.isfinite(result).all());self.assertLessEqual(float(result.abs().max()),1.)
   torch.testing.assert_close(teacher(obs,self.environment(7,True,True)),actor.act_inference(obs),rtol=0,atol=0)
 def test_pulse_time_boundaries_and_mirrored_targets(self):
  from .control_pulse import Teacher as Pulse,DIM as N
  p=torch.zeros(1,N);p[:,0]=-1.;p[:,7:11]=torch.tensor([-.8,.8,.7,.6]);p[:,11:15]=torch.tensor([-.7,-.5,.6,.5]);p[:,15:19]=.3
  teacher=Pulse(Dummy(),p);obs=torch.zeros(1,420)
  first=teacher(obs,self.environment(1,t=.4));second=teacher(obs,self.environment(1,t=.44));third=teacher(obs,self.environment(1,t=.48))
  torch.testing.assert_close(first[0,[1,2,7,8]],torch.tensor([-.8,.8,-.7,-.6]))
  torch.testing.assert_close(second[0,[1,2,7,8]],torch.tensor([-.7,-.5,-.6,-.5]))
  torch.testing.assert_close(third[0,[1,2,7,8]],torch.tensor([.3,.3,-.3,-.3]))
if __name__=='__main__':unittest.main()
