"""解析反例测试，不需要 Isaac，不能替代GPU验收。"""
import ast,math,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from .protocol import FlipState,targets,delta_angle
from .spec import CONTRACT as C
class ProtocolTests(unittest.TestCase):
 def test_action_contract(self):
  x=torch.zeros(3,16);x[0]=-1;x[2]=1;y=targets(x)
  for i,key in enumerate(('action_lower','stand','action_upper')):torch.testing.assert_close(y[i],torch.tensor(C[key]))
 def test_bounds_against_urdf(self):
  import xml.etree.ElementTree as ET
  from .spec import ROOT,LEG_JOINTS
  joints={j.attrib['name']:j for j in ET.parse(ROOT/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf').findall('joint')}
  for name,lo,hi in zip(LEG_JOINTS,C['action_lower'],C['action_upper']):
   limits=joints[name].find('limit').attrib
   self.assertGreaterEqual(lo,float(limits['lower']));self.assertLessEqual(hi,float(limits['upper']))
 def test_quaternion_full_turn(self):
  axis=torch.tensor([[0.,1.,0.]]);prev=torch.tensor([[1.,0,0,0.]]);total=0.
  for a in torch.linspace(0,2*math.pi,1001)[1:]:
   q=torch.tensor([[math.cos(float(a)/2),0,-math.sin(float(a)/2),0]])
   total+=float(delta_angle(q,prev,axis));prev=q
  self.assertAlmostEqual(total,2*math.pi,places=5)
 def test_batch_success_and_counterexamples(self):
  # full flip / no flip / forward / double / collision / brief hop / ground roll / side axis / late fall
  n=9;s=FlipState(n);q0=torch.zeros(n,4);q0[:,0]=1
  s.reset(torch.arange(n),torch.full((n,),.33),torch.zeros(n,2),q0)
  for k in range(1,2501):
   u=k*.001;t=torch.full((n,),u);force=torch.zeros(n,5,3);force[:,:4,2]=30.
   free=.6<=u<1.
   if free:force[:5]=0;force[7:]=0
   if .6<=u<.603:force[5]=0
   com=torch.zeros(n,3);com[:,2]=.33;v=torch.zeros(n,3)
   if free:com[:,2]+=.1+.5*(u-.6)-.4*(u-.6)**2;v[:,2]=.5-.8*(u-.6)
   a=torch.zeros(n);f=min(1,max(0,(u-.6)/.4))
   a[0]=a[4]=a[6]=a[7]=a[8]=f*2*math.pi;a[2]=-f*2*math.pi;a[3]=f*4*math.pi
   q=torch.zeros(n,4);q[:,0]=torch.cos(a/2);q[:,2]=-torch.sin(a/2)
   if free:q[7]=torch.tensor([math.cos(.5),math.sin(.5),0.,0.])
   body=torch.zeros(n)
   if u>=.85:body[4]=10
   if u>=2.3:body[8]=10
   m=dict(com=com,com_vel=v,force=force,force_z=force[:,:4,2],clearance=torch.ones(n,4)*.01,
    tilt=torch.acos((1-2*q[:,2].square()-2*q[:,1].square()).clamp(-1,1)),base_z=torch.ones(n)*.416,body_force=body,torque_peak=torch.zeros(n))
   s.update(.001,t,m,q)
  self.assertTrue(bool(s.success[0]));self.assertFalse(bool(s.success[1:].any()))
  self.assertFalse(bool(s.launched[5]));self.assertFalse(bool(s.launched[6]));self.assertEqual(float(s.angle[6]),0.)
  self.assertAlmostEqual(float(s.flight_time[0]),.4,places=4);self.assertEqual(float(s.stable_time[8]),0.)
  s.reset(torch.tensor([0]),torch.tensor([.33]),torch.zeros(1,2),q0[:1]);self.assertFalse(bool(s.launched[0]));self.assertEqual(float(s.angle[0]),0)
 def test_dropping_not_launch(self):
  s=FlipState(1)
  for k in range(20):
   force=torch.zeros(1,4,3)
   m=dict(com=torch.tensor([[0.,0.,.4]]),com_vel=torch.tensor([[0.,0.,-.5]]),force=force,
    force_z=force[:,:,2],clearance=torch.ones(1,4)*.1,tilt=torch.zeros(1),base_z=torch.ones(1)*.4,body_force=torch.zeros(1),torque_peak=torch.zeros(1))
   s.update(.001,torch.tensor([.6+k*.001]),m,torch.tensor([[1.,0,0,0]]))
  self.assertFalse(bool(s.launched[0]))
 def test_standing_has_no_success_reward(self):
  tree=ast.parse((Path(__file__).parent/'env.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AutoFlipEnv')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards');ns=dict(torch=torch,math=math,C=C,recovery_terms=__import__('r5a_auto_v5.recovery',fromlist=['recovery_terms']).recovery_terms)
  from .shaping import flight_shaping,landing_progress,flight_quality,reward_potentials,preparation_potential
  ns.update(flight_shaping=flight_shaping,landing_progress=landing_progress,flight_quality=flight_quality,reward_potentials=reward_potentials,preparation_potential=preparation_potential)
  exec(compile(ast.Module(body=[method],type_ignores=[]),'reward_test','exec'),ns)
  s=FlipState(1);z=torch.zeros(1);m=dict(tilt=z,com_vel=torch.zeros(1,3),force_z=torch.ones(1,4)*30,base_z=torch.tensor([.416]),com=torch.zeros(1,3),locked_inertia=torch.ones(1))
  e=SimpleNamespace(state=s,current_metrics=m,step_dt=.02,old_potential=torch.zeros(1,5),old_preparation=torch.zeros(1),old_extension=torch.zeros(1),old_launch_score=torch.zeros(1),old_air_rotation=torch.zeros(1),phase_switch=torch.zeros(1,dtype=torch.bool),episode_length_buf=torch.tensor([50]),
   max_episode_length=300,episode_physics_steps=torch.tensor([1000]),physics_dt=.001,actions=torch.zeros(1,16),previous_actions=torch.zeros(1,16),leg_ids=list(range(12)),wheel_ids=list(range(12,16)),
   robot=SimpleNamespace(data=SimpleNamespace(applied_torque=torch.zeros(1,16),joint_vel=torch.zeros(1,16),root_ang_vel_b=torch.zeros(1,3))),
   mass=torch.tensor([11.668]),peak_support=torch.tensor([120.]),extras={'log':{}},episode_rewards={k:torch.zeros(1) for k in C['reward_weights']})
  self.assertLess(float(ns['_get_rewards'](e)),0);self.assertEqual(float(e.reward_terms['stable_rate']),0);self.assertEqual(float(e.reward_terms['terminal_success']),0)
 def test_exact_physics_clock_and_hold(self):
  state=FlipState(1);state.launched.fill_(True);state.landed.fill_(True);state.qualified_landing.fill_(True)
  q=torch.tensor([[1.,0.,0.,0.]])
  force=torch.zeros(1,4,3);force[:,:,2]=30.
  m=dict(com=torch.tensor([[0.,0.,.33]]),com_vel=torch.zeros(1,3),force=force,force_z=force[:,:,2],
   clearance=torch.zeros(1,4),tilt=torch.zeros(1),base_z=torch.tensor([.416]),body_force=torch.zeros(1),torque_peak=torch.zeros(1))
  for k in range(1000):state.update(.001,torch.tensor([1.+k*.001]),m,q)
  self.assertEqual(float(state.stable_time[0]),1.);self.assertTrue(bool(state.success[0]))
  state.reset(torch.tensor([0]),torch.tensor([.33]),torch.zeros(1,2),q)
  state.update(.001,torch.tensor([3.]),m,q)
  self.assertTrue(bool(state.failed[0]));self.assertEqual(int(state.reason[0]),3)
 def test_analytical_momentum(self):
  from .dynamics import self_test
  self_test()
def run_tests():
 from .v2_tests import V2Tests
 from .v5_tests import V5Tests
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (ProtocolTests,V2Tests,V5Tests)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise RuntimeError('CPU tests failed')
 return result.testsRun
if __name__=='__main__':run_tests()
