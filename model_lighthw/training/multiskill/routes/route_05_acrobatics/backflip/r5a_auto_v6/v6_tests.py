"""v6关键反例：预蹲刷分、低跳换转角、相位概率不一致、迁移污染与选模误标。"""
import copy,json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch.distributions import TransformedDistribution,TanhTransform
from .spec import CONTRACT as C
from .shaping import flight_quality,landing_progress,flight_shaping,preparation_potential,takeoff_score
from .recovery import recovery_terms
from .policy import SquashedActorCritic

class V6Tests(unittest.TestCase):
 def test_actor_handoff_latches_only_first_air_control_step(self):
  import ast
  from .protocol import FlipState
  tree=ast.parse(Path(__file__).with_name('env.py').read_text())
  method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='AutoFlipEnv' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
  from .shaping import reward_potentials
  ns=dict(torch=torch,C=C,reward_potentials=reward_potentials,preparation_potential=preparation_potential)
  exec(compile(ast.Module(body=[method],type_ignores=[]),'actual_pre_physics_step','exec'),ns)
  e=NS(extras={},finite=torch.tensor(True),peak_support=torch.zeros(2),state=FlipState(2),mass=torch.ones(2),
   old_potential=torch.zeros(2,5),old_preparation=torch.zeros(2),old_extension=torch.zeros(2),old_launch_score=torch.zeros(2),old_air_rotation=torch.zeros(2),old_compact=torch.zeros(2),
   previous_actions=torch.zeros(2,16),actions=torch.zeros(2,16),episode_physics_steps=torch.full((2,),400),physics_dt=.001,
   air_control_seen=torch.zeros(2,dtype=torch.bool),phase_switch=torch.zeros(2,dtype=torch.bool))
  advance=ns['_pre_physics_step'];advance(e,torch.zeros(2,16));self.assertFalse(bool(e.phase_switch.any()))
  e.state.launched[0]=True;advance(e,torch.ones(2,16));self.assertEqual(e.phase_switch.tolist(),[True,False])
  advance(e,torch.ones(2,16));self.assertFalse(bool(e.phase_switch.any()))
 def trajectory(self,prepared=True,short=False,stale=False):
  """解析接触序列驱动真实1kHz状态机；不代表物理上可达的仿真动作。"""
  from .protocol import FlipState
  s=FlipState(1);q=torch.tensor([[1.,0.,0.,0.]])
  s.reset(torch.tensor([0]),torch.tensor([.33]),torch.zeros(1,2),q)
  release=1330 if stale else 430;land=release+(40 if short else 300)
  total=0.;event_times=[]
  for k in range(400,land+50):
   s.valid_air_event.zero_();s.landing_event.zero_()
   force=torch.zeros(1,4,3);force[:,:,2]=30.
   z=.33;vz=0.
   if prepared and 401<=k<=410:z=.27
   elif prepared and 411<=k<=429:z=.33;vz=2.0
   if k>=release and k<land:force.zero_();z=.45;vz=2.
   m=dict(com=torch.tensor([[0.,0.,z]]),com_vel=torch.tensor([[0.,0.,vz]]),force=force,force_z=force[:,:,2],
    clearance=torch.ones(1,4)*(.02 if release<=k<land else 0),tilt=torch.zeros(1),base_z=torch.tensor([.416]),
    body_force=torch.zeros(1),torque_peak=torch.zeros(1),H_backward=torch.tensor([2.6]),omega_backward=torch.tensor([8.]),locked_inertia=torch.ones(1))
   # 上升到释放之后继续增加COM，满足最小真实跳高。
   if release<k<land:m['com'][0,2]+=(k-release)*.001
   s.update(.001,torch.tensor([k*.001]),m,q)
   terms=flight_shaping(s,m,torch.zeros(1),.001)
   total+=float(terms['preparation_credit'])
   if bool(s.valid_air_event):event_times.append(k)
  return s,total,event_times
 def test_credit_arrives_before_landing_exactly_once(self):
  s,credit,events=self.trajectory()
  self.assertEqual(len(events),1);self.assertLess(events[0],730)
  self.assertGreater(credit,.5);self.assertLessEqual(credit,1.)
  self.assertAlmostEqual(credit,float(s.release_extension),places=6)
 def test_hop_no_compression_and_stale_compression_no_credit(self):
  s,credit,events=self.trajectory(short=True)
  self.assertEqual(credit,0.);self.assertEqual(events,[])
  self.assertEqual(self.trajectory(prepared=False)[1],0.)
  self.assertEqual(self.trajectory(stale=True)[1],0.)
 def test_release_direction_and_vertical_capability_both_needed(self):
  h=torch.tensor([-2.6,2.6,2.6,2.6]);w=torch.tensor([8.,8.,-8.,8.]);vz=torch.tensor([2.4,0.,2.4,2.4])
  r=takeoff_score(h,w,vz)
  self.assertEqual(float(r[0]),0.);self.assertEqual(float(r[1]),0.)
  self.assertGreater(float(r[3]),float(r[2]));self.assertTrue(bool((r<=1).all()))
 def test_partial_landing_feedback_does_not_grant_success(self):
  self.assertAlmostEqual(float(landing_progress(torch.tensor([math.radians(108.)]))),.3,places=6)
  s,_,_=self.trajectory();self.assertFalse(bool(s.qualified_landing));self.assertFalse(bool(s.success))
 def test_checked_candidate_beats_unchecked_and_status_is_honest(self):
  from .cli import select_candidate,summary,behavior_status
  c=dict(success=False,qualified_landing=False,flight_valid=True,launched=True,stable_s=0.,failed=False,flight_s=.3,
   peak_support_bodyweights=10.,flight_rise_m=.2,airborne_angle_deg=120.,recovery_score=.5)
  checked=summary(dict(cases=[dict(c,physics_consistent=True)]))
  unknown=summary(dict(cases=[dict(c,physics_consistent=None,airborne_angle_deg=360.)]))
  self.assertEqual(select_candidate([dict(label='checked',metrics=checked),dict(label='unknown',metrics=unknown)])['label'],'checked')
  self.assertEqual(behavior_status(unknown),'physics_unchecked')
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
  n=4;s=NS(valid_air_event=torch.tensor([False,True,False,True]),release_extension=torch.ones(n),landing_event=torch.tensor([False,True,True,True]),flight_valid=torch.tensor([False,True,False,True]),failed=torch.tensor([False,False,False,True]),release_inertia=torch.ones(n),release_H=torch.ones(n),release_omega=torch.ones(n),launch_vz=torch.ones(n)*2.,release_compression=torch.ones(n)*100,launched=torch.ones(n,dtype=torch.bool),landed=torch.ones(n,dtype=torch.bool),flight_rise=torch.ones(n)*.3,flight_time=torch.ones(n)*.4)
  s.air_validated=torch.tensor([False,True,False,True])
  r=flight_shaping(s,dict(locked_inertia=torch.ones(n)*.5),torch.zeros(n),.02)
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
  with tempfile.TemporaryDirectory() as td:initialize(r,Path(td));meta=json.loads((Path(td)/'bootstrap.json').read_text());self.assertFalse(meta['ground_actor_reinitialized'])
  for k,v in critic.items():torch.testing.assert_close(v,p.critic.state_dict()[k],atol=0,rtol=0)
  self.assertFalse(opt.state)
  obs=torch.randn(96,420)*.05;obs[:,-8]=0.;obs[32:,-8]=1.;obs[:,-7]=0.;obs[64:,-7]=1.
  with torch.no_grad():
   from .bootstrap import reference_actions
   source=torch.load(Path(__file__).parent/'bootstrap/actor_1201.pt',weights_only=True)
   reference=reference_actions(source,obs,C['ground_actor_initial_output_scale']);actual=p.act_inference(obs)
   torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
   for k,v in source.items():
    if k.startswith('actor.'):torch.testing.assert_close(v,p.state_dict()[k],atol=0,rtol=0)
  a=p.act(obs);dist=p.distribution
  self.assertGreaterEqual(float(dist.scale[:32,:12].min()),.29999)
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
