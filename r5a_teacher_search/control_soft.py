"""Blend into landing feedback before contact to reduce a 20ms handoff discontinuity."""
import torch
from .control_split_bound import Teacher as BoundedTeacher,ACTION_LIMIT,score
DIM=40
class Teacher:
 def __init__(self,actor,parameters):
  self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
  self.base=BoundedTeacher(actor,self.parameters[:,:39]);self.actor=actor
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;d=env.robot.data;m=env.current_metrics
  a=self.base(obs,env);capture=torch.zeros_like(a);pose=p[:,26:30]
  for j in (1,4):capture[:,j]=pose[:,0];capture[:,j+1]=pose[:,1]
  for j in (7,10):capture[:,j]=-pose[:,2];capture[:,j+1]=-pose[:,3]
  pitch=p[:,31]*d.projected_gravity_b[:,0]+p[:,32]*d.root_ang_vel_b[:,1]*.1
  capture[:,[1,4,7,10]]+=pitch[:,None]
  capture[:,[0,3,6,9]]=p[:,33,None]*d.projected_gravity_b[:,1,None]
  # Before contact the position-hold reference is the current COM, not the
  # not-yet-initialized landing position. Thus only braking is applied in air.
  speed=-(1+p[:,34]).clamp(0,3)*m['com_vel'][:,0]
  capture[:,12:]=speed[:,None]/(.0875*37.7)
  threshold=(.06+.04*p[:,39]).clamp(.02,.14)
  weight=((threshold-m['clearance'].amin(-1))/threshold).clamp(0,1)*((s.angle-5.3)/.4).clamp(0,1)
  a=torch.where((s.launched&~s.landed)[:,None],a*(1-weight[:,None])+capture*weight[:,None],a)
  return a.clamp(-ACTION_LIMIT,ACTION_LIMIT)
