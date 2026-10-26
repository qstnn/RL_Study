"""Search explicit pre-contact leg opening around a measured complete-turn candidate."""
import json
from pathlib import Path
import torch
from .control_kick import Teacher as KickTeacher
from .control import score
DIM=37
REFERENCE=json.loads(Path(__file__).with_name('open_reference.json').read_text())['parameters'][:44]

class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
  q=self.parameters.new_tensor(REFERENCE).expand(len(self.parameters),-1).clone()
  q[:,:24]+=self.parameters[:,:24]
  self.kick=KickTeacher(actor,q)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;d=env.robot.data
  a=self.kick(obs,env)
  start=(4.4+p[:,24]).clamp(2.5,6.1);width=(.8+.5*p[:,25]).clamp(.2,1.5)
  blend=((s.angle-start)/width).clamp(0,1)
  landing=torch.zeros_like(a)
  pose=p[:,26:30]
  for j in (1,4):landing[:,j]=pose[:,0];landing[:,j+1]=pose[:,1]
  for j in (7,10):landing[:,j]=-pose[:,2];landing[:,j+1]=-pose[:,3]
  a=torch.where((s.launched&~s.landed)[:,None],a*(1-blend[:,None])+landing*blend[:,None],a)
  t=env.episode_physics_steps*env.physics_dt
  hold=(.15+.15*p[:,30]).clamp(.04,.6)
  capture=landing*(1-(t-s.landing_t)/hold).clamp(0,1)[:,None]
  pitch=p[:,31]*d.projected_gravity_b[:,0]+p[:,32]*d.root_ang_vel_b[:,1]*.1
  capture[:,[1,4,7,10]]+=pitch[:,None]
  capture[:,[0,3,6,9]]=p[:,33,None]*d.projected_gravity_b[:,1,None]
  m=env.current_metrics
  speed=p[:,34]*m['com_vel'][:,0]+p[:,35]*(m['com'][:,0]-s.landing_xy[:,0])*5
  capture[:,12:]=speed[:,None]/(.0875*37.7)
  a=torch.where(s.landed[:,None],capture,a)
  # Additional bounded airborne wheel speed can trade internal spin for pose.
  a[:,12:]+=p[:,36,None]*(s.launched&~s.landed)[:,None]
  return a.clamp(-1,1)
