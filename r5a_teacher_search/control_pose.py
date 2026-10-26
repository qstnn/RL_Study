"""Event-based airborne pose teacher; physical joints must reach all commanded targets."""
import torch
from .control import score
DIM=18
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  original=self.actor.act_inference(obs);ground=original.clone()
  for j in (1,4):ground[:,j]+=p[:,0];ground[:,j+1]+=p[:,1]
  for j in (7,10):ground[:,j]-=p[:,2];ground[:,j+1]-=p[:,3]
  ground[:,12:14]+=p[:,14:15];ground[:,14:]+=p[:,15:16]
  open_angle=(4.5+p[:,8]).clamp(2.5,6.1);width=(.4+p[:,9]).clamp(.15,1.5)
  blend=((s.angle-open_angle)/width).clamp(0,1)[:,None]
  pose=p[:,4:8]*(1-blend)+p[:,10:14]*blend
  air=torch.zeros_like(original)
  for j in (1,4):air[:,j]=pose[:,0];air[:,j+1]=pose[:,1]
  for j in (7,10):air[:,j]=-pose[:,2];air[:,j+1]=-pose[:,3]
  air[:,12:14]=p[:,16:17];air[:,14:]=p[:,17:18]
  result=torch.where(s.launched[:,None],air,ground)
  return torch.where((s.landed|(t<.4))[:,None],original,result).clamp(-1,1)
