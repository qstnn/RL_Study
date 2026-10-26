"""Ground 20ms targets and four independently optimized airborne pose knots."""
import torch
from .control import interpolate,score
DIM=36
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  original=self.actor.act_inference(obs);ground=original.clone()
  value=interpolate(p[:,:16].reshape(-1,4,4),t-.4,[0,.02,.04,.06])
  for j in (1,4):ground[:,j]+=value[:,0];ground[:,j+1]+=value[:,1]
  for j in (7,10):ground[:,j]-=value[:,2];ground[:,j+1]-=value[:,3]
  ground[:,12:14]+=p[:,32:33];ground[:,14:]+=p[:,33:34]
  pose=interpolate(p[:,16:32].reshape(-1,4,4),s.angle,[0,1.5,3.5,5.8])
  air=torch.zeros_like(original)
  for j in (1,4):air[:,j]=pose[:,0];air[:,j+1]=pose[:,1]
  for j in (7,10):air[:,j]=-pose[:,2];air[:,j+1]=-pose[:,3]
  air[:,12:14]=p[:,34:35];air[:,14:]=p[:,35:36]
  result=torch.where(s.launched[:,None],air,ground)
  return torch.where((s.landed|(t<.4))[:,None],original,result).clamp(-1,1)
