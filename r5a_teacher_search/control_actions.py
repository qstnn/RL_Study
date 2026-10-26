"""Direct bounded-action residuals allow changes even when the frozen actor saturates."""
import torch
from .control import interpolate,score
DIM=36
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  ground=interpolate(p[:,:16].reshape(-1,4,4),t-.4,[0,.06,.14,.30])
  air=interpolate(p[:,16:32].reshape(-1,4,4),s.angle,[0,1.8,3.8,5.8])
  value=torch.where(s.launched[:,None],air,ground)
  delta=torch.zeros_like(obs[:,:16])
  for j in (1,4):delta[:,j]=value[:,0];delta[:,j+1]=value[:,1]
  for j in (7,10):delta[:,j]=-value[:,2];delta[:,j+1]=-value[:,3]
  wheels=torch.where(s.launched[:,None],p[:,34:36],p[:,32:34]);delta[:,12:14]=wheels[:,:1];delta[:,14:]=wheels[:,1:]
  delta*=((t>=.4)&~s.landed)[:,None]
  return (self.actor.act_inference(obs)+delta).clamp(-1,1)
