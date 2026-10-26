"""Joint-target crouch/push/tuck search; no prescribed root pose, velocity or impulse."""
import torch
from .control import score
DIM=26
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  duration=(.20+.10*p[:,0]).clamp(.08,.6);hold=(.04+.05*p[:,1]).clamp(0,.3)
  start=.4+duration+hold
  crouch=p[:,2:6].clamp(-1,1);push=p[:,8:12].clamp(-1,1)
  down=((t-.4)/duration).clamp(0,1)[:,None]
  ground=crouch*down
  delays=torch.stack((torch.zeros_like(t),p[:,7]*.04,p[:,6]*.08,p[:,6]*.08+p[:,7]*.04),-1)
  mix=((t[:,None]-start[:,None]-delays)/.03).clamp(0,1)
  ground=ground*(1-mix)+push*mix
  angle=(4.5+p[:,16]).clamp(2.5,6.1);width=(.4+p[:,17]).clamp(.15,1.5)
  blend=((s.angle-angle)/width).clamp(0,1)[:,None]
  air=p[:,12:16]*(1-blend)+p[:,18:22]*blend
  pose=torch.where(s.launched[:,None],air,ground)
  result=torch.zeros_like(obs[:,:16])
  for j in (1,4):result[:,j]=pose[:,0];result[:,j+1]=pose[:,1]
  for j in (7,10):result[:,j]=-pose[:,2];result[:,j+1]=-pose[:,3]
  wheel=torch.where(s.launched[:,None],p[:,24:26],p[:,22:24]*(t>=start)[:,None])
  result[:,12:14]=wheel[:,:1];result[:,14:]=wheel[:,1:]
  original=self.actor.act_inference(obs)
  return torch.where((s.landed|(t<.4))[:,None],original,result).clamp(-1,1)
