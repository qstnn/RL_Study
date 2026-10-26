"""Three-pulse push teacher with independently searched crouch and event-based tuck."""
import torch
from .control import score
DIM=33
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  prep=(.15+.15*p[:,0]).clamp(0,.6);start=.4+prep
  duration1=(.04+.02*p[:,5]).clamp(.02,.16);duration2=(.02+.02*p[:,6]).clamp(.02,.16)
  down=((t-.4)/prep.clamp_min(.001)).clamp(0,1)[:,None]
  ground=p[:,1:5]*down
  ground=torch.where((t>=start)[:,None],p[:,7:11],ground)
  ground=torch.where((t>=start+duration1)[:,None],p[:,11:15],ground)
  ground=torch.where((t>=start+duration1+duration2)[:,None],p[:,15:19],ground)
  blend=((s.angle-(4.5+p[:,23]).clamp(2.5,6.1))/(.4+p[:,24]).clamp(.15,1.5)).clamp(0,1)[:,None]
  air=p[:,19:23]*(1-blend)+p[:,25:29]*blend
  pose=torch.where(s.launched[:,None],air,ground)
  result=torch.zeros_like(obs[:,:16])
  for j in (1,4):result[:,j]=pose[:,0];result[:,j+1]=pose[:,1]
  for j in (7,10):result[:,j]=-pose[:,2];result[:,j+1]=-pose[:,3]
  wheels=torch.where(s.launched[:,None],p[:,31:33],p[:,29:31]*(t>=start)[:,None])
  result[:,12:14]=wheels[:,:1];result[:,14:]=wheels[:,1:]
  original=self.actor.act_inference(obs)
  return torch.where((s.landed|(t<.4))[:,None],original,result).clamp(-1,1)
