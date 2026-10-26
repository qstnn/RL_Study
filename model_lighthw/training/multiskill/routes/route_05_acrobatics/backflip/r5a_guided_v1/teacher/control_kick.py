"""Preserve measured asymmetric v5 push pulses; search preparation and airborne knots."""
import json
from pathlib import Path
import torch
from .control import interpolate,score
DIM=44
REFERENCE=json.loads(Path(__file__).with_name('kick_reference.json').read_text())['actions']
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  prep=(.1+.1*p[:,0]).clamp(0,.5);start=.4+prep
  duration=(.02+.02*p[:,5:8]).clamp(.02,.12)
  stage=((t-start)[:,None]>=duration.cumsum(-1)).sum(-1)
  action=p.new_tensor(REFERENCE)[stage]+p[:,8:24]
  crouch=p.new_zeros((len(p),16));v=p[:,1:5]*((t-.4)/prep.clamp_min(.001)).clamp(0,1)[:,None]
  for j in (1,4):crouch[:,j]=v[:,0];crouch[:,j+1]=v[:,1]
  for j in (7,10):crouch[:,j]=-v[:,2];crouch[:,j+1]=-v[:,3]
  action=torch.where((t<start)[:,None],crouch,action)
  pose=interpolate(p[:,24:40].reshape(-1,4,4),s.angle,[0,1.5,3.5,5.8]);air=torch.zeros_like(action)
  for j in (1,4):air[:,j]=pose[:,0];air[:,j+1]=pose[:,1]
  for j in (7,10):air[:,j]=-pose[:,2];air[:,j+1]=-pose[:,3]
  air[:,12:]=p[:,40:44]
  action=torch.where(s.launched[:,None],air,action)
  return torch.where((s.landed|(t<.4))[:,None],self.actor.act_inference(obs),action).clamp(-1,1)
