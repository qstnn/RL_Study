"""Deterministic feedback teacher: frozen v5 actor plus bounded smooth phase residuals.

Search changes joint targets only. Simulation state and physical limits are unchanged.
"""
import math
import torch
# Four front/rear thigh/calf residuals at four ground times and four airborne angles.
DIM=36

def interpolate(values,coordinate,knots):
 knots=coordinate.new_tensor(knots)
 j=torch.searchsorted(knots,coordinate.contiguous(),right=True).clamp(1,len(knots)-1)
 w=((coordinate-knots[j-1])/(knots[j]-knots[j-1])).clamp(0,1)
 ids=torch.arange(len(coordinate),device=coordinate.device)
 return values[ids,j-1]*(1-w[:,None])+values[ids,j]*w[:,None]

class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  ground=interpolate(p[:,:16].reshape(-1,4,4),t-.4,[0,.06,.14,.30])
  air=interpolate(p[:,16:32].reshape(-1,4,4),s.angle,[0,1.8,3.8,5.8])
  value=torch.where(s.launched[:,None],air,ground)
  delta=torch.zeros_like(obs[:,:16])
  # Front/rear axes are mirrored; equal changes preserve left/right symmetry.
  for j in (1,4):delta[:,j]=value[:,0];delta[:,j+1]=value[:,1]
  for j in (7,10):delta[:,j]=-value[:,2];delta[:,j+1]=-value[:,3]
  wheels=torch.where(s.launched[:,None],p[:,34:36],p[:,32:34])
  delta[:,12:14]=wheels[:,:1];delta[:,14:]=wheels[:,1:]
  delta*=((t>=.4)&~s.landed)[:,None]
  # Latent residual and final tanh share exactly the same bounded action range as PPO.
  return torch.tanh(self.actor.latent_mean(obs)+delta)

def score(state,metrics):
 """Search objective, not an acceptance rule; only first-flight progress is counted."""
 s=state
 progress=(s.angle/(2*math.pi)).clamp(0,1)
 airborne=s.air_validated.float()
 tilt=(1+torch.cos(s.landing_tilt))*.5
 score=(2.5*progress+.30*(s.flight_time/.5).clamp(0,1)+.25*(s.flight_rise/.4).clamp(0,1)
        +.15*(s.release_H/7.).clamp(0,1)+.4*tilt*((progress-.5)*2).clamp(0,1))*airborne
 score-=.35*s.failed.float()+.15*(s.axis_error/.35).clamp(0,4)
 score-=.5*(s.angle/(2*math.pi)-1).clamp_min(0)
 score+=2*s.qualified_landing.float()+s.stable_time.clamp_max(1)+10*s.success.float()
 return score
