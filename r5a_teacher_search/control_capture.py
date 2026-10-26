"""Measured kick and airborne pose knots followed by bounded landing capture.

All outputs use the original position/velocity action interface. No external
forces, reset-state changes, or relaxed acceptance are used.
"""
import torch
from .control_kick import Teacher as KickTeacher
from .control import score
DIM=54

class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor
  self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32)
  self.kick=KickTeacher(actor,self.parameters[:,:44])
 def __call__(self,obs,env):
  action=self.kick(obs,env);p=self.parameters;s=env.state;d=env.robot.data
  t=env.episode_physics_steps*env.physics_dt
  duration=(.12+.12*p[:,48]).clamp(.04,.6)
  blend=(1-(t-s.landing_t)/duration).clamp(0,1)
  pose=p[:,44:48]*blend[:,None]
  capture=torch.zeros_like(action)
  for j in (1,4):capture[:,j]=pose[:,0];capture[:,j+1]=pose[:,1]
  for j in (7,10):capture[:,j]=-pose[:,2];capture[:,j+1]=-pose[:,3]
  # Signed gains are searched. Feedback uses physical tilt/rate and COM drift,
  # not a pose overwrite. Wheel commands remain bounded by the original limits.
  pitch=p[:,51]*d.projected_gravity_b[:,0]+p[:,52]*d.root_ang_vel_b[:,1]*.1
  capture[:,[1,4,7,10]]+=pitch[:,None]
  capture[:,[0,3,6,9]]=p[:,53,None]*d.projected_gravity_b[:,1,None]
  m=env.current_metrics
  velocity=p[:,49]*m['com_vel'][:,0]+p[:,50]*(m['com'][:,0]-s.landing_xy[:,0])*5
  capture[:,12:]=velocity[:,None]/(.0875*37.7)
  return torch.where(s.landed[:,None],capture,action).clamp(-1,1)
