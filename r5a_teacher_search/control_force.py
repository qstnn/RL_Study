"""Search desired contact loads via joint targets under the ORIGINAL PD/limits.

Desired forces are only a control parameterization: no external force is applied.
Actual wheel contact and joint torque must be verified in the resulting trajectory.
"""
import xml.etree.ElementTree as ET
import torch
from r5a_auto_v6.protocol import rotate
from r5a_auto_v6.spec import CONTRACT as C,LEG_JOINTS,WHEELS,LEGS
from .control import score
DIM=26
class Teacher:
 def __init__(self,actor,parameters):
  self.actor=actor;self.parameters=torch.as_tensor(parameters,device=next(actor.parameters()).device,dtype=torch.float32);self.geometry=None
 def jacobian(self,env):
  if self.geometry is None:
   from scipy.spatial.transform import Rotation
   import numpy as np
   tree=ET.parse(env.cfg.scene.robot.spawn.asset_path);joints={j.get('name'):j for j in tree.findall('joint')};parents=[];origins=[];axes=[];leg=[]
   for name in LEG_JOINTS+WHEELS:
    j=joints[name];o=j.find('origin');parents.append(env.robot.body_names.index(j.find('parent').get('link')))
    origins.append(np.fromstring(o.get('xyz','0 0 0'),sep=' ').tolist());r=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' '))
    axes.append(r.apply(np.fromstring(j.find('axis').get('xyz'),sep=' ')).tolist());leg.append(LEGS.index(name[:2]))
   self.geometry=(parents,torch.tensor(origins,device=env.device),torch.tensor(axes,device=env.device),leg)
  parents,local,axis,legs=self.geometry;d=env.robot.data;n=env.num_envs
  q=d.body_link_quat_w[:,parents].reshape(-1,4)
  axes=rotate(q,axis[None].expand(n,-1,-1).reshape(-1,3)).reshape(n,16,3)
  origins=rotate(q,local[None].expand(n,-1,-1).reshape(-1,3)).reshape(n,16,3)+d.body_link_pos_w[:,parents]
  up=axes.new_tensor([0.,0.,1.]);wa=axes[:,12:];radial=up-(wa*up).sum(-1,keepdim=True)*wa;radial=radial/radial.norm(dim=-1,keepdim=True).clamp_min(.1)
  contact=d.body_link_pos_w[:,env.wheel_body_ids]-.0875*radial
  return torch.cross(axes,contact[:,legs]-origins,dim=-1),legs
 def __call__(self,obs,env):
  p=self.parameters;s=env.state;t=env.episode_physics_steps*env.physics_dt
  duration=(.2+.1*p[:,0]).clamp(.08,.5);start=.4+duration;switch=start+(.08+.04*p[:,5]).clamp(.02,.2)
  early=torch.stack(((120+80*p[:,6]).clamp(0,400),(120+80*p[:,7]).clamp(0,400),60*p[:,8],60*p[:,9]),-1)
  late=torch.stack(((180+80*p[:,10]).clamp(0,400),(60+80*p[:,11]).clamp(0,400),60*p[:,12],60*p[:,13]),-1)
  load=torch.where((t<switch)[:,None],early,late)
  force=p.new_zeros((len(p),4,3));force[:,:2,2]=load[:,0:1];force[:,2:,2]=load[:,1:2]
  force[:,:2,0]=load[:,2:3];force[:,2:,0]=load[:,3:4]
  force[:,:,0]=force[:,:,0].clamp(-force[:,:,2],force[:,:,2])
  jac,legs=self.jacobian(env);tau=-(jac*force[:,legs]).sum(-1)
  q=env.robot.data.joint_pos[:,env.leg_ids];v=env.robot.data.joint_vel[:,env.leg_ids]
  target=q+(tau[:,:12]+3*v)/80
  neutral=p.new_tensor(C['stand']);lo=p.new_tensor(C['action_lower']);hi=p.new_tensor(C['action_upper'])
  action=(target-neutral)/torch.where(target>=neutral,hi-neutral,neutral-lo)
  result=p.new_zeros((len(p),16));result[:,:12]=action
  result[:,12:]=(env.robot.data.joint_vel[:,env.wheel_ids]+tau[:,12:])/37.7
  down=((t-.4)/duration).clamp(0,1)[:,None];crouch=p[:,1:5]*down
  blend=((s.angle-(4.5+p[:,18]).clamp(2.5,6.1))/(.4+p[:,19]).clamp(.15,1.5)).clamp(0,1)[:,None]
  pose=torch.where(s.launched[:,None],p[:,14:18]*(1-blend)+p[:,20:24]*blend,crouch)
  reference=p.new_zeros((len(p),16))
  for j in (1,4):reference[:,j]=pose[:,0];reference[:,j+1]=pose[:,1]
  for j in (7,10):reference[:,j]=-pose[:,2];reference[:,j+1]=-pose[:,3]
  reference[:,12:14]=p[:,24:25]*s.launched[:,None];reference[:,14:]=p[:,25:26]*s.launched[:,None]
  result=torch.where(((t<start)|s.launched)[:,None],reference,result)
  return torch.where((s.landed|(t<.4))[:,None],self.actor.act_inference(obs),result).clamp(-1,1)
