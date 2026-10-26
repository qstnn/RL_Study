"""冻结回放的1kHz原始轨迹、50Hz模型输入、视频及独立NumPy复算。"""
import math
from pathlib import Path
import numpy as np
from .spec import CONTRACT as C
from .io import write
from .dynamics import centroidal,rotation
from .physics_review import additional_metrics

class Recorder:
 def __init__(self,env,folder,label,video=False):
  import torch
  self.env=env;self.folder=Path(folder);self.label=label;self.start=env._sim_step_counter;self.buffers={};self.count=0;self.policy=[];self.video=None;self.frames=0
  self.capacity=env.max_episode_length*20+1
  if video:
   import os,imageio.v2 as imageio,imageio_ffmpeg
   os.environ['IMAGEIO_FFMPEG_EXE']=imageio_ffmpeg.get_ffmpeg_exe()
   self.video=imageio.get_writer(str(self.folder/(label+'.mp4')),fps=50,codec='libx264',quality=7,macro_block_size=1)
  env.physics_observer=self.sample;env._last_capture_tick=-1;env._capture(self.start)
 def sample(self,e,tick):
  import torch
  d=e.robot.data;m=e.metrics();s=e.state;idx=tick-self.start
  values=dict(com=m['com'][0],com_velocity=m['com_vel'][0],root_state=d.root_state_w[0],
   body_com_position=d.body_com_pos_w[0],body_com_velocity=d.body_com_lin_vel_w[0],
   body_link_quaternion=d.body_link_quat_w[0],body_angular_velocity=d.body_com_ang_vel_w[0],
   contact_force=m['force'][0],wheel_force_z=m['force_z'][0],wheel_clearance=m['clearance'][0],
   body_force=m['body_force'][0],base_height=m['base_z'][0],tilt=m['tilt'][0],
   joint_position=d.joint_pos[0],joint_velocity=d.joint_vel[0],applied_torque=d.applied_torque[0],
   computed_torque=d.computed_torque[0],actions=e.actions[0],leg_targets=e.targets[0],
   online_angle=s.angle[0],online_launched=s.launched[0],online_landed=s.landed[0],online_success=s.success[0],
   online_failed=s.failed[0],online_stable_time=s.stable_time[0])
  for k,v in values.items():
   if k not in self.buffers:self.buffers[k]=torch.empty((self.capacity,*v.shape),device=e.device,dtype=v.dtype)
   self.buffers[k][idx].copy_(v)
  self.count=idx+1
  if self.video is not None and idx%20==0:self.video.append_data(e.render());self.frames+=1
 def policy_step(self,obs,action,critic):self.policy.append((obs[0].detach().clone(),action[0].detach().clone(),critic[0].detach().clone()))
 def finish(self,reason,steps):
  import torch
  self.env.physics_observer=None
  if self.video is not None:self.video.close();self.video=None
  arrays={k:v[:self.count].cpu().numpy() for k,v in self.buffers.items()};arrays['t']=np.arange(self.count)*.001
  np.savez_compressed(self.folder/(self.label+'_trace.npz'),**arrays)
  if self.policy:
   np.savez_compressed(self.folder/(self.label+'_policy.npz'),observation=torch.stack([x[0] for x in self.policy]).cpu().numpy(),raw_action=torch.stack([x[1] for x in self.policy]).cpu().numpy(),critic=torch.stack([x[2] for x in self.policy]).cpu().numpy())
  e=self.env
  meta=dict(contract=C,reason=reason,steps=steps,physics_samples=self.count,video_frames=self.frames,
   body_names=e.robot.body_names,contact_names=e.contacts.body_names,joint_names=e.robot.joint_names,
   masses=e.masses[0].cpu().tolist(),inertias_link_axes=e.inertias[0].cpu().tolist(),
   initial_contact_sample_valid=False,expected_physics_samples=steps*20+1)
  write(self.folder/(self.label+'_trace.json'),meta)
  report=judge(arrays,meta)
  write(self.folder/(self.label+'.json'),report);plot(arrays,report,self.folder/(self.label+'.png'),meta)
  if self.count!=steps*20+1:raise RuntimeError('Incomplete physics recording')
  if self.frames and self.frames!=steps+1:raise RuntimeError('Incomplete video recording')
  if not report['online_agrees']:raise RuntimeError('Independent adjudication disagrees with online state')
  return report

def judge(a,meta):
 """独立分段/积分验收；不读取在线成功标记来决定成功。"""
 if not all(np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite trace')
 t=a['t'];n=len(t);free=(np.linalg.norm(a['contact_force'],axis=-1).max(-1)<2)&(a['wheel_clearance'].min(-1)>0)
 start=None;release=None;end=n-1;count=0
 for i in range(1,n):
  if free[i] and t[i]>.4:
   if count==0 and a['com_velocity'][i,2]>.25:start=i;count=1
   elif count:count+=1
  else:count=0
  if count>=7:release=start;break
 landed=False
 if release is not None:
  for i in range(release+7,n):
   if np.linalg.norm(a['contact_force'][i],axis=-1).max()>=2:end=i;landed=True;break
 rot=rotation(a['root_state'][:,3:7]);axis=rot[0,:,1];angle=0.;axis_error=0.;flight=0.;rise=0.;launch_vz=0.;h_release=None;h_variation=None
 if release is not None:
  flight=t[end]-t[release];rise=max(0.,float(a['com'][release:end+1,2].max()-a['com'][release,2]));launch_vz=float(a['com_velocity'][release,2])
  # Matrix logarithm of each tiny relative rotation, independent of online quaternion arithmetic.
  for j in range(release+1,end if landed else end+1):
   d=rot[j]@rot[j-1].T;v=np.array([d[2,1]-d[1,2],d[0,2]-d[2,0],d[1,0]-d[0,1]])/2
   theta=np.arccos(np.clip((np.trace(d)-1)/2,-1,1));v*=theta/np.sin(theta) if theta>1e-7 else 1.
   angle-=float(v@axis)
  axis_error=float(np.arccos(np.clip(rot[release:end+1,:,1]@axis,-1,1)).max())
  _,_,h,_,_,_=centroidal(a['body_com_position'][release:end],a['body_com_velocity'][release:end],a['body_link_quaternion'][release:end],a['body_angular_velocity'][release:end],meta['masses'],meta['inertias_link_axes'])
  if len(h):h_release=float(-h[0]@axis);h_variation=float(np.ptp(h@axis))
 origin=a['com'][np.flatnonzero(t<=.4)[-1],:2]
 failed=bool(np.any((a['body_force']>5)&(t>.1)) or np.any(np.linalg.norm(a['com'][:,:2]-origin,axis=-1)>1.5))
 if release is None:
  failed |= bool(t[-1]>=3 or np.any((t>.4)&(a['tilt']>math.radians(80))))
 else:
  failed |= bool(np.any((t[:release]>.4)&(a['tilt'][:release]>math.radians(80))))
 valid=bool(landed and flight>=.1-1e-6 and rise>=.02)
 qualified=bool(valid and math.radians(350)<=angle<=math.radians(385) and axis_error<=math.radians(20) and a['tilt'][end]<=math.radians(20))
 stable=0.
 if qualified and not failed:
  for i in range(end,n):
   ok=(a['wheel_force_z'][i]>5).all() and a['tilt'][i]<math.radians(15) and np.linalg.norm(a['com_velocity'][i,:2])<.1 and abs(a['com_velocity'][i,2])<.15 and np.linalg.norm(a['com'][i,:2]-a['com'][end,:2])<.1 and abs(a['base_height'][i]-.416)<.07
   stable=stable+.001 if ok else 0.
 success=bool(qualified and not failed and stable>=1.-1e-5 and meta['reason']=='timeout' and t[-1]>=5.999)
 agrees=(bool(a['online_launched'][-1])==(release is not None) and bool(a['online_landed'][-1])==landed and bool(a['online_failed'][-1])==failed and bool(a['online_success'][-1])==success and abs(float(a['online_angle'][-1])-angle)<2e-4 and abs(float(a['online_stable_time'][-1])-stable)<1e-3)
 extra=additional_metrics(a,meta,release,end,landed)
 return dict(success=success and extra['physics_consistent'] is True,protocol_success=success,**extra,launched=release is not None,landed=landed,failed=failed,flight_valid=valid,qualified_landing=qualified,
  flight_s=float(flight),flight_rise_m=rise,airborne_angle_deg=math.degrees(angle),axis_error_deg=math.degrees(axis_error),
  stable_s=stable,launch_vz=launch_vz,backward_H_takeoff=h_release,airborne_H_pitch_peak_to_peak=h_variation,
  torque_peak=float(np.abs(a['applied_torque'][1:]).max()),duration_s=float(t[-1]),online_agrees=bool(agrees),
  samples=n,hardware_acceptance='not_evaluated')

def plot(a,r,path,meta):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(4,1,figsize=(10,10),sharex=True);t=a['t']
 ax[0].plot(t,a['com'][:,2],label='whole COM');ax[0].plot(t,a['base_height'],label='base');ax[0].set_ylabel('Height (m)')
 ax[1].plot(t,np.degrees(a['online_angle']),label='airborne backward angle');ax[1].axhline(360,color='gray');ax[1].set_ylabel('Angle (deg)')
 for j in range(4):ax[2].plot(t,a['wheel_force_z'][:,j],label=('FR','FL','RR','RL')[j])
 ax[2].set_ylabel('Contact (N)')
 for j,name in enumerate(meta['joint_names']):
  if name.startswith('FL_'):ax[3].plot(t,a['applied_torque'][:,j],label=name)
 ax[3].set_ylabel('Applied (Nm)');ax[3].set_xlabel('Time (s)')
 for x in ax:x.grid(alpha=.3);x.legend(fontsize=8)
 fig.suptitle(f"Success={r['success']} | Independent check={r['online_agrees']}");fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
