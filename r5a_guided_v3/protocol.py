"""1kHz 单次真实腾空的事件状态；CPU/GPU 共用，可用解析反例核查。"""
import math
import torch
from .spec import CONTRACT as C

def quat_mul(a,b):
 aw,av=a[:,:1],a[:,1:];bw,bv=b[:,:1],b[:,1:]
 return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)),dim=-1)

def quat_conj(q):return torch.cat((q[:,:1],-q[:,1:]),dim=-1)

def rotate(q,v):
 t=2*torch.cross(q[:,1:],v,dim=-1)
 return v+q[:,:1]*t+torch.cross(q[:,1:],t,dim=-1)

def delta_angle(q,previous,axis):
 # 最短四元数增量避免 180° 欧拉分支；投影到起始横轴，负方向为后翻。
 dq=quat_mul(q,quat_conj(previous));dq=torch.where((dq[:,:1]<0),-dq,dq)
 norm=dq[:,1:].norm(dim=-1)
 vector=dq[:,1:]*(2*torch.atan2(norm,dq[:,0].clamp_min(0.))/norm.clamp_min(1e-8))[:,None]
 return -(vector*axis).sum(-1)

def targets(actions):
 """策略绝对目标：0 对应站姿，±1 到达上下界；没有固定蹲起时序。"""
 neutral=actions.new_tensor(C['stand']);lo=actions.new_tensor(C['action_lower']);hi=actions.new_tensor(C['action_upper'])
 a=actions[:,:12].clamp(-1,1)
 return neutral+a*torch.where(a>=0,hi-neutral,neutral-lo)

class FlipState:
 FLOATS=('baseline_z','release_t','release_z','flight_time','flight_rise','angle','axis_error',
         'launch_vz','landing_t','landing_tilt','stable_time','push_peak','height_peak','rotation_potential',
         'airtime_peak','compression_peak','release_compression','release_H','release_omega','release_inertia','candidate_H','candidate_omega','candidate_inertia',
         'compression_t','extension_peak','launch_score_peak','air_rotation_peak','release_extension','candidate_extension','candidate_compression',
         'candidate_rotation_peak','candidate_compact_credit','raw_rotation_peak','compact_credit','compact_potential',
         'impact_peak','max_torque','candidate_t','candidate_z','candidate_angle','candidate_vz')
 BOOLS=('launched','landed','failed','flight_valid','air_validated','qualified_landing','success')
 def __init__(self,n,device='cpu'):
  for k in self.FLOATS:setattr(self,k,torch.zeros(n,device=device))
  for k in self.BOOLS:setattr(self,k,torch.zeros(n,dtype=torch.bool,device=device))
  self.count=torch.zeros(n,dtype=torch.long,device=device)
  self.stable_count=torch.zeros_like(self.count)
  self.origin=torch.zeros(n,2,device=device);self.landing_xy=self.origin.clone()
  self.axis=torch.zeros(n,3,device=device);self.axis[:,1]=1.
  self.prev_quat=torch.zeros(n,4,device=device);self.prev_quat[:,0]=1.
  self.reason=torch.zeros(n,dtype=torch.long,device=device)
  self.landing_event=torch.zeros(n,dtype=torch.bool,device=device)
  self.valid_air_event=torch.zeros(n,dtype=torch.bool,device=device)
 def reset(self,ids,z,xy,q):
  for k in self.FLOATS+self.BOOLS:getattr(self,k)[ids]=0
  self.count[ids]=0;self.stable_count[ids]=0;self.reason[ids]=0;self.landing_event[ids]=False
  self.valid_air_event[ids]=False
  self.baseline_z[ids]=z;self.origin[ids]=xy;self.landing_xy[ids]=xy
  self.prev_quat[ids]=q
  y=torch.zeros(len(ids),3,device=q.device);y[:,1]=1.
  self.axis[ids]=rotate(q,y)
 def update(self,dt,t,m,q):
  ticks=torch.round(t/dt).long()
  active=ticks>round(C['settle_seconds']/dt);ground=~self.launched
  self.baseline_z.copy_(torch.where(~active,m['com'][:,2],self.baseline_z))
  self.origin.copy_(torch.where((~active)[:,None],m['com'][:,:2],self.origin))
  delta=delta_angle(q,self.prev_quat,self.axis);self.prev_quat.copy_(q)
  free=(m['force'].norm(dim=-1).amax(-1)<C['airborne_force_N'])&(m['clearance'].amin(-1)>0.)
  start=active&ground&free&(self.count==0)&(m['com_vel'][:,2]>C['min_takeoff_vz'])
  continuing=ground&free&(self.count>0)
  self.count.copy_(torch.where(start,1,torch.where(continuing,self.count+1,0)))
  # 只在策略开放后的有接触阶段测量主动压缩；初始化沉降不计入。
  contact_ground=active&ground&~free&(m['tilt']<math.radians(C['preparation_max_tilt_deg']))&((m['force_z']>C['preparation_support_N']).all(-1))
  compression=torch.where(contact_ground,(self.baseline_z-m['com'][:,2]).clamp_min(0),0.)
  deeper=compression>self.compression_peak+1e-6
  self.compression_t.copy_(torch.where(deeper,t,self.compression_t))
  self.compression_peak.copy_(torch.maximum(self.compression_peak,compression))
  # 仅近期真实预蹲之后的上升蹬伸计分；站着、初始化沉降和过期蹲伏无成果信用。
  recent=(self.compression_peak>=C['preparation_min_depth_m'])&((t-self.compression_t)<=C['preparation_credit_window_s'])
  rebound=(m['com'][:,2]-(self.baseline_z-self.compression_peak)).clamp_min(0)
  supported=active&ground&~free&(m['force_z'].amax(-1)>C['preparation_support_N'])
  extension=(self.compression_peak/C['preparation_depth_scale_m']).clamp(0,1)*(rebound/C['preparation_depth_scale_m']).clamp(0,1)*(m['com_vel'][:,2]/math.sqrt(2*9.81*C['fixed_flight_rise_target_m'])).clamp(0,1)
  self.extension_peak.copy_(torch.maximum(self.extension_peak,torch.where(supported&recent,extension,0.)))
  from .shaping import takeoff_score
  score=takeoff_score(m.get('H_backward',torch.zeros_like(t)),m.get('omega_backward',torch.zeros_like(t)),m['com_vel'][:,2])
  self.launch_score_peak.copy_(torch.maximum(self.launch_score_peak,torch.where(supported,score,0.)))
  self.candidate_compression.copy_(torch.where(start,torch.where(recent,self.compression_peak,0.),self.candidate_compression))
  self.candidate_extension.copy_(torch.where(start,torch.where(recent,self.extension_peak,0.),self.candidate_extension))
  for target,key in (('candidate_H','H_backward'),('candidate_omega','omega_backward'),('candidate_inertia','locked_inertia')):
   value=m.get(key,torch.zeros_like(t))
   getattr(self,target).copy_(torch.where(start,value,getattr(self,target)))
  self.candidate_t.copy_(torch.where(start,t,self.candidate_t))
  self.candidate_z.copy_(torch.where(start,m['com'][:,2],self.candidate_z))
  self.candidate_vz.copy_(torch.where(start,m['com_vel'][:,2],self.candidate_vz))
  self.candidate_angle.copy_(torch.where(start,0.,torch.where(continuing,self.candidate_angle+delta,0.)))
  # 7帧离地确认前也保存早期紧凑旋转，未确认的小跳不兑现。
  candidate_progress=(self.candidate_angle/(2*math.pi)).clamp(0,1)
  candidate_delta=(candidate_progress-self.candidate_rotation_peak).clamp_min(0)
  inertia=m.get('locked_inertia',torch.ones_like(t))
  candidate_compact=((1-inertia/self.candidate_inertia.clamp_min(.05))/C['compact_inertia_reduction_scale']).clamp(0,1)
  self.candidate_compact_credit.copy_(torch.where(continuing,self.candidate_compact_credit+candidate_delta*candidate_compact,0.))
  self.candidate_rotation_peak.copy_(torch.where(continuing,torch.maximum(self.candidate_rotation_peak,candidate_progress),0.))
  takeoff=ground&(self.count>=C['release_samples'])
  for target,source in (('release_H','candidate_H'),('release_omega','candidate_omega'),('release_inertia','candidate_inertia'),('release_compression','candidate_compression'),('release_extension','candidate_extension')):
   getattr(self,target).copy_(torch.where(takeoff,getattr(self,source),getattr(self,target)))
  self.release_t.copy_(torch.where(takeoff,self.candidate_t,self.release_t))
  self.release_z.copy_(torch.where(takeoff,self.candidate_z,self.release_z))
  self.launch_vz.copy_(torch.where(takeoff,self.candidate_vz,self.launch_vz))
  self.raw_rotation_peak.copy_(torch.where(takeoff,self.candidate_rotation_peak,self.raw_rotation_peak))
  self.compact_credit.copy_(torch.where(takeoff,self.candidate_compact_credit,self.compact_credit))
  inflight=self.launched&~self.landed
  # 首次接触帧停止旋转积分，不把触地后滚动计为空中进度。
  contact=m['force'].norm(dim=-1).amax(-1)>=C['airborne_force_N']
  self.angle.copy_(torch.where(takeoff,self.candidate_angle, self.angle+delta*(inflight&~contact)))
  self.launched.logical_or_(takeoff)
  flying=self.launched&~self.landed
  self.flight_time.copy_(torch.where(flying,t-self.release_t,self.flight_time))
  self.flight_rise.copy_(torch.where(flying,torch.maximum(self.flight_rise,m['com'][:,2]-self.release_z),self.flight_rise))
  y=torch.zeros_like(self.axis);y[:,1]=1.
  axis_error=torch.acos((rotate(q,y)*self.axis).sum(-1).clamp(-1.,1.))
  self.axis_error.copy_(torch.where(flying,torch.maximum(self.axis_error,axis_error),self.axis_error))
  landed_now=flying&contact&~takeoff
  self.landing_event.logical_or_(landed_now)
  self.landing_t.copy_(torch.where(landed_now,t,self.landing_t))
  self.landing_tilt.copy_(torch.where(landed_now,m['tilt'],self.landing_tilt))
  self.landing_xy.copy_(torch.where(landed_now[:,None],m['com'][:,:2],self.landing_xy))
  valid=(self.flight_time>=C['min_flight_s']-1e-6)&(self.flight_rise>=C['min_flight_rise_m'])
  self.flight_valid.logical_or_(landed_now&valid)
  lo,hi=[math.radians(x) for x in C['rotation_range_deg']]
  qualifies=valid&(self.angle>=lo)&(self.angle<=hi)&(self.axis_error<=math.radians(C['max_axis_error_deg']))&(m['tilt']<=math.radians(C['max_landing_tilt_deg']))
  self.qualified_landing.logical_or_(landed_now&qualifies)
  self.landed.logical_or_(landed_now)
  # 只保留一次动作；落地后的再次腾空不会增加任何起跳/旋转进度。
  self.push_peak.copy_(torch.maximum(self.push_peak,torch.where(active&ground&~free,m['com_vel'][:,2].clamp(0.,4.),0.)))
  self.height_peak.copy_(torch.maximum(self.height_peak,torch.where(flying,self.flight_rise.clamp(0.,.8),0.)))
  valid_air=flying&(self.flight_time>=C['min_flight_s'])&(self.flight_rise>=C['min_flight_rise_m'])
  self.valid_air_event.logical_or_(valid_air&~self.air_validated)
  self.air_validated.logical_or_(valid_air)
  # 只对第一次无接触飞行中单调新增的转角积分，反复摆动/再次起跳不能刷分。
  compact=((1-inertia/self.release_inertia.clamp_min(.05))/C['compact_inertia_reduction_scale']).clamp(0,1)
  progress=(self.angle/(2*math.pi)).clamp(0,1)
  credit_active=flying&~contact&~self.failed
  increment=torch.where(credit_active,(progress-self.raw_rotation_peak).clamp_min(0),0.)
  self.compact_credit.add_(increment*compact)
  self.raw_rotation_peak.copy_(torch.maximum(self.raw_rotation_peak,torch.where(credit_active,progress,0.)))
  # 达到真实腾空后一次释放历史信用，随后只追加新增值；有效性判据保持原样。
  self.compact_potential.copy_(torch.maximum(self.compact_potential,torch.where(self.air_validated&~self.failed,self.compact_credit,0.)))
  self.air_rotation_peak.copy_(torch.maximum(self.air_rotation_peak,torch.where(valid_air,(self.angle/(2*math.pi)).clamp(0,1),0.)))
  self.airtime_peak.copy_(torch.maximum(self.airtime_peak,torch.where(valid_air,self.flight_time.clamp(0.,C['fixed_flight_time_target_s']),0.)))
  quality=torch.minimum((self.flight_rise/C['fixed_flight_rise_target_m']).clamp(0,1),(self.flight_time/C['fixed_flight_time_target_s']).clamp(0,1))
  from .shaping import landing_progress
  self.rotation_potential.copy_(torch.maximum(self.rotation_potential,torch.where(valid_air,landing_progress(self.angle),0.)))
  illegal=(m['body_force']>C['illegal_contact_N'])&(ticks>round(.1/dt))
  bounds=(m['com'][:,:2]-self.origin).norm(dim=-1)>1.5
  missed=(ticks>=round(C['launch_deadline_s']/dt))&~self.launched
  grounded_fall=active&~self.launched&(m['tilt']>math.radians(80))
  newfail=illegal|bounds|missed|grounded_fall
  self.reason.copy_(torch.where((self.reason==0)&newfail,torch.where(illegal,1,torch.where(bounds,2,torch.where(missed,3,4))),self.reason))
  self.failed.logical_or_(newfail)
  stable=self.qualified_landing&~self.failed&(m['tilt']<math.radians(C['stable_tilt_deg']))&((m['force_z']>5.).all(-1))
  stable &= (m['com_vel'][:,:2].norm(dim=-1)<C['stable_speed_m_s'])&(m['com_vel'][:,2].abs()<C['stable_vertical_speed_m_s'])
  stable &= ((m['com'][:,:2]-self.landing_xy).norm(dim=-1)<C['stable_landing_drift_m'])&((m['base_z']-C['standing_base_z']).abs()<C['standing_height_tolerance_m'])
  self.stable_count.copy_(torch.where(stable,self.stable_count+1,0))
  self.stable_time.copy_(self.stable_count*dt)
  self.success.copy_(stable&(self.stable_time>=C['stable_hold_s']))
  self.impact_peak.copy_(torch.maximum(self.impact_peak,torch.where(self.landed,m['force_z'].clamp_min(0).sum(-1),0.)))
  self.max_torque.copy_(torch.maximum(self.max_torque,m['torque_peak']))
 def phase(self,t):return torch.where(torch.round(t/C['physics_dt']).long()<round(C['settle_seconds']/C['physics_dt']),0,torch.where(~self.launched,1,torch.where(~self.landed,2,3)))
 def report(self,i):
  return {**{k:float(getattr(self,k)[i]) for k in self.FLOATS if not k.startswith('candidate')},
          **{k:bool(getattr(self,k)[i]) for k in self.BOOLS},'failure_code':int(self.reason[i])}
