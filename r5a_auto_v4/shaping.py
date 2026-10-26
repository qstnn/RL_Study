"""结果驱动的有界奖励辅助函数：同时要求真实腾空能力与后旋进度。"""
import math
import torch
from .spec import CONTRACT as C

def flight_quality(rise,time):
 """首次有效腾空使用；返回[0,1]跳高/时间中较弱的一项，目标固定无课程。"""
 return torch.minimum((rise/C['fixed_flight_rise_target_m']).clamp(0,1),(time/C['fixed_flight_time_target_s']).clamp(0,1))

def landing_progress(angle):
 """首次有效触地后启用；平方单圈进度，前翻为零，过转衰减。"""
 return (angle/(2*math.pi)).clamp(0,1).square()*torch.exp(-((angle-2*math.pi).clamp_min(0)/.4).square())

def preparation_potential(state):
 """有接触/四轮承载/低倾角下累计的主动COM压缩势[0,1]，不含初始化沉降。
 奖励调用gamma*Phi(next)-Phi(now)，结束状态强制为0；不跳/反复蹲的折扣总收益为0。
 """
 return (state.compression_peak/C['preparation_depth_scale_m']).clamp(0,1)

def reward_potentials(state,mass):
 """每20ms差分：蹬伸、真实跳高、联合旋转、滞空、着地峰值冲击。均每回合有界。"""
 s=state
 impact=((s.impact_peak/(mass*9.81)-6.).clamp(0,20)/10.).square()
 return torch.stack((s.push_peak/4.,(s.height_peak/C['fixed_flight_rise_target_m']).clamp(0,1),s.rotation_potential,
                     (s.airtime_peak/C['fixed_flight_time_target_s']).clamp(0,1),impact),-1)

def takeoff_score(h_backward,omega_backward,vz):
 """真实起跳时的后旋H/机身转向与垂向速度共同计分；尺度不是速度命令。"""
 vertical=(vz/math.sqrt(2*9.81*C['fixed_flight_rise_target_m'])).clamp(0,1)
 return vertical*torch.tanh(h_backward.clamp_min(0)/2.)*(.5+.5*torch.tanh(omega_backward/8.))

def flight_shaping(state,metrics,rotation_delta,dt):
 """首次有效触地只付一次起跳/准备成果；紧凑项只伴随首次腾空联合进度新增。
 不放宽真正单圈、着地或站稳验收；未起跳不付成果分。
 """
 s=state;touch=s.landing_event&s.flight_valid&~s.failed
 quality=flight_quality(s.flight_rise,s.flight_time)
 compact=(1-metrics['locked_inertia']/s.release_inertia.clamp_min(.05)).clamp(0,1)
 firstair=s.launched&~s.landed&~s.failed&(s.flight_rise>=C['min_flight_rise_m'])
 return dict(takeoff_coordination=touch*takeoff_score(s.release_H,s.release_omega,s.launch_vz),
  preparation_credit=touch*(s.release_compression/C['preparation_depth_scale_m']).clamp(0,1)*quality,
  compact_rotation=rotation_delta.clamp_min(0)*compact*firstair)
