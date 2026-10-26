"""自主起跳/后旋辅助量。只评价动作结果，不指定蹲深、时序或根部速度。"""
import math
import torch
from .spec import CONTRACT as C

def landing_progress(angle):
 """首次有效腾空结束时启用；返回[0,1]连续单圈进度，过转衰减，地面滚动不计入。"""
 return (angle/(2*math.pi)).clamp(0,1).square()*torch.exp(-((angle-2*math.pi).clamp_min(0)/.4).square())

def takeoff_score(h_backward,omega_backward):
 """真实起跳时的整机H与机身转向协同分数[0,1]；正H但机身前旋时降低收益。
 2 kg m²/s和8 rad/s仅为平滑饱和尺度，不是规定速度，也不会施加任何力/速度。
 """
 return torch.tanh(h_backward.clamp_min(0)/2.)*(.5+.5*torch.tanh(omega_backward/8.))

def flight_shaping(state,metrics,rotation_delta,dt):
 """有效首次腾空触地时结算起跳/预蹲；空中仅随新增后旋进度奖励惯量收缩。
 无真实腾空、地面摆腿、反复蹲起或落地后再跳均不能反复获取辅助奖励。
 """
 s=state;touch=s.landing_event&s.flight_valid&~s.failed
 compact=(1-metrics['locked_inertia']/s.release_inertia.clamp_min(.05)).clamp(0,1)
 flight=s.launched&~s.landed&~s.failed&(s.flight_rise>=C['min_flight_rise_m'])
 return dict(takeoff_coordination=touch*takeoff_score(s.release_H,s.release_omega),
  preparation_credit=touch*(s.release_compression/.08).clamp(0,1),
  compact_rotation=rotation_delta.clamp_min(0)*compact*flight)
