"""结果驱动的有界奖励辅助函数：同时要求真实腾空能力与后旋进度。"""
import math
import torch
from .spec import CONTRACT as C

def flight_quality(rise,time):
 """首次有效腾空使用；返回[0,1]跳高/时间中较弱的一项，目标固定无课程。"""
 return torch.minimum((rise/C['fixed_flight_rise_target_m']).clamp(0,1),(time/C['fixed_flight_time_target_s']).clamp(0,1))

def landing_progress(angle):
 """有效腾空触地后的线性单圈学习进度；前翻为零、过转衰减，成功门保持独立。"""
 return (angle/(2*math.pi)).clamp(0,1)*torch.exp(-((angle-2*math.pi).clamp_min(0)/.4).square())

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
 """地面蹬伸/释放时启用；返回[0,1]垂速与后旋H的几何均值及机身转向分。
 H尺度2.6来自v3释放量级，不是理论所需角动量；没有外加力矩/速度。
 """
 vertical=(vz/math.sqrt(2*9.81*C['fixed_flight_rise_target_m'])).clamp(0,1)
 angular=(h_backward/C['takeoff_H_scale']).clamp(0,1)
 return torch.sqrt(vertical*angular)*(.5+.5*torch.tanh(omega_backward/C['takeoff_omega_scale']))

def flight_shaping(state,metrics,rotation_delta,dt):
 """首次腾空满足.1s/.02m时即付一次起跳/蹬伸成果，无需等到落地。
 准备成果要求近期压缩后的带正垂速蹬伸；紧凑项仅伴随首次空中新增转角。
 返回有界事件分与进度增量；不跳/短跳/重复触地不重复支付。
 """
 s=state;validated=s.valid_air_event&~s.failed
 compact=(1-metrics['locked_inertia']/s.release_inertia.clamp_min(.05)).clamp(0,1)
 firstair=s.launched&~s.landed&~s.failed&(s.flight_rise>=C['min_flight_rise_m'])
 return dict(takeoff_coordination=validated*takeoff_score(s.release_H,s.release_omega,s.launch_vz),
  preparation_credit=validated*s.release_extension.clamp(0,1),
  compact_rotation=rotation_delta.clamp_min(0)*compact*firstair)
