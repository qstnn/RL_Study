"""结果驱动的有界奖励辅助函数：同时要求真实腾空能力与后旋进度。"""
import math
import torch
from .spec import CONTRACT as C

def flight_quality(rise,time):
 """首次有效腾空使用；返回[0,1]跳高/时间中较弱的一项，目标固定无课程。"""
 return torch.minimum((rise/C['fixed_flight_rise_target_m']).clamp(0,1),(time/C['fixed_flight_time_target_s']).clamp(0,1))

def landing_progress(angle):
 """首次有效飞行/落地使用；返回[0,1]近整圈进度，与跳高无关。
 保留15%线性梯度，85%四次项加强最后一段；前翻零分、过转衰减，成功门不变。
 """
 p=(angle/(2*math.pi)).clamp(0,1)
 return (.15*p+.85*p.pow(4))*torch.exp(-((angle-2*math.pi).clamp_min(0)/.4).square())

def preparation_potential(state):
 """有接触/四轮承载/低倾角下累计的主动COM压缩势[0,1]，不含初始化沉降。
 奖励调用gamma*Phi(next)-Phi(now)，结束状态强制为0；不跳/反复蹲的折扣总收益为0。
 """
 return (state.compression_peak/C['preparation_depth_scale_m']).clamp(0,1)

def reward_potentials(state,mass):
 """每20ms差分：前四项有界进度、1kHz首触后支撑峰值的对数代价。"""
 s=state
 # 6倍体重内免罚；log1p连续增长，没有高冲击饱和平台。峰值差分只付新增值。
 excess=(s.impact_peak/(mass*9.81)-6.).clamp_min(0)/6.
 impact=torch.log1p(excess)
 return torch.stack((s.push_peak/4.,(s.height_peak/C['fixed_flight_rise_target_m']).clamp(0,1),s.rotation_potential,
                     (s.airtime_peak/C['fixed_flight_time_target_s']).clamp(0,1),impact),-1)

def takeoff_score(h_backward,omega_backward,vz):
 """地面蹬伸/释放时启用；返回[0,1]垂速与后旋H的几何均值及机身转向分。
 H尺度2.6来自v3释放量级，不是理论所需角动量；没有外加力矩/速度。
 """
 vertical=(vz/math.sqrt(2*9.81*C['fixed_flight_rise_target_m'])).clamp(0,1)
 # 平滑有界且不在旧v3量级2.6硬截顶；尺度不是空翻所需角动量。
 positive=h_backward.clamp_min(0)
 angular=positive/(positive+C['takeoff_H_scale'])
 return torch.sqrt(vertical*angular)*(.5+.5*torch.tanh(omega_backward/C['takeoff_omega_scale']))

def flight_shaping(state,metrics,compact_delta,dt):
 """首次腾空满足.1s/.02m时即付一次起跳/蹬伸成果，无需等到落地。
 准备成果要求近期压缩后的带正垂速蹬伸；紧凑项来自1kHz暂存的历史信用增量。
 返回有界事件分与进度增量；不跳/短跳/重复触地不重复支付。
 """
 s=state;validated=s.valid_air_event&~s.failed
 return dict(takeoff_coordination=validated*takeoff_score(s.release_H,s.release_omega,s.launch_vz),
  preparation_credit=validated*s.release_extension.clamp(0,1),
  compact_rotation=compact_delta.clamp_min(0)*s.air_validated*~s.failed)
