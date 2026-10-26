"""首次腾空后的恢复辅助信号；无起跳站立、地面翻滚和二次跳跃不领取恢复收益。"""
import math
import torch

def recovery_terms(state,metrics,wheel_velocity,dt):
 """有效首次腾空并着地后启用，返回逐环境未加权奖励。硬成功判据仍在protocol中。
 角度平滑门允许未完美落地的策略获得纠正信号；各分量相加，避免四轮AND使梯度消失。
 """
 s=state;m=metrics
 gate=(s.flight_valid&s.landed&~s.failed).float()*torch.exp(-((s.angle-2*math.pi)/1.0).square())
 upright=torch.exp(-(m['tilt']/.35).square())
 height=torch.exp(-((m['base_z']-.416)/.10).square())
 load=(m['force_z'].clamp_min(0)/10.).clamp_max(1.)
 support=.5*load.mean(-1)+.5*load.min(-1).values
 quiet=torch.exp(-(m['com_vel'].norm(dim=-1)/.20).square())
 drift=torch.exp(-((m['com'][:,:2]-s.landing_xy).norm(dim=-1)/.15).square())
 return dict(recovery_upright=dt*gate*upright,recovery_height=dt*gate*height,
  recovery_support=dt*gate*support,recovery_quiet=dt*gate*quiet,recovery_drift=dt*gate*drift,
  landing_wheel_spin=dt*gate*(wheel_velocity/37.7).square().mean(-1))
