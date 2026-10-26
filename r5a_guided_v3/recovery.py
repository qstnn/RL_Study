"""首次有效腾空后的承载/制动/低漂移反馈；严守真实单圈与终末站稳门。"""
import torch
from .shaping import landing_progress,flight_quality
from .spec import CONTRACT as C

def recovery_terms(state,metrics,wheel_velocity,dt):
 """首次有效落地未失败时启用；返回按dt/6归一的各项，正项全回合各不超过1。
 组合项同时要求姿态、高度、四轮承载、低速及低漂移，避免仅抬正机身就获得满分。
 """
 s=state;m=metrics;gate=(s.flight_valid&s.landed&~s.failed).float()*landing_progress(s.angle)
 upright=torch.exp(-(m['tilt']/.35).square());height=torch.exp(-((m['base_z']-.416)/.10).square())
 load=(m['force_z'].clamp_min(0)/10.).clamp_max(1.);minimum=load.min(-1).values
 support=.4*load.mean(-1)+.6*minimum
 quiet=torch.exp(-(m['com_vel'].norm(dim=-1)/.15).square())
 drift=torch.exp(-((m['com'][:,:2]-s.landing_xy).norm(dim=-1)/.10).square());scale=dt/C['episode_seconds']
 return dict(recovery_upright=scale*gate*upright,recovery_height=scale*gate*height,recovery_support=scale*gate*support,
  recovery_quiet=scale*gate*quiet,recovery_drift=scale*gate*drift,recovery_combined=scale*gate*upright*height*minimum*quiet*drift,
  landing_wheel_spin=scale*gate*(wheel_velocity/37.7).square().mean(-1))
