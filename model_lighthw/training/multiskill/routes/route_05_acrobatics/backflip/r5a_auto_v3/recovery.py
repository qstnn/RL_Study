"""有界的落地恢复信号；成功判定仍严格要求350–385°及连续站稳1s。"""
import torch
from .shaping import landing_progress
from .spec import CONTRACT as C

def recovery_terms(state,metrics,wheel_velocity,dt):
 """首次有效腾空着地后、未硬失败时启用；返回逐环境奖励率积分。
 平方进度门在约100°时仍可学习，站立/无翻转/地面滚动为零。
 除以整回合时长，使整个6s回合每个正分量累计≤1；不会靠延长恢复刷分。
 """
 s=state;m=metrics
 gate=(s.flight_valid&s.landed&~s.failed).float()*landing_progress(s.angle)
 upright=torch.exp(-(m['tilt']/.35).square())
 height=torch.exp(-((m['base_z']-.416)/.10).square())
 load=(m['force_z'].clamp_min(0)/10.).clamp_max(1.)
 support=.5*load.mean(-1)+.5*load.min(-1).values
 quiet=torch.exp(-(m['com_vel'].norm(dim=-1)/.20).square())
 drift=torch.exp(-((m['com'][:,:2]-s.landing_xy).norm(dim=-1)/.15).square())
 scale=dt/C['episode_seconds']
 return dict(recovery_upright=scale*gate*upright,recovery_height=scale*gate*height,
  recovery_support=scale*gate*support,recovery_quiet=scale*gate*quiet,recovery_drift=scale*gate*drift,
  landing_wheel_spin=scale*gate*(wheel_velocity/37.7).square().mean(-1))
