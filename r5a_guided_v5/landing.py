"""触地前缓冲代价和站稳门诊断；不更改任何动力学或成功判据。"""
import math
import torch
from .spec import CONTRACT as C

FLOATS=('prelanding_speed_cost','touchdown_spin_cost','landing_speed_cost','anchor_position_cost','anchor_return_cost','qualified_postlanding_s',
        'gate_miss_support_s','gate_miss_tilt_s','gate_miss_xy_speed_s','gate_miss_z_speed_s',
        'gate_miss_drift_s','gate_miss_height_s','terminal_tilt_deg','terminal_drift_m',
        'terminal_xy_speed','terminal_z_speed','terminal_base_height')
BOOLS=tuple('terminal_gate_'+k for k in ('support','tilt','xy_speed','z_speed','drift','height'))

def prelanding_gate(s,m):
 """首次离地后、尚未接触、已后旋270°且COM下降时启用；返回布尔门。
 不按固定时钟提前展开，不在起跳/上升/着地后抑制正常发力。
 """
 return s.launched&~s.landed&~s.failed&(s.angle>=math.radians(C['prelanding_angle_deg']))&(m['com_vel'][:,2]<-.15)

def observation_prelanding(obs):
 """与物理门同义的策略观测解析；只用于分阶段采样，不增加观测或教师动作。"""
 return (obs[:,-8]>.5)&(obs[:,-7]<.5)&(obs[:,-11]>=C['prelanding_angle_deg']/360.)&(obs[:,-28]<-.15*.25)

def stable_gates(s,m):
 """逐样本复用原有六项站稳门；仅反馈/诊断，不能放宽验收。"""
 return dict(support=(m['force_z']>5.).all(-1),tilt=m['tilt']<math.radians(C['stable_tilt_deg']),
  xy_speed=m['com_vel'][:,:2].norm(dim=-1)<C['stable_speed_m_s'],
  z_speed=m['com_vel'][:,2].abs()<C['stable_vertical_speed_m_s'],
  drift=(m['com'][:,:2]-s.landing_xy).norm(dim=-1)<C['stable_landing_drift_m'],
  height=(m['base_z']-C['standing_base_z']).abs()<C['standing_height_tolerance_m'])

def anchor_cost_rates(s,m,t):
 """首触后0.25秒且首次落地合格/未失败时启用，返回位置与回返速度非负代价率。
 锚点始终使用第一次接触COM位置，弹跳不重置。参考回返速度只用于打分，不向
 仿真注入速度或力。x²/(1+x²)使两项分别在[0,1)内且随误差严格递增，避免
 远处失稳用无界累计负分驱使策略提前失败。外部按物理dt/6积分。
 """
 active=s.qualified_landing&~s.failed&((t-s.landing_t)>=C['recovery_catch_grace_s'])
 error=m['com'][:,:2]-s.landing_xy;distance=error.norm(dim=-1)
 target=-C['recovery_return_gain_s_inv']*error
 target*= (C['recovery_return_speed_m_s']/target.norm(dim=-1).clamp_min(1e-8)).clamp_max(1)[:,None]
 pos=(distance/C['recovery_anchor_scale_m']).square()
 vel=((m['com_vel'][:,:2]-target).norm(dim=-1)/C['recovery_return_speed_m_s']).square()
 return active*pos/(1+pos),active*vel/(1+vel)

def advance(s,m,dt,t,landed_now):
 """1kHz积分实际关节速度代价、一次首触角速度代价及各门失效秒数。
 速度平方归一于原腿限速后最多4；仅下降末段/首触后0.25秒积分，防止抑制起跳。
 首触角速度为一次事件，最多9；增加到峰值代价，不按20ms漏掉接触事件。
 缺少扩展机械量时只供旧CPU判定测试使用零值；真实环境总是提供实测值。
 """
 speed=m['leg_velocity'] if 'leg_velocity' in m else s.angle.new_zeros((len(t),12))
 angular=m['root_angular_velocity'] if 'root_angular_velocity' in m else s.angle.new_zeros((len(t),3))
 cost=(speed/C['leg_speed_rad_s']).square().mean(-1).clamp_max(4.)
 s.prelanding_speed_cost.add_(dt*prelanding_gate(s,m)*cost)
 s.touchdown_spin_cost.add_(landed_now*(angular.norm(dim=-1)/8.).square().clamp_max(9.))
 catching=s.landed&~s.failed&((t-s.landing_t)<=.25)
 s.landing_speed_cost.add_(dt*catching*cost)
 position,returning=anchor_cost_rates(s,m,t)
 s.anchor_position_cost.add_(dt/C['episode_seconds']*position)
 s.anchor_return_cost.add_(dt/C['episode_seconds']*returning)
 active=s.qualified_landing&~s.failed
 s.qualified_postlanding_s.add_(dt*active)
 for k,ok in stable_gates(s,m).items():
  getattr(s,'gate_miss_'+k+'_s').add_(dt*active*~ok)
  getattr(s,'terminal_gate_'+k).copy_(ok)
 s.terminal_tilt_deg.copy_(m['tilt']*180/math.pi)
 s.terminal_drift_m.copy_((m['com'][:,:2]-s.landing_xy).norm(dim=-1))
 s.terminal_xy_speed.copy_(m['com_vel'][:,:2].norm(dim=-1))
 s.terminal_z_speed.copy_(m['com_vel'][:,2].abs())
 s.terminal_base_height.copy_(m['base_z'])

def costs(s):
 return torch.stack((s.prelanding_speed_cost,s.touchdown_spin_cost,s.landing_speed_cost,s.anchor_position_cost,s.anchor_return_cost),-1)

def terminal_quality(s,m):
 """只有原完整成功才付一次终末姿态裕量奖；平正、站高、低速、低漂移同时更优。
 返回[0,1]，不改成功门、不用姿态奖励替代完整后空翻。
 """
 upright=torch.exp(-(m['tilt']/.15).square())
 height=torch.exp(-((m['base_z']-C['standing_base_z'])/.05).square())
 quiet=torch.exp(-(m['com_vel'].norm(dim=-1)/.08).square())
 drift=torch.exp(-((m['com'][:,:2]-s.landing_xy).norm(dim=-1)/.08).square())
 return s.success*upright*height*quiet*drift
