"""冻结轨迹的全机刚体角动量筛查及末一秒恢复质量，不用在线成功标记作判定。"""
import numpy as np
from .spec import CONTRACT as C
from .dynamics import centroidal

def momentum_screen(h):
 """无接触区间去掉边缘2ms；测量全向H相对首样本偏移，返回诊断而非硬件认证。"""
 h=np.asarray(h)
 if len(h)<25:return dict(momentum_checked=False,physics_consistent=None,H_vector_drift=None,H_drift_limit=None)
 interior=h[2:-2];drift=float(np.linalg.norm(interior-interior[0],axis=-1).max())
 limit=max(C['momentum_absolute_tolerance'],C['momentum_relative_tolerance']*float(np.linalg.norm(interior[0])))
 return dict(momentum_checked=True,physics_consistent=drift<=limit,H_vector_drift=drift,H_drift_limit=limit)

def additional_metrics(a,meta,release,end,landed):
 if release is None:screen=momentum_screen([])
 else:
  _,_,h,_,_,_=centroidal(a['body_com_position'][release:end],a['body_com_velocity'][release:end],
   a['body_link_quaternion'][release:end],a['body_angular_velocity'][release:end],meta['masses'],meta['inertias_link_axes'])
  screen=momentum_screen(h)
 tail=a['t']>=max(0.,a['t'][-1]-1.)
 upright=np.exp(-(a['tilt'][tail]/.35)**2)
 height=np.exp(-((a['base_height'][tail]-.416)/.10)**2)
 load=np.clip(a['wheel_force_z'][tail]/10.,0,1)
 support=.5*load.mean(-1)+.5*load.min(-1)
 quiet=np.exp(-(np.linalg.norm(a['com_velocity'][tail],axis=-1)/.20)**2)
 drift=np.linalg.norm(a['com'][tail,:2]-a['com'][end,:2],axis=-1) if landed else np.zeros(sum(tail))
 score=float(np.mean((upright+height+support+quiet+np.exp(-(drift/.15)**2))/5)) if landed else 0.
 return dict(**screen,recovery_score=score,end_base_height_m=float(a['base_height'][-1]),
  end_tilt_deg=float(np.degrees(a['tilt'][-1])),tail_mean_COM_speed_m_s=float(np.linalg.norm(a['com_velocity'][tail,:2],axis=-1).mean()),
  tail_wheel_support_fraction=(a['wheel_force_z'][tail]>5).mean(0).tolist(),tail_four_wheel_support_fraction=float(np.mean((a['wheel_force_z'][tail]>5).all(-1))),
  tail_max_landing_drift_m=float(drift.max()) if landed else None,
  peak_support_bodyweights=float(a['wheel_force_z'][1:].clip(0).sum(-1).max()/(sum(meta['masses'])*9.81)))
