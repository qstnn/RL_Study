"""由原始轨迹独立计算缓冲/恢复统计；不参与成功判定。"""
import numpy as np
from .spec import CONTRACT as C

def metrics(a,meta,end,landed):
 if not landed:return dict(landed=False)
 t=a['t'];post=np.arange(len(t))>=end
 drift=np.linalg.norm(a['com'][:,:2]-a['com'][end,:2],axis=-1)
 gates=dict(support=(a['wheel_force_z']>5).all(-1),tilt=a['tilt']<np.radians(C['stable_tilt_deg']),
  xy_speed=np.linalg.norm(a['com_velocity'][:,:2],axis=-1)<C['stable_speed_m_s'],
  z_speed=abs(a['com_velocity'][:,2])<C['stable_vertical_speed_m_s'],drift=drift<C['stable_landing_drift_m'],
  height=abs(a['base_height']-C['standing_base_z'])<C['standing_height_tolerance_m'])
 force=a['wheel_force_z'].clip(0).sum(-1);peak=end+int(force[end:].argmax());bw=sum(meta['masses'])*9.81
 return dict(landed=True,first_contact_s=float(t[end]),post_landing_peak_s=float(t[peak]),post_landing_peak_bodyweights=float(force[peak]/bw),
  touchdown_root_angular_speed_rad_s=float(np.linalg.norm(a['root_state'][end,10:13])),
  gate_miss_postcontact_seconds={k:float(np.count_nonzero(post&~v)*C['physics_dt']) for k,v in gates.items()},
  terminal_gate_pass={k:bool(v[-1]) for k,v in gates.items()},
  gate_scope='all recorded samples after first contact; training counters additionally require qualified landing and no failure')
