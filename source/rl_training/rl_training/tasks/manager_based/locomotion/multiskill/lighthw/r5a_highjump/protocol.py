"""真实离地高度、连续无接触时间与事件参考；可在 CPU 独立复算。"""
import torch
try:
    from ..r5a.protocol import JumpState as LegacyJumpState
except ImportError:
    from r5a.protocol import JumpState as LegacyJumpState
from . import spec


class JumpState(LegacyJumpState):
    FLOATS=LegacyJumpState.FLOATS+('release_z','release_vz','release_t','free_flight_time',
        'flight_rise','landing_t','target_height','phase','apex_t','wheel_clearance_peak','first_flight_peak_rise')
    BOOLS=LegacyJumpState.BOOLS+('released','flight_ended','height_qualified','apex_seen')

    def __init__(self,n,device='cpu'):
        super().__init__(n,device)
        self.target_height.fill_(spec.CONTRACT['target_flight_rise_m'])

    def reset(self,ids,com_z):
        super().reset(ids,com_z)
        self.target_height[ids]=spec.CONTRACT['target_flight_rise_m']

    def update(self,dt,t,com_z,com_vz,force_z,clearance,tilt,base_z,speed,drift,body_force,torque_peak):
        """保留 v2 起跳门槛；另记首段物理离地，反弹不增加真实跳高/滞空。

        激活：稳定支撑武装后 t>=.8，四轮法向力<2N、净空>0且向上速度>.25。
        返回原有效腾空掩码和稳定掩码；success 还要求真实高跳与末段稳定。
        """
        was_landed=self.landed.clone()
        prior_success=self.ever_success.clone()
        airborne,stable=super().update(dt,t,com_z,com_vz,force_z,clearance,tilt,base_z,speed,drift,body_force,torque_peak)
        clear=(force_z.abs()<2.).all(-1)&(clearance>0.).all(-1)
        release=self.armed & ~self.released & (t>=.8) & clear & (com_vz>=.25)
        self.release_z.copy_(torch.where(release,com_z,self.release_z))
        self.release_vz.copy_(torch.where(release,com_vz,self.release_vz))
        self.release_t.copy_(torch.where(release,t,self.release_t))
        self.released |= release
        active=self.released & ~self.flight_ended & clear
        self.free_flight_time += active.float()*dt
        self.flight_rise.copy_(torch.maximum(self.flight_rise,torch.where(active,com_z-self.release_z,0.)))
        self.first_flight_peak_rise.copy_(torch.maximum(self.first_flight_peak_rise,torch.where(active,com_z-self.baseline_z,0.)))
        self.wheel_clearance_peak.copy_(torch.maximum(self.wheel_clearance_peak,torch.where(active,clearance.amin(-1),0.)))
        self.flight_ended |= self.released & ~clear
        apex=self.released & ~self.apex_seen & (com_vz<=0.)
        self.apex_t.copy_(torch.where(apex,t,self.apex_t));self.apex_seen |= apex
        self.landing_t.copy_(torch.where(self.landed & ~was_landed,t,self.landing_t))
        self.height_qualified.copy_((self.flight_rise>=spec.SCREEN['min_flight_rise_m']) &
            (self.free_flight_time>=spec.SCREEN['min_free_flight_s']-1e-6) &
            (self.first_flight_peak_rise>=spec.SCREEN['min_first_flight_peak_rise_m']) &
            (self.wheel_clearance_peak>=spec.SCREEN['min_peak_wheel_clearance_m']))
        self.success &= self.height_qualified & (self.impact_peak<=spec.SCREEN['max_landing_impact_N'])
        # Parent ever_success describes old low-jump success; recompute for the high task.
        self.new_success.copy_(self.success & ~prior_success)
        self.ever_success.copy_(prior_success | self.success)
        self.phase.copy_(phase_ids(t,self).float())
        return airborne,stable

    def report(self,i):
        return {n:bool(getattr(self,n)[i]) for n in self.BOOLS if n!='new_success'} | {
            n:float(getattr(self,n)[i]) for n in self.FLOATS}


def phase_ids(t,state):
    start=spec.CONTRACT['push_start_s']+min(0.,spec.REFERENCE_PROFILES[spec.CONTRACT['reference_profile']].get('calf_delay_s',0.))
    phase=torch.where(t<start,0,1)
    phase=torch.where(state.released,2,phase)
    phase=torch.where(state.landed,3,phase)
    return torch.where(state.landed & (t-state.landing_t>=spec.CONTRACT['cushion_seconds']),4,phase)


def reference_pose(t,state=None,profiles=None):
    """1kHz 关节参考：蹬伸直到真实离地，再收腿、准备触地和缓冲恢复。"""
    n=t.numel();c=spec.CONTRACT
    if profiles is None: profiles=torch.full((n,),c['reference_profile'],device=t.device,dtype=torch.long)
    table=torch.tensor([[p['crouch_thigh'],p['crouch_calf'],p['push_duration_s'],p['extend_thigh'],p['extend_calf'],p.get('crouch_end_s',.85),p.get('tuck_fraction',0.),p.get('landing_delay_s',0.),p.get('thigh_duration_s',p['push_duration_s']),p.get('calf_duration_s',p['push_duration_s']),p.get('calf_delay_s',0.),p.get('prepare_duration_s',.08)] for p in spec.REFERENCE_PROFILES],device=t.device,dtype=t.dtype)
    params=table[profiles]
    stand=torch.tensor(spec.STAND,device=t.device,dtype=t.dtype).expand(n,-1)
    extend=torch.tensor(spec.EXTEND,device=t.device,dtype=t.dtype).expand(n,-1).clone()
    for j,sign in ((0,1),(3,1),(6,-1),(9,-1)):
        extend[:,j+1]=sign*params[:,3];extend[:,j+2]=sign*params[:,4]
    crouch=stand.clone()
    for j,sign in ((0,1),(3,1),(6,-1),(9,-1)):
        crouch[:,j+1]=sign*params[:,0];crouch[:,j+2]=sign*params[:,1]
    lerp=lambda a,b,alpha:a+(b-a)*alpha.clamp(0.,1.)[:,None]
    pose=lerp(stand,crouch,(t-.5)/(params[:,5]-.5))
    push=lerp(crouch,extend,(t-c['push_start_s'])/params[:,2])
    for j in (0,3,6,9):
        push[:,j+1]=crouch[:,j+1]+(extend[:,j+1]-crouch[:,j+1])*((t-c['push_start_s'])/params[:,8]).clamp(0.,1.)
        push[:,j+2]=crouch[:,j+2]+(extend[:,j+2]-crouch[:,j+2])*((t-c['push_start_s']-params[:,10])/params[:,9]).clamp(0.,1.)
    pose=torch.where((t>=c['push_start_s'])[:,None],push,pose)
    early=(params[:,10]<0.) & (t>=c['push_start_s']+params[:,10])
    for j in (0,3,6,9):pose[:,j+2]=torch.where(early,push[:,j+2],pose[:,j+2])
    if state is not None:
        age=t-state.release_t
        tuck_target=stand+params[:,6,None]*(crouch-stand)
        tucked=lerp(extend,tuck_target,age/c['tuck_transition_s'])
        prepare=lerp(tuck_target,stand,(t-state.apex_t-params[:,7])/params[:,11])
        tucked=torch.where((state.apex_seen & (t-state.apex_t>=params[:,7]))[:,None],prepare,tucked)
        pose=torch.where(state.released[:,None],tucked,pose)
        # 触地后先缓冲，随后连续恢复站姿；不在触地瞬间锁住关节。
        age=t-state.landing_t
        cushion=stand+.25*(crouch-stand)
        landing=lerp(stand,cushion,age/.08)
        landing=torch.where((age>.08)[:,None],lerp(cushion,stand,(age-.08)/.32),landing)
        pose=torch.where(state.landed[:,None],landing,pose)
    return pose
