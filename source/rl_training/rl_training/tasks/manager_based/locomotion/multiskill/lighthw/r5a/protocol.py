"""起跳与落地判定。训练与原始物理轨迹复核共用；不依赖 Isaac Sim。"""
import math
import torch
from .spec import SCREEN, STAND, CROUCH, EXTEND, CONTRACT


def reference_pose(t):
    """按回合秒数返回 12 关节参考角；只给关节目标，不改根状态或施加外力。"""
    poses = [STAND, STAND, CROUCH, EXTEND, EXTEND, STAND, STAND]
    times = CONTRACT['reference_times_s']
    out = torch.tensor(STAND, device=t.device, dtype=t.dtype).expand(t.numel(), -1).clone()
    for start, end, left, right in zip(times[:-1], times[1:], poses[:-1], poses[1:]):
        alpha = ((t-start)/(end-start)).clamp(0.,1.)
        # 连续位置参考；端点取精确值，速度变化由实际 PD 与驱动限幅承担。
        a = torch.tensor(left, device=t.device, dtype=t.dtype)
        b = torch.tensor(right, device=t.device, dtype=t.dtype)
        out = torch.where(((t>=start)&(t<=end))[:,None], a+alpha[:,None]*(b-a), out)
    return out


class JumpState:
    """有状态的单次起跳判据；出生落下、原地站立或重置均不能算成功。"""
    FLOATS = ('support_time','baseline_sum','baseline_count','baseline_z','flight_time','takeoff_vz',
              'takeoff_t','peak_rise','stable_time','impact_peak','torque_peak','max_tilt','stable_peak')
    BOOLS = ('armed','launched','landed','failed','success','new_success','ever_success')

    def __init__(self, n, device='cpu'):
        self.device = device
        for name in self.FLOATS:
            setattr(self,name,torch.zeros(n,device=device))
        for name in self.BOOLS:
            setattr(self,name,torch.zeros(n,device=device,dtype=torch.bool))

    def reset(self, ids, com_z):
        for name in self.FLOATS+self.BOOLS:
            getattr(self,name)[ids] = 0
        self.baseline_z[ids] = com_z

    def update(self, dt, t, com_z, com_vz, force_z, clearance, tilt, base_z, speed, drift, body_force, torque_peak):
        """每次采样更新一次。接触为轮法向力；高度/速度必须使用整机质量加权 COM。"""
        s = SCREEN
        support = (force_z>s['support_force_N']).all(-1)
        settling = (t>=.20)&(t<=.50)
        self.baseline_sum += torch.where(settling,com_z,0.)
        self.baseline_count += settling.float()
        self.baseline_z.copy_(torch.where(settling,self.baseline_sum/self.baseline_count.clamp_min(1.),self.baseline_z))
        prelaunch = t<.8
        self.support_time.copy_(torch.where(prelaunch & support,self.support_time+dt,torch.where(prelaunch,0.,self.support_time)))
        self.armed |= self.support_time >= .20-1e-6
        clear = (force_z.abs()<s['airborne_force_N']).all(-1)&(clearance>s['min_wheel_clearance_m']).all(-1)
        takeoff = self.armed & ~self.launched & (t>=.8) & clear & (com_vz>=s['min_takeoff_vz'])
        self.takeoff_vz.copy_(torch.where(takeoff,com_vz,self.takeoff_vz))
        self.takeoff_t.copy_(torch.where(takeoff,t,self.takeoff_t))
        self.launched |= takeoff
        airborne = self.launched & ~self.landed & clear
        self.flight_time += airborne.float()*dt
        self.peak_rise.copy_(torch.maximum(self.peak_rise,torch.where(self.launched,com_z-self.baseline_z,0.)))
        touchdown = self.launched & (t>self.takeoff_t+.03) & (force_z>s['support_force_N']).any(-1)
        self.landed |= touchdown
        self.impact_peak.copy_(torch.maximum(self.impact_peak,torch.where(self.landed,force_z.clamp_min(0).sum(-1),0.)))
        self.torque_peak.copy_(torch.maximum(self.torque_peak,torque_peak))
        self.max_tilt.copy_(torch.maximum(self.max_tilt,tilt))
        stable = (self.landed & support & (tilt<math.radians(s['max_stable_tilt_deg']))
                  & ((base_z-s['base_height_m']).abs()<s['base_height_tolerance_m'])
                  & (speed<s['max_stable_speed_m_s']) & (com_vz.abs()<s['max_stable_vertical_speed_m_s'])
                  & (drift<s['max_landing_drift_m']))
        self.stable_time.copy_(torch.where(stable,self.stable_time+dt,0.))
        self.stable_peak.copy_(torch.maximum(self.stable_peak,self.stable_time))
        invalid = ((body_force>s['illegal_contact_N']) | (tilt>math.radians(s['fail_tilt_deg']))
                   | (base_z<s['fail_base_height_m']) | (drift>s['fail_drift_m'])
                   | ((t>=s['no_takeoff_deadline_s']) & ~self.launched))
        self.failed |= invalid
        qualified = ((self.peak_rise>=s['min_com_rise_m']) & (self.flight_time>=s['min_air_seconds']-1e-6)
                     & (self.stable_time>=s['min_stable_seconds']-1e-6) & ~self.failed)
        # success 表示当前连续保持达标；ever_success 仅保留历史，不能替代终末验收。
        self.new_success.copy_(qualified & ~self.ever_success)
        self.ever_success |= qualified
        self.success.copy_(qualified & ~self.failed)
        return airborne, stable

    def critic(self):
        return torch.stack((self.armed,self.launched,self.landed,self.flight_time,self.peak_rise,self.stable_time),dim=-1).float()

    def report(self, i):
        return {name:bool(getattr(self,name)[i]) for name in self.BOOLS if name!='new_success'} | {
            name:float(getattr(self,name)[i]) for name in ('baseline_z','flight_time','takeoff_vz','takeoff_t','peak_rise',
                                                          'stable_time','impact_peak','torque_peak','max_tilt','stable_peak')}
