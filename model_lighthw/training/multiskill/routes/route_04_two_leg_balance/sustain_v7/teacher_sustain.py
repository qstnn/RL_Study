"""合法腿位置/轮速接口下的研究教师；COM/载荷为特权量，不作为Actor输入。"""
import math
import torch
from two_leg_balance_split_v4.spec import STAND, LOW, HIGH, LEG_JOINTS


def pitch_and_rate(gravity, angular_velocity):
    """目的：按投影重力同一坐标定义俯仰速度；全阶段激活，返回rad与rad/s。"""
    derivative = -torch.linalg.cross(angular_velocity, gravity, dim=-1)
    gx, gz = gravity[:, 0], gravity[:, 2]
    pitch = torch.atan2(gx, -gz)
    rate = (-gz * derivative[:, 0] + gx * derivative[:, 2]) / (gx.square() + gz.square()).clamp_min(1e-6)
    return pitch, rate


def inverse_targets(target):
    """把PD位置偏置映射回原动作；不额外施加扭矩或改变执行器参数。"""
    stand = target.new_tensor(STAND)
    span = torch.where(target >= stand, target.new_tensor(HIGH) - stand, stand - target.new_tensor(LOW))
    return ((target - stand) / span).clamp(-1., 1.)


class SustainTeacher:
    def __init__(self, rows, geometry, device):
        self.rows = rows
        self.device = device
        self.n = len(rows)
        self.p = {key: torch.tensor([r[key] for r in rows], device=device) for key in rows[0] if key != 'id'}
        self.triggered = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.trigger_t = torch.full((self.n,), -1., device=device)
        self.start = torch.zeros(self.n, 16, device=device)
        self.hold_end_action = self.start.clone()
        self.return_started = self.triggered.clone()
        self.target_pitch = float(geometry['candidates'][1]['pitch_rad'])
        torque = torch.tensor([geometry['candidates'][1]['static_torques_Nm'][n] for n in LEG_JOINTS], device=device)
        # 理想静力平衡需要q_target=q+tau/Kp；只作为有界原PD目标偏置对照，非物理注入。
        self.ff = inverse_targets(torque / 80. + torque.new_tensor(STAND))
        geo=geometry['candidates'][1]
        delta=torch.tensor(geo['com_m'],device=device)-torch.tensor(geo['wheel_centers_m'],device=device)[2:].mean(0)
        c,s=math.cos(self.target_pitch),math.sin(self.target_pitch)
        self.upright_delta_x=c*delta[0]-s*delta[2]
        self.upright_delta_z=s*delta[0]+c*delta[2]

    def __call__(self, b, task_t):
        m = b.metrics()
        p = self.p
        pitch, rate = pitch_and_rate(m['gravity'], m['ang_vel'])
        x = ((task_t - 1.) / p['ramp_s']).clamp(0., 1.)
        smooth = x.square() * (3. - 2. * x)
        initial = torch.zeros(self.n, 16, device=self.device)
        initial[:, [1, 4]] = (p['rise_front'] * smooth)[:, None]
        initial[:, [7, 10]] = (p['rise_rear'] * smooth)[:, None]
        initial[:, 14:16] = (p['rise_wheel'] * smooth)[:, None]
        fire = ~self.triggered & (task_t >= 1.) & (pitch <= torch.deg2rad(p['trigger_pitch_deg']))
        self.triggered |= fire
        self.trigger_t = torch.where(fire, task_t, self.trigger_t)
        self.start = torch.where(fire[:, None], initial, self.start)
        u = ((task_t - self.trigger_t) / p['blend_s']).clamp(0., 1.)
        blend = u.square() * (3. - 2. * u)
        final = initial.new_zeros(self.n, 16)
        final[:, [1, 4]] = p['final_front'][:, None]
        final[:, [7, 10]] = p['final_rear'][:, None]
        final[:, :12] += p['gravity_compensation'][:, None] * self.ff
        lateral = m['wheel_pos'][:, 3, :2] - m['wheel_pos'][:, 2, :2]
        forward = torch.stack((lateral[:, 1], -lateral[:, 0]), -1)
        forward /= forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rear_center = m['wheel_pos'][:, 2:, :2].mean(1)
        gap = ((m['com'][:, :2] - rear_center) * forward).sum(-1)
        velocity = (m['com_vel'][:, :2] * forward).sum(-1)
        drift = ((m['com'][:, :2] - b.state.hold_origin) * forward).sum(-1)
        drift = torch.where(task_t >= 4., drift, 0.)
        # 正COM误差意味着重心在后轮之前；后轮向前追上COM，并用速度/位置反馈制动。
        wheel = (p['line_gain'] * gap + p['velocity_gain'] * velocity + p['drift_gain'] * drift
                 + p['pitch_gain'] * (pitch - self.target_pitch) + p['pitch_rate_gain'] * rate)
        final[:, 14:16] = wheel[:, None]
        final[:, [7, 10]] += (p['leg_line_gain'] * gap.clamp(-.12, .12))[:, None]
        action = torch.where(self.triggered[:, None], self.start * (1. - blend[:, None]) + final * blend[:, None], initial)
        # 返回是明确待验的合法动作序列：平滑退回四轮目标；绝不把保持成功当完整成功。
        return_elapsed = b.physics_steps * b.physics_dt - b.state.return_time
        returning_now = b.state.return_time >= 0.
        fire_return = ~self.return_started & returning_now
        self.hold_end_action = torch.where(fire_return[:, None], action, self.hold_end_action)
        self.return_started |= fire_return
        r = (return_elapsed / 3.).clamp(0., 1.)
        return_blend = r.square() * (3. - 2. * r)
        action = torch.where(returning_now[:, None], self.hold_end_action * (1. - return_blend[:, None]), action)
        if 'return_mode' in p:
            # 用URDF几何的目标COM/后轴水平间距引导放下前轮；仅输出原轮速/腿位置目标。
            r=(return_elapsed/p['return_duration']).clamp(0.,1.)
            blend=r.square()*(3.-2.*r)
            theta=self.target_pitch*(1.-blend)
            theta_rate=-self.target_pitch*6.*r*(1.-r)/p['return_duration']
            reference=self.upright_delta_x*torch.cos(theta)+self.upright_delta_z*torch.sin(theta)
            reference_rate=(-self.upright_delta_x*torch.sin(theta)+self.upright_delta_z*torch.cos(theta))*theta_rate
            returning=self.hold_end_action*(1.-blend[:,None])
            wheel=(p['return_line_gain']*(gap-reference)+p['return_velocity_gain']*velocity
                   -p['return_feedforward']*reference_rate/(15.*.0875)
                   +.05*(pitch-theta)+.05*(rate-theta_rate))
            # 四轮已经放下且接近直立后停止驱动；避免稳定站立仍追逐后轴参考。
            four=(m['force_z']>5.).all(-1)
            wheel=torch.where((four & (pitch.abs()<math.radians(10.))) | (return_elapsed>=3.),0.,wheel)
            returning[:,14:16]=wheel[:,None]
            action=torch.where(returning_now[:,None],returning,action)
        return action.clamp(-1., 1.)
