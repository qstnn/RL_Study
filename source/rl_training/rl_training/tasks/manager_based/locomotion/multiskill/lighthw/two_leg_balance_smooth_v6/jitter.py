"""1kHz因果高通及指数RMS；检测实际状态振动，不过滤Actor或物理动作。"""
import math
import torch

CUTOFF_HZ = 5.
RMS_TAU_S = .10
LIMITS = (.02, .05, math.radians(.2), .50)
NAMES = ('com_velocity_m_s', 'angular_velocity_rad_s', 'leg_position_rad', 'wheel_velocity_rad_s')


def coefficients(dt=.001):
    """二阶Butterworth高通系数；截止5Hz附近为平滑过渡，非理想砖墙频带。"""
    k = math.tan(math.pi * CUTOFF_HZ * dt)
    b0 = 1. / (1. + math.sqrt(2.) * k + k * k)
    return b0, -2. * b0, b0, 2. * (k * k - 1.) * b0, (1. - math.sqrt(2.) * k + k * k) * b0


class Smoothness:
    def __init__(self, count, device):
        self.x1 = torch.zeros(count, 22, device=device)
        self.x2 = torch.zeros_like(self.x1)
        self.y1 = torch.zeros_like(self.x1)
        self.y2 = torch.zeros_like(self.x1)
        self.offset = torch.zeros_like(self.x1)
        self.energy = torch.zeros_like(self.x1)
        self.rms = torch.zeros(count, 4, device=device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        for value in (self.x1, self.x2, self.y1, self.y2, self.offset, self.energy, self.rms):
            value[ids] = 0.
        self.initialized[ids] = False

    def update(self, dt, m):
        """全物理步更新；返回[N,4]实际COM/角速度向量RMS及最差腿/轮RMS。"""
        raw = torch.cat((m['com_vel'], m['ang_vel'], m['leg_position'], m['wheel_velocity']), -1)
        first = ~self.initialized
        self.offset[first] = raw[first]
        self.initialized[:] = True
        x = raw - self.offset
        b0, b1, b2, a1, a2 = coefficients(dt)
        y = b0 * x + b1 * self.x1 + b2 * self.x2 - a1 * self.y1 - a2 * self.y2
        self.x2.copy_(self.x1)
        self.x1.copy_(x)
        self.y2.copy_(self.y1)
        self.y1.copy_(y)
        alpha = 1. - math.exp(-dt / RMS_TAU_S)
        self.energy.lerp_(y.square(), alpha)
        self.rms = torch.stack((self.energy[:, :3].sum(-1).sqrt(), self.energy[:, 3:6].sum(-1).sqrt(),
            self.energy[:, 6:18].amax(-1).sqrt(), self.energy[:, 18:].amax(-1).sqrt()), -1)
        return self.rms

    def accepted(self):
        return (self.rms <= self.rms.new_tensor(LIMITS)).all(-1)
