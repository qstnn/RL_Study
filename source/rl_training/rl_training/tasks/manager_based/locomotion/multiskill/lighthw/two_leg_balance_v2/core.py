"""纯 Torch 的阶段、动作映射和严格验收；无需导入 Isaac Sim，可独立复算轨迹。"""
import math
import torch
from .spec import STAND, LOW, HIGH


def targets(actions):
    """目的：保留零动作为四轮站姿。全阶段激活，返回12个URDF范围内腿位置目标(rad)。"""
    neutral = actions.new_tensor(STAND)
    a = actions[:, :12].clamp(-1., 1.)
    span = torch.where(a >= 0, actions.new_tensor(HIGH) - neutral, neutral - actions.new_tensor(LOW))
    return neutral + a * span


def schedule(t, hold_seconds):
    """返回阶段0站立/1转移/2双支撑/3返回/4恢复，以及平滑倾角幅度[0,1]。"""
    end_hold = 5. + hold_seconds
    end_return = end_hold + 3.
    phase = ((t >= 1.).long() + (t >= 4.).long()
             + (t >= end_hold).long() + (t >= end_return).long())
    up = ((t - 1.) / 3.).clamp(0, 1)
    down = ((end_return - t) / 3.).clamp(0, 1)
    x = torch.minimum(up, down)
    return phase, x.square() * (3. - 2. * x)


def desired_gravity(pitch):
    return torch.stack((pitch.sin(), torch.zeros_like(pitch), -pitch.cos()), -1)


def support_geometry(com, wheels, mode):
    """前模式0选FR/FL，后模式1选RR/RL；返回COM投影到有限支撑线段的距离。"""
    ids = torch.arange(com.shape[0], device=com.device)
    a, b = wheels[ids, 2 * mode, :2], wheels[ids, 2 * mode + 1, :2]
    line = b - a
    u = ((com[:, :2] - a) * line).sum(-1) / line.square().sum(-1).clamp_min(1e-8)
    closest = a + u.clamp(0, 1)[:, None] * line
    return (com[:, :2] - closest).norm(dim=-1)


def masks(m, mode, target_g):
    ids = torch.arange(mode.numel(), device=mode.device)
    stance = torch.stack((2 * mode, 2 * mode + 1), -1)
    swing = (stance + 2) % 4
    f = m["force_z"]
    sf, af = f[ids[:, None], stance], f[ids[:, None], swing]
    clear = m["clearance"][ids[:, None], swing].amin(-1)
    attitude = torch.acos((m["gravity"] * target_g).sum(-1).clamp(-1., 1.))
    support = (sf > 5.).all(-1)
    pair = support & (af.abs() < 2.).all(-1) & (clear >= .03)
    quiet = (m["com_vel"].norm(dim=-1) <= .12) & (m["ang_vel"].norm(dim=-1) <= .25)
    balance = pair & quiet & (m["line_error"] <= .06) & (attitude <= math.radians(12.))
    four = (f > 5.).all(-1)
    recovered = four & quiet & (m["gravity"][:, 2] < -math.cos(math.radians(12.)))
    recovered &= (m["base_z"] - .416).abs() <= .07
    return dict(support=support, pair=pair, balance=balance, recovered=recovered,
                clearance=clear, attitude=attitude, quiet=quiet)


class BalanceState:
    """1kHz累计连续保持，任何中断清零；成功必须先双支撑，再在终末恢复四轮。"""
    FLOATS = ("entry_time", "hold_time", "best_hold", "recovery_time", "best_recovery",
              "max_drift", "max_torque", "max_impact", "entry_best", "pair_time", "best_pair",
              "entry_samples", "entry_four", "entry_linear", "entry_angular", "entry_pose")
    BOOLS = ("entered", "held", "failed", "success")

    def __init__(self, count, device, hold_seconds):
        self.hold_seconds = hold_seconds
        self.start_time = torch.full((count,), -1., device=device)
        for name in self.FLOATS:
            setattr(self, name, torch.zeros(count, device=device))
        for name in self.BOOLS:
            setattr(self, name, torch.zeros(count, dtype=torch.bool, device=device))
        self.reason = torch.zeros(count, dtype=torch.long, device=device)
        self.last_phase = torch.zeros_like(self.reason)
        self.origin = torch.zeros(count, 2, device=device)
        self.hold_origin = torch.zeros_like(self.origin)
        self.recovery_origin = torch.zeros_like(self.origin)
        self.hold_event = torch.zeros_like(self.failed)

    def reset(self, ids, origin):
        for name in self.FLOATS + self.BOOLS:
            getattr(self, name)[ids] = 0
        self.start_time[ids] = -1.
        self.reason[ids] = self.last_phase[ids] = 0
        self.origin[ids] = self.hold_origin[ids] = self.recovery_origin[ids] = origin
        self.hold_event[ids] = False

    def clock(self, t):
        """初态等待不推进转移时钟；只有真实站稳事件才能开启完整任务。"""
        return torch.where(self.start_time >= 0., t - self.start_time + 1., t.clamp(max=.999))

    def finished(self, t):
        return (self.start_time >= 0.) & (self.clock(t) >= 10. + self.hold_seconds - 1e-5)

    def update(self, dt, t, m, mode, target_g):
        phase, _ = schedule(self.clock(t), self.hold_seconds)
        first_hold = (phase == 2) & (self.last_phase != 2)
        first_recovery = (phase == 4) & (self.last_phase != 4)
        self.hold_origin[first_hold] = m["com"][first_hold, :2]
        self.recovery_origin[first_recovery] = m["com"][first_recovery, :2]
        self.last_phase.copy_(phase)
        v = masks(m, mode, target_g)
        drift = (m["com"][:, :2] - self.origin).norm(dim=-1)
        bad_contact = (t > .15) & (m["body_force"] > 5.)
        fall = (m["base_z"] < .18) | (m["gravity"][:, 2] > -math.cos(math.radians(80.)))
        outside = drift > 1.5
        finite = torch.ones_like(self.failed)
        for tensor in m.values():
            finite &= torch.isfinite(tensor.reshape(t.numel(), -1)).all(-1)
        code = torch.where(~finite, 4, torch.where(bad_contact, 1, torch.where(fall, 2, torch.where(outside, 3, 0))))
        fresh = ~self.failed & (code > 0)
        self.reason[fresh] = code[fresh]
        self.failed |= code > 0
        valid = ~self.failed
        entry = (phase == 0) & v["recovered"] & valid
        self.entry_time = torch.where(entry, self.entry_time + dt, 0.)
        self.entry_best = torch.maximum(self.entry_best, self.entry_time)
        self.entered |= self.entry_time >= .5 - 1e-5
        waiting = (phase == 0) & valid
        self.entry_samples += waiting.float()
        self.entry_four += (waiting & (m["force_z"] > 5.).all(-1)).float()
        self.entry_linear += (waiting & (m["com_vel"].norm(dim=-1) <= .12)).float()
        self.entry_angular += (waiting & (m["ang_vel"].norm(dim=-1) <= .25)).float()
        pose = (m["gravity"][:, 2] < -math.cos(math.radians(12.))) & ((m["base_z"] - .416).abs() <= .07)
        self.entry_pose += (waiting & pose).float()
        # 允许最早1s开始，避免出生瞬态掩盖稳定要求；3s仍未进入则显式失败。
        start = (self.start_time < 0.) & self.entered & valid & (t >= 1.)
        self.start_time[start] = t[start]
        entry_timeout = (self.start_time < 0.) & (t >= 3. - 1e-6) & valid
        self.reason[entry_timeout] = 5
        self.failed |= entry_timeout
        valid = ~self.failed
        self.pair_time = torch.where(v["pair"] & valid, self.pair_time + dt, 0.)
        self.best_pair = torch.maximum(self.best_pair, self.pair_time)
        hd = (m["com"][:, :2] - self.hold_origin).norm(dim=-1)
        balance = (phase == 2) & v["balance"] & (hd <= .10) & self.entered & valid
        self.hold_time = torch.where(balance, self.hold_time + dt, 0.)
        self.best_hold = torch.maximum(self.best_hold, self.hold_time)
        now_held = self.best_hold >= self.hold_seconds - 1e-4
        self.hold_event |= now_held & ~self.held
        self.held |= now_held
        rd = (m["com"][:, :2] - self.recovery_origin).norm(dim=-1)
        recovery = (phase == 4) & v["recovered"] & (rd <= .10) & self.held & valid
        self.recovery_time = torch.where(recovery, self.recovery_time + dt, 0.)
        self.best_recovery = torch.maximum(self.best_recovery, self.recovery_time)
        # 不能用早先出现过一次稳定，掩盖回合结束前又摔倒或失稳。
        self.success = self.held & (self.recovery_time >= 1. - 1e-4) & valid
        self.max_drift = torch.maximum(self.max_drift, drift)
        self.max_torque = torch.maximum(self.max_torque, m["torque_peak"])
        self.max_impact = torch.maximum(self.max_impact, m["force_z"].clamp_min(0).sum(-1))
        return phase, v
