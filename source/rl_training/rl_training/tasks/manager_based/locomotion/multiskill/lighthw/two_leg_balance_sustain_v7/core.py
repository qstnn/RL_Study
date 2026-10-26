"""1kHz持续站立状态：显式返回、不可抹去的首中断、独立观察截断。"""
import math
import torch
from two_leg_balance_smooth_v6.core import BalanceState, masks, desired_gravity, targets


class SustainState(BalanceState):
    FLOATS = BalanceState.FLOATS + ('post_capture_s', 'stable_step_s', 'recovery_step_s',
                                  'capture_time', 'first_break_time', 'return_time', 'break_reason')
    BOOLS = BalanceState.BOOLS + ('stand_request', 'interrupted', 'boundary', 'observation_complete')

    def __init__(self, count, device, observation_s=10.):
        super().__init__(count, device, 2.)
        # 长时累计使用float64；绝对时钟仍来自整数物理步，保留旧前缀的float32计算。
        for name in self.FLOATS:
            setattr(self, name, torch.zeros(count, device=device, dtype=torch.float64))
        self.observation_s = observation_s
        self.capture_time = torch.full((count,), -1., device=device)
        self.first_break_time = self.capture_time.clone()
        self.return_time = self.capture_time.clone()
        self.return_amplitude = torch.ones(count, device=device)
        self.break_reason = torch.zeros(count, dtype=torch.long, device=device)
        self.stand_request[:] = True

    def reset(self, ids, origin):
        super().reset(ids, origin)
        self.capture_time[ids] = self.first_break_time[ids] = self.return_time[ids] = -1.
        self.return_amplitude[ids] = 1.
        self.break_reason[ids] = 0
        self.stand_request[ids] = True

    def schedule(self, t):
        """局部起身参考完成后无限保持；仅返回事件能改变最终目标。"""
        task = self.clock(t)
        x = ((task - 1.) / 3.).clamp(0., 1.)
        amplitude = x.square() * (3. - 2. * x)
        phase = (task >= 1.).long() + (task >= 4.).long()
        returning = self.return_time >= 0.
        r = ((t - self.return_time) / 3.).clamp(0., 1.)
        amplitude = torch.where(returning, self.return_amplitude * (1. - r.square() * (3. - 2. * r)), amplitude)
        phase = torch.where(returning, 3 + (r >= 1.).long(), phase)
        return phase, amplitude

    def request_return(self, ids, t):
        """外部明确命令；幂等，不以保持时长或回合进度隐式触发。"""
        ids = ids[self.stand_request[ids]]
        self.return_amplitude[ids] = self.schedule(t)[1][ids]
        self.return_time[ids] = t[ids]
        self.stand_request[ids] = False

    def update(self, dt, t, m, mode, target_g):
        phase, _ = self.schedule(t)
        first_hold = (phase == 2) & (self.last_phase != 2)
        self.hold_origin[first_hold] = m['com'][first_hold, :2]
        self.last_phase.copy_(phase)
        v = masks(m, mode, target_g)
        self.jitter.update(dt, m)
        smooth = self.jitter.accepted()
        finite = torch.ones_like(self.failed)
        for tensor in m.values():
            finite &= torch.isfinite(tensor.reshape(t.numel(), -1)).all(-1)
        contact = (t > .15) & (m['body_force'] > 5.)
        fall = (m['base_z'] < .18) | (m['gravity'][:, 2] > -math.cos(math.radians(80.)))
        code = torch.where(~finite, 4, torch.where(contact, 1, torch.where(fall, 2, 0)))
        fresh = ~self.failed & (code > 0)
        self.reason[fresh] = code[fresh]
        self.failed |= code > 0
        # 400m地面留2m余量，使用实际COM和全部轮位置；不以相对出生点距离判摔倒。
        outside = (m['com'][:, :2].abs() >= 198.).any(-1)
        outside |= (m['wheel_pos'][..., :2].abs() >= 198.).any(-1).any(-1)
        self.boundary |= outside
        valid = ~self.failed & ~self.boundary
        waiting = (phase == 0) & valid
        entry = waiting & v['recovered']
        self.entry_time = torch.where(entry, self.entry_time + dt, 0.)
        self.entry_best = torch.maximum(self.entry_best, self.entry_time)
        self.entered |= self.entry_time >= .5 - 1e-5
        self.entry_samples += waiting
        self.entry_four += waiting & (m['force_z'] > 5.).all(-1)
        self.entry_linear += waiting & (m['com_vel'].norm(dim=-1) <= .12)
        self.entry_angular += waiting & (m['ang_vel'].norm(dim=-1) <= .25)
        pose = (m['gravity'][:, 2] < -math.cos(math.radians(12.))) & ((m['base_z'] - .416).abs() <= .07)
        self.entry_pose += waiting & pose
        start = (self.start_time < 0.) & self.entered & valid & (t >= 1.)
        self.start_time[start] = t[start]
        entry_timeout = (self.start_time < 0.) & (t >= 3. - 1e-6) & valid
        self.reason[entry_timeout] = 5
        self.failed |= entry_timeout
        valid = ~self.failed & ~self.boundary
        self.pair_time = torch.where(v['pair'] & valid, self.pair_time + dt, 0.)
        self.best_pair = torch.maximum(self.best_pair, self.pair_time)
        stable = (phase == 2) & v['balance'] & smooth & self.entered & valid & self.stand_request
        was_held = self.held.clone()
        self.hold_time = torch.where(stable, self.hold_time + dt, 0.)
        self.best_hold = torch.maximum(self.best_hold, self.hold_time)
        newly = (self.hold_time >= 2. - 1e-8) & ~was_held
        self.capture_time[newly] = t[newly]
        self.hold_event |= newly
        self.held |= newly
        # 首个中断永久锁存；恢复后仍给逐步稳定奖励，但不恢复本回合长稳通过资格。
        broken = was_held & self.stand_request & ~stable & ~self.interrupted
        bits = ((~v['pair']).long() + 2 * (~v['quiet']).long()
                + 4 * (m['line_error'] > .06).long() + 8 * (v['attitude'] > math.radians(12.)).long()
                + 16 * (~smooth).long() + 32 * self.failed.long() + 64 * self.boundary.long())
        self.first_break_time[broken] = t[broken]
        self.break_reason[broken] = bits[broken]
        self.interrupted |= broken
        self.post_capture_s += dt * (was_held & stable & ~self.interrupted)
        self.stable_step_s += dt * stable
        # 返回不要求曾成功保持；但它只能由命令开启，返回成功与长稳验收分开。
        recovery = (phase == 4) & v['recovered'] & smooth & valid & ~self.stand_request
        self.recovery_time = torch.where(recovery, self.recovery_time + dt, 0.)
        self.recovery_step_s += dt * recovery
        self.best_recovery = torch.maximum(self.best_recovery, self.recovery_time)
        timeout = (self.start_time >= 0.) & (t - self.start_time >= 8. - 1e-6) & ~self.held & self.stand_request & valid
        self.reason[timeout] = 6
        self.failed |= timeout
        self.observation_complete = (self.post_capture_s >= self.observation_s - 1e-8) & ~self.interrupted & ~self.failed & ~self.boundary
        # success不由timeout自动决定，也不锁存；若继续观察中出现中断即失效。
        self.success = self.observation_complete & self.stand_request
        self.max_drift = torch.maximum(self.max_drift, (m['com'][:, :2] - self.origin).norm(dim=-1).double())
        self.max_torque = torch.maximum(self.max_torque, m['torque_peak'].double())
        self.max_impact = torch.maximum(self.max_impact, m['force_z'].clamp_min(0).sum(-1).double())
        return phase, v
