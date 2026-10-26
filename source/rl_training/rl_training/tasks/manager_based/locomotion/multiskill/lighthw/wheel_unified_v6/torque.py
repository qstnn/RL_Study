"""关节输出侧力矩：读取真实PD上限，逐物理步采集并按工况统计。"""
import numpy as np
import torch


def actuator_limits(robot, validate=False):
    """按仿真关节顺序组装执行器软件限值；不读取PhysX求解器保护上限。"""
    limits = torch.zeros_like(robot.data.applied_torque)
    coverage = torch.zeros(limits.shape[1], dtype=torch.long, device=limits.device) if validate else None
    for actuator in robot.actuators.values():
        indices = actuator.joint_indices
        limits[:, indices] = actuator.effort_limit
        if coverage is not None:
            coverage[indices] += 1
    if validate and (not bool(torch.isfinite(limits).all()) or not bool((limits > 0).all())
                     or not bool((coverage == 1).all())):
        raise ValueError('Every robot joint must have exactly one finite positive actuator effort limit')
    return limits


class PhysicsTorqueRecorder:
    """在Articulation写力矩后、物理积分前采集；重置重复写入不算物理步。"""
    def __init__(self, env, robot, joint_ids):
        self.env, self.robot, self.joint_ids = env, robot, joint_ids
        shape = (env.num_envs, env.cfg.decimation, len(joint_ids))
        self.applied = torch.zeros(shape, device=env.device)
        self.requested = torch.zeros_like(self.applied)
        self.velocity = torch.zeros_like(self.applied)
        self.enabled = False
        self.cursor = 0
        self.last_physics_counter = -1

    def begin(self):
        self.cursor = 0
        self.enabled = True

    def sample(self):
        counter = self.env._sim_step_counter
        if not self.enabled or counter == self.last_physics_counter:
            return
        if self.cursor >= self.applied.shape[1]:
            raise RuntimeError('Torque recorder sampled more than one value per physics step')
        j = self.joint_ids
        self.applied[:, self.cursor] = self.robot.data.applied_torque[:, j]
        self.requested[:, self.cursor] = self.robot.data.computed_torque[:, j]
        self.velocity[:, self.cursor] = self.robot.data.joint_vel[:, j]
        self.cursor += 1
        self.last_physics_counter = counter

    def finish(self):
        self.enabled = False
        if self.cursor != self.applied.shape[1]:
            raise RuntimeError(f'Missing physics torque samples: {self.cursor}/{self.applied.shape[1]}')
        if not all(bool(torch.isfinite(x).all()) for x in (self.applied, self.requested, self.velocity)):
            raise RuntimeError('Non-finite physics torque measurement')

    def summaries(self, limits):
        """每个20ms控制步的真实20个子步；返回每环境、每关节指标。"""
        return {
            'peak': self.applied.abs().amax(1),
            'rms': self.applied.square().mean(1).sqrt(),
            'request_peak': self.requested.abs().amax(1),
            'saturation': (self.applied.abs() >= .99 * limits[:, None, :]).float().mean(1),
            'clipped': (self.requested.abs() > limits[:, None, :] + 1.e-5).float().mean(1),
        }

    def numpy_trace(self, env_index=0):
        """固定评估的1kHz原始数据；只读复制，不改变动作或环境状态。"""
        return {name: getattr(self, name)[env_index].detach().cpu().numpy().copy()
                for name in ('applied', 'requested', 'velocity')}


class TorqueWindow:
    """按PPO日志窗口聚合1kHz样本；RMS从平方和计算，峰值取最大而非均值。"""
    def __init__(self, joint_names, dt):
        self.joint_names = joint_names
        self.dt = dt
        self.sums = None

    def update(self, recorder, masks, limits):
        names = list(masks)
        weights = torch.stack([masks[k] for k in names]).float()
        applied, requested, velocity = recorder.applied, recorder.requested, recorder.velocity
        count = weights.sum(-1) * applied.shape[1]
        terms = {
            'square': applied.square().sum(1),
            'absolute': applied.abs().sum(1),
            'saturation': (applied.abs() >= .99 * limits[:, None, :]).float().sum(1),
            'clipped': (requested.abs() > limits[:, None, :] + 1.e-5).float().sum(1),
            'positive_work': (applied * velocity).clamp_min(0.).sum(1) * self.dt,
            'negative_work': (applied * velocity).clamp_max(0.).abs().sum(1) * self.dt,
        }
        maxima = {'peak_Nm': applied.abs().amax(1), 'request_peak_Nm': requested.abs().amax(1),
                  'joint_speed_peak_rad_s': velocity.abs().amax(1),
                  'mechanical_power_peak_W': (applied * velocity).abs().amax(1)}
        if self.sums is None:
            self.groups = names
            self.count = torch.zeros_like(count)
            self.sums = {k: torch.zeros((len(names), applied.shape[2]), device=applied.device) for k in terms}
            self.peaks = {k: torch.zeros_like(self.sums['square']) for k in maxima}
            self.limit_min = limits.amin(0)
            self.limit_max = limits.amax(0)
        if names != self.groups:
            raise RuntimeError('Torque condition group order changed')
        self.count += count
        for name, values in terms.items():
            self.sums[name] += weights @ values
        for name, values in maxima.items():
            peak = torch.stack([torch.where(masks[group][:, None], values, 0.).amax(0) for group in names])
            self.peaks[name] = torch.maximum(self.peaks[name], peak)

    @torch.inference_mode()
    def drain(self):
        if self.sums is None:
            return []
        counts = self.count.cpu().tolist()
        sums = {k: v.cpu().numpy() for k, v in self.sums.items()}
        peaks = {k: v.cpu().numpy() for k, v in self.peaks.items()}
        lo, hi = self.limit_min.cpu().tolist(), self.limit_max.cpu().tolist()
        rows = []
        for i, (group, count) in enumerate(zip(self.groups, counts)):
            if not count:
                continue
            for j, joint in enumerate(self.joint_names):
                row = dict(group=group, joint=joint, physics_samples=int(count),
                           limit_min_Nm=lo[j], limit_max_Nm=hi[j],
                           rms_Nm=float(np.sqrt(sums['square'][i, j] / count)),
                           mean_abs_Nm=float(sums['absolute'][i, j] / count),
                           saturation_fraction=float(sums['saturation'][i, j] / count),
                           clipped_fraction=float(sums['clipped'][i, j] / count),
                           saturated_time_s=float(sums['saturation'][i,j]*self.dt),
                           clipped_time_s=float(sums['clipped'][i,j]*self.dt),
                           positive_work_J=float(sums['positive_work'][i, j]),
                           negative_work_J=float(sums['negative_work'][i, j]))
                row.update({k: float(v[i, j]) for k, v in peaks.items()})
                rows.append(row)
        self.count.zero_()
        for values in self.sums.values():
            values.zero_()
        for values in self.peaks.values():
            values.zero_()
        return rows


def summarize_trace(trace, joint_names, limits, success, dt=.001):
    """冻结工况逐关节力矩报告；只有完整通过的轨迹可作为满足任务的需求证据。"""
    applied, requested, velocity = (np.asarray(trace[k]) for k in ('applied', 'requested', 'velocity'))
    limits = np.asarray(limits)
    if applied.ndim != 2 or applied.shape != requested.shape or applied.shape != velocity.shape:
        raise ValueError('Expected time-by-joint torque traces of equal shape')
    if applied.shape[1] != len(joint_names) or limits.shape != (len(joint_names),):
        raise ValueError('Trace and actuator joint order differ')
    if not all(np.isfinite(v).all() for v in (applied, requested, velocity, limits)) or (limits <= 0).any():
        raise ValueError('Invalid torque trace or limits')
    result = {'sample_hz': round(1 / dt), 'samples': len(applied), 'duration_s': len(applied) * dt,
              'torque_frame': 'joint output shaft; no gearbox/efficiency conversion',
              'task_passed': bool(success), 'requirement_evidence_valid': bool(success),
              'note': 'Observed demand of this policy, not a proven minimum motor rating. Failed cases are diagnostics only. RMS is mechanical torque RMS, not a validated thermal rating.',
              'joints': []}
    for j, name in enumerate(joint_names):
        magnitude = abs(applied[:, j])
        saturated = magnitude >= .99 * limits[j]
        above90 = magnitude >= .9 * limits[j]
        edges = np.diff(np.r_[False, above90, False].astype(int))
        durations = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        power = applied[:, j] * velocity[:, j]
        result['joints'].append(dict(joint=name, limit_Nm=float(limits[j]), peak_Nm=float(magnitude.max()),
            p95_abs_Nm=float(np.percentile(magnitude, 95)), p99_abs_Nm=float(np.percentile(magnitude, 99)),
            p99_9_abs_Nm=float(np.percentile(magnitude, 99.9)), rms_Nm=float(np.sqrt(np.mean(magnitude**2))),
            saturation_fraction=float(saturated.mean()), clipped_fraction=float((abs(requested[:, j]) > limits[j] + 1.e-5).mean()),
            peak_censored_by_limit=bool(saturated.any()), request_peak_Nm=float(abs(requested[:, j]).max()),
            longest_above90_s=float(durations.max() * dt) if len(durations) else 0.,
            joint_speed_peak_rad_s=float(abs(velocity[:, j]).max()), mechanical_power_peak_W=float(abs(power).max()),
            positive_work_J=float(np.maximum(power, 0.).sum() * dt), negative_work_J=float(np.maximum(-power, 0.).sum() * dt)))
    return result
