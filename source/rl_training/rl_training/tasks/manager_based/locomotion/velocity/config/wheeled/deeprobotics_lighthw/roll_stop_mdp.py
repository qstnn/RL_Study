"""LightHW 独立平地 PPO：有界动作、重复起停命令及精简奖励。"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .roll_stop_core import (ACCELERATION, CYCLE_SECONDS, LEG_LIMITS, STOP_SECONDS,
                             phase_spec, thresholds)


class RollStopLegAction(JointPositionAction):
    """raw [-1,1] 线性映射默认角周围的小幅目标；最终再受真实关节边界保护。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.limits = torch.tensor(LEG_LIMITS, device=self.device)
        self.previous_targets = self._offset.clone()
        self._processed_actions.copy_(self._offset)

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        self.previous_targets.copy_(self._processed_actions)
        target = self._offset + actions.clamp(-1.0, 1.0) * self.limits
        bounds = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions.copy_(target.clamp(min=bounds[..., 0], max=bounds[..., 1]))

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0
        self._processed_actions[env_ids] = self._offset[env_ids]
        self.previous_targets[env_ids] = self._offset[env_ids]


@configclass
class RollStopLegActionCfg(JointPositionActionCfg):
    class_type: type = RollStopLegAction


class RollStopCommand(UniformVelocityCommand):
    """20秒循环：15秒前进/加减速，5秒停车；所有轮目标仍由 PPO 产生。

    命令仅写入观测及奖励，不写入动作。每个速度依次训练斜坡、斜坡与阶跃
    混合两个阶段。指标在 reward 时刻、reset 前采集，不能混入重置后的姿态。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stage = 0
        self.external_command = False
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.peak = self.phase.clone()
        self.ramp = self.phase.clone()
        self.step_cycle = torch.zeros_like(self.phase, dtype=torch.bool)
        self.armed = torch.zeros_like(self.step_cycle)
        self.braking = torch.zeros_like(self.step_cycle)
        self.brake_done = torch.zeros_like(self.step_cycle)
        self.stop_anchor_set = torch.zeros_like(self.step_cycle)
        self.brake_distance = self.phase.clone()
        self.brake_elapsed = self.phase.clone()
        self.slow_time = self.phase.clone()
        self.drift = self.phase.clone()
        self.cycle_error = self.phase.clone()
        self.cycle_samples = self.phase.clone()
        self.cycle_support = self.phase.clone()
        self.cycle_bad = torch.zeros_like(self.step_cycle)
        self.last_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.stop_xy = self.last_xy.clone()
        self._sums = torch.zeros(20, dtype=torch.float64, device=self.device)
        self._last_record_step = -1
        self.set_stage(0)

    def set_stage(self, stage):
        self.stage = int(stage)
        cap, _ = phase_spec(self.stage)
        self.cfg.ranges.lin_vel_x = (0.0, cap)

    def reset_window(self):
        self._sums.zero_()

    def _clear_cycle(self, ids):
        for value in (self.braking, self.brake_done, self.stop_anchor_set, self.brake_distance,
                      self.brake_elapsed, self.slow_time, self.drift, self.cycle_error,
                      self.cycle_samples, self.cycle_support, self.cycle_bad):
            value[ids] = 0

    def _sample_peak(self, ids):
        cap, step_probability = phase_spec(self.stage)
        sample = torch.rand(len(ids), device=self.device)
        # 一半为当前速度上限，另一半连续覆盖低速，避免只会某个锚点速度。
        self.peak[ids] = torch.where(sample < 0.5, cap,
                                    cap * torch.empty_like(sample).uniform_(0.25, 1.0))
        self.step_cycle[ids] = torch.rand_like(sample) < step_probability
        self.ramp[ids] = torch.where(self.step_cycle[ids], 0.0, self.peak[ids] / ACCELERATION)

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self._clear_cycle(ids)
        self.armed[ids] = False
        # 出生先停车1~5秒，随后每个完整周期精确25%停车；分散相位便于采样。
        self.phase[ids] = torch.empty(len(ids), device=self.device).uniform_(15.0, 19.0)
        self._sample_peak(ids)
        self.vel_command_b[ids] = 0
        self.is_standing_env[ids] = True
        self.is_heading_env[ids] = False
        self.last_xy[ids] = self.robot.data.root_pos_w[ids, :2]

    def _update_command(self):
        if self.external_command:
            return
        self.phase.add_(self._env.step_dt)
        ids = (self.phase >= CYCLE_SECONDS).nonzero(as_tuple=False).flatten()
        if len(ids):
            self._finish_cycles(ids)
            self.phase[ids] -= CYCLE_SECONDS
            self._clear_cycle(ids)
            self.armed[ids] = True
            self._sample_peak(ids)
        active_time = CYCLE_SECONDS - STOP_SECONDS
        ramp = self.ramp.clamp_min(self._env.step_dt)
        factor = torch.minimum(self.phase / ramp, (active_time - self.phase) / ramp).clamp(0, 1)
        # 阶跃周期15秒恒速、5秒零命令；斜坡周期的加减速度均为0.8 m/s²。
        factor = torch.where(self.step_cycle, (self.phase < active_time).float(), factor)
        self.vel_command_b[:, 0] = self.peak * factor
        self.vel_command_b[:, 1:] = 0
        self.is_standing_env.copy_(self.phase >= active_time)

    def _finish_cycles(self, ids):
        valid = self.armed[ids]
        error = self.cycle_error[ids] / self.cycle_samples[ids].clamp_min(1)
        support = self.cycle_support[ids] / self.cycle_samples[ids].clamp_min(1)
        # 从开始减速起的路程；斜坡允许物理制动距离+0.25m，阶跃单独用1m/s²基准。
        braking_limit = 0.25 + self.peak[ids].square() / (2 * torch.where(
            self.step_cycle[ids], 1.0, ACCELERATION))
        good = (valid & (self.cycle_samples[ids] >= 25) & ~self.cycle_bad[ids]
                & (error <= torch.maximum(torch.full_like(error, 0.08), 0.12 * self.peak[ids]))
                & (support >= 0.9) & self.brake_done[ids]
                & (self.brake_distance[ids] <= braking_limit)
                & (self.brake_elapsed[ids] <= self.ramp[ids] + 2.0)
                & (self.drift[ids] <= 0.04))
        step = valid & self.step_cycle[ids]
        values = [valid.sum(), good.sum(), step.sum(), (good & step).sum(),
                  (self.brake_distance[ids] * valid).sum(), (self.drift[ids] * valid).sum()]
        self._sums[12:18].add_(torch.stack(values).double())

    def record(self):
        if self.external_command or self._last_record_step == self._env.common_step_counter:
            return
        self._last_record_step = self._env.common_step_counter
        robot, dt = self.robot, self._env.step_dt
        sensor = self._env.scene["contact_forces"]
        if not hasattr(self, "wheel_ids"):
            self.wheel_ids = sensor.find_bodies(".*_wheel_link")[0]
            self.other_ids = [i for i, name in enumerate(sensor.body_names) if "wheel" not in name]
        force = sensor.data.net_forces_w
        support = (force[:, self.wheel_ids, 2] > 5.0).all(-1).float()
        nonwheel = (sensor.data.net_forces_w_history[:, :, self.other_ids].norm(dim=-1)
                    .amax(dim=(1, 2)) > 1.0)
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        bad = nonwheel | (tilt > 0.174533)
        vel = robot.data.root_lin_vel_b
        speed = vel[:, :2].norm(dim=-1)
        xy = robot.data.root_pos_w[:, :2]
        distance = (xy - self.last_xy).norm(dim=-1)
        self.last_xy.copy_(xy)
        cap, _ = phase_spec(self.stage)
        cruise = (self.phase >= self.ramp + 1.0) & (self.phase < 15.0 - self.ramp)
        fast = cruise & (self.peak >= 0.9 * cap)
        stop = self.phase >= 17.0
        error = (vel[:, 0] - self.vel_command_b[:, 0]).abs()
        n = self.num_envs
        self._sums[:12].add_(torch.stack([
            fast.sum(), (error * fast).sum(), (vel[:, 0] * fast).sum(),
            (self.vel_command_b[:, 0] * fast).sum(), (support * fast).sum(),
            stop.sum(), (speed * stop).sum(), nonwheel.sum(), (tilt > 0.174533).sum(),
            self._env.termination_manager.terminated.sum(), torch.as_tensor(n, device=self.device),
            (support * stop).sum(),
        ]).double())
        self.cycle_error.add_(error * cruise)
        self.cycle_samples.add_(cruise)
        self.cycle_support.add_(support * cruise)
        self.cycle_bad.logical_or_(bad)
        begin = (self.phase >= 15.0 - self.ramp) & ~self.braking
        self.braking.logical_or_(begin)
        pending = self.braking & ~self.brake_done
        self.brake_distance.add_(distance * pending)
        self.brake_elapsed.add_(dt * pending)
        self.slow_time.copy_(torch.where((speed < 0.05) & (self.phase >= 15), self.slow_time + dt, 0.0))
        self.brake_done.logical_or_(pending & (self.slow_time >= 0.2))
        anchor = (self.phase >= 18.0) & ~self.stop_anchor_set
        self.stop_xy[anchor] = xy[anchor]
        self.stop_anchor_set.logical_or_(anchor)
        self.drift.copy_(torch.maximum(self.drift,
                         (xy - self.stop_xy).norm(dim=-1) * self.stop_anchor_set))

    def snapshot(self):
        s = self._sums.cpu().tolist()
        div = lambda a, b: s[a] / s[b] if s[b] > 0 else None
        return dict(fast_samples=s[0], speed_mae=div(1, 0), actual_speed=div(2, 0),
                    command_speed=div(3, 0), support=div(4, 0), stop_samples=s[5],
                    stop_speed=div(6, 5), nonwheel=div(7, 10), tilt_bad=div(8, 10),
                    failure=div(9, 10), stop_support=div(11, 5), cycles=s[12],
                    cycle_success=div(13, 12), step_cycles=s[14], step_success=div(15, 14),
                    brake_distance=div(16, 12), stop_drift=div(17, 12))


@configclass
class RollStopCommandCfg(UniformVelocityCommandCfg):
    class_type: type = RollStopCommand


def velocity_tracking(env):
    """全部命令生效；奖励实际前向速度，返回0~1，翻倒时衰减为0。"""
    robot = env.scene["robot"]
    error = robot.data.root_lin_vel_b[:, 0] - env.command_manager.get_command("base_velocity")[:, 0]
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
    return torch.exp(-error.square() / 0.25) * upright


def body_motion(env):
    """全程限制横漂、垂向冲击及摇摆；偏航0.15rad/s内无处罚，返回非负代价。"""
    data = env.scene["robot"].data
    return (data.root_lin_vel_b[:, 1].square() + 0.2 * data.root_lin_vel_b[:, 2].square()
            + 0.05 * data.root_ang_vel_b[:, :2].square().sum(-1)
            + 0.1 * (data.root_ang_vel_b[:, 2].abs() - 0.15).clamp_min(0).square())


def leg_regularization(env):
    """全程一个腿姿态约束；返回归一化偏角与raw越界代价，不要求精确机身高度。"""
    term = env.action_manager.get_term("joint_pos")
    delta = (env.scene["robot"].data.joint_pos[:, term._joint_ids] - term._offset) / term.limits
    return delta.square().mean(-1) + 0.2 * (term.raw_actions.abs() - 0.8).clamp_min(0).square().mean(-1)


def wheel_support(env, sensor_cfg):
    """仅平地全程生效；返回四轮最低载荷的0~1软支撑分数，不强迫四轮均载。"""
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces.amin(-1) / 5.0).clamp(0, 1)


def stop_motion(env):
    """仅真正零命令时生效；处罚停后机身漂移和轮转动，返回非负代价。"""
    term = env.action_manager.get_term("joint_vel")
    data = env.scene["robot"].data
    stop = env.command_manager.get_command("base_velocity").norm(dim=-1) < 0.01
    return stop * (data.root_lin_vel_b[:, :2].square().sum(-1)
                   + 0.02 * data.root_ang_vel_b[:, 2].square()
                   + 0.005 * data.joint_vel[:, term._joint_ids].square().mean(-1))


def action_smoothness(env):
    """全程按实际下发目标平滑；返回无量纲代价，腿/轮分组平均避免维数偏置。"""
    leg = env.action_manager.get_term("joint_pos")
    wheel = env.action_manager.get_term("joint_vel")
    # 不替换持久tensor，避免 inference tensor 在PPO更新外原地修改错误。
    if not hasattr(wheel, "roll_stop_previous"):
        with torch.inference_mode(False):
            wheel.roll_stop_previous = torch.zeros_like(wheel.processed_actions)
    value = ((leg.processed_actions - leg.previous_targets) / leg.limits).square().mean(-1)
    value += ((wheel.processed_actions - wheel.roll_stop_previous) / 10.0).square().mean(-1)
    wheel.roll_stop_previous.copy_(wheel.processed_actions)
    return value


def metrics(env):
    """每控制步、reset前记录验收量；返回0，不产生奖励或控制输入。"""
    env.command_manager.get_term("base_velocity").record()
    return torch.zeros(env.num_envs, device=env.device)
