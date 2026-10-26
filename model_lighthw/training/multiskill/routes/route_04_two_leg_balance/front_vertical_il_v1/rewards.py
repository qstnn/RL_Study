"""R4b：90度前腿持续积分奖励；不奖励短时完成或累计漂移。"""
import torch
from .gates import masks
from two_leg_balance_smooth_v6.spec import STAND, TRANSFER_GATE_FLOOR, MAX_SPEED
from .settings import PROFILE
REWARD_PROFILES = (PROFILE,)
from two_leg_balance_smooth_v6.jitter import LIMITS


def task_rewards(m, state, phase, mode, full_target, mass, dt, joint_pos, leg_torque,
                 leg_speed, wheel_speed, actions, previous_actions, timeouts, missed_hold,
                 reward_profile="front_vertical_v1", older_actions=None):
    """目的：把收益集中到重心转移、指定双轮承载和另一对离地，减少四轮原地收益。

    激活：入场单独奖励站稳；转移/保持用最终目标；返回/恢复只在外部返回命令后激活。
    返回：每项形状[N]的未加权奖励；连续项已乘dt，完成事件项为0；平衡/恢复使用1kHz合格时间积分。
    姿态/COM过渡保留25%卸载门底值；捕获项要求允许速度内平移且实际高频振动较小。
    此处连续引导允许尚未达标的中间状态取得部分收益；严格验收仍由BalanceState负责。
    """
    if reward_profile not in REWARD_PROFILES:
        raise ValueError(f"unknown R4 reward profile: {reward_profile}")
    if older_actions is None:
        raise ValueError('Smooth reward needs two previous executed actions')
    v = masks(m, mode, full_target)
    valid = (~state.failed).float()
    active = ((phase == 1) | (phase == 2)).float() * valid
    entry = (phase == 0).float() * valid
    returning = (phase >= 3).float() * (~state.stand_request).float() * valid
    ids = torch.arange(mode.numel(), device=mode.device)
    stance = torch.stack((2 * mode, 2 * mode + 1), -1)
    swing = (stance + 2) % 4
    force = m['force_z'].clamp_min(0.)
    sf, af = force[ids[:, None], stance], force[ids[:, None], swing]
    weight = mass * 9.81
    # 两支撑轮均须承载，且总轮载不能过低，防止腾空或单轮接触得到卸载奖励。
    bearing = (sf.amin(-1) / (.25 * weight)).clamp(0., 1.)
    if reward_profile == "front_vertical_v1":
        # 动态合格瞬间可能低于65%体重；连续承载门保留该区间梯度。
        # 腾空时总载荷为0，错误支撑或仅一只指定轮承载时sf.amin为0，仍无正收益。
        bearing *= (force.sum(-1) / (.65 * weight)).clamp(0., 1.)
    else:
        bearing *= (force.sum(-1) >= .65 * weight)
    unloading = (1. - af.sum(-1) / (.5 * weight)).clamp(0., 1.)
    # 正式验收要求两只悬空轮分别低于2N；诊断用每轮连续载荷门控，避免一只轮卸载后另一只仍承重时继续获得姿态/COM引导。
    per_wheel_load = (af / (.25 * weight)[:, None]).clamp(0., 1.)
    pair_unloading = (1. - per_wheel_load).amin(-1)
    load_asymmetry = (per_wheel_load[..., 0] - per_wheel_load[..., 1]).abs().clamp(0., 1.)
    if reward_profile in ("unload_diag_v1", "front_vertical_v1"):
        unloading_gate = pair_unloading
        swing_load = per_wheel_load.amax(-1)
    else:
        unloading_gate = unloading
        swing_load = (af.sum(-1) / weight).clamp(0., 1.)
    target_angle = torch.acos((-full_target[:, 2]).clamp(-1., 1.)).clamp_min(.1)
    # 相对直立姿态的最终倾角进度：站直为0，接近目标递增；不再全回合奖励宽高斯。
    tilt_progress = (1. - v['attitude'] / target_angle).clamp(0., 1.)
    com_progress = (1. - m['line_error'] / .45).clamp(0., 1.)
    com_quality = torch.exp(-(m['line_error'] / .12).square())
    pose_quality = torch.exp(-(v['attitude'] / .12).square())
    # 净空收益同时要求卸载；只抬轮中心但仍有较大接触力不能拿满收益。
    clearance = (v['clearance'] / .08).clamp(0., 1.) * unloading * bearing
    quiet = torch.exp(-(m['ang_vel'].norm(dim=-1) / .35).square()
                      -((m['com_vel'].norm(dim=-1) - MAX_SPEED).clamp_min(0.) / .18).square())
    four = (m['force_z'] > 5.).all(-1).float()
    upright = torch.exp(-(torch.acos((-m['gravity'][:, 2]).clamp(-1., 1.)) / .3).square())
    stand_pose = torch.exp(-((joint_pos - actions.new_tensor(STAND)) / .4).square().mean(-1))
    # 慢漂移不受出生原点约束，实际场地边界单独截断。
    drift_cost = torch.zeros_like(valid)
    stable_phase = ((phase == 2) | (phase == 4)).float() * valid
    jitter_ratio = state.jitter.rms / actions.new_tensor(LIMITS)
    jitter_cost = jitter_ratio.square().clamp(max=4.).mean(-1)
    smooth_quality = torch.exp(-jitter_cost)
    guidance_gate = unloading_gate
    capture_stability = torch.ones_like(bearing)
    hold_smooth_quality = torch.ones_like(bearing)
    if reward_profile == "front_vertical_v1":
        # 目的：前轮尚承载时也能学习倾身和COM转移；仅两个引导项使用底值。
        # 激活：转移/保持；范围[0.25,1]，不能为支撑、净空或成功项放宽卸载要求。
        guidance_gate = TRANSFER_GATE_FLOOR + (1. - TRANSFER_GATE_FLOOR) * pair_unloading
        hold_smooth_quality = torch.where(phase == 2, smooth_quality, torch.ones_like(bearing))
        # 捕获需要同时满足速度上限与低振动，不要求回到保持原点。
        capture_stability = quiet * hold_smooth_quality

    return dict(
        attitude=dt * active * bearing * guidance_gate * tilt_progress,
        com_line=dt * active * bearing * guidance_gate * com_progress,
        support=dt * active * bearing * unloading_gate,
        clearance=dt * active * clearance * unloading_gate,
        capture=dt * active * bearing * unloading_gate * com_quality * pose_quality * capture_stability,
        pair=dt * active * v['pair'].float() * com_quality * pose_quality * capture_stability,
        swing_load=dt * active * swing_load,
        swing_asymmetry=dt * active * load_asymmetry,
        quiet=dt * valid * ((phase == 2) * v['pair'] * com_quality * pose_quality * hold_smooth_quality
                           + (phase == 4) * (~state.stand_request) * four) * quiet,
        entry_quiet=dt * entry * four * quiet,
        entry_pose=dt * entry * four * upright * stand_pose,
        return_pose=dt * returning * stand_pose,
        return_attitude=dt * returning * upright,
        balance=state.stable_step_s.float(),
        recovery=state.recovery_step_s.float(),
        hold_once=torch.zeros_like(valid),
        success_once=torch.zeros_like(valid),
        missed_hold=torch.zeros_like(valid),
        action_rate=dt * (actions - previous_actions).square().mean(-1),
        torque=dt * (leg_torque / 48.).square().mean(-1),
        joint_speed=dt * (leg_speed / 12.57).square().mean(-1),
        wheel_speed=dt * ((phase == 2) | (phase == 4)) * (wheel_speed / 15.).square().mean(-1),
        saturation=dt * (actions.abs() - .9).clamp_min(0).mean(-1),
        drift=dt * drift_cost,
        # 目的：压制已站稳后的实际振动；激活保持/恢复，返回[N]非负成本。
        jitter=dt * stable_phase * jitter_cost,
        # 目的：抑制指令逐步交替；同样只在稳定段计分，不压制起身的大动作。
        action_curvature=dt * stable_phase * (actions - 2. * previous_actions + older_actions).square().mean(-1),
        # 允许速度死区内平移，超出0.12m/s才增加成本；不能靠快滚获取稳定收益。
        slow_speed=dt * stable_phase * ((m['com_vel'].norm(dim=-1) - MAX_SPEED).clamp_min(0.) / MAX_SPEED).square().clamp(max=4.),
        failure=state.failed.float(), time=dt * (phase == 1) * valid,
    )
