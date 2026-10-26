"""R4拆分任务奖励：纯Torch函数，可用记录的状态离线复算；不改变物理或成功判据。"""
import torch
from .core import masks
from .spec import STAND, REWARD_PROFILES, TRANSFER_GATE_FLOOR, HOLD_DRIFT_WIDTH_M


def task_rewards(m, state, phase, mode, full_target, mass, dt, joint_pos, leg_torque,
                 leg_speed, wheel_speed, actions, previous_actions, timeouts, missed_hold,
                 reward_profile="baseline_v4"):
    """目的：把收益集中到重心转移、指定双轮承载和另一对离地，减少四轮原地收益。

    激活：入场单独奖励站稳；转移/保持用最终目标；返回/恢复必须先严格保持成功。
    返回：每项形状[N]的未加权奖励；连续项已乘dt，一次性事件不乘dt。
    baseline_v4沿用总悬空载荷信号；unload_diag_v1按两只悬空轮分别门控并加入左右差异项。
    soft_transfer_v2只给姿态/COM过渡保留25%卸载门底值，捕获项同时要求低速度/低漂移。
    此处连续引导允许尚未达标的中间状态取得部分收益；严格验收仍由BalanceState负责。
    """
    if reward_profile not in REWARD_PROFILES:
        raise ValueError(f"unknown R4 reward profile: {reward_profile}")
    v = masks(m, mode, full_target)
    valid = (~state.failed).float()
    active = ((phase == 1) | (phase == 2)).float() * valid
    entry = (phase == 0).float() * valid
    returning = (phase >= 3).float() * state.held.float() * valid
    ids = torch.arange(mode.numel(), device=mode.device)
    stance = torch.stack((2 * mode, 2 * mode + 1), -1)
    swing = (stance + 2) % 4
    force = m['force_z'].clamp_min(0.)
    sf, af = force[ids[:, None], stance], force[ids[:, None], swing]
    weight = mass * 9.81
    # 两支撑轮均须承载，且总轮载不能过低，防止腾空或单轮接触得到卸载奖励。
    bearing = (sf.amin(-1) / (.25 * weight)).clamp(0., 1.)
    if reward_profile == "soft_transfer_v2":
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
    if reward_profile in ("unload_diag_v1", "soft_transfer_v2"):
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
    pose_quality = torch.exp(-(v['attitude'] / .35).square())
    # 净空收益同时要求卸载；只抬轮中心但仍有较大接触力不能拿满收益。
    clearance = (v['clearance'] / .08).clamp(0., 1.) * unloading * bearing
    quiet = torch.exp(-(m['ang_vel'].norm(dim=-1) / .35).square()
                      -(m['com_vel'].norm(dim=-1) / .18).square())
    four = (m['force_z'] > 5.).all(-1).float()
    upright = torch.exp(-(torch.acos((-m['gravity'][:, 2]).clamp(-1., 1.)) / .3).square())
    stand_pose = torch.exp(-((joint_pos - actions.new_tensor(STAND)) / .4).square().mean(-1))
    phase_drift = torch.where((phase == 2)[:, None], m['com'][:, :2] - state.hold_origin,
                             m['com'][:, :2] - state.recovery_origin).norm(dim=-1)
    drift_cost = torch.where((phase == 2) | (phase == 4), (phase_drift - .1).clamp_min(0).square(),
                            ((m['com'][:, :2] - state.origin).norm(dim=-1) - .6).clamp_min(0).square())
    guidance_gate = unloading_gate
    capture_stability = torch.ones_like(bearing)
    hold_drift_quality = torch.ones_like(bearing)
    if reward_profile == "soft_transfer_v2":
        # 目的：前轮尚承载时也能学习倾身和COM转移；仅两个引导项使用底值。
        # 激活：转移/保持；范围[0.25,1]，不能为支撑、净空或成功项放宽卸载要求。
        guidance_gate = TRANSFER_GATE_FLOOR + (1. - TRANSFER_GATE_FLOOR) * pair_unloading
        hold_drift_quality = torch.where(
            phase == 2, torch.exp(-(phase_drift / HOLD_DRIFT_WIDTH_M).square()),
            torch.ones_like(bearing))
        # 捕获需要同时安静且接近保持起点，避免高速度经过目标或持续滚走也拿满收益。
        capture_stability = quiet * hold_drift_quality
        # 在10cm硬门之前就提供连续成本；上界4避免大漂移压倒所有任务项。
        drift_cost = torch.where(phase == 2, (phase_drift / HOLD_DRIFT_WIDTH_M).square().clamp(max=4.), drift_cost)
    return dict(
        attitude=dt * active * bearing * guidance_gate * tilt_progress,
        com_line=dt * active * bearing * guidance_gate * com_progress,
        support=dt * active * bearing * unloading_gate,
        clearance=dt * active * clearance * unloading_gate,
        capture=dt * active * bearing * unloading_gate * com_quality * pose_quality * capture_stability,
        pair=dt * active * v['pair'].float() * com_quality * pose_quality * capture_stability,
        swing_load=dt * active * swing_load,
        swing_asymmetry=dt * active * load_asymmetry,
        quiet=dt * valid * ((phase == 2) * v['pair'] * com_quality * pose_quality * hold_drift_quality
                           + (phase == 4) * state.held * four) * quiet,
        entry_quiet=dt * entry * four * quiet,
        entry_pose=dt * entry * four * upright * stand_pose,
        return_pose=dt * returning * stand_pose,
        return_attitude=dt * returning * upright,
        balance=dt * valid * (state.hold_time > 0),
        recovery=dt * valid * (state.recovery_time > 0),
        hold_once=state.hold_event.float() * valid,
        success_once=(timeouts & state.success).float(),
        missed_hold=missed_hold.float(),
        action_rate=dt * (actions - previous_actions).square().mean(-1),
        torque=dt * (leg_torque / 48.).square().mean(-1),
        joint_speed=dt * (leg_speed / 12.57).square().mean(-1),
        wheel_speed=dt * ((phase == 2) | (phase == 4)) * (wheel_speed / 15.).square().mean(-1),
        saturation=dt * (actions.abs() - .9).clamp_min(0).mean(-1),
        drift=dt * drift_cost,
        failure=state.failed.float(), time=torch.full_like(m['base_z'], dt),
    )
