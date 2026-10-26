"""独立NumPy/SciPy长稳验收，另复算CPU动作和保持积分奖励。"""
from types import SimpleNamespace
import numpy as np
import torch
from scipy.signal import butter, lfilter
from common_sustain import (read, write, sha, require, check_legacy, ACTOR, BASELINE, CONTROL,
    load_actor, actor_network, candidate)


def streak(mask):
    ids = np.arange(len(mask))
    return np.where(mask, ids - np.maximum.accumulate(np.where(~mask, ids, -1)), 0)


def analyse(d, c, geometry, observation_s):
    t = d['t'].astype(np.float64)
    n = len(t)
    norm = lambda a: np.linalg.norm(a, axis=-1)
    np.testing.assert_allclose(np.diff(np.r_[0., t]), .001, atol=4e-5, rtol=0)
    force = d['force_z']
    four = (force > 5.).all(-1)
    linear, angular = norm(d['com_vel']) <= .12, norm(d['ang_vel']) <= .25
    pose = (d['gravity'][:, 2] < -np.cos(np.deg2rad(12.))) & (abs(d['base_z'] - .416) <= .07)
    code = np.where((t > .15) & (d['body_force'] > 5.), 1,
        np.where((d['base_z'] < .18) | (d['gravity'][:, 2] > -np.cos(np.deg2rad(80.))), 2, 0))
    physical = np.maximum.accumulate(code > 0)
    outside = (abs(d['com'][:, :2]) >= 198.).any(-1) | (abs(d['wheel_pos'][..., :2]) >= 198.).any((-1, -2))
    outside = np.maximum.accumulate(outside)
    entry = streak(four & linear & angular & pose & ~physical & ~outside) >= 500
    starts = np.flatnonzero(np.maximum.accumulate(entry) & (t >= 1.) & (t <= 3.000001) & ~physical & ~outside)
    start = float(t[starts[0]]) if len(starts) else -1.
    task = np.where(t >= start, t - start + 1., np.minimum(t, .999)) if start >= 0 else np.minimum(t, .999)
    np.testing.assert_allclose(task, d['task_t'], atol=4e-5, rtol=0)
    return_t = d['return_time']
    returning = return_t >= 0.
    x = np.clip((task - 1.) / 3., 0., 1.)
    amplitude = x*x*(3.-2.*x)
    phase = (task >= 1.).astype(int) + (task >= 4.)
    if returning.any():
        ix = np.flatnonzero(returning)[0]
        local = np.clip((t - return_t) / 3., 0., 1.)
        start_amp = np.clip((float(return_t[ix]) - start) / 3., 0., 1.)
        start_amp = start_amp**2*(3.-2.*start_amp)
        amplitude = np.where(returning, start_amp * (1. - local*local*(3.-2.*local)), amplitude)
        phase = np.where(returning, 3 + (local >= 1.).astype(int), phase)
    pitch = np.float32(geometry['candidates'][1]['pitch_rad']) * amplitude
    target = np.stack((np.sin(pitch), np.zeros(n), -np.cos(pitch)), -1)
    np.testing.assert_allclose(target, d['target_gravity'], atol=3e-6, rtol=0)
    np.testing.assert_array_equal(phase, d['phase'])
    np.testing.assert_array_equal(~returning, d['stand_request'])
    a, b = d['wheel_pos'][:, 2, :2], d['wheel_pos'][:, 3, :2]
    line = b-a
    u = np.sum((d['com'][:, :2]-a)*line, -1) / np.maximum(np.sum(line*line, -1), 1e-8)
    line_error = norm(d['com'][:, :2] - (a+np.clip(u, 0., 1.)[:, None]*line))
    np.testing.assert_allclose(line_error, d['line_error'], atol=1e-5, rtol=1e-4)
    clear = d['wheel_pos'][:, :, 2] - .087501
    np.testing.assert_allclose(clear, d['clearance'], atol=1e-6, rtol=0)
    pair = (force[:, 2:] > 5.).all(-1) & (abs(force[:, :2]) < 2.).all(-1) & (clear[:, :2].min(-1) >= .03)
    angle = np.arccos(np.clip(np.sum(d['gravity']*d['target_gravity'], -1), -1., 1.))
    raw = np.concatenate((d['com_vel'], d['ang_vel'], d['leg_position'], d['wheel_velocity']), -1).astype(np.float64)
    b, a = butter(2, 5., fs=1000., btype='highpass')
    hp = lfilter(b, a, raw-raw[:1], axis=0)
    alpha = 1.-np.exp(-.001/.1)
    energy = lfilter([alpha], [1., -(1.-alpha)], hp*hp, axis=0)
    rms = np.stack((np.sqrt(energy[:, :3].sum(-1)), np.sqrt(energy[:, 3:6].sum(-1)),
        np.sqrt(energy[:, 6:18].max(-1)), np.sqrt(energy[:, 18:].max(-1))), -1)
    np.testing.assert_allclose(rms, d['smooth_rms'], atol=5e-5, rtol=2e-3)
    smooth = (rms <= [.02, .05, np.deg2rad(.2), .5]).all(-1)
    qualified = (phase == 2) & pair & linear & angular & (line_error <= .06) & (angle <= np.deg2rad(12.)) & smooth & ~physical & ~outside & (start >= 0.)
    initial_captures = np.flatnonzero(streak(qualified) >= 2000)
    capture = int(initial_captures[0]) if len(initial_captures) else None
    if start < 0:
        code = np.where((code == 0) & (t >= 3.-1e-6), 5, code)
    elif capture is None or t[capture] > start + 8.000001:
        code = np.where((code == 0) & (t >= start+8.-1e-6) & ~returning, 6, code)
    failed = np.maximum.accumulate(code > 0)
    qualified &= ~failed
    captures = np.flatnonzero(streak(qualified) >= 2000)
    capture = int(captures[0]) if len(captures) else None
    break_ids = np.flatnonzero((np.arange(n) > capture) & ~qualified & ~returning) if capture is not None else []
    first_break = int(break_ids[0]) if len(break_ids) else None
    tail = (np.arange(n) > capture) & qualified if capture is not None else np.zeros(n, bool)
    if first_break is not None:
        tail[first_break:] = False
    elapsed = np.cumsum(tail) * .001
    captured_t = float(t[capture]) if capture is not None else -1.
    broken_t = float(t[first_break]) if first_break is not None else -1.
    success = bool(elapsed[-1] >= observation_s - 1e-8 and first_break is None and not failed[-1] and not outside[-1] and not returning[-1])
    first_code = int(code[np.flatnonzero(code)[0]]) if code.any() else 0
    for key, expected in dict(capture_time=captured_t, first_break_time=broken_t, post_capture_s=elapsed[-1],
                              best_hold=streak(qualified).max()*.001, start_time=start).items():
        if abs(c[key]-expected) > .002:
            raise AssertionError(f'{c["trace"]} slot{c["trace_env"]}: {key} {c[key]} vs {expected}')
    assert bool(c['success']) == success and bool(c['failed']) == bool(failed[-1])
    assert c['reason'] == first_code and bool(c['interrupted']) == (first_break is not None)
    np.testing.assert_allclose(elapsed, d['post_capture_s'], atol=.002, rtol=0)
    if captured_t >= 0:
        np.testing.assert_allclose(d['capture_time'][capture+2:], captured_t, atol=.002, rtol=0)
    gate_names = ['pair', 'speed', 'angular_speed', 'com_line', 'attitude', 'actual_hf']
    gates = [pair, linear, angular, line_error <= .06, angle <= np.deg2rad(12.), smooth]
    reason = [name for name, mask in zip(gate_names, gates) if first_break is not None and not mask[first_break]]
    stable_speed = norm(d['com_vel'])[tail]
    stats = dict(trace=c['trace'], slot=c['trace_env'], blend_s=c['blend_s'], duration_s=float(t[-1]),
        capture_time_s=captured_t, post_capture_uninterrupted_s=float(elapsed[-1]), first_break_time_s=broken_t,
        first_break_gates=reason, observation_pass=success, physical_failure=bool(failed[-1]), reason=first_code,
        max_body_force_N=float(d['body_force'].max()), max_drift_m=float(norm(d['com'][:, :2]-d['origin']).max()),
        max_stable_speed_m_s=float(stable_speed.max()) if len(stable_speed) else None,
        max_angular_speed_after_capture=float(norm(d['ang_vel'])[capture:].max()) if capture is not None else None,
        last_phase=int(phase[-1]), returned=bool(returning.any()), physics_samples=n,
        longest_qualified_s=float(streak(qualified).max()*.001))
    return stats, qualified


def verify(run):
    require(run)
    check_legacy()
    result, request = read(run/'result.json'), read(run/'request.json')
    geometry = read(run/'geometry.json')
    model = actor_network().eval()
    model.load_state_dict(load_actor(ACTOR)['actor_state_dict'])
    with np.load(BASELINE/'trace_seed43.npz') as z:
        baseline = {k: z[k] for k in z.files}
    checks, stats = [], []
    for path in sorted(run.glob('trace_seed*.npz')):
        assert sha(path) == result['trace_hashes'][path.name]
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in data.values())
        group = [c for c in result['cases'] if c['trace'] == path.name]
        assert len(group) == 2
        qualified = np.zeros(data['active'].shape, bool)
        for c in group:
            slot = c['trace_env']; active = data['active'][:, slot]
            d = {k: v[active, slot] for k, v in data.items() if not k.startswith('policy_')}
            row, good = analyse(d, c, geometry, request['hold_seconds'])
            if request['return_after_s'] is None:
                assert not row['returned']
            elif row['returned']:
                returned_at = float(d['return_time'][d['return_time'] >= 0.][0])
                assert abs(returned_at - row['capture_time_s'] - request['return_after_s']) < .021
            stats.append(row)
            qualified[active, slot] = good
        # 旧Actor在新协议中仍接收独立旧观测；公共物理前缀和切换输入应完全相同。
        length = int(round(request['switch_s']*1000))
        common = [k for k in baseline if not k.startswith('policy_') and k in data]
        for key in common:
            np.testing.assert_array_equal(data[key][:length], baseline[key][:length], err_msg='prefix '+key)
        controls = int(round(request['switch_s']*50))+1
        np.testing.assert_array_equal(data['policy_legacy_observation'][:controls], baseline['policy_observation'][:controls])
        np.testing.assert_array_equal(data['policy_actor_action'][:controls], baseline['policy_actor_action'][:controls])
        active = data['policy_active'].astype(bool)
        np.testing.assert_array_equal(active, data['active'][::20])
        with torch.no_grad():
            mu = model(torch.from_numpy(data['policy_legacy_observation']).float()).numpy()
        actor_error = float(abs(mu[active]-data['policy_actor_action'][active]).max())
        assert actor_error < 2e-6
        from teacher_sustain import SustainTeacher
        teacher = SustainTeacher([candidate(group[0]['blend_s'])]*2, geometry, 'cpu')
        labels = []
        for k in range(len(active)):
            frame = max(0, k*20-1)
            m = {name: torch.from_numpy(data[name][frame]).float() for name in
                ('gravity','ang_vel','wheel_pos','com','com_vel','force_z')}
            mock = SimpleNamespace(metrics=lambda: m, physics_steps=torch.round(torch.from_numpy(data['policy_physics_time'][k])/.001).long(),
                physics_dt=.001, state=SimpleNamespace(hold_origin=torch.from_numpy(data['policy_hold_origin'][k]),
                    return_time=torch.from_numpy(data['policy_return_time'][k])))
            labels.append(teacher(mock, torch.from_numpy(data['policy_time'][k])).numpy())
        labels = np.stack(labels)
        teacher_error = float(abs(labels[active]-data['policy_teacher_action'][active]).max())
        assert teacher_error < 2e-5, teacher_error
        expected = np.where(data['policy_takeover'][..., None], data['policy_teacher_action'], data['policy_actor_action'])
        np.testing.assert_array_equal(expected[active], data['policy_action'][active])
        expected = expected.clip(-1.,1.); expected[:10] = 0.
        np.testing.assert_array_equal(expected[active], data['actions'][::20][active])
        # 每个历史帧的末通道只能是站立命令0/1，不得包含旧回合进度。
        command = data['policy_observation'][..., 60::61]
        assert np.isin(command[active], [0.,1.]).all()
        if request['return_after_s'] is None:
            assert (command[active] == 1.).all()
        names = read(run/'reward_names.json')
        terms = data['policy_reward_terms']
        np.testing.assert_allclose(terms.sum(-1)[active], data['policy_reward'][active], atol=2e-6, rtol=1e-5)
        expected_balance = qualified.reshape(-1,20,2).sum(1) * .001 * 12.
        np.testing.assert_allclose(terms[..., names.index('balance')][active], expected_balance[active], atol=.02401, rtol=0)
        for name in ('hold_once','success_once','missed_hold','drift'):
            assert (terms[..., names.index(name)][active] == 0.).all()
        if request['return_after_s'] is None:
            for name in ('return_pose','return_attitude','recovery'):
                assert (terms[..., names.index(name)][active] == 0.).all()
        # 0.50秒新协议在原7秒以前与同教师旧记录逐值一致，隔离取消返回的作用。
        control_fields = 0
        if group[0]['blend_s'] == .5:
            with np.load(CONTROL/'trace_seed43_trigger_m05.npz') as z:
                for key in common:
                    np.testing.assert_array_equal(data[key][:7000], z[key][:7000], err_msg='pre-return '+key)
                    control_fields += 1
        checks.append(dict(trace=path.name, prefix_arrays=len(common), old_7s_control_arrays=control_fields,
            actor_max_error=actor_error, teacher_max_error=teacher_error, executed_action_error=0.))
    assert len(stats) == len(result['cases'])
    report = dict(cases=stats, actions=checks, physics_samples=sum(c['physics_samples'] for c in stats),
        independent_numpy_scipy_protocol=True, balance_integral_recomputed=True,
        old_actor_is_diagnostic_prefix_only=True, ppo_admission=False, ppo_updates=0)
    write(run/'verification.json', report)
    return report
