"""独立NumPy/SciPy长稳验收，另复算CPU动作和保持积分奖励。"""
from types import SimpleNamespace
import numpy as np
import torch
from pathlib import Path
from .common import read, write, sha, require
from .model import Features
from scipy.signal import butter, lfilter


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
        np.where((d['base_z'] < .18) | ((np.arctan2(d['gravity'][:, 0], -d['gravity'][:, 2]) < np.deg2rad(-35.)) | (np.arctan2(d['gravity'][:, 0], -d['gravity'][:, 2]) > np.deg2rad(110.)) | (abs(d['gravity'][:, 1]) > np.sin(np.deg2rad(35.)))), 2, 0))
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
    pitch = np.float32(geometry['selected']['pitch_rad']) * amplitude
    target = np.stack((np.sin(pitch), np.zeros(n), -np.cos(pitch)), -1)
    np.testing.assert_allclose(target, d['target_gravity'], atol=3e-6, rtol=0)
    np.testing.assert_array_equal(phase, d['phase'])
    np.testing.assert_array_equal(~returning, d['stand_request'])
    a, b = d['wheel_pos'][:, 0, :2], d['wheel_pos'][:, 1, :2]
    line = b-a
    u = np.sum((d['com'][:, :2]-a)*line, -1) / np.maximum(np.sum(line*line, -1), 1e-8)
    line_error = norm(d['com'][:, :2] - (a+np.clip(u, 0., 1.)[:, None]*line))
    np.testing.assert_allclose(line_error, d['line_error'], atol=1e-5, rtol=1e-4)
    clear = d['wheel_pos'][:, :, 2] - .087501
    np.testing.assert_allclose(clear, d['clearance'], atol=1e-6, rtol=0)
    pair = (force[:, :2] > 5.).all(-1) & (abs(force[:, 2:]) < 2.).all(-1) & (clear[:, 2:].min(-1) >= .03)
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
    qualified = (phase == 2) & pair & linear & angular & (line_error <= .06) & (angle <= np.deg2rad(5.)) & smooth & ~physical & ~outside & (start >= 0.)
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
    gates = [pair, linear, angular, line_error <= .06, angle <= np.deg2rad(5.), smooth]
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


def verify(run, stage):
    run = Path(run); require(run); folder = run/stage
    result = read(folder/'result.json'); geometry = read(run/'geometry.json')
    if not result['complete']: raise ValueError('Partial smoke cannot count as behavior success')
    if sha(folder/'trace.npz') != result['trace_sha256']: raise ValueError('Trace changed')
    with np.load(folder/'trace.npz') as z: data = {k: z[k] for k in z.files}
    if not all(np.isfinite(v).all() for v in data.values()): raise ValueError('Nonfinite trace')
    stats = []
    for case in result['cases']:
        slot = case['trace_env']; active = data['active'][:, slot]
        trace = {k: v[active, slot] for k, v in data.items() if not k.startswith('policy_')}
        if not (trace['mode'] == 0).all(): raise ValueError('Wrong support pair')
        row, _ = analyse(trace, case, geometry, result['hold_seconds'])
        torque = abs(trace['torque'])
        if (torque[:, :12] > 48.001).any() or (torque[:, 12:] > 24.001).any():
            raise ValueError('Actuator effort limit exceeded')
        row['candidate'] = case['candidate']; stats.append(row)
    # Check real executed labels; no hypothetical teacher action is added to the dataset.
    active = data['policy_active']; expected = data['policy_action'].clip(-1, 1).copy()
    expected[:10] = 0
    np.testing.assert_allclose(expected[active], data['actions'][::20][active], atol=1e-7, rtol=0)
    rows = sorted(result['cases'], key=lambda c: c['trace_env'])
    features = Features(len(rows), 'cpu', [5.]*len(rows), [c['blend_s'] for c in rows])
    model_error = feature_error = 0.
    if result['teacher_off']:
        from .learning import load_actor
        actor, _ = load_actor(run/'student.pt')
    with torch.no_grad():
        for k, raw in enumerate(data['policy_raw']):
            clear = np.ones(len(rows), bool) if k == 0 else (active[k-1] & ~active[k])
            obs = features(torch.from_numpy(raw), torch.from_numpy(clear))
            if active[k].any():
                feature_error = max(feature_error, float(abs(obs.numpy()-data['policy_observation'][k])[active[k]].max()))
                if result['teacher_off']:
                    model_error = max(model_error, float(abs(actor(obs).numpy()-data['policy_action'][k])[active[k]].max()))
    if feature_error > 3e-5 or model_error > 3e-5:
        raise ValueError(f'Feature/action reconstruction mismatch: {feature_error}, {model_error}')
    report = dict(teacher_off=result['teacher_off'], cases=stats, passes=sum(r['observation_pass'] for r in stats),
        case_count=len(stats), hold_seconds=result['hold_seconds'], seed=result['seed'],
        trace_sha256=sha(folder/'trace.npz'), independent_physics_verified=True,
        feature_error=feature_error, actor_error=model_error, executed_actions_verified=True,
        physics_samples=sum(r['physics_samples'] for r in stats), ppo_admission=False)
    write(folder/'verification.json', report)
    return report

