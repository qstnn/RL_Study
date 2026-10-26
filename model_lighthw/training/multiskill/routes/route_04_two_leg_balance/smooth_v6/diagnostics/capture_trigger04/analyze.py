"""接管诊断的门条件分解、终末四轮与切换跃变量；只分析，不收集数据或训练。"""
from pathlib import Path
import argparse
import os

import numpy as np
from compare import HERE, BASELINE, DEFAULT_RUN, arrays, read, write, sha, check_comparison_sources, successful
from verify_smooth import streak


def longest(mask):
    return float(streak(mask).max() * .001)


def analyze(run):
    check_comparison_sources(run)
    assert read(run / 'session.json')['status'] == 'completed'
    result, verified = read(run / 'result.json'), read(run / 'comparison_verification.json')
    assert verified['verification']['protocol_recomputed'] and verified['verification']['jitter_recomputed']
    cfg = read(run / 'contract.json')
    limits = np.array([cfg['jitter_filter']['limits'][k] for k in (
        'com_velocity_m_s', 'angular_velocity_rad_s', 'leg_position_rad', 'wheel_velocity_rad_s')])
    rows = []
    for name, digest in result['trace_hashes'].items():
        assert sha(run / name) == digest
        data = arrays(run / name)
        for case in (c for c in result['cases'] if c['trace'] == name):
            slot = case['trace_env']
            active = data['active'][:, slot]
            d = {k: a[active, slot] for k, a in data.items() if not k.startswith('policy_')}
            t = d['task_t']
            speed = np.linalg.norm(d['com_vel'], axis=-1)
            angular = np.linalg.norm(d['ang_vel'], axis=-1)
            angle = np.rad2deg(np.arccos(np.clip((d['gravity'] * d['target_gravity']).sum(-1), -1, 1)))
            tilt = np.rad2deg(np.arccos(np.clip(-d['gravity'][:, 2], -1, 1)))
            drift = np.linalg.norm(d['com'][:, :2] - d['origin'], axis=-1)
            failure = ((d['t'] > .15) & (d['body_force'] > 5)) | (d['base_z'] < .18) | (tilt > 80) | (drift > 1.5)
            live = ~np.maximum.accumulate(failure)
            four = (d['force_z'] > 5).all(-1)
            rear = (d['force_z'][:, 2:] > 5).all(-1)
            unloaded = (abs(d['force_z'][:, :2]) < 2).all(-1)
            clearance = (d['clearance'][:, :2] >= .03).all(-1)
            hf = d['smooth_rms'] <= limits
            gates = dict(rear_support=rear, front_unloaded=unloaded, front_clearance=clearance,
                         com_line=d['line_error'] <= .06, attitude=angle <= 12,
                         com_speed=speed <= .12, angular_speed=angular <= .25,
                         hf_com=hf[:, 0], hf_angular=hf[:, 1], hf_legs=hf[:, 2], hf_wheels=hf[:, 3])
            hold = (t >= 4) & (t < 7) & live
            physical = np.logical_and.reduce([v for k, v in gates.items() if not k.startswith('hf_')])
            accepted = hold & physical & hf.all(-1)
            lengths = streak(accepted)
            end = int(lengths.argmax())
            selected = np.zeros(len(t), bool)
            if lengths[end]:
                selected[end - lengths[end] + 1:end + 1] = True
            qualified = np.flatnonzero(lengths >= 2000)
            after = hold & (np.arange(len(t)) >= qualified[0]) if len(qualified) else np.zeros(len(t), bool)
            terminal = (t >= 10) & live & four & (speed <= .12) & (angular <= .25)
            terminal &= (tilt < 12) & (abs(d['base_z'] - .416) <= .07) & hf.all(-1)
            switch = case['switch_s']
            step, tick = round(switch / .02), round(switch / .001) - 1
            # tick为最后一个自主物理帧，step为随后首个教师控制步；二者间无reset。
            jump = data['policy_action'][step, slot] - data['actions'][tick, slot]
            changes = data['leg_targets'][tick + 1, slot] - data['leg_targets'][tick, slot]
            row = {k: case[k] for k in ('seed', 'trace_env', 'switch_s', 'trigger_pitch_deg', 'duration_s', 'best_hold',
                'best_recovery', 'held', 'failed', 'success', 'timeout', 'reason')}
            row.update(full_success=successful(case), strict_hold_recomputed_s=float(lengths.max() * .001),
                longest_hold_without_hf_s=longest(hold & physical),
                best_pair_s=longest(rear & unloaded & clearance & live),
                terminal_four_stability_without_prior_hold_s=longest(terminal),
                hold_hf_rejected_samples=int((hold & physical & ~hf.all(-1)).sum()),
                hold_rear_support_invalid_ms=int((hold & ~rear).sum()),
                hold_both_rear_below_5N_ms=int((hold & (d['force_z'][:, 2:] <= 5).all(-1)).sum()),
                hold_gate_pass_fraction={k: float(v[hold].mean()) if hold.any() else None for k, v in gates.items()},
                hold_last_violation_s={k: float(t[hold & ~v][-1]) if (hold & ~v).any() else None for k, v in gates.items()},
                hold_with_one_gate_removed_s={k: longest(hold & np.logical_and.reduce([v for key, v in gates.items() if key != k])) for k in gates},
                post_held_samples=int(after.sum()),
                post_held_hf_violations=int((after & ~hf.all(-1)).sum()) if len(qualified) else None,
                stable_start_s=float(t[selected][0]) if selected.any() else None,
                stable_end_s=float(t[selected][-1]) if selected.any() else None,
                stable_max_speed_m_s=float(speed[selected].max()) if selected.any() else None,
                stable_net_drift_m=float(np.linalg.norm(d['com'][selected][-1, :2] - d['com'][selected][0, :2])) if selected.any() else None,
                stable_max_hf_rms=d['smooth_rms'][selected].max(0).tolist() if selected.any() else None,
                switch_state=dict(task_s=float(t[tick]), attitude_error_deg=float(angle[tick]),
                    com_speed_m_s=float(speed[tick]), angular_speed_rad_s=float(angular[tick]),
                    line_error_m=float(d['line_error'][tick]), force_z=d['force_z'][tick].tolist(),
                    max_leg_action_jump=float(abs(jump[:12]).max()), max_wheel_action_jump=float(abs(jump[12:]).max()),
                    max_leg_target_jump_rad=float(abs(changes).max())),
                first_physical_failure_s=float(d['t'][np.flatnonzero(failure)[0]]) if failure.any() else None)
            capture = (t >= switch) & (t < 7) & live
            row['capture_peak_com_speed_m_s'] = float(speed[capture].max()) if capture.any() else None
            row['capture_peak_angular_speed_rad_s'] = float(angular[capture].max()) if capture.any() else None
            row['hold_peak_hf_rms'] = d['smooth_rms'][hold].max(0).tolist() if hold.any() else None
            row['body_force_max_N'] = float(d['body_force'].max())
            # 教师读取的是控制前一物理帧；重建触发时刻，避免把接管误当捕获反馈已全量启用。
            previous = np.maximum(0, 20 * np.arange(len(data['policy_time'])) - 1)
            gravity = data['gravity'][previous, slot]
            angular_velocity = data['ang_vel'][previous, slot]
            pitch = np.arctan2(gravity[:, 0], -gravity[:, 2])
            derivative = -np.cross(angular_velocity, gravity)
            rate = (-gravity[:, 2] * derivative[:, 0] + gravity[:, 0] * derivative[:, 2]) / np.maximum(
                gravity[:, 0] ** 2 + gravity[:, 2] ** 2, 1e-6)
            policy_t = data['policy_time'][:, slot]
            policy_active = data['policy_active'][:, slot]
            fire = np.flatnonzero(policy_active & (policy_t >= 1) & (pitch <= np.deg2rad(case['trigger_pitch_deg'])))
            row['teacher_capture_trigger_s'] = float(policy_t[fire[0]]) if len(fire) else None
            row['teacher_blend_complete_s'] = row['teacher_capture_trigger_s'] + result['base_candidate']['blend_s'] if len(fire) else None
            row['teacher_blend_completed_before_termination'] = bool(len(fire) and t[-1] >= row['teacher_blend_complete_s'])
            if len(fire):
                trigger_tick = previous[fire[0]]
                row['capture_trigger_state'] = dict(
                    pitch_deg=float(np.rad2deg(pitch[fire[0]])), pitch_rate_rad_s=float(rate[fire[0]]),
                    com_speed_m_s=float(np.linalg.norm(data['com_vel'][trigger_tick, slot])),
                    angular_speed_rad_s=float(np.linalg.norm(data['ang_vel'][trigger_tick, slot])),
                    line_error_m=float(data['line_error'][trigger_tick, slot]),
                    force_z=data['force_z'][trigger_tick, slot].tolist(),
                    front_clearance_m=data['clearance'][trigger_tick, slot, :2].tolist())
            else:
                row['capture_trigger_state'] = None
            row['pitch_rate_at_switch_rad_s'] = float(rate[step])
            row['recorded_first_pitch_threshold_crossings_s'] = {}
            for threshold in (-20, -10, -5):
                crossings = np.flatnonzero(policy_active & (policy_t >= 1) & (pitch <= np.deg2rad(threshold)))
                row['recorded_first_pitch_threshold_crossings_s'][str(threshold)] = float(policy_t[crossings[0]]) if len(crossings) else None
            assert abs(row['strict_hold_recomputed_s'] - case['best_hold']) < .002
            row['eligible_for_future_dataset'] = row['full_success'] and row['post_held_hf_violations'] == 0
            # 仅报告可采集数量；本入口不导出训练数据，不将失败或Actor前缀标成示范。
            row['eligible_actual_suffix_frames'] = int((data['policy_active'][:, slot] & data['policy_takeover'][:, slot]).sum()) if row['eligible_for_future_dataset'] else 0
            rows.append(row)
    for row in rows:
        reference = next(r for r in rows if r['trigger_pitch_deg'] == -20. and r['trace_env'] == row['trace_env'])
        row['hold_gain_vs_m20_s'] = row['strict_hold_recomputed_s'] - reference['strict_hold_recomputed_s']
        row['stable_start_advance_vs_m20_s'] = (reference['stable_start_s'] - row['stable_start_s']
            if row['stable_start_s'] is not None and reference['stable_start_s'] is not None else None)
    reference_trace = arrays(run / next(c['trace'] for c in result['cases'] if c['trigger_pitch_deg'] == -20.))
    for name in result['trace_hashes']:
        current = arrays(run / name)
        for case in (c for c in result['cases'] if c['trace'] == name):
            slot = case['trace_env']
            row = next(r for r in rows if r['trigger_pitch_deg'] == case['trigger_pitch_deg'] and r['trace_env'] == slot)
            count = min(len(current['policy_time']), len(reference_trace['policy_time']))
            changed = (current['policy_action'][:count, slot] != reference_trace['policy_action'][:count, slot]).any(-1)
            changed &= current['policy_active'][:count, slot] & reference_trace['policy_active'][:count, slot]
            first = np.flatnonzero(changed)
            row['first_action_difference_vs_m20_s'] = float(current['policy_time'][first[0], slot]) if len(first) else None
    pairs = []
    for trigger in sorted({r['trigger_pitch_deg'] for r in rows}):
        group = [r for r in rows if r['trigger_pitch_deg'] == trigger]
        assert {r['trace_env'] for r in group} == {0, 1}
        pairs.append(dict(trigger_pitch_deg=trigger,
            full_successes=sum(r['full_success'] for r in group),
            minimum_hold_s=min(r['strict_hold_recomputed_s'] for r in group),
            both_slots_eligible=all(r['eligible_for_future_dataset'] for r in group)))
    eligible_pairs = [p for p in pairs if p['both_slots_eligible']]
    selected_pair = max(eligible_pairs, key=lambda p: p['minimum_hold_s']) if eligible_pairs else None
    summary = dict(cases=sorted(rows, key=lambda c: (c['trigger_pitch_deg'], c['trace_env'])),
        pairs=pairs, recommended_next_teacher_trigger_deg=selected_pair['trigger_pitch_deg'] if selected_pair else None,
        diagnostic_reference_trigger_deg=max(pairs, key=lambda p: p['minimum_hold_s'])['trigger_pitch_deg'],
        teacher_selection_rule='Largest minimum hold among complete two-slot pairs with no post-held HF violations; no autonomous admission.',
        full_successes=sum(r['full_success'] for r in rows),
        eligible_future_suffix_frames=sum(r['eligible_actual_suffix_frames'] for r in rows),
        collected_frames=0, new_training_runs=0, ppo_updates=0, autonomous_ppo_admission=False,
        verified_physics_samples=verified['verification']['physics_samples'],
        source_hashes={name: sha(run / name) for name in ('result.json', 'comparison_verification.json', 'request.json')},
        analysis_sha256=sha(Path(__file__).resolve()),
        note='Gate exclusions describe recorded trajectories only; they do not change acceptance.')
    write(run / 'review.json', summary)
    plot(run, result, limits)
    return summary


def plot(run, result, limits):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/lighthw_smooth_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
    traces = [(BASELINE / 'trace_seed43.npz', 'Autonomous baseline', 'grey')]
    traces += [(run / name, f'Capture trigger {next(c["trigger_pitch_deg"] for c in result["cases"] if c["trace"] == name):.0f} deg', None)
               for name in sorted(result['trace_hashes'], key=lambda name: next(
                   c['trigger_pitch_deg'] for c in result['cases'] if c['trace'] == name))]
    for path, label, color in traces:
        z = arrays(path)
        for i in (0, 1):
            use = z['active'][:, i]
            t = z['task_t'][use, i]
            f = z['force_z'][use, i]
            pair = (f[:, 2:] > 5).all(-1) & (abs(f[:, :2]) < 2).all(-1) & (z['clearance'][use, i, :2] >= .03).all(-1)
            angle = np.rad2deg(np.arccos(np.clip((z['gravity'][use, i] * z['target_gravity'][use, i]).sum(-1), -1, 1)))
            values = (angle, np.linalg.norm(z['com_vel'][use, i], axis=-1),
                      np.linalg.norm(z['ang_vel'][use, i], axis=-1), z['line_error'][use, i],
                      pair.astype(float), (z['smooth_rms'][use, i] / limits).max(-1))
            for j, y in enumerate(values):
                axes[j, i].plot(t, y, label=label, color=color, lw=.9, alpha=.8)
    labels = ('Attitude error (deg)', 'COM speed (m/s)', 'Angular speed (rad/s)',
              'COM line error (m)', 'Rear pair + clearance', 'Max HF RMS / limit')
    for i in (0, 1):
        axes[0, i].set_title(f'Slot {i}, fixed seed 43')
        for j, label in enumerate(labels):
            axes[j, i].set_ylabel(label)
            axes[j, i].grid(alpha=.2)
            axes[j, i].axvspan(4, 7, color='green', alpha=.07)
            axes[j, i].axvspan(10, 12, color='blue', alpha=.04)
        for j, limit in ((0, 12), (1, .12), (2, .25), (3, .06), (5, 1)):
            axes[j, i].axhline(limit, color='black', ls='--', lw=.8)
        axes[0, i].legend(fontsize=8)
        axes[-1, i].set_xlabel('Task time (s)')
    fig.suptitle('R4a fourth Actor: capture trigger, fixed 3.2 s takeover')
    fig.tight_layout()
    fig.savefig(run / 'behavior.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, nargs='?', default=DEFAULT_RUN)
    result = analyze(parser.parse_args().run.resolve())
    for c in result['cases']:
        print(c['trigger_pitch_deg'], c['trace_env'], 'hold', c['best_hold'], 'without HF', c['longest_hold_without_hf_s'],
              'success', c['full_success'], 'end', c['duration_s'], 'reason', c['reason'])
    print('Eligible future suffix frames:', result['eligible_future_suffix_frames'], '; actually collected: 0; PPO: 0')
