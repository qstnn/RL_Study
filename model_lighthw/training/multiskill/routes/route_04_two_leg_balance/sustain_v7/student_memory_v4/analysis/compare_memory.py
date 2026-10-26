"""只读比较实际自主状态与名义参考；影子教师查询不作为训练标签。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_memory import parent, read, write, sha
from teacher_sustain import SustainTeacher
from verify_sustain import streak


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def features(d):
    return [np.rad2deg(np.arctan2(d['gravity'][:, 0], -d['gravity'][:, 2])),
            np.linalg.norm(d['com_vel'], axis=-1), d['line_error'],
            (d['smooth_rms'] / [.02, .05, np.deg2rad(.2), .5]).max(-1)]


@torch.inference_mode()
def compare(run):
    torch.set_num_threads(4)
    paths = {
        'Nominal hybrid teacher': HERE.parent / 'runs/2026-10-22_sustain_30s_blend800/trace_seed43_blend800.npz',
        'Memory zero': HERE / 'runs/2026-10-23_zero_screen/trace_autonomous_seed43.npz',
        'Memory exposed': run / 'trace_autonomous_seed43.npz',
    }
    data = {name: load(path) for name, path in paths.items()}
    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex='col')
    rows = []
    reference = data['Nominal hybrid teacher']
    for name, raw in data.items():
        for slot in (0, 1):
            active = raw['active'][:, slot]
            d = {k: v[active, slot] for k, v in raw.items() if not k.startswith('policy_')}
            values = features(d)
            plot = d['t'] <= 6.
            for ax, value, label in zip(axes[:4, slot], values,
                                       ['Pitch (deg)', 'COM speed (m/s)', 'COM line error (m)', 'Actual HF / threshold']):
                ax.plot(d['t'][plot], value[plot], label=name, lw=.9)
                ax.set_ylabel(label)
                ax.grid(alpha=.25)
            valid = raw['policy_active'][:, slot] & (raw['policy_physics_time'][:, slot] <= 6.)
            axes[4, slot].plot(raw['policy_physics_time'][valid, slot], raw['actions'][::20, slot, 14:][valid].mean(-1), label=name, lw=.9)
            if name == 'Nominal hybrid teacher':
                continue
            n = min(len(d['t']), len(reference['t']))
            gravity = np.rad2deg(np.arccos(np.clip((d['gravity'][:n] * reference['gravity'][:n, slot]).sum(-1), -1, 1)))
            offset = (d['com'][:n] - d['com'][0]) - (reference['com'][:n, slot] - reference['com'][0, slot])
            departure = np.flatnonzero((gravity > 5.) | (np.linalg.norm(offset, axis=-1) > .03))
            force = d['force_z']
            pair = (force[:, 2:] > 5.).all(-1) & (abs(force[:, :2]) < 2.).all(-1) & (d['clearance'][:, :2].min(-1) >= .03)
            attitude = np.rad2deg(np.arccos(np.clip((d['gravity'] * d['target_gravity']).sum(-1), -1, 1)))
            gates = dict(pair=pair, speed=values[1] <= .12, angular_speed=np.linalg.norm(d['ang_vel'], axis=-1) <= .25,
                         com_line=values[2] <= .06, attitude=attitude <= 12., actual_hf=values[3] <= 1.)
            window = (d['phase'] == 2) & ~d['failed']
            without_hf = window & np.logical_and.reduce([v for k, v in gates.items() if k != 'actual_hf'])
            checkpoints = []
            for when in [2.5, 2.8, 3.2, 3.6, 4., 4.4, 5., float(d['t'][-1])]:
                if when > d['t'][-1] + .0001:
                    continue
                j = int(np.argmin(abs(d['t'] - when)))
                checkpoints.append(dict(time=float(d['t'][j]), pitch_deg=float(values[0][j]),
                    speed_m_s=float(values[1][j]), angular_speed_rad_s=float(np.linalg.norm(d['ang_vel'][j])),
                    com_line_m=float(values[2][j]), hf_ratio=float(values[3][j]), attitude_error_deg=float(attitude[j]),
                    rear_load_N=force[j, 2:].tolist(), front_clearance_m=d['clearance'][j, :2].tolist()))
            rows.append(dict(model=name, slot=slot, first_reference_departure_s=float(d['t'][departure[0]]) if len(departure) else None,
                longest_qualified_without_hf_s=float(streak(without_hf).max() * .001),
                hold_phase_samples=int(window.sum()), gate_pass_samples={k: int((v & window).sum()) for k, v in gates.items()},
                checkpoints=checkpoints))
    queries = []
    for name in ('Memory zero', 'Memory exposed'):
        raw = data[name]
        teacher = SustainTeacher([parent.candidate(.8)] * 2, read(paths[name].parent / 'geometry.json'), 'cpu')
        for k in range(len(raw['policy_time'])):
            j = max(0, k * 20 - 1)
            m = {key: torch.from_numpy(raw[key][j]) for key in ['gravity', 'ang_vel', 'wheel_pos', 'com', 'com_vel', 'force_z']}
            mock = SimpleNamespace(metrics=lambda: m,
                physics_steps=torch.round(torch.from_numpy(raw['policy_physics_time'][k]) / .001).long(), physics_dt=.001,
                state=SimpleNamespace(hold_origin=torch.from_numpy(raw['policy_hold_origin'][k]),
                                      return_time=torch.from_numpy(raw['policy_return_time'][k])))
            action = teacher(mock, torch.from_numpy(raw['policy_time'][k])).numpy()
            if k in [125, 140, 160, 180, 200, 220, 250]:
                for slot in (0, 1):
                    if raw['policy_active'][k, slot]:
                        queries.append(dict(model=name, slot=slot, time=float(raw['policy_physics_time'][k, slot]),
                            teacher_rear=action[slot, 14:].tolist(), student_rear=raw['policy_actor_action'][k, slot, 14:].tolist(),
                            leg_max_difference=float(abs(action[slot, :12] - raw['policy_actor_action'][k, slot, :12]).max())))
    for slot in (0, 1):
        axes[0, slot].set_title(f'Slot {slot}')
        for row, limit in [(1, .12), (2, .06), (3, 1.)]:
            axes[row, slot].axhline(limit, color='gray', linestyle='--', lw=.7)
        axes[4, slot].set_ylabel('Mean rear wheel action')
        axes[4, slot].set_xlabel('Physical time (s)')
        axes[4, slot].grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / 'comparison.png', dpi=150)
    plt.close(fig)
    result = dict(rows=rows, actual_state_shadow_queries=queries,
        departure_definition='same physical time gravity difference >5deg OR relative COM displacement difference >3cm',
        trace_hashes={str(p): sha(p) for p in paths.values()}, analysis_source_sha256=sha(Path(__file__)),
        limitation='Descriptive differences and unexecuted shadow teacher queries only; no causal claim, successful recovery label, or altered acceptance criterion')
    write(run / 'comparison.json', result)
    print({(r['model'], r['slot']): (r['first_reference_departure_s'], r['longest_qualified_without_hf_s']) for r in rows})
    print([q for q in queries if abs(q['time'] - 4.) < .001])


if __name__ == '__main__':
    compare(Path(sys.argv[1]).resolve())
