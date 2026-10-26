"""Small static overview; full CSV/JSONL/TensorBoard remain the authoritative data."""
import csv
import json
from pathlib import Path


def make_plots(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    made=[]
    for path in sorted((folder/'runs').glob('*/training_metrics.csv')):
        rows=list(csv.DictReader(path.open()))
        if not rows: continue
        fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
        x=[int(r['iteration']) for r in rows]
        specs=[('tilt_deg','Mean tilt (deg)'),('height','Mean base height (m)'),
               ('four_wheel','Four-wheel support fraction'),('torque_saturation','Torque saturation fraction'),
               ('terminated','Terminations per update'),('speed','Mean horizontal speed (m/s)')]
        for axis,(key,title) in zip(axes.flat,specs):
            y=[float(r[key]) if r.get(key) not in ('',None) else float('nan') for r in rows]
            axis.plot(x,y); axis.set_title(title);axis.set_xlabel('PPO update');axis.grid(alpha=.3)
        fig.suptitle('Training diagnostics - averages do not establish behavior acceptance')
        target=path.parent/'training_overview.png';fig.savefig(target,dpi=130);plt.close(fig)
        made.append(str(target.relative_to(folder)))
        updates=[]
        for line in (path.parent/'ppo_updates.jsonl').read_text().splitlines() if (path.parent/'ppo_updates.jsonl').exists() else []:
            try: updates.append(json.loads(line))
            except ValueError: continue  # partial last line remains in the original file for diagnosis
        if updates:
            fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
            xx=[r['iteration'] for r in updates]
            for axis,key,title in zip(axes.flat,('episode_reward_last100','episode_length_steps_last100','learning_rate','policy_noise_std'),
                                     ('Completed-episode reward (last 100)','Completed-episode length (steps, last 100)','Learning rate','Policy noise std')):
                axis.plot(xx,[r.get(key) if r.get(key) is not None else float('nan') for r in updates]);axis.set_title(title);axis.set_xlabel('PPO update');axis.grid(alpha=.3)
            target=path.parent/'ppo_overview.png';fig.savefig(target,dpi=130);plt.close(fig)
            made.append(str(target.relative_to(folder)))
    for path in sorted((folder/'runs').glob('*/*_trace.csv')):
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'payload_kg' not in reader.fieldnames:
                continue
            rows = list(reader)
        if not rows:
            continue
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
        x = [float(r['t']) for r in rows]
        axes[0, 0].step(x, [float(r['payload_kg']) for r in rows], where='post')
        axes[0, 0].set_title('Payload (kg) - keyboard changes')
        for key in ('roll_deg', 'pitch_deg'):
            axes[0, 1].plot(x, [float(r[key]) for r in rows], label=key)
        axes[0, 1].set_title('Body attitude (deg)'); axes[0, 1].legend()
        axes[1, 0].plot(x, [float(r['height']) for r in rows])
        axes[1, 0].axhline(.416, linestyle='--', color='gray')
        axes[1, 0].set_title('Body height (m)')
        axes[1, 1].plot(x, [float(r['four_wheel']) for r in rows])
        axes[1, 1].set_title('Four-wheel support (0 or 1)')
        joints = [k for k in rows[0] if k.startswith('torque_') and k not in ('torque_peak', 'torque_saturation', 'torque_clipped')]
        for axis, wheel, title in ((axes[2, 0], False, 'Leg torque (Nm)'), (axes[2, 1], True, 'Wheel torque (Nm)')):
            for key in joints:
                if ('wheel' in key) == wheel:
                    axis.plot(x, [float(r[key]) for r in rows], label=key.removeprefix('torque_'))
            axis.set_title(title)
            axis.legend(fontsize=6, ncol=2)
        for axis in axes.flat:
            axis.set_xlabel('Trial simulation time (s)'); axis.grid(alpha=.3)
        fig.suptitle('Interactive payload response - no fixed-load pass/fail claim')
        target = path.with_name(path.stem.removesuffix('_trace')+'_payload_response.png')
        fig.savefig(target, dpi=130); plt.close(fig)
        made.append(str(target.relative_to(folder)))
    return made
