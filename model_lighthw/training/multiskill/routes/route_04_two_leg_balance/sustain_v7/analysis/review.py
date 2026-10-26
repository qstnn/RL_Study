"""持续诊断的图表与首失稳定位；独立于仿真执行来源，记录脚本及输入SHA。"""
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_sustain import read,write,sha,now


def review(run):
    result=read(run/'result.json')
    request=read(run/'request.json')
    verified=read(run/'verification.json')
    names=read(run/'reward_names.json')
    rows=[]
    fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
    for trace in sorted(result['trace_hashes']):
        assert sha(run/trace)==result['trace_hashes'][trace]
        with np.load(run/trace) as z:
            data={k:z[k] for k in z.files}
        for c in [r for r in result['cases'] if r['trace']==trace]:
            i=c['trace_env'];active=data['active'][:,i]
            d={k:v[active,i] for k,v in data.items() if not k.startswith('policy_')}
            t=d['t'];label=f"blend {c['blend_s']:.2f}s"
            pitch=np.rad2deg(np.arctan2(d['gravity'][:,0],-d['gravity'][:,2]))
            speed=np.linalg.norm(d['com_vel'],axis=-1)
            omega=np.linalg.norm(d['ang_vel'],axis=-1)
            ratios=d['smooth_rms']/np.array([.02,.05,np.deg2rad(.2),.5])
            hold=(t>c['capture_time']) & (c['capture_time']>=0)
            valid_hold=hold & ((t<c['first_break_time']) | (c['first_break_time']<0)) & d['stand_request']
            hold_ids=np.flatnonzero(valid_hold)
            def window(mask):
                if not mask.any():return None
                return dict(samples=int(mask.sum()),speed_mean=float(speed[mask].mean()),
                    speed_max=float(speed[mask].max()),ang_speed_max=float(omega[mask].max()),
                    hf_ratio_max=ratios[mask].max(0).tolist(),line_error_max=float(d['line_error'][mask].max()))
            first5=valid_hold & (t<=t[hold_ids[0]]+5.) if len(hold_ids) else valid_hold
            last5=valid_hold & (t>=t[hold_ids[-1]]-5.) if len(hold_ids) else valid_hold
            old_window=(t>=4.)&(t<7.)
            rear=d['force_z'][old_window,2:]
            capture=(t>=3.2)&((t<=c['capture_time']) if c['capture_time']>=0 else True)
            joint_peak=np.abs(d['torque']).max(axis=0)
            pc=data['policy_active'][:,i].astype(bool)
            terms=data['policy_reward_terms'][pc,i]
            sums={name:float(terms[:,j].sum()) for j,name in enumerate(names)}
            windows=[]
            if hold.any():
                for start in np.arange(c['capture_time'],float(t[-1]),5.):
                    mask=(t>=start)&(t<start+5)&d['stand_request']
                    if not mask.any():continue
                    windows.append(dict(start_s=float(start),samples=int(mask.sum()),
                        speed_max=float(speed[mask].max()),speed_mean=float(speed[mask].mean()),
                        ang_speed_max=float(omega[mask].max()),
                        hf_ratio_max=ratios[mask].max(0).tolist(),
                        line_error_max=float(d['line_error'][mask].max())))
            row=dict(trace=trace,slot=i,blend_s=c['blend_s'],capture_time_s=c['capture_time'],
                uninterrupted_after_capture_s=c['post_capture_s'],first_break_time_s=c['first_break_time'],
                first_break_bits=c['break_reason'],failed=c['failed'],reason=c['reason'],
                success=c['success'],max_leg_torque_Nm=float(joint_peak[:12].max()),
                max_wheel_torque_Nm=float(joint_peak[12:].max()),
                capture_peak_speed=float(speed[capture].max()) if capture.any() else None,
                capture_peak_ang_speed=float(omega[capture].max()) if capture.any() else None,
                qualified_hold_speed_max=float(speed[valid_hold].max()) if valid_hold.any() else None,
                qualified_hold_hf_ratio_max=ratios[valid_hold].max(0).tolist() if valid_hold.any() else None,
                hold_net_displacement_m=float(np.linalg.norm(d['com'][hold_ids[-1],:2]-d['com'][hold_ids[0],:2])) if len(hold_ids) else None,
                hold_first_5s=window(first5),hold_last_5s=window(last5),
                rear_support_bad_ms_4_7=int((rear<=5).any(-1).sum()),
                both_rear_unloaded_ms_4_7=int((rear<=5).all(-1).sum()),
                hold_windows=windows,reward_sums=sums)
            rows.append(row)
            for ax,y in zip(axes[:,i],(pitch,speed,omega,ratios.max(1))):
                ax.plot(t[::10],y[::10],label=label,linewidth=1)
                ax.axvline(7.,color='gray',linestyle=':',linewidth=.7)
                if c['capture_time']>=0:ax.axvline(c['capture_time'],color='green',alpha=.15)
                if c['first_break_time']>=0:ax.axvline(c['first_break_time'],color='red',alpha=.25)
    for i in range(2):
        axes[0,i].set_title(f'Rear support / slot {i}')
        axes[0,i].legend(fontsize=8)
        for row,limit in ((1,.12),(2,.25),(3,1.)):
            axes[row,i].axhline(limit,color='black',linestyle='--',linewidth=.8)
        axes[-1,i].set_xlabel('Physics time (s); dotted line: old automatic return at 7s')
    for ax,label in zip(axes[:,0],('Pitch (deg)','COM speed (m/s)','Body angular speed (rad/s)','Worst actual HF RMS / limit')):
        ax.set_ylabel(label)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('R4a continuous standing: privileged teacher takeover at 3.2s')
    fig.tight_layout()
    fig.savefig(run/'behavior.png',dpi=160);plt.close(fig)
    out=dict(at=now(),analysis_script_sha256=sha(Path(__file__)),input_trace_sha256=result['trace_hashes'],
        independent_verification_sha256=sha(run/'verification.json'),cases=rows,
        teacher_only=True,ppo_admission=False,request=request)
    write(run/'review.json',out)
    print([(r['blend_s'],r['slot'],r['uninterrupted_after_capture_s'],r['first_break_time_s'],r['success']) for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    review(p.parse_args().run.resolve())
