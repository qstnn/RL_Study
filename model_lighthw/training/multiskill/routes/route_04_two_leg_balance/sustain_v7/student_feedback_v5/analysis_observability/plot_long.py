"""Export actual 1 kHz post-capture traces; no averaging or simulator use."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SUSTAIN=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=SUSTAIN/'student_feedback_v5/structured_v1/runs/2026-10-23_full_01_60s')
    a=p.parse_args();run=a.run.resolve()
    if not (run/'session.json').exists() or json.loads((run/'session.json').read_text())['status']!='completed':
        print('PENDING: completed long run required');return 2
    path=run/'trace_autonomous_seed43.npz'
    with np.load(path) as z:d={k:z[k] for k in z.files if not k.startswith('policy_')}
    fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
    summaries=[]
    for slot in (0,1):
        valid=d['active'][:,slot].astype(bool);capture=d['capture_time'][valid,slot]
        ct=float(capture[capture>=0][0]);valid &= d['t'][:,slot]>=ct
        t=d['t'][valid,slot]-ct
        speed=np.linalg.norm(d['com_vel'][valid,slot],axis=-1)
        wheel=d['wheel_pos'][valid,slot];com=d['com'][valid,slot]
        lateral=wheel[:,3,:2]-wheel[:,2,:2]
        forward=np.stack((lateral[:,1],-lateral[:,0]),-1)
        forward/=np.maximum(np.linalg.norm(forward,axis=-1,keepdims=True),1e-6)
        signed=((com[:,:2]-wheel[:,2:,:2].mean(1))*forward).sum(-1)
        distance=d['line_error'][valid,slot]
        f=d['force_z'][valid,slot]
        hf=d['smooth_rms'][valid,slot]/[.02,.05,np.deg2rad(.2),.5]
        ax=axes[0,slot]
        ax.plot(t,speed,lw=.8,rasterized=True,label='Actual COM speed (all 1 kHz samples)')
        ax.axhline(.12,color='red',ls='--',lw=1,label='Acceptance: 0.12 m/s')
        ax.text(.02,.78,f'Actual peak: {speed.max():.6f} m/s',transform=ax.transAxes,fontsize=8)
        ax.set_ylabel('COM speed (m/s)');ax.set_ylim(bottom=0);ax.set_title(f'Slot {slot}; capture at {ct:.3f} s')
        ax=axes[1,slot]
        ax.plot(t,distance*1000,lw=.9,rasterized=True,label='Finite support-segment distance')
        ax.plot(t,signed*1000,lw=.7,rasterized=True,label='Signed forward gap')
        ax.axhline(60,color='red',ls='--',lw=1,label='Distance acceptance: 60 mm')
        ax.text(.02,.78,f'Actual distance peak: {distance.max()*1000:.3f} mm',transform=ax.transAxes,fontsize=8)
        ax.set_ylabel('COM support gap (mm)')
        ax=axes[2,slot]
        for i,name in enumerate(('FR airborne','FL airborne','RR support','RL support')):
            ax.plot(t,f[:,i],lw=.7,rasterized=True,label=name)
        ax.axhline(5,color='red',ls='--',lw=.9,label='Rear >5 N')
        ax.axhline(2,color='gray',ls=':',lw=.9,label='Front |Fz|<2 N')
        ax.text(.02,.74,f'Rear minimum: {f[:,2:].min():.3f} N\nFront absolute peak: {abs(f[:,:2]).max():.3f} N',transform=ax.transAxes,fontsize=8)
        ax.set_ylabel('Signed wheel load Fz (N)')
        ax=axes[3,slot]
        for i,name in enumerate(('COM velocity / 0.02 m/s','Angular velocity / 0.05 rad/s','Leg position / 0.2 deg','Wheel speed / 0.5 rad/s')):
            ax.plot(t,hf[:,i],lw=.7,rasterized=True,label=name)
        ax.axhline(1,color='red',ls='--',lw=1,label='All four ratios <=1')
        ax.text(.02,.78,f'Actual maximum ratio: {hf.max():.6f}',transform=ax.transAxes,fontsize=8)
        ax.set_ylabel('Actual high-pass RMS / limit');ax.set_xlabel('Time since 2 s capture milestone (s)');ax.set_ylim(bottom=0)
        summaries.append(dict(slot=slot,capture_time_s=ct,post_capture_samples=len(t),duration_s=float(t[-1]),
                              actual_speed_peak_m_s=float(speed.max()),finite_segment_distance_peak_m=float(distance.max()),
                              rear_load_min_N=float(f[:,2:].min()),front_abs_load_peak_N=float(abs(f[:,:2]).max()),
                              actual_hf_ratio_peak=float(hf.max())))
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=6,loc='upper right');ax.set_xlim(left=0)
    fig.suptitle('R4a structured feedback: teacher off, native physics, every 1 kHz sample',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(HERE/'actual_60s.png',dpi=180)
    fig.savefig(HERE/'actual_60s.pdf',dpi=180)
    plt.close(fig)
    (HERE/'actual_60s_plot.json').write_text(json.dumps(dict(trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=summaries,
        note='Raw pointwise traces and peaks, no time averaging; signed gap is not a substitute for finite-segment distance.'),indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
