"""按真实状态比较学生与名义示范，保留首偏离及分项门证据。"""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_student import read,write,sha


def review(run):
    reference=HERE.parent/'runs/2026-10-22_sustain_30s_blend800/trace_seed43_blend800.npz'
    with np.load(reference) as z:
        teacher={k:z[k] for k in z.files}
    with np.load(run/'trace_seed43.npz') as z:
        student={k:z[k] for k in z.files}
    fig,axes=plt.subplots(4,2,figsize=(13,10),sharex='col')
    rows=[]
    for slot in (0,1):
        s={k:v[student['active'][:,slot],slot] for k,v in student.items() if not k.startswith('policy_')}
        limit=s['t'][-1]
        t={k:v[(teacher['t'][:,slot]<=limit)&teacher['active'][:,slot],slot] for k,v in teacher.items() if not k.startswith('policy_')}
        def features(d):
            return [np.rad2deg(np.arctan2(d['gravity'][:,0],-d['gravity'][:,2])),
                np.linalg.norm(d['com_vel'],axis=-1),d['line_error'],
                (d['smooth_rms']/[.02,.05,np.deg2rad(.2),.5]).max(-1)]
        for data,label in ((t,'Successful hybrid reference'),(s,'Student execution')):
            for ax,values,name in zip(axes[:,slot],features(data),('Pitch (deg)','COM speed (m/s)','COM line error (m)','Actual HF / threshold')):
                ax.plot(data['t'],values,label=label,linewidth=1.)
                ax.set_ylabel(name);ax.grid(alpha=.25)
        axes[0,slot].set_title(f'Slot {slot}')
        for row,threshold in ((1,.12),(2,.06),(3,1.)):
            axes[row,slot].axhline(threshold,color='gray',linestyle='--',linewidth=.8)
        axes[-1,slot].set_xlabel('Physical time (s)')
        n=min(len(s['t']),len(t['t']))
        gravity_error=np.rad2deg(np.arccos(np.clip((s['gravity'][:n]*t['gravity'][:n]).sum(-1),-1,1)))
        com_error=np.linalg.norm((s['com'][:n]-s['com'][0])-(t['com'][:n]-t['com'][0]),axis=-1)
        departed=np.flatnonzero((gravity_error>5)|(com_error>.03))
        checkpoints=[]
        for when in [1.,2.,3.2,4.,5.,6.,float(limit)]:
            if when>limit+.001:continue
            i=int(np.argmin(abs(s['t']-when)))
            checkpoints.append(dict(t=float(s['t'][i]),pitch_deg=float(features(s)[0][i]),
                speed_m_s=float(features(s)[1][i]),line_error_m=float(s['line_error'][i]),
                attitude_error_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(s['gravity'][i],s['target_gravity'][i]),-1,1)))),
                rear_load_N=s['force_z'][i,2:].tolist(),front_clearance_m=s['clearance'][i,:2].tolist(),
                max_hf_ratio=float(features(s)[3][i])))
        # 这里只描述轨迹分歧，不把阈值越界当作因果证明。
        rows.append(dict(slot=slot,first_reference_departure_s=float(s['t'][departed[0]]) if len(departed) else None,
            departure_definition='gravity angle >5deg OR relative COM difference >3cm at same physics time',
            checkpoints=checkpoints))
    axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(run/'behavior.png',dpi=150);plt.close(fig)
    report=dict(reference=str(reference),reference_sha256=sha(reference),student_trace_sha256=sha(run/'trace_seed43.npz'),
        rows=rows,caveat='Reference divergence is descriptive; this is not a causal intervention or robustness result')
    write(run/'behavior_analysis.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    review(Path(sys.argv[1]).resolve())
