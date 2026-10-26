"""短段采集/纯学生的分项物理图与交接状态；不改变验收。"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_segments import read,write,sha


def review(run):
    result=read(run/'result.json');rows=[]
    for name in sorted({c['trace'] for c in result['cases'] if c['seed']==43}):
        with np.load(run/name) as z:data={k:z[k] for k in z.files}
        fig,axes=plt.subplots(5,2,figsize=(13,11),sharex='col')
        for slot in (0,1):
            active=data['active'][:,slot]
            d={k:v[active,slot] for k,v in data.items() if not k.startswith('policy_')}
            t=d['t'];norm=lambda x:np.linalg.norm(x,axis=-1)
            pitch=np.rad2deg(np.arctan2(d['gravity'][:,0],-d['gravity'][:,2]))
            desired=np.rad2deg(np.arctan2(d['target_gravity'][:,0],-d['target_gravity'][:,2]))
            hf=(d['smooth_rms']/[.02,.05,np.deg2rad(.2),.5]).max(-1)
            for ax,values,label in zip(axes[:4,slot],[pitch,norm(d['com_vel']),d['line_error'],hf],
                                      ['Pitch (deg)','COM speed (m/s)','COM line error (m)','Actual HF / threshold']):
                ax.plot(t,values,lw=1);ax.set_ylabel(label);ax.grid(alpha=.25)
            axes[0,slot].plot(t,desired,'--',lw=.8,label='Reference');axes[0,slot].set_title(f'{name}: slot {slot}')
            for row,limit in [(1,.12),(2,.06),(3,1.)]:axes[row,slot].axhline(limit,color='gray',linestyle='--',lw=.7)
            valid=data['policy_active'][:,slot];pt=data['policy_physics_time'][valid,slot]
            mask=data['policy_takeover'][valid,slot]
            axes[4,slot].step(pt,mask.astype(int),where='post');axes[4,slot].set_yticks([0,1],['Student','Teacher'])
            axes[4,slot].set_xlabel('Physical time (s)');axes[4,slot].set_ylim(-.1,1.1)
            switch=np.flatnonzero(np.diff(np.r_[False,mask]))
            transitions=[]
            for k in switch:
                j=max(0,k*20-1)
                transitions.append(dict(physical_time=float(pt[k]),teacher=bool(mask[k]),
                    speed_m_s=float(norm(d['com_vel'])[j]),pitch_deg=float(pitch[j]),
                    rear_teacher=data['policy_teacher_action'][k,slot,14:].tolist(),
                    rear_student=data['policy_actor_action'][k,slot,14:].tolist()))
            rows.append(dict(trace=name,slot=slot,transitions=transitions,
                terminal_t=float(t[-1]),max_body_force_N=float(d['body_force'].max()),
                final_hf_ratio=float(hf[-1])))
        fig.tight_layout();fig.savefig(run/(Path(name).stem+'_behavior.png'),dpi=140);plt.close(fig)
    write(run/'behavior_analysis.json',dict(rows=rows,source_sha256=sha(Path(__file__)),
        trace_hashes=result['trace_hashes'],scope='descriptive physics and control handoff; no new success definition'))


if __name__=='__main__':review(Path(sys.argv[1]).resolve())
