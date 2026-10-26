"""Read-only 10s/60s shared execution-prefix and late-state comparison."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
SUSTAIN=HERE.parents[1]
RUNS=SUSTAIN/'student_feedback_v5/structured_v1/runs'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short',type=Path,default=RUNS/'2026-10-23_full_01_screen')
    parser.add_argument('--long',type=Path,default=RUNS/'2026-10-23_full_01_60s')
    args=parser.parse_args()
    for run in (args.short,args.long):
        if not (run/'session.json').exists() or read(run/'session.json')['status']!='completed':
            print(f'PENDING: {run}');return 2
    sr,lr=read(args.short/'request.json'),read(args.long/'request.json')
    assert sr['actor_sha256']==lr['actor_sha256']
    assert sr['return_after_s'] is lr['return_after_s'] is None
    assert sr['autonomous'] and lr['autonomous']
    short_path=args.short/'trace_autonomous_seed43.npz';long_path=args.long/'trace_autonomous_seed43.npz'
    with np.load(short_path) as z:a={k:z[k] for k in z.files}
    with np.load(long_path) as z:b={k:z[k] for k in z.files}
    physical=('com','com_vel','wheel_pos','force_z','clearance','gravity','ang_vel','base_z','line_error','body_force',
              'torque_peak','leg_position','wheel_velocity','smooth_rms','t','task_t','mode','origin','target_gravity','phase','start_time',
              'actions','leg_targets','torque','joint_velocity','capture_time','first_break_time','break_reason','return_time','stand_request','post_capture_s','held','interrupted','failed','boundary')
    policy=('policy_raw_observation','policy_observation','policy_capture_memory','policy_trigger_action','policy_estimated_feedback',
            'policy_actor_action','policy_teacher_action','policy_action','policy_time','policy_takeover','policy_physics_time','policy_return_time','policy_hold_origin')
    cases=[]
    for slot in (0,1):
        take=a['active'][:,slot].astype(bool)
        indices=np.flatnonzero(take);last=int(indices[-1])+1
        nctl=len(a['policy_active']);control=a['policy_active'][:,slot].astype(bool)
        pcheck={k:bool(np.array_equal(a[k][:last,slot][take[:last]],b[k][:last,slot][take[:last]])) for k in physical}
        ccheck={k:bool(np.array_equal(a[k][:nctl,slot][control],b[k][:nctl,slot][control])) for k in policy}
        failures={k:float(abs(a[k][:last,slot].astype(float)-b[k][:last,slot].astype(float)).max()) for k in physical if not pcheck[k]}
        valid=b['active'][:,slot].astype(bool)
        t=b['t'][valid,slot];com=b['com'][valid,slot];vel=b['com_vel'][valid,slot];wheel=b['wheel_pos'][valid,slot]
        f=b['force_z'][valid,slot];clear=b['clearance'][valid,slot]
        pair=(f[:,2:]>5).all(-1)&(abs(f[:,:2])<2).all(-1)&(clear[:,:2].min(-1)>=.03)
        lateral=wheel[:,3,:2]-wheel[:,2,:2]
        forward=np.stack((lateral[:,1],-lateral[:,0]),-1);forward/=np.maximum(np.linalg.norm(forward,axis=-1,keepdims=True),1e-6)
        gap=((com[:,:2]-wheel[:,2:,:2].mean(1))*forward).sum(-1)
        speed=np.linalg.norm(vel,axis=-1);angular=np.linalg.norm(b['ang_vel'][valid,slot],axis=-1)
        hf=(b['smooth_rms'][valid,slot]/[.02,.05,np.deg2rad(.2),.5]).max(-1)
        captured=b['capture_time'][valid,slot];ct=float(captured[captured>=0][0])
        elapsed_ticks=np.rint((t.astype(np.float64)-ct)*1000).astype(np.int64)
        windows=[]
        for start,end in [(0,10),(10,30),(30,50),(50,60.1)]:
            mask=(elapsed_ticks>round(start*1000))&(elapsed_ticks<=round(end*1000))
            if not mask.any():continue
            windows.append(dict(after_capture_s=[start,end],samples=int(mask.sum()),pair_fraction=float(pair[mask].mean()),
                com_speed_mean_m_s=float(speed[mask].mean()),com_speed_max_m_s=float(speed[mask].max()),
                signed_gap_mean_m=float(gap[mask].mean()),signed_gap_abs_max_m=float(abs(gap[mask]).max()),
                angular_speed_max_rad_s=float(angular[mask].max()),actual_hf_max_ratio=float(hf[mask].max()),
                physical_failure=bool(b['failed'][valid,slot][mask].any())))
        cases.append(dict(slot=slot,physics_frames_compared=last,policy_inputs_compared=int(control.sum()),
            physical_prefix_exact=pcheck,physical_mismatch_max_error=failures,policy_prefix_exact=ccheck,
            administrative_observation_complete_excluded=True,late_windows=windows))
    result=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        short_run=str(args.short),long_run=str(args.long),actor_sha256=sr['actor_sha256'],
        short_trace_sha256=sha(short_path),long_trace_sha256=sha(long_path),
        all_shared_actor_inputs_outputs_and_physics_exact=all(all(c['physical_prefix_exact'].values()) and all(c['policy_prefix_exact'].values()) for c in cases),
        horizon_observation_complete_is_administrative_not_actor_input=True,cases=cases,
        simulator_runs_added=0,robot_ppo_updates=0,analysis_sha256=sha(__file__))
    (HERE/'horizon_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
