"""跨观察时长的输入/行为一致性，以及明确返回命令后的独立终末验收。"""
from pathlib import Path
import sys
import numpy as np
from scipy.signal import butter,lfilter

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_sustain import read,write,sha,require


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def horizon(short,long):
    require(short);require(long)
    a=arrays(short/'trace_seed43_blend800.npz');b=arrays(long/'trace_seed43_blend800.npz')
    physics=[k for k in a if not k.startswith('policy_') and k not in ('active','observation_complete')]
    policy=[k for k in a if k.startswith('policy_') and k!='policy_active']
    # 只比较两个槽均未reset的公共前缀，避免另一槽reset改变整批接触求解布局。
    n=int(a['active'].all(-1).sum());m=int(a['policy_active'].all(-1).sum())
    for slot in (0,1):
        for key in physics:np.testing.assert_array_equal(a[key][:n,slot],b[key][:n,slot],err_msg=key)
        for key in policy:np.testing.assert_array_equal(a[key][:m,slot],b[key][:m,slot],err_msg=key)
    result=dict(physics_arrays_exact=len(physics),policy_arrays_exact=len(policy),
        observation_horizons_s=[10,30],includes_new_policy_observations=True,includes_rewards=True,
        shared_physics_steps_per_slot=n,shared_policy_steps_per_slot=m,
        short_result_sha256=sha(short/'result.json'),long_result_sha256=sha(long/'result.json'),
        analysis_source_sha256=sha(Path(__file__)))
    write(long/'horizon_invariance.json',result)
    return result


def check_return(run):
    require(run)
    request=read(run/'request.json');assert request['return_after_s'] is not None
    records=[]
    for c in read(run/'result.json')['cases']:
        all_data=arrays(run/c['trace']);slot=c['trace_env'];use=all_data['active'][:,slot]
        d={k:v[use,slot] for k,v in all_data.items() if not k.startswith('policy_')}
        assert np.isfinite(np.concatenate([v.reshape(len(use[use]),-1) for v in d.values()],-1)).all()
        return_time=c['return_time'];assert return_time>=0
        assert abs(return_time-c['capture_time']-request['return_after_s'])<.021
        raw=np.concatenate((d['com_vel'],d['ang_vel'],d['leg_position'],d['wheel_velocity']),-1).astype(np.float64)
        b,a=butter(2,5.,fs=1000.,btype='highpass');hp=lfilter(b,a,raw-raw[:1],axis=0)
        alpha=1.-np.exp(-.001/.1);energy=lfilter([alpha],[1.,-(1.-alpha)],hp*hp,axis=0)
        rms=np.stack((np.sqrt(energy[:,:3].sum(-1)),np.sqrt(energy[:,3:6].sum(-1)),
            np.sqrt(energy[:,6:18].max(-1)),np.sqrt(energy[:,18:].max(-1))),-1)
        smooth=(rms<=[.02,.05,np.deg2rad(.2),.5]).all(-1)
        four=(d['force_z']>5.).all(-1)
        quiet=(np.linalg.norm(d['com_vel'],axis=-1)<=.12)&(np.linalg.norm(d['ang_vel'],axis=-1)<=.25)
        upright=(d['gravity'][:,2]<-np.cos(np.deg2rad(12.)))&(abs(d['base_z']-.416)<=.07)
        recovered=(d['t']>=return_time+3.-1e-6)&four&quiet&upright&smooth&~d['failed']&~d['boundary']
        index=np.arange(len(recovered));streak=np.where(recovered,index-np.maximum.accumulate(np.where(~recovered,index,-1)),0)*.001
        assert abs(streak[-1]-c['recovery_time'])<.002
        assert abs(streak.max()-c['best_recovery'])<.002
        # success字段专指仍在执行站立命令时的长稳完成；返回能力必须单列判断。
        passed=bool(c['observation_complete'] and not c['interrupted'] and not c['failed'] and not c['boundary'] and streak[-1]>=1.)
        moving=(d['t']>return_time)&(d['t']<return_time+3.)
        weight=read(run/'geometry.json')['mass_kg']*9.81
        records.append(dict(slot=slot,capture_time_s=c['capture_time'],return_command_s=return_time,
            extra_hold_before_return_s=c['post_capture_s'],terminal_four_wheel_s=float(streak[-1]),
            explicit_return_pass=passed,standing_at_end_flag=c['success'],
            return_peak_speed_m_s=float(np.linalg.norm(d['com_vel'][moving],axis=-1).max()),
            return_peak_ang_speed_rad_s=float(np.linalg.norm(d['ang_vel'][moving],axis=-1).max()),
            return_peak_hf_ratio=float((rms[moving]/[.02,.05,np.deg2rad(.2),.5]).max()),
            return_peak_wheel_load_bodyweights=float(d['force_z'][moving].clip(min=0).sum(-1).max()/weight),
            max_body_force_N=float(d['body_force'].max())))
    result=dict(cases=records,passed=sum(c['explicit_return_pass'] for c in records),
        independent_numpy_scipy_recovery=True,request_sha256=sha(run/'request.json'),
        analysis_source_sha256=sha(Path(__file__)),ppo_admission=False)
    write(run/'return_verification.json',result)
    return result


if __name__=='__main__':
    short=HERE/'runs/2026-10-22_sustain_blend_6_verified'
    long=HERE/'runs/2026-10-22_sustain_30s_blend800'
    returning=HERE/'runs/2026-10-22_sustain_explicit_return'
    print(horizon(short,long))
    print(check_return(returning))
