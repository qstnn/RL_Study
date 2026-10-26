"""v6独立NumPy/SciPy验收：取消稳定段累计漂移门，复算1kHz实际振动。"""
import argparse
from pathlib import Path
import numpy as np
from common_smooth import read, write
from scipy.signal import butter, lfilter


def streak(mask):
    index = np.arange(len(mask))
    return np.where(mask, index - np.maximum.accumulate(np.where(~mask, index, -1)), 0)


def verify(folder):
    report, c = read(folder / 'evaluation.json'), read(folder / 'contract.json')
    count = samples = 0
    for path in sorted(folder.glob('trace_seed*.npz')):
        with np.load(path) as z: data = {k:z[k] for k in z.files}
        if not all(np.isfinite(a).all() for a in data.values()): raise ValueError('Nonfinite trace')
        for case in [r for r in report['cases'] if r['trace'] == path.name]:
            i = case['trace_env']; use = data['active'][:,i]
            d = {k:v[use,i] for k,v in data.items() if not k.startswith('policy_')}
            t = d['t']; n=len(t)
            if not np.allclose(np.diff(np.r_[0.,t]),.001,atol=2e-6,rtol=0): raise ValueError('Trace clock gap')
            mode = int(d['mode'][0]); stance=[2*mode,2*mode+1]; swing=[(j+2)%4 for j in stance]
            norm = lambda v: np.linalg.norm(v,axis=-1)
            drift = norm(d['com'][:,:2]-d['origin'])
            four = (d['force_z']>5).all(-1)
            linear = norm(d['com_vel'])<=c['max_com_speed_m_s']; angular=norm(d['ang_vel'])<=.25
            pose=(d['gravity'][:,2]<-np.cos(np.deg2rad(12))) & (abs(d['base_z']-.416)<=.07)
            code = np.where((t>.15)&(d['body_force']>5),1,np.where((d['base_z']<.18)|(d['gravity'][:,2]>-np.cos(np.deg2rad(80))),2,np.where(drift>1.5,3,0)))
            valid=~np.maximum.accumulate(code>0)
            entry_candidate=streak(four&linear&angular&pose&valid)*.001
            first_entry = np.flatnonzero((entry_candidate>=.5-1e-5)&(t<=3.000001))
            entered = bool(len(first_entry))
            start_ids = np.flatnonzero((t>=1.)&(np.arange(n)>=first_entry[0])&valid) if entered else []
            start=float(t[start_ids[0]]) if len(start_ids) else -1.
            if start>3.000001: start=-1.;entered=False
            if start<0: code=np.where((code==0)&(t>=3.-1e-6),5,code)
            failed=np.maximum.accumulate(code>0)
            first_code=int(code[np.flatnonzero(code)[0]]) if np.any(code) else 0
            task=np.minimum(t,.999) if start<0 else np.where(t<start,np.minimum(t,.999),t-start+1.)
            # 存档clock经过float32张量计算，容许最后一位舍入误差。
            np.testing.assert_allclose(task,d['task_t'],atol=2e-6,rtol=0)
            phase=(task>=1).astype(int)+(task>=4)+(task>=5+c['hold_seconds'])+(task>=8+c['hold_seconds'])
            a=d['wheel_pos'][:,stance[0],:2]; b=d['wheel_pos'][:,stance[1],:2]; line=b-a
            u=np.sum((d['com'][:,:2]-a)*line,-1)/np.maximum(np.sum(line**2,-1),1e-8)
            line_error=norm(d['com'][:,:2]-(a+np.clip(u,0,1)[:,None]*line))
            np.testing.assert_allclose(line_error,d['line_error'],atol=1e-5,rtol=1e-4)
            clearance=d['wheel_pos'][:,:,2]-.087501
            np.testing.assert_allclose(clearance,d['clearance'],atol=1e-6,rtol=0)
            amp=np.minimum(np.clip((task-1)/3,0,1),np.clip((8+c['hold_seconds']-task)/3,0,1));amp=amp**2*(3-2*amp)
            geometry=read(folder/'geometry.json');pitch=np.float32(geometry['candidates'][mode]['pitch_rad'])*amp
            target=np.stack((np.sin(pitch),np.zeros(n),-np.cos(pitch)),-1)
            np.testing.assert_allclose(target,d['target_gravity'],atol=2e-6,rtol=0)
            angle=np.arccos(np.clip(np.sum(d['gravity']*d['target_gravity'],-1),-1,1))
            pair=(d['force_z'][:,stance]>5).all(-1)&(abs(d['force_z'][:,swing])<2).all(-1)&(clearance[:,swing].min(-1)>=.03)
            ho=d['com'][np.argmax(phase==2),:2];ro=d['com'][np.argmax(phase==4),:2]
            # 独立SciPy滤波，不调用训练状态机或Torch滤波器。
            settings=c['jitter_filter']
            limits=np.array([settings['limits'][key] for key in ('com_velocity_m_s', 'angular_velocity_rad_s', 'leg_position_rad', 'wheel_velocity_rad_s')])
            raw=np.concatenate((d['com_vel'],d['ang_vel'],d['leg_position'],d['wheel_velocity']),axis=-1).astype(np.float64)
            b,a=butter(2,settings['cutoff_hz'],fs=1./settings['sample_dt'],btype='highpass')
            hp=lfilter(b,a,raw-raw[:1],axis=0)
            alpha=1.-np.exp(-settings['sample_dt']/settings['rms_time_constant_s'])
            energy=lfilter([alpha],[1.,-(1.-alpha)],hp**2,axis=0)
            rms=np.stack((np.sqrt(energy[:,:3].sum(-1)),np.sqrt(energy[:,3:6].sum(-1)),
                np.sqrt(energy[:,6:18].max(-1)),np.sqrt(energy[:,18:].max(-1))),-1)
            np.testing.assert_allclose(rms,d['smooth_rms'],atol=5e-5,rtol=2e-3)
            smooth=(rms<=limits).all(-1)
            hold=streak((phase==2)&pair&linear&angular&(line_error<=.06)&(angle<=np.deg2rad(12))&smooth&entered&~failed)*.001
            held=np.maximum.accumulate(hold>=c['hold_seconds']-1e-4)
            recovery=streak((phase==4)&four&linear&angular&pose&smooth&held&~failed)*.001
            success=bool(held[-1] and recovery[-1]>=1-1e-4 and not failed[-1])
            results=dict(entered=entered,held=bool(held[-1]),failed=bool(failed[-1]),success=success,reason=first_code,
                         best_hold=float(hold.max()),best_recovery=float(recovery.max()),start_time=start,
                         best_pair=float((streak(pair&~failed)*.001).max()))
            for key,val in results.items():
                if abs(val-case[key])>.002:raise AssertionError(f'{path.name} slot{i} {key}: {val} vs {case[key]}')
            # 终末成功必须由任务时钟触发自然timeout。
            timeout=bool(start>=0 and task[-1]>=10+c['hold_seconds']-1e-5)
            if timeout != case['timeout']:raise AssertionError('Timeout clock mismatch')
            count+=1;samples+=n
    if count!=len(report['cases']):raise AssertionError('Missing trace cases')
    result=dict(cases=count,physics_samples=samples,protocol_recomputed=True,support_geometry_recomputed=True,
                implementation='independent NumPy/SciPy; no training core/filter import',jitter_recomputed=True,behavior_accepted=False)
    write(folder/'trace_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    print(verify(parser.parse_args().folder))
