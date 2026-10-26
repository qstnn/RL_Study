"""NumPy独立重建因果记忆；复算动作、路由与原1kHz长稳判据。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common_memory import read,sha,require,load_actor


def reconstruct_memory(raw):
    assert raw.ndim==3 and raw.shape[-1]==305
    output=np.zeros((*raw.shape[:2],2),np.float32)
    for slot in range(raw.shape[1]):
        trigger_index=None
        for k in range(len(raw)):
            f=raw[k,slot,-61:]
            if f[55]>=.5 or f[60]<=.5:trigger_index=None
            elif trigger_index is None and np.arctan2(f[3],-f[5])<=np.deg2rad(-5.):trigger_index=k
            if trigger_index is not None:
                u=min((k-trigger_index)/40.,1.)
                output[k,slot]=[1.,u*u*(3.-2.*u)]
    return output


def verify(run):
    from verify_sustain import analyse
    require(run);result=read(run/'result.json');request=read(run/'request.json')
    assert result['autonomous'] and request['autonomous']
    assert all(p['windows'] is None for p in request['programs'])
    assert result['actor_sha256']==request['actor_sha256']==sha(request['actor'])
    actor,value=load_actor(request['actor']);assert value['memory_mode']==request['memory_mode']
    cases=[];checks=[];arrays=[]
    for path in sorted(run.glob('trace_*.npz')):
        assert sha(path)==result['trace_hashes'][path.name]
        with np.load(path) as z:d={k:z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in d.values())
        group=[c for c in result['cases'] if c['trace']==path.name];assert len(group)==2
        for c in group:
            slot=c['trace_env'];active=d['active'][:,slot]
            row,_=analyse({k:v[active,slot] for k,v in d.items() if not k.startswith('policy_')},c,read(run/'geometry.json'),request['hold_seconds'])
            assert not row['returned'] and not c['teacher'] and c['windows'] is None
            row.update(seed=c['seed'],program=c['program']);cases.append(row)
        active=d['policy_active'];np.testing.assert_array_equal(active,d['active'][::20])
        raw=d['policy_raw_observation'];obs=d['policy_observation']
        assert raw.shape[-1]==305 and obs.shape[-1]==307
        np.testing.assert_array_equal(raw,obs[...,:305])
        assert (raw[...,60::61][active]==1).all()
        if len(raw)>1:
            np.testing.assert_allclose(raw[1:,...,-58:-55][active[1:]],d['gravity'][19::20][:-1][active[1:]],atol=1e-6,rtol=0)
        memory=reconstruct_memory(raw)
        np.testing.assert_allclose(memory[active],d['policy_capture_memory'][active],atol=3e-7,rtol=0)
        np.testing.assert_array_equal(memory[...,0][active],d['policy_capture_memory'][...,0][active])
        recorded_tail=d['policy_capture_memory'] if actor.mode=='exposed' else np.zeros_like(memory)
        np.testing.assert_array_equal(obs[...,305:][active],recorded_tail[active])
        assert ((memory[active]>=0)&(memory[active]<=1)).all()
        expected_memory=memory if actor.mode=='exposed' else np.zeros_like(memory)
        np.testing.assert_allclose(obs[...,305:][active],expected_memory[active],atol=3e-7,rtol=0)
        with torch.no_grad():
            mu=actor(torch.from_numpy(obs)).numpy();h=actor.trunk(torch.from_numpy(obs))
            rise=actor.rise(h).numpy();capture=actor.capture(h).numpy()
        f=raw[...,-61:];gate=(np.arctan2(f[...,3],-f[...,5])<=np.deg2rad(-5.))&(f[...,55]<.5)&(f[...,60]>.5)
        np.testing.assert_allclose(mu,np.where(gate[...,None],capture,rise),atol=2e-6,rtol=0)
        error=float(abs(mu[active]-d['policy_actor_action'][active]).max());assert error<2e-6
        assert not d['policy_teacher_action'].any() and not d['policy_takeover'].any()
        np.testing.assert_allclose(mu[active],d['policy_action'][active],atol=2e-6,rtol=0)
        executed=d['policy_action'].clip(-1,1).copy();executed[d['policy_physics_time']<.2-1e-6]=0
        np.testing.assert_array_equal(executed[active],d['actions'][::20][active])
        from two_leg_balance_smooth_v6.core import targets
        a=torch.from_numpy(d['actions'][::20]);target=targets(a.reshape(-1,16)).reshape(*a.shape[:-1],12).numpy()
        np.testing.assert_allclose(target[active],d['leg_targets'][::20][active],atol=1e-6,rtol=0)
        state_rows=[]
        for slot in (0,1):
            valid=active[:,slot];pt=d['policy_physics_time'][valid,slot];m=memory[valid,slot]
            on=np.flatnonzero(m[:,0]>0);complete=np.flatnonzero(m[:,1]>=1.)
            changes=np.flatnonzero(np.diff(gate[valid,slot].astype(int)))+1
            state_rows.append(dict(slot=slot,first_trigger_s=float(pt[on[0]]) if len(on) else None,
                blend_complete_s=float(pt[complete[0]]) if len(complete) else None,
                routing_switch_times=pt[changes].tolist(),final_memory=m[-1].tolist()))
        checks.append(dict(trace=path.name,memory_mode=actor.mode,independent_memory_max_error=float(abs(memory[active]-d['policy_capture_memory'][active]).max()),
            actor_max_error=error,executed_action_error=0,teacher_frames=0,states=state_rows));arrays.append(d)
    passes=sum(c['observation_pass'] for c in cases);assert passes==result['observation_passes']
    same=bool(len(arrays)>1 and all(all(np.array_equal(v,arrays[0][k]) for k,v in d.items()) for d in arrays[1:]))
    gate=bool(request['seeds']==[43,44,45] and len(cases)==6 and passes==6 and request['hold_seconds']>=10)
    return dict(independent_physics_verified=True,independent_memory_verified=True,autonomous=True,teacher_constructed=False,
        cases=cases,action_checks=checks,physics_samples=sum(c['physics_samples'] for c in cases),
        fixed_seeds_identical=same,autonomous_10s_gate=gate,ppo_admission=False,robot_ppo_updates=0)


def main():
    # 独立离线精度审查；保留原运行失败、原判据与原冻结文件。
    from common_memory import write,now
    run=Path(sys.argv[1]).resolve()
    report=verify(run)
    report.update(review_version='v4.1_float32_precision',memory_atol=3e-7,
        original_verifier_sha256=sha(Path(__file__).resolve().parents[1]/'verify_memory.py'),
        reviewer_sha256=sha(__file__),reviewed_at=now(),physics_thresholds_unchanged=True,
        latch_and_input_tail_bit_exact=True,float32_epsilon=float(np.finfo(np.float32).eps),
        original_session_status=read(run/'session.json')['status'])
    original=read(run/'worker_error.json') if (run/'worker_error.json').exists() else None
    if report['original_session_status']=='failed':
        assert original is not None and 'atol=1e-07' in original['error']
        assert 'policy_capture_memory' in original['traceback']
    write(run/'verification_v4_1.json',report)
    print({k:report[k] for k in ['physics_samples','autonomous_10s_gate','ppo_admission','original_session_status']})
    print(report['action_checks'])


if __name__=='__main__':main()
