"""独立NumPy FK与因果记忆重建；原1kHz物理门保持。"""
import numpy as np
import torch
from common_feedback import read,sha,require,load_actor,BodyEstimator,URDF


def verify(run):
    from verify_sustain import analyse
    from verify_memory import reconstruct_memory
    require(run);request=read(run/'request.json');result=read(run/'result.json')
    assert request['autonomous'] and result['autonomous']
    assert result['actor_sha256']==request['actor_sha256']==sha(request['actor'])
    assert all(p['windows'] is None for p in request['programs'])
    actor,value=load_actor(request['actor']);assert value['variant']==request['variant']
    estimator=BodyEstimator(URDF);cases=[];checks=[];arrays=[]
    for path in sorted(run.glob('trace_*.npz')):
        assert sha(path)==result['trace_hashes'][path.name]
        with np.load(path) as z:d={k:z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in d.values())
        group=[c for c in result['cases'] if c['trace']==path.name];assert len(group)==2
        for c in group:
            slot=c['trace_env'];active=d['active'][:,slot]
            row,_=analyse({k:v[active,slot] for k,v in d.items() if not k.startswith('policy_')},c,read(run/'geometry.json'),request['hold_seconds'])
            assert not row['returned'] and not c['teacher']
            row.update(seed=c['seed'],program=c['program']);cases.append(row)
        active=d['policy_active'];raw=d['policy_raw_observation'];obs=d['policy_observation']
        np.testing.assert_array_equal(active,d['active'][::20]);assert obs.shape[-1]==325
        np.testing.assert_array_equal(obs[...,:305],raw);assert (raw[...,60::61][active]==1).all()
        if len(raw)>1:np.testing.assert_allclose(raw[1:,...,-58:-55][active[1:]],d['gravity'][19::20][:-1][active[1:]],atol=1e-6,rtol=0)
        memory=reconstruct_memory(raw)
        np.testing.assert_array_equal(memory[...,0][active],obs[...,305][active])
        np.testing.assert_allclose(memory[active],obs[...,305:307][active],atol=3e-7,rtol=0)
        estimates=estimator.estimate(raw.reshape(-1,305))
        ev=np.stack((estimates['gap_m'],estimates['velocity_m_s']),-1).reshape(*raw.shape[:-1],2)
        np.testing.assert_allclose(ev[active],obs[...,323:325][active],atol=2e-5,rtol=0)
        with torch.no_grad():
            prefix=actor.prefix(torch.from_numpy(np.concatenate((raw,memory),-1))).numpy()
            mu=actor(torch.from_numpy(obs)).numpy()
        start=np.zeros((*raw.shape[:2],16),np.float32)
        for slot in (0,1):
            stored=np.zeros(16,np.float32);previous=False
            for k in range(len(raw)):
                on=bool(memory[k,slot,0])
                if not on:stored=np.zeros(16,np.float32)
                elif not previous:stored=raw[k,slot,-27:-11].copy()
                start[k,slot]=stored;previous=on
        np.testing.assert_allclose(start[active],obs[...,307:323][active],atol=2e-6,rtol=0)
        # 独立标量公式重建低维头；不调用其feedback_basis。
        f=raw[...,-61:];g=f[...,3:6].astype(np.float64);w=f[...,:3].astype(np.float64)/.2
        derivative=-np.cross(w,g);pitch=np.arctan2(g[...,0],-g[...,2])
        rate=(-g[...,2]*derivative[...,0]+g[...,0]*derivative[...,2])/np.maximum(g[...,0]**2+g[...,2]**2,1e-6)
        x=np.stack((np.ones_like(pitch),ev[...,0],ev[...,1],pitch,rate,ev[...,0].clip(-.12,.12)),-1)
        final=x@(actor.coefficients.detach().numpy()*actor.mask.numpy()).T
        blend=memory[...,1:2];expected=np.where(memory[...,0:1]>.5,start*(1-blend)+final*blend,prefix).clip(-1,1)
        error=float(abs(expected[active]-d['policy_actor_action'][active]).max());assert error<5e-5,error
        np.testing.assert_allclose(mu[active],d['policy_actor_action'][active],atol=2e-5,rtol=0)
        assert not d['policy_takeover'].any() and not d['policy_teacher_action'].any()
        np.testing.assert_array_equal(d['policy_actor_action'][active],d['policy_action'][active])
        executed=d['policy_action'].clip(-1,1).copy();executed[d['policy_physics_time']<.2-1e-6]=0
        np.testing.assert_array_equal(executed[active],d['actions'][::20][active])
        from two_leg_balance_smooth_v6.core import targets
        a=torch.from_numpy(d['actions'][::20]);target=targets(a.reshape(-1,16)).reshape(*a.shape[:-1],12).numpy()
        np.testing.assert_allclose(target[active],d['leg_targets'][::20][active],atol=1e-6,rtol=0)
        checks.append(dict(trace=path.name,independent_actor_max_error=error,independent_feature_max_error=float(abs(ev[active]-obs[...,323:325][active]).max()),
            teacher_frames=0,executed_action_error=0));arrays.append(d)
    passes=sum(c['observation_pass'] for c in cases);assert passes==result['observation_passes']
    gate=request['seeds']==[43,44,45] and len(cases)==6 and passes==6 and request['hold_seconds']>=10
    return dict(independent_physics_verified=True,independent_proprioception_verified=True,autonomous=True,teacher_constructed=False,
        cases=cases,passes=passes,action_checks=checks,physics_samples=sum(c['physics_samples'] for c in cases),
        autonomous_10s_gate=bool(gate),ppo_admission=False,robot_ppo_updates=0,
        limitation='finite fixed-condition repeat only; rolling estimate does not replace true COM velocity acceptance')
