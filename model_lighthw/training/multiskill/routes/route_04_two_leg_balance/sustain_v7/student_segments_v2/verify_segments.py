"""独立复算1kHz验收、CPU教师/学生、接管释放及真实前缀。"""
from types import SimpleNamespace
import numpy as np
import torch
from common_segments import read,sha,require,load_actor,parent,ACTOR,BASELINE


def verify(run):
    from verify_sustain import analyse
    require(run)
    result,request=read(run/'result.json'),read(run/'request.json')
    assert result['actor_sha256']==request['actor_sha256']==sha(request['actor'])
    actor,_=load_actor(request['actor'])
    geometry=read(run/'geometry.json')
    autonomous=request['autonomous']
    assert result['autonomous']==autonomous
    programs={p['id']:p['windows'] for p in request['programs']}
    assert autonomous == (len(programs)==1 and list(programs.values())==[None])
    stats,checks,arrays=[],[],[]
    for path in sorted(run.glob('trace_*.npz')):
        assert sha(path)==result['trace_hashes'][path.name]
        with np.load(path) as z:data={k:z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in data.values())
        group=[c for c in result['cases'] if c['trace']==path.name]
        assert len(group)==2
        windows=programs[group[0]['program']]
        for c in group:
            slot=c['trace_env'];active=data['active'][:,slot]
            row,good=analyse({k:v[active,slot] for k,v in data.items() if not k.startswith('policy_')},c,geometry,request['hold_seconds'])
            assert not row['returned'] and c['teacher']==(not autonomous) and c['windows']==windows
            row.update(program=c['program'],seed=c['seed']);stats.append(row)
        active=data['policy_active']
        np.testing.assert_array_equal(active,data['active'][::20])
        assert (data['policy_observation'][...,60::61][active]==1).all()
        assert 'policy_legacy_observation' not in data
        with torch.no_grad():mu=actor(torch.from_numpy(data['policy_observation']).float()).numpy()
        error=float(abs(mu[active]-data['policy_actor_action'][active]).max());assert error<2e-6
        # 用NumPy从请求重新构建窗口，不调用执行器的掩码函数。
        expected_mask=np.zeros_like(active)
        if windows is not None:
            for start,end in windows:
                condition=data['policy_time']>=start-1e-6
                if end is not None:condition &= data['policy_time']<end-1e-6
                expected_mask |= condition
        np.testing.assert_array_equal(expected_mask[active],data['policy_takeover'][active])
        transitions=[]
        for c in group:
            slot=c['trace_env'];valid=active[:,slot]
            mask=data['policy_takeover'][valid,slot]
            changed=np.flatnonzero(np.diff(np.r_[False,mask]))
            times=data['policy_time'][valid,slot][changed].tolist()
            if c['success'] and not autonomous:assert len(times)==2*len(windows)-1
            transitions.append(dict(slot=slot,task_times=times,teacher_frames=int(mask.sum()),student_frames=int((~mask).sum())))
        if autonomous:
            assert not data['policy_teacher_action'].any() and not data['policy_takeover'].any()
            expected=mu;teacher_error=0.
        else:
            from teacher_sustain import SustainTeacher
            teacher=SustainTeacher([parent.candidate(.8)]*2,geometry,'cpu')
            labels=[]
            for k in range(len(active)):
                frame=max(0,k*20-1)
                metrics={name:torch.from_numpy(data[name][frame]).float() for name in ('gravity','ang_vel','wheel_pos','com','com_vel','force_z')}
                mock=SimpleNamespace(metrics=lambda:metrics,
                    physics_steps=torch.round(torch.from_numpy(data['policy_physics_time'][k])/.001).long(),physics_dt=.001,
                    state=SimpleNamespace(hold_origin=torch.from_numpy(data['policy_hold_origin'][k]),return_time=torch.from_numpy(data['policy_return_time'][k])))
                labels.append(teacher(mock,torch.from_numpy(data['policy_time'][k])).numpy())
            labels=np.stack(labels)
            teacher_error=float(abs(labels[active]-data['policy_teacher_action'][active]).max());assert teacher_error<2e-5
            expected=np.where(expected_mask[...,None],labels,mu)
        np.testing.assert_allclose(expected[active],data['policy_action'][active],atol=2e-6,rtol=0)
        executed=data['policy_action'].clip(-1,1).copy()
        executed[data['policy_physics_time']<.2-1e-6]=0
        np.testing.assert_array_equal(executed[active],data['actions'][::20][active])
        from two_leg_balance_smooth_v6.core import targets
        a=torch.from_numpy(data['actions'][::20])
        target=targets(a.reshape(-1,16)).reshape(*a.shape[:-1],12).numpy()
        np.testing.assert_allclose(target[active],data['leg_targets'][::20][active],atol=1e-6,rtol=0)
        prefix_count=None
        if not autonomous and request['actor']==str(ACTOR):
            # 只比较首次接管前；交接以后必须保持真实连续演化。
            length=round(windows[0][0]*1000);controls=round(windows[0][0]*50)+1
            with np.load(BASELINE/'trace_seed43.npz') as old:
                keys=[k for k in old.files if not k.startswith('policy_')]
                for k in keys:np.testing.assert_array_equal(old[k][:length],data[k][:length],err_msg='prefix '+k)
                for k in ['policy_observation','policy_actor_action']:
                    np.testing.assert_array_equal(old[k][:controls],data[k][:controls],err_msg='switch '+k)
                prefix_count=len(keys)
        checks.append(dict(trace=path.name,actor_max_error=error,teacher_max_error=teacher_error,
            executed_action_error=0,prefix_arrays_exact=prefix_count,transitions=transitions))
        arrays.append(data)
    success=sum(c['observation_pass'] for c in stats)
    assert success==result['observation_passes']
    repeats=bool(autonomous and len(arrays)>1 and all(all(np.array_equal(v,arrays[0][k]) for k,v in d.items()) for d in arrays[1:]))
    gate=bool(autonomous and request['hold_seconds']>=10 and request['seeds']==[43,44,45] and len(stats)==6 and success==6)
    return dict(independent_physics_verified=True,autonomous=autonomous,teacher_constructed=not autonomous,
        cases=stats,action_checks=checks,physics_samples=sum(c['physics_samples'] for c in stats),
        fixed_seeds_identical=repeats,autonomous_10s_gate=gate,ppo_admission=False,robot_ppo_updates=0)
