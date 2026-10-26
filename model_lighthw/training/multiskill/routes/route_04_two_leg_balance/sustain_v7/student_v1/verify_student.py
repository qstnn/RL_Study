"""独立1kHz物理验收加CPU学生动作复算，不用拟合误差替代闭环成功。"""
from types import SimpleNamespace
import numpy as np
import torch
from common_student import read, sha, require, load_actor, parent


def verify(run):
    from verify_sustain import analyse
    require(run)
    result, request = read(run/'result.json'), read(run/'request.json')
    assert result['actor_sha256'] == request['actor_sha256'] == sha(request['actor'])
    actor, _ = load_actor(request['actor'])
    geometry = read(run/'geometry.json')
    autonomous = request['switch_s'] is None
    assert result['autonomous'] == autonomous
    stats, checks, arrays = [], [], []
    for path in sorted(run.glob('trace_seed*.npz')):
        assert sha(path) == result['trace_hashes'][path.name]
        with np.load(path) as z:
            data = {k:z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in data.values())
        group = [c for c in result['cases'] if c['trace'] == path.name]
        assert len(group) == 2
        for c in group:
            slot = c['trace_env']; active = data['active'][:,slot]
            d = {k:v[active,slot] for k,v in data.items() if not k.startswith('policy_')}
            row, good = analyse(d,c,geometry,request['hold_seconds'])
            assert not row['returned'] and c['teacher'] == (not autonomous)
            stats.append(row)
        active = data['policy_active']
        np.testing.assert_array_equal(active,data['active'][::20])
        assert (data['policy_observation'][...,60::61][active] == 1).all()
        assert 'policy_legacy_observation' not in data
        with torch.no_grad():
            mu = actor(torch.from_numpy(data['policy_observation']).float()).numpy()
        error = float(abs(mu[active]-data['policy_actor_action'][active]).max())
        assert error < 2e-6, error
        if autonomous:
            assert not data['policy_takeover'].any() and not data['policy_teacher_action'].any()
            expected = mu
            teacher_error = 0.
        else:
            from teacher_sustain import SustainTeacher
            teacher = SustainTeacher([parent.candidate(.8)]*2,geometry,'cpu')
            labels=[]
            for k in range(len(active)):
                frame=max(0,k*20-1)
                metrics={name:torch.from_numpy(data[name][frame]).float() for name in ('gravity','ang_vel','wheel_pos','com','com_vel','force_z')}
                mock=SimpleNamespace(metrics=lambda:metrics,
                    physics_steps=torch.round(torch.from_numpy(data['policy_physics_time'][k])/.001).long(),physics_dt=.001,
                    state=SimpleNamespace(hold_origin=torch.from_numpy(data['policy_hold_origin'][k]),return_time=torch.from_numpy(data['policy_return_time'][k])))
                labels.append(teacher(mock,torch.from_numpy(data['policy_time'][k])).numpy())
            labels=np.stack(labels)
            teacher_error=float(abs(labels[active]-data['policy_teacher_action'][active]).max())
            assert teacher_error < 2e-5
            np.testing.assert_array_equal(data['policy_takeover'][active],(data['policy_time'] >= request['switch_s']-1e-6)[active])
            expected=np.where(data['policy_takeover'][...,None],labels,mu)
        np.testing.assert_allclose(expected[active],data['policy_action'][active],atol=2e-6,rtol=0)
        executed=data['policy_action'].clip(-1,1).copy()
        executed[data['policy_physics_time'] < .2-1e-6]=0
        np.testing.assert_array_equal(executed[active],data['actions'][::20][active])
        from two_leg_balance_smooth_v6.core import targets
        executed_tensor=torch.from_numpy(data['actions'][::20])
        target=targets(executed_tensor.reshape(-1,16)).reshape(*executed_tensor.shape[:-1],12).numpy()
        np.testing.assert_allclose(target[active],data['leg_targets'][::20][active],atol=1e-6,rtol=0)
        checks.append(dict(trace=path.name,actor_max_error=error,teacher_max_error=teacher_error,executed_action_error=0))
        arrays.append(data)
    repeats = bool(len(arrays)>1 and all(set(d)==set(arrays[0]) and all(np.array_equal(v,arrays[0][k]) for k,v in d.items()) for d in arrays[1:]))
    success=sum(s['observation_pass'] for s in stats)
    assert success == result['observation_passes']
    gate=bool(autonomous and request['hold_seconds'] >= 10 and request['seeds']==[43,44,45] and success==6)
    return dict(independent_physics_verified=True, autonomous=autonomous, teacher_constructed=not autonomous,
        cases=stats, action_checks=checks, physics_samples=sum(s['physics_samples'] for s in stats),
        fixed_seeds_identical=repeats, autonomous_10s_gate=gate, ppo_admission=False, robot_ppo_updates=0)
