"""新纠偏只监督实际执行教师段；BC结构与上轮一致。"""
from pathlib import Path
import math
import numpy as np
import torch
from common_segments import (HERE,parent,baseline,read,write,now,sha,snapshot,require,contract,sources,actor_network,load_actor,NOMINAL,PREVIOUS)


def selected_cases(result):
    """纠偏组必须双槽完整成功；失败组全部保留作负例档案，不进入示范。"""
    traces=sorted({c['trace'] for c in result['cases']})
    return [c for name in traces for c in result['cases'] if c['trace']==name and
        len([v for v in result['cases'] if v['trace']==name])==2 and
        all(v['success'] and v['return_time']<0 for v in result['cases'] if v['trace']==name)]


def prepare(output,collections):
    from verify_sustain import verify as verify_nominal
    from verify_student import verify as verify_previous
    from verify_segments import verify
    assert len({p.resolve() for p in collections})==len(collections)
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    records=[];episodes=[];excluded=[];new_cases=0
    for folder,kind,checker in [(NOMINAL,0,verify_nominal),(PREVIOUS,1,verify_previous)]+[(p,2,verify) for p in collections]:
        checker(folder)
        result=read(folder/'result.json');accepted=selected_cases(result)
        if kind<2:assert len(accepted)==len(result['cases'])==2
        else:new_cases+=len(accepted)
        excluded.extend(dict(run=str(folder),trace=c['trace'],slot=c['trace_env'],reason='paired program not fully successful') for c in result['cases'] if c not in accepted)
        for c in accepted:
            path=folder/c['trace'];slot=c['trace_env']
            with np.load(path) as d:
                active=d['policy_active'][:,slot].astype(bool)
                np.testing.assert_array_equal(active,d['active'][::20,slot])
                teacher=d['policy_takeover'][:,slot].astype(bool)
                selected=active if kind==0 else active & teacher
                assert selected.any()
                obs=d['policy_observation'][selected,slot]
                actions=d['actions'][::20,slot][selected]
                requested=d['policy_action'][selected,slot].clip(-1,1).copy()
                time=d['policy_physics_time'][selected,slot]
                requested[time<.2-1e-6]=0
                np.testing.assert_array_equal(actions,requested)
                if kind>0:
                    assert teacher[selected].all()
                    np.testing.assert_array_equal(actions,d['policy_teacher_action'][selected,slot])
                assert obs.shape[1:]==(305,) and (obs[:,60::61]==1).all()
                task=d['policy_time'][selected,slot]
                stage=np.where(task<1,0,np.where(task<3.2,1,np.where(time<c['capture_time'],2,3)))
                episode=len(episodes);n=len(obs)
                records.append(dict(observation=obs,action=actions,stage=stage.astype(np.int64),time=time,
                    teacher_executed=teacher[selected],episode=np.full(n,episode,dtype=np.int64),
                    policy_index=np.flatnonzero(selected),source_kind=np.full(n,kind,dtype=np.int64)))
                episodes.append(dict(run=str(folder),trace=c['trace'],trace_sha256=sha(path),slot=slot,
                    seed=c['seed'],program=c.get('program'),source_kind=kind,frames=n,
                    student_execution_excluded=int((active & ~teacher).sum()) if kind>0 else 0,
                    capture_time=c['capture_time'],verification_sha256=sha(folder/'verification.json')))
    assert new_cases>=2,'No successful paired short-segment evidence; do not train'
    data={key:np.concatenate([r[key] for r in records]) for key in records[0]}
    assert all(np.isfinite(v).all() for v in data.values())
    assert data['teacher_executed'][data['source_kind']>0].all()
    np.savez_compressed(output/'dataset.npz',**data)
    meta=dict(created_at=now(),frames=len(data['stage']),episodes=episodes,rejected_cases=excluded,
        stage_counts=np.bincount(data['stage'],minlength=4).tolist(),source_counts=np.bincount(data['source_kind'],minlength=3).tolist(),
        dataset_sha256=sha(output/'dataset.npz'),teacher_executed_frames=int(data['teacher_executed'].sum()),
        student_execution_excluded=sum(e['student_execution_excluded'] for e in episodes),
        source_roles=['frozen nominal full hybrid','previous teacher actual execution only','new segments teacher actual execution only'],
        independent_validation=False,unexecuted_teacher_labels_used=0,return_frames_used=0,
        limitations='Fixed conditions; accepted teacher-controlled success does not prove autonomy')
    write(output/'dataset.json',meta);require(output)
    print('[DATASET]',meta['frames'],meta['stage_counts'],meta['source_counts'],flush=True)

def train(output, dataset, steps, device, initial):
    require(dataset)
    meta = read(dataset/'dataset.json')
    assert sha(dataset/'dataset.npz') == meta['dataset_sha256']
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    started = now()
    torch.manual_seed(42)
    model = actor_network().to(device)
    if initial is not None:
        previous, _ = load_actor(initial)
        model.load_state_dict(previous.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    with np.load(dataset/'dataset.npz') as z:
        x = torch.tensor(z['observation'], device=device)
        y = torch.tensor(z['action'], device=device)
        groups = [torch.tensor(np.flatnonzero(z['stage'] == i), device=device) for i in range(4)]
    assert all(len(g) for g in groups)

    @torch.no_grad()
    def metrics():
        error = (model(x)-y).square().mean(-1)
        return dict(mse=float(error.mean()), stage_mse=[float(error[g].mean()) for g in groups],
                    action_max_error=float((model(x)-y).abs().max()))

    before = metrics()
    torch.save(model.state_dict(), output/'initial_actor_state.pt')
    history = []
    for step in range(steps):
        # 每阶段128帧，固定1:1:1:1，避免长保持吞没起身/捕获梯度。
        ids = torch.cat([g[torch.randint(len(g), (128,), device=device)] for g in groups])
        lr = 3e-4 * (.1 + .9*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.param_groups[0]['lr'] = lr
        loss = (model(x[ids])-y[ids]).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step+1) % 1000 == 0 or step+1 == steps:
            row = dict(step=step+1, lr=lr, **metrics())
            history.append(row)
            print('[BC]', row, flush=True)
    state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    value = dict(kind='sustain_segment_bc', contract=contract(), sources=sources(), parent_sources=parent.sources(), baseline_sources=baseline.sources(),
        actor_state_dict=state, optimizer_state_dict=optimizer.state_dict(),
        dataset=str(dataset), dataset_sha256=meta['dataset_sha256'], steps=steps, seed=42,
        initialization=str(initial) if initial else 'fresh seed42; no old Actor/Critic/Adam',
        initialization_sha256=sha(initial) if initial else None)
    torch.save(value, output/'actor.pt')
    write(output/'result.json', dict(started_at=started, finished_at=now(), steps=steps, batch_size=512,
        before=before, after=metrics(), history=history, actor_sha256=sha(output/'actor.pt'),
        dataset_frames=len(x), dataset_sha256=meta['dataset_sha256'], device=device,
        autonomous_verified=False, ppo_updates=0, ppo_admission=False))
    require(output)
    load_actor(output/'actor.pt')
