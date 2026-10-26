"""只学习成功真实执行轨迹；阶段均衡的小预算BC不代表自主通过。"""
from pathlib import Path
import math
import numpy as np
import torch
from common_student import HERE, parent, read, write, now, sha, snapshot, require, contract, sources, actor_network, load_actor


def prepare(output, inputs):
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    records, provenance = [], []
    for run in inputs:
        if (run/'student_contract.json').exists():
            from verify_student import verify
        else:
            from verify_sustain import verify
        evidence = verify(run)
        result = read(run/'result.json')
        assert result['cases'] and all(c['success'] for c in result['cases']), 'Only complete success is a demonstration'
        assert all(c['return_time'] < 0 for c in result['cases']), 'Exclude unoptimized return'
        for c in result['cases']:
            path = run/c['trace']
            with np.load(path) as z:
                slot = c['trace_env']
                active = z['policy_active'][:, slot].astype(bool)
                np.testing.assert_array_equal(active, z['active'][::20, slot])
                obs = z['policy_observation'][active, slot]
                action = z['actions'][::20, slot][active]
                command = z['policy_action'][active, slot].clip(-1, 1).copy()
                time = z['policy_physics_time'][active, slot]
                command[time < .2-1e-6] = 0
                np.testing.assert_array_equal(action, command)
                assert (obs[:, 60::61] == 1).all()
                assert obs.shape[1:] == (305,) and action.shape[1:] == (16,)
                takeover = z['policy_takeover'][active, slot].astype(bool)
                task = z['policy_time'][active, slot]
                # 保持阶段只有在完成2秒捕获里程碑后开始；此前捕获瞬态单独采样。
                stage = np.where(task < 1, 0, np.where(task < 3.2, 1, np.where(time < c['capture_time'], 2, 3)))
                episode = len(provenance)
                records.append(dict(observation=obs, action=action, stage=stage.astype(np.int64),
                    time=time, teacher_executed=takeover, episode=np.full(len(obs), episode, dtype=np.int64)))
                provenance.append(dict(run=str(run), trace=c['trace'], trace_sha256=sha(path),
                    slot=slot, seed=c['seed'], frames=len(obs), capture_time=c['capture_time'],
                    prefix='student' if (run/'student_contract.json').exists() else 'legacy Actor through legacy adapter',
                    suffix='executed teacher only', verification_sha256=sha(run/'verification.json')))
    data = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    assert all(np.isfinite(v).all() for v in data.values())
    np.savez_compressed(output/'dataset.npz', **data)
    meta = dict(created_at=now(), frames=len(data['stage']), episodes=provenance,
        stages=['entry', 'rise before 3.2s', 'capture', 'post-capture hold'],
        stage_counts=np.bincount(data['stage'], minlength=4).tolist(),
        teacher_executed_frames=int(data['teacher_executed'].sum()),
        dataset_sha256=sha(output/'dataset.npz'), independent_validation=False,
        limitations='Fixed successful trajectories; repeats and static frames do not add state coverage',
        old_dataset_frames_used=0, return_frames_used=0)
    write(output/'dataset.json', meta)
    print('[DATASET]', meta['frames'], meta['stage_counts'], flush=True)


def train(output, dataset, steps, device, initial=None):
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
    value = dict(kind='sustain_student_bc', contract=contract(), sources=sources(), parent_sources=parent.sources(),
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
