"""新协议真实恢复示范→保留历史数据的有界纠偏；不自动评估或PPO。"""
from pathlib import Path
import numpy as np
import torch
from common_smooth import (require, read, write, sha, arrays, contract, sources, input_actor,
    actor_network, finite, legacy_actor, legacy_dataset, LEGACY_ACTOR, LEGACY_DATASET,
    TEACHER, teacher_evidence, feedback_candidate, prefix_check, successful)
from common_takeover import successful_suffix
from learning import validate_arrays, imitation_weights, corrected_trace_arrays
from verify_smooth import verify


def evidence_check(folder):
    folder = Path(folder).resolve()
    require(folder)
    result, request = read(folder / 'result.json'), read(folder / 'request.json')
    if result['actor_sha256'] != sha(request['actor']):
        raise ValueError('Evidence actor changed')
    saved = input_actor(request['actor'])
    assert result['cases'] == read(folder / 'evaluation.json')['cases']
    assert {c['trace'] for c in result['cases']} == set(result['trace_hashes'])
    assert len({(c['seed'], c['switch_s'], c['trace_env']) for c in result['cases']}) == len(result['cases'])
    is_teacher = result['kind'] == 'smooth_teacher_replay'
    if is_teacher:
        assert request['mode'] == 'replay' and not result['autonomous']
        baseline = Path(request['baseline'])
        assert sha(baseline / 'result.json') == request['baseline_result_sha256']
        baseline_result = read(baseline / 'result.json')
        assert baseline_result['actor_sha256'] == result['actor_sha256']
        for name, digest in baseline_result['trace_hashes'].items():
            assert sha(baseline / name) == digest
        candidate = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
        assert result['candidate'] == candidate and result['teacher_result_sha256'] == sha(TEACHER / 'result.json')
    else:
        assert result['kind'] == 'smooth_autonomous_evaluation' and result['autonomous']
        assert request['mode'] == 'evaluate' and result['candidate'] is None
    model = actor_network().eval()
    model.load_state_dict(saved['actor_state_dict'])
    checks = []
    for name, digest in result['trace_hashes'].items():
        assert sha(folder / name) == digest
        d = arrays(folder / name)
        active = d['policy_active'].astype(bool)
        np.testing.assert_array_equal(active, d['active'][::20])
        with torch.no_grad():
            expected_actor = model(torch.from_numpy(d['policy_observation']).float()).numpy()
        actor_error = float(abs(expected_actor[active] - d['policy_actor_action'][active]).max())
        assert actor_error < 2e-6
        cases = [c for c in result['cases'] if c['trace'] == name]
        assert {c['trace_env'] for c in cases} == {0, 1}
        if is_teacher:
            switch = cases[0]['switch_s']
            np.testing.assert_array_equal(d['policy_takeover'][active], (d['policy_time'] >= switch - 1e-6)[active])
            labels = corrected_trace_arrays(d, candidate, read(folder / 'geometry.json'))
            teacher_error = float(abs(labels['actions'] - d['policy_teacher_action'][active]).max())
            assert teacher_error < 2e-5
            prefix = prefix_check(d, arrays(Path(request['baseline']) / f'trace_seed{cases[0]["seed"]}.npz'), switch)
        else:
            assert not d['policy_takeover'].any() and not d['policy_teacher_action'].any()
            assert all(c['switch_s'] is None for c in cases)
            teacher_error, prefix = 0., {}
        expected = np.where(d['policy_takeover'][..., None], d['policy_teacher_action'], d['policy_actor_action'])
        np.testing.assert_array_equal(expected[active], d['policy_action'][active])
        expected = expected.clip(-1, 1)
        expected[:10] = 0
        np.testing.assert_array_equal(expected[active], d['actions'][::20][active])
        checks.append(dict(trace=name, actor_error=actor_error, teacher_error=teacher_error,
                           prefix_max_error=max(prefix.values(), default=0.), executed_error=0.))
    verification = verify(folder)
    return dict(verification=verification, action_checks=checks, cases=len(result['cases']),
        full_successes=sum(successful(c) for c in result['cases']),
        autonomous_ppo_admission=(not is_teacher and len(result['cases']) >= 6 and all(successful(c) for c in result['cases'])))


def assemble(diagnosis):
    audit = evidence_check(diagnosis)
    result = read(Path(diagnosis) / 'result.json')
    if result['kind'] != 'smooth_teacher_replay' or not audit['full_successes']:
        raise ValueError('Need actual full-success teacher suffixes under smooth protocol')
    data, meta = legacy_dataset(LEGACY_DATASET)
    # 原5472帧完整保留。旧协议来源不能改名为新协议成功样本。
    data['protocol_id'] = np.zeros(len(data['phase']), dtype=np.int64)
    additions = []
    for c in result['cases']:
        if not successful(c):
            continue
        new = successful_suffix(arrays(Path(diagnosis) / c['trace']), c, int(data['episode'].max()) + 1)
        new['protocol_id'] = np.ones(len(new['phase']), dtype=np.int64)
        data = {k: np.concatenate((v, new[k])) for k, v in data.items()}
        additions.append(dict(seed=c['seed'], slot=c['trace_env'], switch_s=c['switch_s'], frames=len(new['phase'])))
    validate_arrays(data)
    return data, additions, audit


def dataset_check(folder):
    folder = Path(folder).resolve()
    require(folder)
    meta = read(folder / 'result.json')
    assert meta['kind'] == 'smooth_corrective_dataset'
    assert sha(folder / 'dataset.npz') == meta['dataset_sha256']
    assert sha(LEGACY_DATASET / 'dataset.npz') == meta['parent_dataset_sha256']
    assert sha(Path(meta['diagnosis']) / 'result.json') == meta['diagnosis_result_sha256']
    expected, _, _ = assemble(meta['diagnosis'])
    actual = arrays(folder / 'dataset.npz')
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    return actual, meta


def collect(run, request):
    data, additions, audit = assemble(request['diagnosis'])
    np.savez_compressed(run / 'dataset.npz', **data)
    write(run / 'result.json', dict(kind='smooth_corrective_dataset', diagnosis=request['diagnosis'],
        diagnosis_result_sha256=sha(Path(request['diagnosis']) / 'result.json'),
        parent_dataset=str(LEGACY_DATASET), parent_dataset_sha256=sha(LEGACY_DATASET / 'dataset.npz'),
        dataset_sha256=sha(run / 'dataset.npz'), frames=len(data['phase']),
        sample_kind_counts=np.bincount(data['sample_kind'], minlength=3).tolist(),
        protocol_counts=np.bincount(data['protocol_id'], minlength=2).tolist(),
        additions=additions, verification=audit, ppo_updates=0))
    dataset_check(run)


def temporal_indices(data):
    """只用同一实际示范中连续三帧稳定段，不跨接管、阶段、reset或反事实片段。"""
    idx = np.arange(2, len(data['phase']))
    valid = np.ones(len(idx), dtype=bool)
    for offset in (0, 1, 2):
        j = idx - offset
        valid &= data['episode'][j] == data['episode'][idx]
        valid &= data['phase'][j] == data['phase'][idx]
        valid &= data['sample_kind'][j] != 1
        valid &= np.isclose(data['time'][idx] - data['time'][j], offset * .02, atol=2e-5)
    valid &= np.isin(data['phase'][idx], [2, 4])
    return idx[valid]


def fit(data, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = actor_network()
    model.load_state_dict(legacy_actor(LEGACY_ACTOR)['actor_state_dict'])
    x, y = torch.from_numpy(data['observations']).float(), torch.from_numpy(data['actions']).float()
    phase, time = torch.from_numpy(data['phase']).long(), torch.from_numpy(data['time']).float()
    weight = imitation_weights(phase, time)
    weight *= torch.where(torch.from_numpy(data['is_correction']).bool(), 2., 1.)
    weight /= weight.mean()
    components = x.new_tensor([1.] * 14 + [4., 4.])
    components /= components.mean()
    temporal = torch.zeros(len(x), dtype=torch.bool)
    temporal[temporal_indices(data)] = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = []
    for epoch in range(epochs):
        for batch in torch.randperm(len(x)).split(512):
            output = model(x[batch])
            loss = ((output - y[batch]).square() * components).mean(-1).mul(weight[batch]).mean()
            ids = batch[temporal[batch]]
            if len(ids):
                predicted = model(x[ids]) - 2. * model(x[ids - 1]) + model(x[ids - 2])
                target = y[ids] - 2. * y[ids - 1] + y[ids - 2]
                loss += .1 * ((predicted - target).square() * components).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                error = (model(x) - y).square()
                history.append(dict(epoch=epoch + 1, weighted_component_mse=float((error * components).mean(-1).mul(weight).mean()),
                    leg_mse=float(error[:, :12].mean()), rear_wheel_mse=float(error[:, 14:].mean()),
                    by_phase=[dict(leg_mse=float(error[phase == i, :12].mean()), rear_wheel_mse=float(error[phase == i, 14:].mean())) for i in range(5)]))
    finite(model.state_dict())
    return model, history, int(temporal.sum())


def pretrain(run, request):
    data, meta = dataset_check(request['dataset'])
    model, history, triplets = fit(data, request['epochs'], request['seed'])
    path = run / 'actor.pt'
    torch.save(dict(kind='smooth_bc_actor', contract=contract(), sources=sources(), actor_state_dict=model.state_dict(),
        dataset=request['dataset'], dataset_sha256=meta['dataset_sha256'],
        initialization_actor=str(LEGACY_ACTOR), initialization_actor_sha256=sha(LEGACY_ACTOR),
        critic_transferred=False, optimizer_transferred=False, behavior_accepted=False), path)
    write(run / 'result.json', dict(kind='smooth_actor_imitation', actor_sha256=sha(path), history=history,
        epochs=request['epochs'], initialization='explicit third Actor warm start, fresh BC Adam', lr=1e-4,
        rear_wheel_component_weight=4., temporal_loss_weight=.1, temporal_triplets=triplets,
        automatic_evaluations=0, autonomous_behavior_tested=False, ppo_updates=0,
        review_status='awaiting_user_request'))
    from common_smooth import load_actor
    load_actor(path)
