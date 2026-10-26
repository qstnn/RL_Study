"""隔离的教师接管与第三轮纠偏合同，不改原guided_v5的22份执行源码。"""
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GUIDED = HERE.parent
sys.path.insert(0, str(GUIDED))
from common_guided import (REPO, read, write, sha, now, contract as physical_contract,
                           sources as original_sources, snapshot as original_snapshot)
from common import asset_hashes
from learning import (load_actor as original_actor, teacher_evidence, correction_evidence,
                      dataset_check as original_dataset, validate_arrays, actor_network, finite)
from verify_trace import verify

FILES = ('common_takeover.py', 'worker_takeover.py', 'run_takeover.py', 'r4a_takeover.sh')


FEEDBACK = {'base': {}, 'velocity': {'velocity_gain': 2.},
            'rate': {'pitch_rate_gain': .15},
            'combined': {'velocity_gain': 2., 'pitch_rate_gain': .10}}


def feedback_candidate(name, base):
    """只调教师捕获反馈，不改物理或姿态/轮速边界；返回阶段参数保持。"""
    if name not in FEEDBACK:
        raise ValueError('Unknown bounded feedback variant')
    return {**base, **FEEDBACK[name]}


def contract():
    return dict(version='r4a_guided_takeover_v2', actor_dim=305, action_dim=16,
        physical_contract=physical_contract(), teacher_shadow_from_reset=True,
        physics_reset_at_takeover=False, successful_teacher_suffix_only=True,
        feedback_variants=FEEDBACK,
        automatic_evaluation=False, ppo_enabled=False,
        sample_kind={'original_teacher': 0, 'counterfactual': 1, 'verified_takeover': 2})


def sources():
    return {**original_sources(True), **{str((HERE / f).relative_to(REPO)): sha(HERE / f) for f in FILES}}


def snapshot(folder):
    original_snapshot(folder, learning=True)
    for rel in sources():
        dest = folder / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    write(folder / 'source_manifest.json', sources())
    write(folder / 'workflow_contract.json', contract())


def require(folder):
    folder = Path(folder)
    if read(folder / 'source_manifest.json') != sources() or read(folder / 'workflow_contract.json') != contract():
        raise ValueError('Takeover source/workflow mismatch')
    if read(folder / 'contract.json') != physical_contract() or read(folder / 'asset_manifest.json') != asset_hashes():
        raise ValueError('Takeover physical contract/assets changed')
    if any(sha(folder / 'source_snapshot' / k) != v for k, v in sources().items()):
        raise ValueError('Takeover archived sources changed')


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def successful(case):
    return bool(case['held'] and case['success'] and case['timeout'] and not case['failed'])


def successful_suffix(data, case, episode):
    """仅收完整协议成功回合中已实际执行的教师后缀；失败与Actor前缀绝不冒充示范。"""
    if not successful(case):
        raise ValueError('Failed/partial recovery is not a verified demonstration')
    i = case['trace_env']
    active = data['policy_active'][:, i].astype(bool)
    use = active & data['policy_takeover'][:, i].astype(bool)
    if not use.any():
        raise ValueError('Successful case has no applied teacher suffix')
    if np.any(np.diff(data['policy_takeover'][active, i].astype(int)) < 0):
        raise ValueError('Takeover must remain latched until episode ends')
    x, y = data['policy_observation'][use, i], data['policy_action'][use, i]
    np.testing.assert_array_equal(y, data['policy_teacher_action'][use, i])
    return dict(observations=x, actions=y, phase=x[:, -6:-1].argmax(-1),
        episode=np.full(len(x), episode, dtype=np.int64), time=data['policy_time'][use, i],
        is_correction=np.ones(len(x), dtype=np.int64), sample_kind=np.full(len(x), 2, dtype=np.int64))


def prefix_check(data, baseline, switch_s):
    """核对接管前实际物理/观测/Actor动作；控制步切换后的帧不参与前缀比较。"""
    steps = int(round(switch_s / .02))
    ticks = steps * 20
    errors = {}
    for key in ('t', 'task_t', 'gravity', 'com', 'com_vel', 'wheel_pos', 'force_z', 'actions',
                'policy_observation', 'policy_action'):
        n = steps if key.startswith('policy_') else ticks
        a, b = data[key][:n], baseline[key][:n]
        if a.shape != b.shape:
            raise ValueError('Missing comparable autonomous prefix')
        delta = float(np.max(np.abs(a - b)))
        if delta > 2e-6:
            raise ValueError(f'Autonomous prefix changed: {key}, error={delta}')
        errors[key] = delta
    return errors


def diagnosis_check(folder):
    """重新计算接管结果、前缀与执行标签，不能仅信任success标记。"""
    folder = Path(folder).resolve()
    require(folder)
    result, request = read(folder / 'result.json'), read(folder / 'request.json')
    if result.get('kind') != 'teacher_takeover_diagnosis':
        raise ValueError('Not a takeover diagnosis')
    original_actor(request['actor'])
    evidence = correction_evidence(request['baseline'])
    teacher = teacher_evidence(request['teacher'])
    candidate = feedback_candidate(request['feedback'], teacher['candidate'])
    if result.get('candidate') != candidate:
        raise ValueError('Recorded feedback parameters do not match request')
    if (result['actor_sha256'] != sha(request['actor']) or
            evidence['actor_sha256'] != sha(request['actor']) or
            result['teacher_result_sha256'] != sha(Path(request['teacher']) / 'result.json')):
        raise ValueError('Takeover actor/teacher lineage changed')
    if result['cases'] != read(folder / 'evaluation.json')['cases']:
        raise ValueError('Takeover reports differ')
    verification = verify(folder)
    model = actor_network().eval()
    model.load_state_dict(original_actor(request['actor'])['actor_state_dict'])
    from learning import corrected_trace_arrays
    audits = []
    for name, digest in result['trace_hashes'].items():
        if sha(folder / name) != digest:
            raise ValueError('Takeover trace changed')
        cases = [c for c in result['cases'] if c['trace'] == name]
        d = arrays(folder / name)
        baseline = arrays(Path(request['baseline']) / f'trace_seed{cases[0]["seed"]}.npz')
        errors = prefix_check(d, baseline, cases[0]['switch_s'])
        with torch.no_grad():
            predicted = model(torch.from_numpy(d['policy_observation']).float()).numpy()
        active = d['policy_active'].astype(bool)
        np.testing.assert_array_equal(active, d['active'][::20])
        actor_error = float(abs(predicted[active] - d['policy_actor_action'][active]).max())
        if actor_error > 2e-6:
            raise ValueError('Actor outputs not reproducible')
        labels = corrected_trace_arrays(d, candidate, read(folder / 'geometry.json'))
        teacher_error = float(abs(labels['actions'] - d['policy_teacher_action'][active]).max())
        if teacher_error > 2e-5:
            raise ValueError(f'Teacher shadow continuity mismatch: {teacher_error}')
        expected_takeover = d['policy_time'] >= cases[0]['switch_s'] - 1e-6
        np.testing.assert_array_equal(d['policy_takeover'][active], expected_takeover[active])
        selected = np.where(d['policy_takeover'][..., None], d['policy_teacher_action'], d['policy_actor_action'])
        np.testing.assert_array_equal(selected[active], d['policy_action'][active])
        executed = selected.clip(-1, 1)
        executed[:10] = 0  # 原环境出生后200ms零动作。
        np.testing.assert_array_equal(executed[active], d['actions'][::20][active])
        audits.append(dict(trace=name, prefix_max_error=max(errors.values()), actor_max_error=actor_error,
            teacher_max_error=teacher_error, executed_action_error=0.))
    return dict(verification=verification, traces=audits,
        full_successes=sum(successful(c) for c in result['cases']), cases=len(result['cases']))


def assemble(parent, diagnosis):
    """保留全部旧数据，新增经过物理成功验收的教师后缀；不新增离线反事实标签。"""
    data, meta = original_dataset(parent)
    audit = diagnosis_check(diagnosis)
    if audit['full_successes'] == 0:
        raise ValueError('No physically successful takeover; third correction is locked')
    request = read(Path(diagnosis) / 'request.json')
    if Path(meta['teacher']).resolve() != Path(request['teacher']).resolve():
        raise ValueError('Parent data and takeover use different teacher')
    data['sample_kind'] = data['is_correction'].astype(np.int64).copy()
    additions = []
    result = read(Path(diagnosis) / 'result.json')
    for c in result['cases']:
        if not successful(c):
            continue
        new = successful_suffix(arrays(Path(diagnosis) / c['trace']), c, int(data['episode'].max()) + 1)
        data = {k: np.concatenate((v, new[k])) for k, v in data.items()}
        additions.append(dict(trace=c['trace'], slot=c['trace_env'], switch_s=c['switch_s'], frames=len(new['phase'])))
    validate_arrays(data)
    return data, additions, audit


def dataset_check(folder):
    folder = Path(folder).resolve()
    require(folder)
    meta = read(folder / 'result.json')
    if meta.get('kind') != 'verified_takeover_dataset' or sha(folder / 'dataset.npz') != meta['dataset_sha256']:
        raise ValueError('Invalid takeover dataset')
    if sha(Path(meta['parent_dataset']) / 'dataset.npz') != meta['parent_dataset_sha256']:
        raise ValueError('Parent dataset changed')
    if sha(Path(meta['diagnosis']) / 'result.json') != meta['diagnosis_result_sha256']:
        raise ValueError('Diagnosis changed')
    expected, _, _ = assemble(meta['parent_dataset'], meta['diagnosis'])
    actual = arrays(folder / 'dataset.npz')
    if set(actual) != set(expected):
        raise ValueError('Dataset schema changed')
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    return actual, meta


def load_actor(path):
    path = Path(path).resolve()
    require(path.parent)
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('kind') != 'bc_takeover_actor' or value.get('workflow') != contract() or value.get('sources') != sources():
        raise ValueError('Actor is not this exact takeover workflow')
    if sha(path) != read(path.parent / 'result.json')['actor_sha256']:
        raise ValueError('Actor changed')
    finite(value['actor_state_dict'])
    actor_network().load_state_dict(value['actor_state_dict'], strict=True)
    _, meta = dataset_check(value['dataset'])
    if value['dataset_sha256'] != meta['dataset_sha256']:
        raise ValueError('Actor dataset changed')
    return value
