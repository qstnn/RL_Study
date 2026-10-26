"""离线BC/学生接口CPU测试；只有人工数组的3次优化，不是机器人训练。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1 import bc, student
from migration_v1.contract import SPEC_PATH, _load, install_actor_mask, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.policy import FrozenBaseTowingResidualActor, tensor_fingerprint


@pytest.fixture
def synthetic(tmp_path):
    torch.manual_seed(72)
    base = bc._mlp()
    install_actor_mask(base)
    x = np.random.default_rng(73).normal(size=(32, 300)).astype(np.float32)
    x[:, 298] = 0.; x[:, 299] = 1.
    with torch.no_grad():
        target = base(torch.from_numpy(x)).numpy()+.2
    arrays = dict(observation=x, action=target, episode_id=np.repeat([0, 1, 2, 3], 8),
        group_id=np.tile(np.repeat([0, 1, 2, 3], 2), 4),
        case_key=np.repeat(['up', 'down', 'up', 'down'], 8),
        stage=np.tile(np.repeat(['prepare', 'move', 'brake', 'hold'], 2), 4),
        successful=np.ones(32, dtype=bool))
    declared = deepcopy(_load(SPEC_PATH, '_r1_test_bc_spec').CONTRACT)
    declared['version'] = 'lighthw_r1_unified_v5'
    path = tmp_path/'synthetic_baseline.pt'
    torch.save(dict(model_state_dict={'actor.'+key: value for key, value in base.state_dict().items()},
        infos={'r1_unified_contract': declared}, iter=0), path)
    return base, arrays, path


def test_baseline_hash_semantics_mask_and_frozen_source(synthetic):
    base, _, path = synthetic
    actor, provenance = bc.load_baseline(path, sha256(path))
    assert tensor_fingerprint(base) == provenance['frozen_base_tensor_sha256']
    assert provenance['optimizer_restored'] is False
    assert all(not p.requires_grad for p in actor.base_actor.parameters())
    with pytest.raises(EvidenceError, match='SHA'):
        bc.load_baseline(path, '0'*64)
    saved = torch.load(path, weights_only=True)
    saved['infos']['r1_unified_contract']['wheel_damping'] = 1.
    torch.save(saved, path)
    with pytest.raises(EvidenceError, match='semantic mismatch'):
        bc.load_baseline(path, sha256(path))


def test_whole_episode_split_without_neighbor_leakage(synthetic):
    _, arrays, _ = synthetic
    train, validation, info = bc.split_episodes(arrays, [2, 3])
    assert set(arrays['episode_id'][train]) == {0, 1}
    assert set(arrays['episode_id'][validation]) == {2, 3}
    assert not set(train) & set(validation)
    assert info['neighboring_frames_split'] is False
    _, validation, info = bc.split_episodes(arrays)
    assert not len(validation)
    assert info['validation_kind'] == 'unavailable_no_independent_holdout'
    with pytest.raises(EvidenceError, match='required uphill/downhill'):
        bc.split_episodes(arrays, [1, 3])
    with pytest.raises(EvidenceError, match='unknown episode'):
        bc.split_episodes(arrays, [4])


def test_stage_balance_does_not_follow_longest_trajectory():
    group = np.array([0]*1+[1]*19+[2]*180)
    rows = bc.balanced_indices(group, np.arange(200), 12000, np.random.default_rng(17))
    counts = np.bincount(group[rows], minlength=3)
    assert np.max(abs(counts-4000)) < 220
    eligible = np.arange(20)
    rows = bc.balanced_indices(group, eligible, 100, np.random.default_rng(17))
    assert set(rows) <= set(eligible)


def test_synthetic_three_step_kernel_changes_only_residual(synthetic):
    base, arrays, _ = synthetic
    actor = FrozenBaseTowingResidualActor(base)
    before = tensor_fingerprint(actor.base_actor)
    protected = torch.from_numpy(arrays['observation']).clone()
    protected[:, 299] = 0.; protected[::2, 298] = 1.
    original = actor(protected).detach().clone()
    actor, report = bc._fit_arrays(actor, arrays, steps=3, batch_size=32, learning_rate=.001,
        seed=15, validation_episode_ids=[2, 3])
    assert report['optimizer_steps'] == 3  # Synthetic CPU optimizer only, no real teacher data.
    assert report['final_train']['equal_group_mean'] < report['initial_train']['equal_group_mean']
    assert report['held_out'] is not None
    assert report['formal_ppo_authorized'] is False
    assert tensor_fingerprint(actor.base_actor) == before
    assert torch.equal(actor(protected), original)
    actor.assert_invariants()


def test_real_entry_rejects_missing_teacher_before_load_or_output(monkeypatch, tmp_path):
    def refused(_):
        raise EvidenceError('Missing complete downhill teacher')
    monkeypatch.setattr(bc, 'read_teacher_bundle', refused)
    monkeypatch.setattr(bc, 'load_baseline', lambda *_: pytest.fail('baseline must not load before evidence'))
    output = tmp_path/'never_created'
    with pytest.raises(EvidenceError, match='Missing complete downhill'):
        bc.fit_from_runs(['unqualified'], 'absent.pt', output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['failed_label', 'float_success', 'nontowing', 'nonfinite'])
def test_kernel_rejects_unfit_labels(synthetic, kind):
    base, arrays, _ = synthetic
    if kind == 'failed_label':
        arrays['successful'][0] = False
    elif kind == 'float_success':
        arrays['successful'] = np.ones(32)
    elif kind == 'nontowing':
        arrays['observation'][0, 299] = 0.
    else:
        arrays['action'][0, 0] = np.nan
    with pytest.raises(EvidenceError):
        bc._fit_arrays(FrozenBaseTowingResidualActor(base), arrays, steps=1, batch_size=4, learning_rate=.001, seed=15)


def test_student_interface_has_only_observation_input_and_no_takeover(synthetic):
    base, arrays, _ = synthetic
    actor = FrozenBaseTowingResidualActor(base)
    interface = student.StudentActionInterface(actor, inference_mode='batch')
    observation = torch.from_numpy(arrays['observation'])
    assert torch.equal(interface(observation), actor(observation))
    record = interface.execution_record()
    assert record['actor_calls'] == 1 and record['teacher_calls'] == 0
    assert record['runtime_environment_verified'] is False
    assert record['formal_ppo_authorized'] is False
    with pytest.raises(TypeError):
        interface(observation, teacher='not_allowed')
    row = student.StudentActionInterface(actor, inference_mode='row')
    assert torch.allclose(row(observation), actor(observation), atol=1.e-6, rtol=1.e-6)
    assert student.StudentActionInterface(actor).inference_mode == 'row'
    assert torch.equal(row(observation), torch.cat([base(value[None]) for value in observation]))


def test_synthetic_packaging_roundtrip_is_not_physical_acceptance(synthetic, monkeypatch, tmp_path):
    _, arrays, path = synthetic
    episodes = []
    for index in range(2):
        directory = tmp_path/f'synthetic_source_{index}'
        directory.mkdir()
        (directory/'request.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(path))))
        episodes.append(dict(directory=str(directory), source_files={}))
    teacher = dict(schema='synthetic_interface_test_only_not_real_teacher_evidence', episodes=episodes)
    monkeypatch.setattr(bc, 'read_teacher_bundle', lambda _: (arrays, teacher))
    monkeypatch.setattr(student, 'read_teacher_bundle', lambda _: (arrays, teacher))
    output = tmp_path/'synthetic_candidate_only'
    result = bc.fit_from_runs(['synthetic_up', 'synthetic_down'], path, output,
        steps=1, batch_size=16, learning_rate=.001)
    assert result['teacher_bundle']['schema'].startswith('synthetic_interface_test_only')
    assert not result['autonomous_behavior_accepted'] and not result['formal_ppo_authorized']
    interface, metadata = student.load_student(output/'student.pt')
    assert interface(torch.from_numpy(arrays['observation'])).shape == (32, 16)
    assert metadata['critic_in_artifact'] is False
    with (output/'student.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(EvidenceError, match='SHA'):
        student.load_student(output/'student.pt')
