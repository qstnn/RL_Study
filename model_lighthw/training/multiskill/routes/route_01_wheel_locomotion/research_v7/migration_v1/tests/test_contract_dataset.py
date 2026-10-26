"""合同/观测/拒绝门CPU测试；人工元数据不是教师成功证据，不启动仿真或拟合。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

V7 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V7))
contract = importlib.import_module('migration_v1.contract')
dataset = importlib.import_module('migration_v1.dataset')
FIXTURES = Path(__file__).parent/'fixtures'


@pytest.fixture
def excerpt():
    metadata = json.loads((FIXTURES/'ordinary_observation_excerpt.json').read_text())
    source = FIXTURES/'ordinary_observation_excerpt.npz'
    assert contract.sha256(source) == metadata['fixture_sha256']
    with np.load(source, allow_pickle=False) as z:
        policy = {key.removeprefix('policy_'): z[key] for key in z.files if key.startswith('policy_')}
        physics = {key.removeprefix('physics_'): z[key] for key in z.files if key.startswith('physics_')}
    return metadata, policy, physics


def test_layout_is_derived_from_actual_source_and_named_order():
    c = contract.checked_contract()
    assert c['ignored_policy_columns'] == [205, 285]
    assert c['layout']['joint_pos']['latest'] == [109, 125]
    assert c['layout']['task_context']['latest'] == [298, 300]
    assert c['observation_to_action_indices'] == [0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15]
    assert c['teacher_critic_dim'] == 252  # V7 actually executes V6; do not silently claim V8 268.
    assert c['formal_ppo_authorized'] is False


def test_real_observation_contents_history_and_mask(excerpt):
    metadata, policy, physics = excerpt
    contract.validate_runtime(metadata['runtime'], metadata['network_contract'])
    errors = dataset.validate_observation_trace(policy, physics, metadata['initial'])
    assert max(errors.values()) < 1.e-6
    original = policy['observation'].copy()
    masked = contract.masked_observations(original)
    np.testing.assert_array_equal(original, policy['observation'])
    np.testing.assert_array_equal(masked[:, :205], original[:, :205])
    np.testing.assert_array_equal(masked[:, 285:], original[:, 285:])
    assert not masked[:, 205:285].any()


@pytest.mark.parametrize('column', [12, 27, 42, 109, 189, 269, 289, 299])
def test_privileged_or_reordered_observation_corruption_is_rejected(excerpt, column):
    metadata, policy, physics = excerpt
    policy['observation'] = policy['observation'].copy()
    policy['observation'][4, column] += .1
    with pytest.raises(dataset.EvidenceError, match='contents do not reproduce'):
        dataset.validate_observation_trace(policy, physics, metadata['initial'])


def test_nonfinite_execution_label_rejected(excerpt):
    metadata, policy, physics = excerpt
    policy['executed_action'] = policy['executed_action'].copy()
    policy['executed_action'][4, 0] = np.nan
    with pytest.raises(dataset.EvidenceError, match='Nonfinite'):
        dataset.validate_observation_trace(policy, physics, metadata['initial'])


def test_actor_weights_gradients_and_existing_adam_mask():
    import torch
    torch.manual_seed(3)
    actor = torch.nn.Sequential(torch.nn.Linear(300, 512))
    optimizer = torch.optim.Adam(actor.parameters())
    weight = actor[0].weight
    optimizer.state[weight]['exp_avg'] = torch.ones_like(weight)
    optimizer.state[weight]['exp_avg_sq'] = torch.ones_like(weight)
    contract.install_actor_mask(actor, optimizer)
    actor(torch.ones((2, 300))).sum().backward()  # Synthetic gradient only; no optimizer.step or fitting.
    assert not torch.count_nonzero(weight.grad[:, 205:285])
    assert torch.count_nonzero(weight.grad[:, :205])
    contract.validate_actor_mask(actor, optimizer)
    optimizer.state[weight]['exp_avg'][0, 205] = 1.
    with pytest.raises(ValueError, match='Adam'):
        contract.validate_actor_mask(actor, optimizer)


def gpu_metadata(excerpt):
    """Artificial unit metadata, never an archived or admitted teacher episode."""
    metadata = deepcopy(excerpt[0])
    request = metadata['request_metadata']
    request['execution_physics']['device'] = 'cuda:0'
    request['position_iterations'], request['velocity_iterations'] = 16, 4
    request['cart_solver_iterations'] = [16, 4]
    request['base_network_contract'] = metadata['network_contract']
    metadata['initial'].update(robot_tensor_device='cuda:0', simulation_device='cuda:0', cuda_available=True)
    for name, values in metadata['solver'].items():
        if name == '/physicsScene':
            values['physxScene:maxPositionIterationCount'] = 16
            values['physxScene:maxVelocityIterationCount'] = 4
        else:
            values['physxArticulation:solverPositionIterationCount'] = 16
            values['physxArticulation:solverVelocityIterationCount'] = 4
    return metadata


def check_metadata(metadata):
    return dataset.validate_physics(metadata['request_metadata'], metadata['runtime'], metadata['initial'],
                                    metadata['solver'], metadata['geometry'])


def test_physics_metadata_validator_checks_content_not_device_label(excerpt):
    fake = gpu_metadata(excerpt)
    assert check_metadata(fake)['robot_solver'] == [16, 4]
    fake['initial']['robot_tensor_device'] = 'cpu'
    with pytest.raises(dataset.EvidenceError, match='confirmed GPU'):
        check_metadata(fake)


@pytest.mark.parametrize('change,expected', [
    ('loaded', 'ordinary'), ('solver', 'Actual USD'), ('nonzero_velocity', 'reset content'),
    ('height', 'reset content'), ('quaternion', 'reset content'), ('noise', 'clean'), ('hitch', 'geometry')])
def test_reject_invalid_physical_evidence(excerpt, change, expected):
    fake = gpu_metadata(excerpt)
    if change == 'loaded': fake['request_metadata']['initial_state_mode'] = 'loaded'
    if change == 'solver': fake['solver']['/World/envs/env_0/Robot/Base_link']['physxArticulation:solverVelocityIterationCount'] = 1
    if change == 'nonzero_velocity': fake['initial']['robot_root_state'][7] = -.2
    if change == 'height': fake['initial']['robot_root_state'][2] -= .1
    if change == 'quaternion': fake['initial']['robot_root_state'][3:7] = [1., 0., 0., 0.]
    if change == 'noise': fake['request_metadata']['noise'] = True
    if change == 'hitch': fake['runtime']['robot_hitch_local_m'][2] += .1
    with pytest.raises((dataset.EvidenceError, ValueError), match=expected):
        check_metadata(fake)


def test_snapshot_tampering_and_path_escape_rejected(tmp_path):
    relative = 'scripts/controller.py'
    source = tmp_path/'source_snapshot'/relative
    source.parent.mkdir(parents=True)
    source.write_text('original\n')
    manifest = {relative: contract.sha256(source)}
    for name in ('source_manifest.json', 'research_source_manifest.json'):
        (tmp_path/name).write_text(json.dumps(manifest))
    assert dataset.verify_snapshot(tmp_path) == manifest
    with pytest.raises(dataset.EvidenceError, match='Required execution source absent'):
        dataset.verify_snapshot(tmp_path, required=['scripts/missing.py'])
    source.write_text('modified\n')
    with pytest.raises(dataset.EvidenceError, match='Archived'):
        dataset.verify_snapshot(tmp_path)
    (tmp_path/'source_manifest.json').write_text(json.dumps({'../escape.py': 'x'}))
    with pytest.raises(dataset.EvidenceError, match='unsafe'):
        dataset.verify_snapshot(tmp_path)


def fake_episode(slope, profile='same'):
    return dict(observation=np.zeros((2, 300), np.float32), action=np.zeros((2, 16), np.float32),
        stage=np.array(['start', 'stop_hold']),
        manifest=dict(case=dict(slope_deg=slope, speed=.3), physics_profile=dict(test_only=profile)))


def test_bundle_missing_direction_or_mixed_physics_is_closed(tmp_path, monkeypatch):
    up, down = tmp_path/'up', tmp_path/'down'
    up.mkdir(); down.mkdir()
    monkeypatch.setattr(dataset, 'read_episode', lambda path: fake_episode(25. if path == up else -20., str(path)))
    with pytest.raises(dataset.EvidenceError, match='Both ordinary'):
        dataset.read_teacher_bundle([up])
    with pytest.raises(dataset.EvidenceError, match='different teacher physical'):
        dataset.read_teacher_bundle([up, down])


def test_even_complete_unit_bundle_cannot_authorize_student_or_ppo(tmp_path, monkeypatch):
    up, down = tmp_path/'up', tmp_path/'down'
    up.mkdir(); down.mkdir()
    monkeypatch.setattr(dataset, 'read_episode', lambda path: fake_episode(25. if path == up else -20.))
    arrays, manifest = dataset.read_teacher_bundle([up, down])
    assert arrays['episode_id'].tolist() == [0, 0, 1, 1]
    assert arrays['cache_kind'].item() == 'teacher_labels_for_offline_fit_only'
    assert not any(manifest[key] for key in ('student_training_ready', 'unified_actor_accepted', 'formal_ppo_authorized'))
    with pytest.raises(dataset.EvidenceError, match='Duplicate'):
        dataset.read_teacher_bundle([up, up, down])


def test_failed_behavior_never_reaches_observation_or_packing(tmp_path, monkeypatch, excerpt):
    # Unit-only admission plumbing. No synthetic run is written as a successful archive.
    fake = gpu_metadata(excerpt)
    request = fake['request_metadata']
    request.update(checkpoint=str(tmp_path/'checkpoint'), checkpoint_sha256=contract.sha256(FIXTURES/'ordinary_observation_excerpt.npz'))
    (tmp_path/'checkpoint').write_bytes((FIXTURES/'ordinary_observation_excerpt.npz').read_bytes())
    (tmp_path/'assets').mkdir(); (tmp_path/'assets/cart.urdf').write_text('unit only')
    data = {'request.json': request, 'runtime_check.json': fake['runtime'], 'initial_state_runtime.json': fake['initial'],
            'solver_runtime_usd.json': fake['solver'], 'rounded_cart_geometry.json': fake['geometry']}
    for name, value in data.items(): (tmp_path/name).write_text(json.dumps(value))
    for name in ['source_manifest.json', 'research_source_manifest.json', 'seed43_case00.csv', 'seed43_case00_physics_trace.npz', 'seed43_case00_policy_trace.npz', 'seed43_case00_motor_trace.npz']:
        (tmp_path/name).write_bytes(b'unit rejection sentinel')
    monkeypatch.setattr(dataset, 'verify_snapshot', lambda run, **kwargs: {'model_lighthw/lighthw_urdf/urdf/lighthw.urdf': fake['network_contract']['urdf_sha256']})
    monkeypatch.setattr(dataset, '_behavior_audit', lambda run: {'move_brake_hold_pass': False, 'checks': {'cart_net_route_1m': False}})
    with pytest.raises(dataset.EvidenceError, match='failed complete.*cart_net_route_1m'):
        dataset.read_episode(tmp_path)
