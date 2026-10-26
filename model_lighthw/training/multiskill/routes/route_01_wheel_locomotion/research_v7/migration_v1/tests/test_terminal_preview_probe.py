"""Synthetic CPU archive/comparison checks; zero simulator or PPO calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1 import terminal_preview_probe as probe
from migration_v1.dataset import EvidenceError


@pytest.fixture
def arrays():
    reference = dict(initial_observation=np.zeros((64, 300)), pre_observation=np.zeros((2, 64, 300)),
        pre_episode_steps=np.broadcast_to([[0], [1]], (2, 64)).copy(),
        post_episode_steps=np.broadcast_to([[1], [2]], (2, 64)).copy(),
        actor_raw=np.zeros((2, 64, 16)), manager_raw=np.zeros((2, 64, 16)),
        leg_target=np.zeros((2, 64, 12)), wheel_target=np.zeros((2, 64, 4)),
        robot_joint_pos=np.full((2, 64, 16), .1), robot_joint_vel=np.full((2, 64, 16), .2),
        initial_robot_joint_pos=np.full((64, 16), .1), initial_robot_joint_vel=np.zeros((64, 16)),
        returned_critic=np.zeros((2, 64, 252)), projected_critic=np.zeros((2, 64, 252)),
        held_critic=np.zeros((2, 64, 252)), terminated=np.zeros((2, 64), dtype=bool),
        truncated=np.zeros((2, 64), dtype=bool), captured=np.zeros((2, 64), dtype=bool),
        previewed=np.zeros((2, 64), dtype=bool))
    reference['returned_critic'][-1, :, 250] = .04/30.
    timeout = deepcopy(reference)
    timeout['projected_critic'][-1] = reference['returned_critic'][-1]
    timeout['returned_critic'][-1] = 0.
    timeout['post_episode_steps'][-1] = 0
    for name in ('truncated', 'captured', 'previewed'):
        timeout[name][-1] = True
    return timeout, reference


def test_exact_pair_accepts_only_terminal_semantics_never_behavior_or_ppo(arrays):
    result = probe.compare_arrays(*arrays)
    assert result['terminal_preview_gpu_pair_verified']
    assert result['actual_eligible_timeout_count'] == 64
    assert result['projected_vs_returned_reset_max_abs_error'] > 0
    assert result['pre_ppo_behavior_ready'] is False and result['formal_ppo_authorized'] is False
    assert result['ppo_updates'] == result['optimizer_steps'] == 0


def test_tiny_cross_app_physical_difference_reported_with_fixed_tolerance(arrays):
    arrays[0]['robot_joint_pos'] += 1.e-8
    result = probe.compare_arrays(*arrays)
    assert result['terminal_preview_gpu_pair_verified']
    assert 0 < result['differences']['robot_joint_pos']['max_abs_error'] < 1.e-7
    assert result['differences']['robot_joint_pos']['atol'] == 1.e-5


@pytest.mark.parametrize('name', ['robot_joint_pos', 'robot_joint_vel', 'initial_robot_joint_pos',
    'actor_raw', 'pre_observation', 'projected_critic'])
def test_separate_apps_diverging_cannot_be_forced_to_pass(arrays, name):
    arrays[0][name] += .001
    result = probe.compare_arrays(*arrays)
    assert not result['terminal_preview_gpu_pair_verified'] and result['errors']


def test_real_failure_is_separate_from_timeout_and_not_bootstrapped(arrays):
    arrays[0]['terminated'][-1, 3] = True
    result = probe.compare_arrays(*arrays)
    assert not result['terminal_preview_gpu_pair_verified']
    assert result['actual_eligible_timeout_count'] == 63
    assert any('natural physical termination' in error for error in result['errors'])


@pytest.mark.parametrize('mutation', ['missing_capture', 'early_timeout', 'reference_timeout',
    'counter_override', 'no_reset_difference', 'nonfinite'])
def test_incomplete_or_uninformative_diagnostic_rejected(arrays, mutation):
    a, b = arrays
    if mutation == 'missing_capture': a['captured'][-1, 0] = False
    if mutation == 'early_timeout': a['truncated'][0, 0] = True
    if mutation == 'reference_timeout': b['truncated'][-1, 0] = True
    if mutation == 'counter_override': a['pre_episode_steps'][0] = 1
    if mutation == 'no_reset_difference': a['returned_critic'][-1] = a['projected_critic'][-1]
    if mutation == 'nonfinite': a['robot_joint_pos'][0, 0, 0] = np.nan
    result = probe.compare_arrays(a, b)
    assert not result['terminal_preview_gpu_pair_verified']


@pytest.fixture
def run_pair(tmp_path, arrays):
    sources = {str(path.resolve()): probe.sha256(path) for path in (Path(probe.__file__),
        Path(probe.__file__).with_name('terminal_capture.py'))}
    directories = []
    for mode, data in zip(probe.MODES, arrays):
        out = tmp_path/mode; out.mkdir()
        for name in sources:
            target = out/'source_snapshot'/probe._relative(name)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, target)
        trace = out/'terminal_preview_trace.npz'; np.savez_compressed(trace, **data)
        request = dict(artifact_kind=probe.KIND, mode=mode, checkpoint_sha256='synthetic',
            checkpoint_manifest_sha256='synthetic', seed=43, num_envs=64,
            installed_commands={}, installed_rewards={}, original_physics_profile={},
            episode_counter_override=False, reset_physical_state_override=False,
            sample_actions=False, action_override=False, teacher_calls=0, ppo_updates=0, optimizer_steps=0,
            device='cuda:0', source_episode_length_s=65., original_command_methods=True,
            actor_inference='row', observation_noise=False, physical_randomization=False,
            requested_control_steps=2, diagnostic_episode_horizon_override=mode == 'short_timeout',
            actual_episode_length_s=.04 if mode == 'short_timeout' else 65.)
        review = dict(artifact_kind=probe.KIND, trace_sha256=probe.sha256(trace),
            actual_conditions_count=64, runtime_first_batch_verified=True)
        for filename, obj in [('request.json', request), ('terminal_preview_review.json', review),
            ('terminal_preview_source_manifest.json', sources)]:
            (out/filename).write_text(json.dumps(obj))
        directories.append(out)
    return directories


def test_archive_pair_verifies_sources_and_raw_sha(run_pair):
    result = probe.compare_runs(*run_pair)
    assert result['terminal_preview_gpu_pair_verified']
    assert result['source_count'] == 2  # Synthetic fixture, never actual GPU evidence.
    with (run_pair[0]/'terminal_preview_trace.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(EvidenceError, match='raw trace SHA'):
        probe.compare_runs(*run_pair)


def test_unknown_horizon_or_counter_override_not_allowed(run_pair):
    path = run_pair[0]/'request.json'
    request = json.loads(path.read_text()); request['episode_counter_override'] = True
    path.write_text(json.dumps(request))
    with pytest.raises(EvidenceError, match='override'):
        probe.compare_runs(*run_pair)


def test_prepare_is_read_only_and_requires_explicit_diagnostic_mode(tmp_path, monkeypatch):
    with pytest.raises(EvidenceError, match='mode'):
        probe.prepare('x', tmp_path/'no_out', 'formal')
    monkeypatch.setattr(probe.environment_probe, 'prepare', lambda checkpoint, output, steps:
        dict(checkpoint=checkpoint, output=Path(output), sources={}, steps=steps))
    for mode in probe.MODES:
        result = probe.prepare('synthetic', tmp_path/mode, mode)
        assert result['steps'] == 2 and result['num_envs'] == 64
        assert result['diagnostic_episode_horizon_override'] == (mode == 'short_timeout')
        assert not result['output'].exists()


def test_failed_projection_artifact_preserves_arrays_without_success_label(tmp_path):
    import torch
    from migration_v1 import terminal_capture
    held = torch.zeros(2, 252); projected = held.clone(); projected[1, 240] = 1.e-8
    physical = torch.ones(252, dtype=torch.bool)
    snapshots = {'first': {'q': torch.tensor([2.])}, 'last': {'q': torch.tensor([2.])}}
    error = terminal_capture._physical_column_error(held, projected, physical, snapshots, {'q': torch.tensor([2.])})
    report = probe.save_error_diagnostic(tmp_path, error)
    assert report['projection_diagnostic']['changed_columns'] == [240]
    assert report['projection_diagnostic']['strict_equality_required']
    assert report['ppo_updates'] == 0 and not report['failed_projection_trace']['successful_preview']
    path = tmp_path/report['failed_projection_trace']['file']
    assert probe.sha256(path) == report['failed_projection_trace']['sha256']
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved['projection/projected_critic'], projected.numpy())
        assert saved['projection/original_input_snapshot/q'][0] == 2.
    assert not (tmp_path/'terminal_preview_review.json').exists()
