"""Independent CPU clock/command integration checks; no PhysX or PPO proof."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

V7 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7))
from exploration_v1 import environment as exploration
from exploration_v1.contract import append_task_clock
from migration_v1.dataset import EvidenceError
from migration_v1.schedule import make_schedule


def clocks(steps=(0, 0, 0), preparation=(150, 374, 150)):
    term = SimpleNamespace(declared_preparation_steps=torch.tensor(preparation))
    def forbidden(*args, **kwargs):
        raise AssertionError('Observation composition must not read privileged state or advance native history')
    manager = SimpleNamespace(get_term=lambda name: term, compute=forbidden)
    base = SimpleNamespace(num_envs=len(steps), step_dt=.02,
        episode_length_buf=torch.tensor(steps), command_manager=manager,
        observation_manager=SimpleNamespace(compute=forbidden, compute_group=forbidden))
    return base, term


def test_declared_all64_commands_keep_towing_and_non_towing_protocols_separate():
    schedule = make_schedule(64)
    before = deepcopy(schedule)
    commands = exploration.declared_commands(schedule)
    cases = np.asarray(schedule['conditions'])
    towing = cases[:, 2] > 0
    assert len(set(map(tuple, cases))) == 64
    np.testing.assert_array_equal(commands['preparation_seconds'], np.where(towing & (cases[:, 0] < 0), 7.48, 3.))
    np.testing.assert_array_equal(commands['loaded_acceleration_mps2'], np.where(towing, .05, .3))
    np.testing.assert_array_equal(commands['goal_distance_m'], np.where(towing, 1.2, 1.))
    assert not commands['live_teacher_or_contact_ready']
    assert not commands['teacher_runtime_required']
    assert not commands['non_anchor_behavior_verified']
    assert schedule == before


def test_observe_is_pure_causal_composition_and_cannot_advance_history_twice():
    base, term = clocks((0, 149, 373))
    original = torch.arange(900, dtype=torch.float32).reshape(3, 300)
    saved = original.clone(), base.episode_length_buf.clone(), term.declared_preparation_steps.clone()
    first = exploration.observe(base, original)
    second = exploration.observe(base, original)
    expected = append_task_clock(original, base.episode_length_buf*.02, term.declared_preparation_steps*.02)
    assert torch.equal(first, second) and torch.equal(first, expected)
    assert torch.equal(first[:, :300], original)
    assert torch.equal(original, saved[0])
    assert torch.equal(base.episode_length_buf, saved[1])
    assert torch.equal(term.declared_preparation_steps, saved[2])
    assert first.data_ptr() != original.data_ptr()


def test_mixed_episode_resets_zero_only_the_reset_slot_clock():
    base, term = clocks((175, 600, 2100))
    original = torch.zeros(3, 300)
    before = exploration.observe(base, original)
    base.episode_length_buf[1] = 0
    after = exploration.observe(base, original)
    assert after[1, 300] == 0
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    assert torch.equal(after[:, 301], before[:, 301])
    assert before[2, 300] == 1.  # Explicit schema saturation after40s.
    assert torch.equal(term.declared_preparation_steps, torch.tensor([150, 374, 150]))


@pytest.mark.parametrize('mutate', [
    lambda base, obs: (setattr(base, 'step_dt', .01), obs)[1],
    lambda base, obs: (setattr(base, 'episode_length_buf', torch.tensor([-1, 0, 0])), obs)[1],
    lambda base, obs: (setattr(base, 'episode_length_buf', torch.tensor([0, 0])), obs)[1],
    lambda base, obs: (setattr(base, 'episode_length_buf', torch.tensor([float('inf'), 0., 0.])), obs)[1],
    lambda base, obs: (setattr(base.command_manager.get_term('base_velocity'), 'declared_preparation_steps',
                              torch.tensor([float('inf'), 374., 150.])), obs)[1],
    lambda base, obs: (base.command_manager.get_term('base_velocity').declared_preparation_steps.fill_(0), obs)[1],
    lambda base, obs: obs.double(),
    lambda base, obs: obs[:, :299],
    lambda base, obs: obs.fill_(float('nan')),
])
def test_malformed_clocks_and_original_observation_are_rejected(mutate):
    base, _ = clocks()
    with pytest.raises(EvidenceError):
        exploration.observe(base, mutate(base, torch.zeros(3, 300)))


@pytest.fixture(scope='module')
def native_fixture():
    """Reuse the source-AST harness, not a rewritten command state machine."""
    path = V7/'migration_v1/tests/test_terminal_preview.py'
    spec = importlib.util.spec_from_file_location('_independent_exploration_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    methods = module.native_methods.__wrapped__()
    return module, methods


def test_actual_native_command_body_starts_at_clock_boundary_and_anchors_after_preparation(native_fixture):
    module, methods = native_fixture
    base = module.env.__wrapped__(methods)
    term = base.command_manager.term
    original = torch.zeros(2, 300)
    # Arbitrarily large preparation displacement is never counted as task travel.
    base.scene['robot'].data.root_pos_w[:, 0] = torch.tensor([2., -3.])
    base.episode_length_buf[:] = torch.tensor([149, 373])
    term._update_command()
    assert torch.equal(base.r1_eval_phase, torch.zeros(2, dtype=torch.long))
    assert torch.equal(term.command, torch.zeros(2, 3))
    base.episode_length_buf += 1
    term._update_command()
    after = exploration.observe(base, original)
    assert torch.equal(base.r1_eval_phase, torch.ones(2, dtype=torch.long))
    assert torch.equal(base.r1_eval_progress, torch.zeros(2))
    assert torch.equal(base.r1_eval_origin, base.scene['robot'].data.root_pos_w)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([.001, .001]), rtol=0., atol=1.e-9)
    torch.testing.assert_close(after[:, 300], torch.tensor([3./40., 7.48/40.]), rtol=0., atol=2.e-8)
    torch.testing.assert_close(after[:, 301], torch.tensor([3./8., 7.48/8.]), rtol=0., atol=6.e-8)
    # The bridge has no access to native command/history computation.
    held_command = term.command.clone()
    repeated = exploration.observe(base, original)
    assert torch.equal(after, repeated) and torch.equal(term.command, held_command)


def test_native_completion_is_clock_event_not_behavior_acceptance(native_fixture):
    module, methods = native_fixture
    base = module.env.__wrapped__(methods)
    term = base.command_manager.term
    base.episode_length_buf[:] = 1000
    base.r1_eval_phase[:] = 3
    term.phase_start_step[:] = 650  # Seven seconds since command reached zero.
    term.vel_command_b.zero_()
    term._update_command()
    assert term.completed_hold_this_step.all()
    # No measured body velocity, rolling slip orcart progress participates here.
    assert not hasattr(base, 'route_behavior_pass')
    observed = exploration.observe(base, torch.zeros(2, 300))
    assert observed.shape == (2, 302)


def test_configure_wires_declared_commands_and_rewards_without_promoting_readiness(monkeypatch, tmp_path):
    schedule = make_schedule(64)
    cfg = SimpleNamespace()
    physical = {'schedule': schedule, 'runtime_pre_action_verified': False}
    calls = []
    monkeypatch.setattr(exploration.env_adapter, 'configure_environment',
        lambda output, num_envs, seed: (cfg, object(), physical))
    monkeypatch.setattr(exploration.command_adapter, 'apply_command_overrides',
        lambda actual_cfg, parameters: calls.append(('commands', actual_cfg, deepcopy(parameters))))
    monkeypatch.setattr(exploration.reward_adapter, 'apply_reward_overrides',
        lambda actual_cfg: calls.append(('rewards', actual_cfg)) or {'changed_terms': []})
    result, _, manifest = exploration.configure(tmp_path, 64, 43)
    assert result is cfg and all(call[1] is cfg for call in calls)
    assert calls[0][0] == 'commands' and calls[1][0] == 'rewards'
    assert calls[0][2] == exploration.declared_commands(schedule)
    assert cfg.episode_length_s == 65.
    assert manifest['physical'] is physical
    assert manifest['policy_dimension'] == 302 and manifest['critic_dimension'] == 252
    assert not manifest['training_runtime_ready'] and not manifest['final_behavior_pass']
    assert not manifest['observation_noise'] and not manifest['physical_randomization']
    for bad in (True, 39.98, float('nan'), float('inf')):
        with pytest.raises(EvidenceError): exploration.configure(tmp_path, episode_seconds=bad)


def test_make_environment_rejects_missing_configuration_and_disabled_terminal_preview():
    with pytest.raises(EvidenceError): exploration.make_environment(SimpleNamespace())
    with pytest.raises(EvidenceError):
        exploration.make_environment(SimpleNamespace(r1_exploration_configuration={}), terminal_preview=False)


@pytest.fixture
def command_readback(native_fixture):
    _, methods = native_fixture
    schedule = make_schedule(64)
    declared = exploration.declared_commands(schedule)
    cases = np.asarray(schedule['conditions'])
    speeds = torch.tensor(np.where(cases[:, 1:].sum(-1) == 0., 3.1, .3), dtype=torch.float32)
    term = object.__new__(methods['MigrationVelocityCommand'])
    preparation = np.asarray(declared['preparation_seconds'])
    term.declared_preparation_steps = torch.tensor(np.rint(preparation/.02).astype(np.int64))
    term.declared_cycle_offsets = torch.tensor(np.rint((preparation-3.)/.02).astype(np.int64))
    term.requested_vx = speeds.clone()
    cfg = SimpleNamespace(migration_schedule=schedule, command_profile='slew', unloaded_acceleration=1.,
        loaded_acceleration=torch.tensor(declared['loaded_acceleration_mps2']),
        eval_distance_m=torch.tensor(declared['goal_distance_m']), fixed_speed_mps=speeds)
    return SimpleNamespace(num_envs=64, step_dt=.02, cfg=cfg, episode_length_buf=torch.zeros(64, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda _: term), r1_terminal_preview_enabled=True)


def test_actual_command_source_and_arrays_pass_separately_from_physical_readback(command_readback):
    actual = exploration.verify_commands(command_readback)
    assert actual == exploration.declared_commands(command_readback.cfg.migration_schedule)
    assert not actual['non_anchor_behavior_verified']


@pytest.mark.parametrize('owner,field', [
    ('term', 'declared_preparation_steps'), ('term', 'declared_cycle_offsets'),
    ('term', 'requested_vx'), ('cfg', 'loaded_acceleration'), ('cfg', 'eval_distance_m'),
    ('cfg', 'fixed_speed_mps'),
])
def test_each_runtime_command_array_is_checked_not_just_metadata(command_readback, owner, field):
    target = command_readback.command_manager.get_term('base_velocity') if owner == 'term' else command_readback.cfg
    getattr(target, field)[0] += 1
    with pytest.raises(EvidenceError, match='vector differs'):
        exploration.verify_commands(command_readback)


@pytest.mark.parametrize('field,value', [('command_profile', 'step'), ('unloaded_acceleration', .3)])
def test_native_slew_and_unloaded_acceleration_cannot_silently_change(command_readback, field, value):
    setattr(command_readback.cfg, field, value)
    with pytest.raises(EvidenceError, match='slew configuration'):
        exploration.verify_commands(command_readback)


def test_forged_command_class_name_with_other_function_source_is_rejected(command_readback):
    old = command_readback.command_manager.get_term('base_velocity')
    wrong_type = type('MigrationVelocityCommand', (), {'_update_command': lambda self: None})
    wrong = wrong_type()
    wrong.__dict__.update(vars(old))
    command_readback.command_manager.get_term = lambda _: wrong
    with pytest.raises(EvidenceError, match='implementation'):
        exploration.verify_commands(command_readback)


def test_pre_action_delegates_physical_readback_and_checks_actual_reward_wiring(command_readback, monkeypatch):
    base = command_readback
    calls = []
    def verify_native(env, obs, output):
        calls.append((env, obs.clone(), output))
        return {'runtime_pre_action_verified': True, 'behavior_accepted': False}
    monkeypatch.setattr(exploration.env_adapter, 'verify_pre_action', verify_native)
    term_cfg = SimpleNamespace(func=exploration.reward_adapter.tracking, weight=2.)
    base.cfg.r1_exploration_configuration = {'rewards': {'changed_terms': ['track_broad'],
        'term_weights': {'track_broad': 2.}}}
    base.reward_manager = SimpleNamespace(get_term_cfg=lambda _: term_cfg)
    wrapped = SimpleNamespace(unwrapped=base)
    obs = exploration.observe(base, torch.zeros(64, 300))
    result = exploration.verify_pre_action(wrapped, obs)
    assert len(calls) == 1 and calls[0][0] is wrapped
    assert torch.equal(calls[0][1], obs[:, :300])
    assert result['actual_clocks_verified'] and result['actual_reward_functions_verified']
    assert not result['final_behavior_pass'] and not result['behavior_accepted']
    term_cfg.weight = 3.
    with pytest.raises(EvidenceError, match='reward override'):
        exploration.verify_pre_action(wrapped, obs)
    term_cfg.weight = 2.
    term_cfg.func = lambda env: 0.
    with pytest.raises(EvidenceError, match='reward override'):
        exploration.verify_pre_action(wrapped, obs)
