"""CPU fake native ordering only; zero Isaac steps and zero RSL PPO updates."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1 import ppo_runtime as runtime
from migration_v1.dataset import EvidenceError
from migration_v1.terminal_capture import (CRITIC_TERMS, PreResetCriticCapture,
    PreResetCriticCaptureMixin, terminal_semantics, validate_capture_manager)


def reader(name):
    def pure_reader(env):
        return env.physical_critic
    pure_reader.__name__ = name
    return pure_reader


class FakeManager:
    def __init__(self, env):
        self.env = env
        self.active_terms = {'critic': [item[0] for item in CRITIC_TERMS]}
        self.group_obs_dim = {'critic': (252,)}
        self._group_obs_concatenate = {'critic': True}
        self._group_obs_term_dim = {'critic': [(item[1],) for item in CRITIC_TERMS]}
        self._group_obs_term_cfgs = {'critic': [SimpleNamespace(history_length=0,
            noise=None, modifiers=None, func=reader(function), scale=scale,
            clip=None if name == 'context' else (-1., 1.) if name == 'height_scan' else (-100., 100.))
            for name, _, function, scale in CRITIC_TERMS]}
        self.calls = []

    def compute_group(self, group, update_history=False):
        self.calls.append((group, update_history))
        assert group == 'critic' and update_history is False
        return self.env.physical_critic  # Deliberate alias: caller must copy before reset.


class FakeNativeEnv:
    """Explicit fake of only the read/reset/command/observation ordering."""
    def __init__(self):
        self.num_envs = 3
        self.device = 'cpu'
        self.physical_critic = torch.zeros(3, 252)
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.reset_terminated = torch.zeros(3, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(3, dtype=torch.bool)
        self.observation_manager = FakeManager(self)
        self.next_terminated = torch.zeros(3, dtype=torch.bool)
        self.next_truncated = torch.zeros(3, dtype=torch.bool)
        self.native_steps = self.command_updates = self.policy_history_updates = 0
        self.policy_history = torch.zeros(3, 5)
        self._reset_idx(torch.arange(3))

    def _reset_idx(self, ids):
        self.physical_critic[ids] = -100.
        self.episode_length_buf[ids] = 0

    def step(self, action):
        self.native_steps += 1
        self.episode_length_buf += 1
        self.physical_critic[:] = action[:, :1]  # Fake physical end state.
        self.reset_terminated = self.next_terminated.clone()
        self.reset_time_outs = self.next_truncated.clone()
        ids = (self.reset_terminated | self.reset_time_outs).nonzero().flatten()
        if len(ids):
            self._reset_idx(ids)
        self.command_updates += 1
        self.physical_critic[:, 9:12] += 1.  # Post-reset command update, absent in capture.
        self.policy_history_updates += 1
        self.policy_history = self.policy_history.roll(-1, dims=1)
        self.policy_history[:, -1] = self.physical_critic[:, 0]
        obs = {'policy': self.policy_history.clone(), 'critic': self.physical_critic.clone()}
        return obs, torch.zeros(3), self.reset_terminated, self.reset_time_outs, {}


class CaptureEnv(PreResetCriticCaptureMixin, FakeNativeEnv):
    pass


def test_capture_before_native_reset_is_distinct_from_returned_next_episode():
    env = CaptureEnv()
    assert not hasattr(env, 'r1_terminal_capture')
    env.next_terminated[0] = True
    env.next_truncated[1] = True
    result = env.step(torch.tensor([[10.], [20.], [30.]]))
    capture = env.r1_terminal_capture.latest
    assert capture.captured.tolist() == [True, True, False]
    assert capture.terminal_episode_steps.tolist() == [1, 1, -1]
    assert capture.held_command_critic[:, 0].tolist() == [10., 20., 0.]
    assert capture.held_command_critic[:, 9].tolist() == [10., 20., 0.]
    assert result[0]['critic'][:, 0].tolist() == [-100., -100., 30.]
    assert result[0]['critic'][:, 9].tolist() == [-99., -99., 31.]
    assert env.observation_manager.calls == [('critic', False)]
    with pytest.raises(EvidenceError, match='held command'):
        capture.for_timeout_bootstrap()
    env.physical_critic.zero_()
    assert capture.held_command_critic[1, 0] == 20.  # Not a view into native/reset storage.


def test_extra_critic_read_preserves_fake_native_outputs_and_actor_history():
    original, capture = FakeNativeEnv(), CaptureEnv()
    for env in (original, capture):
        env.next_terminated[2] = True
    action = torch.tensor([[2.], [4.], [6.]])
    a, b = original.step(action), capture.step(action)
    assert torch.equal(a[0]['policy'], b[0]['policy'])
    assert torch.equal(a[0]['critic'], b[0]['critic'])
    assert capture.native_steps == capture.command_updates == capture.policy_history_updates == 1
    assert torch.equal(original.policy_history, capture.policy_history)
    assert capture.r1_terminal_capture.latest.for_timeout_bootstrap() is None


def test_true_terminal_wins_over_timeout_and_no_timeout_value_is_used():
    env = CaptureEnv()
    env.next_terminated[0] = env.next_truncated[0] = True
    env.step(torch.ones(3, 1))
    assert env.r1_terminal_capture.latest.for_timeout_bootstrap() is None


def test_no_done_makes_no_extra_read_and_explicit_reset_is_not_terminal():
    env = CaptureEnv()
    env.step(torch.ones(3, 1))
    first = env.r1_terminal_capture.latest
    assert not first.captured.any() and not env.observation_manager.calls
    env._reset_idx(torch.arange(3))
    assert env.r1_terminal_capture.latest is first
    env.step(torch.ones(3, 1))
    assert env.r1_terminal_capture.latest.step_number == 2
    assert not env.observation_manager.calls


@pytest.mark.parametrize('field,value', [('history_length', 1), ('noise', object()),
    ('modifiers', [object()]), ('scale', .8), ('clip', (-5., 5.)),
    ('func', lambda env: None)])
def test_unsafe_critic_contract_rejects_before_any_native_step(field, value):
    env = CaptureEnv()
    setattr(env.observation_manager._group_obs_term_cfgs['critic'][0], field, value)
    with pytest.raises(EvidenceError):
        env.step(torch.ones(3, 1))
    assert env.native_steps == 0 and not env.observation_manager.calls


def test_critic_semantic_columns_and_widths_are_explicit():
    result = terminal_semantics()
    assert result['command_dependent_columns'] == dict(velocity_commands=[9, 12],
        hold_route_displacement=[242, 243], hold_lateral_displacement=[243, 244],
        hold_elapsed=[244, 245], requested_vx=[245, 246], phase=[246, 250],
        phase_elapsed=[250, 251], remaining_distance=[251, 252])
    assert result['timeout_bootstrap_ready'] is False
    env = CaptureEnv()
    env.observation_manager._group_obs_term_dim['critic'][0] = (4,)
    with pytest.raises(EvidenceError, match='widths'):
        validate_capture_manager(env.observation_manager)


def test_missing_or_duplicate_terminal_capture_is_rejected():
    env = FakeNativeEnv()
    capture = PreResetCriticCapture()
    capture.begin(env)
    flags = torch.tensor([True, False, False])
    with pytest.raises(EvidenceError, match='disagree'):
        capture.finish(flags, torch.zeros(3, dtype=torch.bool))
    capture.abort()
    capture.begin(env)
    env.reset_terminated = flags
    capture.before_reset(env, torch.tensor([0]))
    with pytest.raises(EvidenceError, match='twice'):
        capture.before_reset(env, torch.tensor([0]))
    capture.abort()
    assert not capture.active and capture.latest is None


def test_nonfinite_terminal_state_aborts_without_valid_capture():
    env = CaptureEnv()
    env.next_terminated[0] = True
    with pytest.raises(EvidenceError, match='Nonfinite'):
        env.step(torch.full((3, 1), float('nan')))
    assert env.r1_terminal_capture.latest is None and not env.r1_terminal_capture.active
    assert env.command_updates == 0  # Failure occurred before reset and command update.


@pytest.fixture
def failed_admission(monkeypatch):
    report = dict(pre_ppo_behavior_ready=False, errors=['raw behavior failed'], formal_ppo_authorized=False)
    monkeypatch.setattr(runtime.admission, 'verify_behavior_admission', lambda *args: deepcopy(report))
    def unexpected(*args, **kwargs):
        raise AssertionError('No configuration/policy load is permitted after a rejected gate')
    monkeypatch.setattr(runtime.environment_probe, 'prepare', unexpected)
    monkeypatch.setattr(runtime.environment_probe, 'verify_sources', unexpected)
    return report


def test_behavior_failure_precedes_output_app_optimizer_and_other_options(tmp_path, failed_admission):
    out = tmp_path/'never_created'
    imports = set(sys.modules)
    with pytest.raises(runtime.BehaviorAdmissionRejected) as error:
        runtime.prepare('unloaded_candidate.pt', [], [], out, base_std=[float('nan')], rollout_steps=0)
    assert error.value.report == failed_admission
    assert not out.exists()
    assert not any(name.startswith(('isaaclab.app', 'isaacsim', 'omni.', 'rsl_rl'))
                   for name in set(sys.modules)-imports)


def test_prepared_pass_flag_cannot_bypass_new_raw_admission(tmp_path, failed_admission):
    forged = dict(checkpoint='x', anchor_runs=[], retention_runs=[], output=tmp_path/'no_output',
        behavior_review=dict(pre_ppo_behavior_ready=True))
    with pytest.raises(runtime.BehaviorAdmissionRejected):
        runtime.recheck_before_runtime(forged)
    assert not forged['output'].exists()


@pytest.fixture
def synthetic_passing_gate(monkeypatch, tmp_path):
    # Synthetic dependency substitution only; does not accept an actual artifact.
    checkpoint = tmp_path/'synthetic_candidate'; checkpoint.write_text('not a model')
    digest = runtime.sha256(checkpoint)
    monkeypatch.setattr(runtime.admission, 'verify_behavior_admission', lambda *args:
        dict(pre_ppo_behavior_ready=True, errors=[], checkpoint_sha256=digest))
    monkeypatch.setattr(runtime.environment_probe, 'prepare', lambda checkpoint, output, steps:
        dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, output=Path(output), sources={}))
    monkeypatch.setattr(runtime.environment_probe, 'verify_sources', lambda prepared: None)
    return checkpoint


def test_even_synthetic_passing_gate_is_plan_only_and_runtime_refuses(tmp_path, synthetic_passing_gate):
    output = tmp_path/'no_app_no_out'
    prepared = runtime.prepare(synthetic_passing_gate, ['a', 'b'], list(map(str, range(12))),
        output, base_std=[.05]*16)
    assert prepared['planned_num_envs'] == 64 and prepared['planned_update_attempts'] == 2
    assert prepared['planned_rollout_steps'] == 24
    assert prepared['runtime_loop_implemented'] is False
    assert prepared['real_ppo_updates_performed'] == 0
    assert not output.exists()
    with pytest.raises(EvidenceError, match='post-command Critic preview'):
        runtime.recheck_before_runtime(prepared)
    assert not output.exists()


@pytest.mark.parametrize('steps', [True, 0, 65, 2.5])
def test_bounded_rollout_plan_validation(tmp_path, synthetic_passing_gate, steps):
    with pytest.raises(EvidenceError, match='bounded'):
        runtime.prepare(synthetic_passing_gate, [], [], tmp_path/'no_out', base_std=[.05]*16,
            rollout_steps=steps)


@pytest.mark.parametrize('standard', [[.1]*15, [0.]*16, [float('inf')]*16])
def test_base_std_must_be_explicit_finite_positive(tmp_path, synthetic_passing_gate, standard):
    with pytest.raises(EvidenceError, match='base_std'):
        runtime.prepare(synthetic_passing_gate, [], [], tmp_path/'no_out', base_std=standard)
