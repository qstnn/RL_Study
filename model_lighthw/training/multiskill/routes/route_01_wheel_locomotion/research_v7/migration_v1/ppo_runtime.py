"""Pre-App behavior gate and pending two-update integration plan; no training CLI.

There is deliberately no executable PPO loop yet. Isolated terminal command
preview has CPU source-method checks, but no actual GPU timeout validation.
Code/CPU checks cannot grant admission or replace raw behavior acceptance.
"""
from pathlib import Path

import numpy as np

from . import admission, environment_probe
from .contract import REPO, sha256
from .dataset import EvidenceError
from .terminal_capture import terminal_semantics


class BehaviorAdmissionRejected(EvidenceError):
    def __init__(self, report):
        self.report = report
        super().__init__('Pre-PPO raw behavior admission failed; no output/App/PPO was created: '
            + '; '.join(report.get('errors', ())))


def prepare(checkpoint, anchor_runs, retention_runs, output, *, base_std, rollout_steps=24):
    """Recompute strict raw behavior BEFORE output/App or policy optimizer work.

Even a passing result is only a source-frozen plan. It does not authorize or
    start PPO; terminal command-preview readback still requires independent
    GPU verification. No public supplied-pass or skip-gate option exists.
"""
    anchors, retention = list(anchor_runs), list(retention_runs)
    review = admission.verify_behavior_admission(checkpoint, anchors, retention)
    if review.get('pre_ppo_behavior_ready') is not True:
        raise BehaviorAdmissionRejected(review)
    if type(rollout_steps) is not int or not 1 <= rollout_steps <= 64:
        raise EvidenceError('Only a bounded one-to-64-step rollout is planned')
    standard = np.asarray(base_std, dtype=np.float32)
    if standard.shape != (16,) or not np.isfinite(standard).all() or np.any(standard <= 0):
        raise EvidenceError('Explicit base_std must contain 16 finite positive values')
    prepared = environment_probe.prepare(checkpoint, output, steps=2)
    if review.get('checkpoint_sha256') != prepared['checkpoint_sha256']:
        raise EvidenceError('Behavior review and strictly loaded candidate differ')
    sources = dict(prepared['sources'])
    files = [Path(__file__), Path(__file__).with_name('terminal_capture.py'),
             Path(__file__).with_name('admission.py'), Path(__file__).with_name('ppo_policy.py'),
             Path(__file__).with_name('ppo_adapter.py')]
    native = REPO.parent/'Isaaclab/source/isaaclab/isaaclab'
    files += [native/'envs/manager_based_rl_env.py', native/'managers/observation_manager.py',
              native/'managers/termination_manager.py', native/'managers/command_manager.py',
              native/'envs/mdp/observations.py', native/'utils/math.py']
    sources.update({str(path.resolve()): sha256(path) for path in files})
    prepared.update(artifact_kind='migration64_two_update_plan_only', sources=sources,
        anchor_runs=[str(Path(path).resolve()) for path in anchors],
        retention_runs=[str(Path(path).resolve()) for path in retention], behavior_review=review,
        base_std=standard.tolist(), base_std_origin='explicit caller vector; old std equality not asserted',
        planned_rollout_steps=rollout_steps, planned_update_attempts=2, planned_num_envs=64,
        planned_adapter='ControlledPPOAdapter/runtime_short_validation/full_first_batch',
        planned_resume='save empty rollout, new adapter load, compare policy/Adam/RNG, second rollout',
        terminal_semantics=terminal_semantics(), runtime_loop_implemented=False,
        real_ppo_updates_performed=0, optimizer_steps=0, formal_ppo_authorized=False)
    return prepared


def recheck_before_runtime(prepared):
    """A prepared dictionary or stale report can never substitute for raw evidence."""
    review = admission.verify_behavior_admission(prepared['checkpoint'],
        prepared['anchor_runs'], prepared['retention_runs'])
    if review.get('pre_ppo_behavior_ready') is not True:
        raise BehaviorAdmissionRejected(review)
    environment_probe.verify_sources(prepared)
    if review.get('checkpoint_sha256') != prepared['checkpoint_sha256']:
        raise EvidenceError('Candidate changed after preparation')
    if Path(prepared['output']).exists():
        raise EvidenceError('A future PPO short validation must use a fresh output directory')
    raise EvidenceError('PPO runtime is not implemented: terminal post-command Critic preview still '
        'requires actual GPU validation; code/CPU checks cannot grant runtime admission')
