"""Explicit 302-D deployable observation schema; original artifacts remain unchanged."""
from copy import deepcopy
from pathlib import Path

import torch

from migration_v1.contract import CONTRACT as BASE_CONTRACT, REPO, V7, sha256

ACTOR_DIM = 302
BASE_OBS_DIM = 300
CRITIC_DIM = 252
ACTION_DIM = 16
RESET_TIME_SCALE_S = 40.
PREPARATION_SCALE_S = 8.
MASK_START, MASK_END = BASE_CONTRACT['ignored_policy_columns']
PAYLOAD_COLUMN, TOWED_COLUMN = 298, 299
VERSION = 'r1_exploration_v1_clock302_dual_residual'


def contract():
    return dict(version=VERSION, policy_dim=ACTOR_DIM, base_policy_dim=BASE_OBS_DIM,
        critic_dim=CRITIC_DIM, action_dim=ACTION_DIM, base_contract=deepcopy(BASE_CONTRACT),
        clock_columns=[300, 302], reset_elapsed_scale_s=RESET_TIME_SCALE_S,
        preparation_scale_s=PREPARATION_SCALE_S, clock_clip=[0., 1.],
        clock_semantics=['elapsed seconds since actual reset / 40',
                         'fixed declared preparation seconds / 8'],
        actor_privileged_phase_or_progress=False, teacher_ready_input=False,
        ignored_policy_columns=[MASK_START, MASK_END],
        architecture='frozen_base300 + nominal_towing_selected_trainable_residual302',
        non_towing_residual_trainable=True, initial_function_is_not_behavior_acceptance=True,
        behavior_accepted=False, formal_ppo_authorized=False)


def append_task_clock(observations, reset_elapsed_s, preparation_seconds):
    """Pure tensor composition; caller must prove clocks originate from reset/task config.

    Negative/nonfinite clocks are errors. Values above the declared scales clip;
    no teacher readiness, physical phase, force or position is read here.
    """
    if (not isinstance(observations, torch.Tensor) or observations.ndim != 2
            or observations.shape[1] != BASE_OBS_DIM or not len(observations)
            or observations.dtype != torch.float32 or not torch.isfinite(observations).all()):
        raise ValueError('Base observations require finite nonempty float32 [N,300]')
    fields = []
    for value, scale in ((reset_elapsed_s, RESET_TIME_SCALE_S),
                         (preparation_seconds, PREPARATION_SCALE_S)):
        tensor = torch.as_tensor(value, device=observations.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.expand(len(observations))
        if tensor.shape != (len(observations),) or not torch.isfinite(tensor).all() or (tensor < 0).any():
            raise ValueError('Task clocks require finite nonnegative scalar or N-vector seconds')
        fields.append((tensor/scale).clamp(0., 1.)[:, None])
    return torch.cat((observations, *fields), dim=-1)


def source_files():
    """Policy/strict-initialization dependencies only, independent of runner edits.

    The runtime must additionally freeze environment, native RSL, simulator and
    its own runner dependencies. This inventory is not a simulation source gate.
    """
    root = Path(__file__).resolve().parent
    migration = root.parent/'migration_v1'
    files = [root/name for name in ('__init__.py', 'contract.py', 'policy.py', 'checkpoint.py')]
    files += [migration/name for name in ('__init__.py', 'contract.py', 'policy.py', 'bc.py',
        'student.py', 'dataset.py', 'preparation_sampling.py', 'ppo_policy.py')]
    from migration_v1.bc import source_signatures as bc_sources
    files += [Path(path) for path in bc_sources()]
    return sorted(set(path.resolve() for path in files))


def source_signatures():
    return {str(path): sha256(path) for path in source_files()}
