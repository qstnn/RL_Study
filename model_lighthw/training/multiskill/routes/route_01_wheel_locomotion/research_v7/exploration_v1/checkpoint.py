"""Strict new-schema initialization/export; no old artifact is re-signed or resumed.

The old preparation BC candidate is an optional *failed-behavior initialization*.
It is not required for zero-residual exploration and never transfers Adam/Critic.
"""
from copy import deepcopy
import json
from pathlib import Path

import torch

from migration_v1.bc import load_baseline
from migration_v1.student import load_student
from .contract import REPO, V7, VERSION, source_files, source_signatures, sha256
from .policy import UnifiedResidualActor, ExplorationActorCritic

BASELINE_PATH = V7.parent/'unified_v5/initialization/selected/model_init.pt'
BASELINE_SHA256 = '3cd3eac8bf1f33ac915d03feb73feb1c2c215065ade56ff00320d3c1aabb1d30'
PREPARATION_PATH = V7/'results/2026-10-23_admission_continuation/bc_preparation_pilot_cpu/student.pt'
PREPARATION_SHA256 = '9f406bf2f10b41b83f980a3adf4b4ba660fb4e7949dcd26d3b70db3bef83225b'
ARTIFACT_SCHEMA = 'r1_exploration_v1_policy_checkpoint'


def initialize_policy(*, towing_initialization='zero', base_std, towing_std=None,
                      device='cpu', inference_mode='row', critic_hidden_dims=(512,256,128),
                      baseline_path=BASELINE_PATH, preparation_checkpoint=PREPARATION_PATH):
    """Return (new policy, verified provenance); no optimizer or fitting is created."""
    if towing_initialization not in ('zero','bc_preparation_9f406'):
        raise ValueError('Choose explicit zero or pinned bc_preparation_9f406 initialization')
    baseline_path=Path(baseline_path).resolve(strict=True)
    with torch.random.fork_rng(devices=[]):
        base,baseline=load_baseline(baseline_path,BASELINE_SHA256)
    parent=None; residual=None
    if towing_initialization=='bc_preparation_9f406':
        path=Path(preparation_checkpoint).resolve(strict=True)
        if sha256(path)!=PREPARATION_SHA256: raise ValueError('Optional towing initializer is not the pinned preparation candidate')
        source,manifest=load_student(path,device='cpu',inference_mode='row')
        if manifest['baseline']!=baseline: raise ValueError('Towing candidate and frozen base provenance differ')
        residual=source.actor.residual_actor
        parent=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),
            manifest_sha256=sha256(path.with_name('manifest.json')),manifest=deepcopy(manifest),
            known_autonomous_behavior='failed_both_anchor_directions',
            old_optimizer_or_critic_restored=False)
    actor=UnifiedResidualActor(base.base_actor,towing_residual=residual).to(device)
    policy=ExplorationActorCritic(actor,base_std=base_std,towing_std=towing_std,
        critic_hidden_dims=critic_hidden_dims,inference_mode=inference_mode)
    provenance=dict(kind='strict_pinned_initialization',checkpoint_provenance_verified=True,
        towing_initialization=towing_initialization,baseline=baseline,towing_parent=parent,
        new_clock_columns_initialization='zero_in_imported_towing_first_layer' if parent else 'zero_output_residual',
        original_base_inputs=300,non_towing_residual_initial_output='zero',
        initial_non_towing_std=list(map(float,base_std)),
        initial_towing_std=list(map(float,base_std if towing_std is None else towing_std)),
        no_old_critic_or_optimizer_restored=True,behavior_accepted=False,formal_ppo_authorized=False)
    policy.source_record=json.loads(json.dumps(provenance,allow_nan=False))
    policy.assert_invariants()
    return policy,deepcopy(policy.source_record)


def save_policy(path,policy,optimizer=None,*,metadata=None):
    """Fresh artifact only; adapter owns rollout-boundary enforcement and counters."""
    if type(policy) is not ExplorationActorCritic or policy.source_record.get('checkpoint_provenance_verified') is not True:
        raise ValueError('Export requires strict initialized exploration policy')
    path=Path(path).resolve()
    payload=dict(schema=ARTIFACT_SCHEMA,policy=policy.checkpoint(optimizer),
        metadata={} if metadata is None else deepcopy(metadata),
        environment_state_saved=False,behavior_accepted=False,formal_ppo_authorized=False)
    with path.open('xb') as stream: torch.save(payload,stream)
    return dict(path=str(path),sha256=sha256(path),schema=ARTIFACT_SCHEMA,
                behavior_accepted=False,formal_ppo_authorized=False)


def load_policy(path,*,device='cpu',restore_optimizer=False):
    """Revalidate old provenance then strict restore; return(policy,optimizer,metadata).

    The exported file is a numerical artifact, not a PhysX-state checkpoint or
    training permission. Caller separately verifies run/source/budget evidence.
    """
    saved=torch.load(Path(path).resolve(strict=True),map_location='cpu',weights_only=True)
    if (saved.get('schema')!=ARTIFACT_SCHEMA or saved.get('environment_state_saved') is not False
            or saved.get('behavior_accepted') is not False or saved.get('formal_ppo_authorized') is not False):
        raise ValueError('Not a declared exploration numerical artifact')
    state=saved['policy']['model_state_dict']; extra=state['_extra_state']; source=extra['source']
    if source.get('checkpoint_provenance_verified') is not True or source.get('kind')!='strict_pinned_initialization':
        raise ValueError('Artifact lacks strict initialization provenance')
    kwargs=dict(towing_initialization=source['towing_initialization'],base_std=source['initial_non_towing_std'],
        towing_std=source['initial_towing_std'],device=device,inference_mode=extra['inference_mode'],
        critic_hidden_dims=tuple(extra['critic_hidden_dims']),baseline_path=source['baseline']['path'])
    if source.get('towing_parent') is not None: kwargs['preparation_checkpoint']=source['towing_parent']['checkpoint']
    policy,current=initialize_policy(**kwargs)
    if current!=source: raise ValueError('Initialization provenance changed since export')
    optimizer=policy.make_optimizer() if restore_optimizer else None
    snapshot=deepcopy(saved['policy'])
    if not restore_optimizer: snapshot['optimizer_state_dict']=None
    policy.restore_checkpoint(snapshot,optimizer)
    return policy,optimizer,deepcopy(saved['metadata'])
