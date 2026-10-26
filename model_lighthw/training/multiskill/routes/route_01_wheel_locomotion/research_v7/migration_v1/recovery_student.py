"""Strict new-schema recovery student loading; no old-schema relabeling."""
import json
from pathlib import Path

import torch

from .contract import sha256
from .dataset import EvidenceError
from .recovery_bc import SCHEMA, ARTIFACT_KIND, source_signatures, validate_holdout_lineage
from .recovery_dataset import read_recovery_bundle, source_student_provenance
from .student import StudentActionInterface


def load_recovery_student(checkpoint,*,device='cpu',inference_mode='row'):
    checkpoint=Path(checkpoint).resolve(strict=True)
    manifest=json.loads(checkpoint.with_name('manifest.json').read_text())
    expected=manifest.pop('student_checkpoint_sha256')
    if sha256(checkpoint)!=expected:raise EvidenceError('Recovery student checkpoint SHA differs')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved.get('metadata')!=manifest or manifest.get('schema')!=SCHEMA:
        raise EvidenceError('Recovery student schema/embedded provenance differs')
    if source_signatures()!=manifest.get('sources'):raise EvidenceError('Recovery student execution source signatures changed')
    if (manifest.get('artifact_kind')!=ARTIFACT_KIND or manifest.get('policy_dim')!=300 or manifest.get('action_dim')!=16
            or manifest.get('teacher_runtime_required') is not False or manifest.get('critic_in_artifact') is not False
            or manifest.get('formal_ppo_authorized') is not False or manifest.get('autonomous_behavior_accepted') is not False):
        raise EvidenceError('Not a deployable new-schema offline recovery candidate')
    declared=manifest['recovery_bundle'];warm=manifest['source_student']['path']
    arrays,current=read_recovery_bundle([e['directory'] for e in declared['teacher_bundle']['episodes']],
        [e['directory'] for e in declared['recovery_episodes']],warm,balance_preparation=declared['balance_preparation'])
    if current!=declared or manifest['teacher_bundle']!=declared['teacher_bundle']:
        raise EvidenceError('Current complete recovery/nominal evidence differs from fitted provenance')
    interface,parent,provenance=source_student_provenance(warm)
    if provenance!=manifest['source_student'] or parent['baseline']!=manifest['baseline']:
        raise EvidenceError('Recovery warm-start/base provenance differs')
    lineage=validate_holdout_lineage(arrays,current,parent,manifest['fitting']['split']['validation_episode_ids'])
    if lineage!=manifest['fitting']['validation_lineage']:
        raise EvidenceError('Recovery validation ancestry changed')
    interface.actor.load_state_dict(saved['actor_state_dict'],strict=True)
    interface.actor.to(device).eval().assert_invariants()
    return StudentActionInterface(interface.actor,inference_mode=inference_mode),manifest
