"""Strict isolated constrained-head loader. Unknown/recovery schemas are rejected."""
import json
from pathlib import Path

import torch

from .constrained_refit import SCHEMA, HEAD_KEYS, source_signatures
from .contract import CONTRACT, sha256
from .dataset import EvidenceError
from .student import StudentActionInterface, load_student as load_parent


def load_student(checkpoint,*,device='cpu',inference_mode='row'):
    checkpoint = Path(checkpoint).resolve(strict=True)
    manifest = json.loads(checkpoint.with_name('manifest.json').read_text())
    digest = manifest.pop('student_checkpoint_sha256')
    if sha256(checkpoint) != digest:
        raise EvidenceError('Constrained checkpoint SHA differs from manifest')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema') != SCHEMA or saved.get('metadata') != manifest or manifest.get('schema') != SCHEMA:
        raise EvidenceError('Unknown constrained schema or embedded provenance mismatch')
    if manifest.get('sources') != source_signatures():
        raise EvidenceError('Constrained implementation/parent source changed')
    if (manifest.get('artifact_kind') != 'offline_constrained_student_candidate' or manifest.get('contract') != CONTRACT
            or manifest.get('policy_dim') != 300 or manifest.get('action_dim') != 16
            or manifest.get('critic_in_artifact') is not False or manifest.get('teacher_runtime_required') is not False
            or manifest.get('deployment_inputs_only') is not True or manifest.get('runtime_first_frame_override') is not False
            or manifest.get('autonomous_behavior_accepted') is not False
            or manifest.get('formal_ppo_authorized') is not False):
        raise EvidenceError('Constrained deployment contract differs')
    parent = manifest['parent_checkpoint']; path = Path(parent['path']).resolve(strict=True)
    if path == checkpoint or sha256(path) != parent['sha256'] or sha256(path.with_name('manifest.json')) != parent['manifest_sha256']:
        raise EvidenceError('Constrained parent checkpoint/manifest source differs')
    interface,parent_manifest = load_parent(path,device='cpu',inference_mode='row')
    if (parent_manifest['schema'] != manifest['parent_schema'] or parent_manifest['teacher_bundle'] != manifest['teacher_bundle']
            or parent_manifest['baseline'] != manifest['baseline']):
        raise EvidenceError('Constrained source teacher/base differs from strict parent')
    actor = interface.actor; state = saved['actor_state_dict']; original = actor.state_dict()
    if set(state) != set(original) or any(not torch.equal(value.cpu(),original[key].cpu()) for key,value in state.items()
            if key not in HEAD_KEYS and isinstance(value,torch.Tensor)):
        raise EvidenceError('Constrained artifact changed protected base or residual prefix')
    fitting = manifest['fitting']
    if (fitting.get('optimized_parameters') != sorted(HEAD_KEYS) or fitting.get('offline_fit_checks_pass') is not True
            or fitting.get('gradient_updates') != 0 or fitting.get('optimizer_steps') != 0
            or not fitting.get('checks') or not all(fitting['checks'].values())):
        raise EvidenceError('Missing constrained offline fitting checks')
    actor.load_state_dict(state,strict=True)
    actor.to(device).eval().assert_invariants()
    return StudentActionInterface(actor,inference_mode=inference_mode),manifest
