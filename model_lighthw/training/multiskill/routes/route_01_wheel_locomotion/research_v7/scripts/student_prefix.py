"""Explicit student prefix followed by executed teacher recovery, never autonomy.

Only actions are selected; physical states are never overwritten. Shadow
teacher actions during the prefix are diagnostics and must not become labels.
"""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINED_SCHEMA = 'r1_migration_v1_constrained_head_candidate_v1'


def load_prefix_student(checkpoint):
    """Explicit schema routing; each loader validates its complete own ancestry."""
    sys.path.insert(0, str(ROOT))
    from migration_v1.bc import SCHEMA as nominal_schema
    path = Path(checkpoint).resolve(strict=True)
    schema = json.loads(path.with_name('manifest.json').read_text()).get('schema')
    if schema == nominal_schema:
        from migration_v1.student import load_student
    elif schema == CONSTRAINED_SCHEMA:
        from migration_v1.constrained_student import load_student
    else:
        raise ValueError('Unsupported student-prefix checkpoint schema')
    policy, manifest = load_student(path, device='cpu', inference_mode='row')
    if manifest.get('schema') != schema:
        raise ValueError('Student-prefix strict loader returned a different schema')
    return policy, manifest


def prefix_steps(seconds):
    value = float(seconds)
    if not np.isfinite(value) or not .02 <= value <= 1. or abs(value/.02-round(value/.02)) > 1.e-8:
        raise ValueError('Student prefix must be 0.02 to 1 second on the original 20ms clock')
    return round(value/.02)


class StudentPrefix:
    def __init__(self, checkpoint, seconds, slope_deg, move_start):
        sys.path.insert(0, str(ROOT))
        from migration_v1.runtime_checks import check_source_preparation
        from migration_v1.contract import sha256
        self.steps = prefix_steps(seconds)
        self.policy, manifest = load_prefix_student(checkpoint)
        clock = check_source_preparation(manifest, move_start, case_slope=slope_deg)
        sources = dict(manifest['sources'])
        for path in (Path(__file__), ROOT/'migration_v1/runtime_checks.py'):
            sources[str(path.resolve())] = sha256(path)
        checkpoint = Path(checkpoint).resolve(strict=True)
        self.metadata = dict(version='r1_executed_student_prefix_then_teacher_v1',
            student_checkpoint=str(checkpoint), student_checkpoint_sha256=sha256(checkpoint),
            student_manifest_sha256=sha256(checkpoint.with_name('manifest.json')),
            source_sha256=sources, prefix_seconds=float(seconds), prefix_steps=self.steps,
            fixed_command_clock=clock, inference_mode='row',
            prefix_teacher_actions_are_shadow_only=True, student_prefix_is_teacher_label=False,
            teacher_executed_actions_require_complete_episode_pass=True,
            autonomous_behavior_accepted=False, formal_ppo_authorized=False)
        # Preserve the legacy metadata shape/source set for nominal BC parents.
        if manifest['schema'] == CONSTRAINED_SCHEMA:
            self.metadata.update(student_schema=CONSTRAINED_SCHEMA,
                student_loader='migration_v1.constrained_student.load_student')
        self.calls = 0; self.teacher_frames = 0

    def snapshot(self, output):
        from migration_v1.contract import REPO, sha256
        for name, digest in self.metadata['source_sha256'].items():
            source = Path(name)
            if sha256(source) != digest:
                raise ValueError('Student-prefix source changed before execution')
            target = Path(output)/'source_snapshot'/source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != digest:
                raise ValueError('Student-prefix source snapshot collision')
            if not target.exists():
                shutil.copy2(source, target)

    def select(self, elapsed, observation, teacher_action):
        """Called exactly once at each actual pre-action boundary, including takeover."""
        index = round(float(elapsed)/.02)
        if abs(float(elapsed)-index*.02) > 1.e-6 or index != self.calls+self.teacher_frames:
            raise ValueError('Student-prefix action clock is discontinuous')
        if observation.shape != (1, 300) or teacher_action.shape != (1, 16):
            raise ValueError('Student prefix requires original single-slot deployed shapes')
        active = index < self.steps
        student_action = torch.zeros_like(teacher_action)
        if active:
            self.policy.actor.to(observation.device)
            student_action = self.policy(observation)
            self.calls += 1
        else:
            self.teacher_frames += 1
        action = student_action if active else teacher_action
        scalar = lambda value: action.new_tensor([value])
        return action, dict(student_prefix_active=scalar(float(active)),
            teacher_executed=scalar(float(not active)), student_prefix_frame_index=scalar(index),
            student_prefix_takeover_frame=scalar(float(index == self.steps)),
            student_prefix_student_raw=student_action.clone(),
            student_prefix_teacher_raw=teacher_action.clone())

    def execution_record(self):
        from migration_v1.contract import sha256
        if any(sha256(name) != digest for name, digest in self.metadata['source_sha256'].items()):
            raise ValueError('Student-prefix source changed during execution')
        if self.calls != self.policy.actor_calls or self.policy.teacher_calls != 0:
            raise ValueError('Student-prefix inference call accounting mismatch')
        return dict(self.metadata, actual_student_frames=self.calls,
            actual_teacher_frames=self.teacher_frames, complete_success_verified=False)
