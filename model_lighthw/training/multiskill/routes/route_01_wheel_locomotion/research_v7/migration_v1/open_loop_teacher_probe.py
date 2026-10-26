"""Diagnostic replay of complete recorded teacher actions, with no live policy.

The shared student's physical worker must explicitly support this version
before execution is allowed. Loading/CPU checks never start an application.
Open-loop physical success cannot authorize autonomous behavior or PPO, nor
provide a new successful teacher/recovery training episode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError, read_teacher_bundle
from migration_v1.runtime_checks import select_case, check_source_preparation
from migration_v1 import student_worker


VERSION = 'r1_open_loop_teacher_sequence_v1'
ARTIFACT_KIND = 'diagnostic_open_loop_teacher_action_replay'


def action_digest(actions):
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


class RecordedTeacherSequence:
    """Original float32 actions indexed by absolute 50 Hz step, never observation."""
    def __init__(self, actions, source_time_s):
        actions = np.asarray(actions)
        times = np.asarray(source_time_s)
        if (actions.dtype != np.float32 or actions.ndim != 2 or actions.shape[1] != 16
                or not len(actions) or not np.isfinite(actions).all()
                or times.shape != (len(actions),) or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(len(times))*.02, atol=1.e-8, rtol=0.)):
            raise EvidenceError('Recorded replay requires finite original float32 N-by-16 actions on the absolute 50Hz grid')
        self.actions = actions.copy(); self.actions.flags.writeable = False
        self.source_time_s = times.copy(); self.source_time_s.flags.writeable = False
        self.original_action_sha256 = action_digest(self.actions)
        self.calls = 0

    def next_action(self, index, elapsed, device):
        if (isinstance(index, bool) or not isinstance(index, int) or index != self.calls
                or index < 0 or index >= len(self.actions)
                or not np.isfinite(elapsed) or abs(float(elapsed)-index*.02) > 1.e-8):
            raise EvidenceError('Open-loop action index/reset clock is discontinuous or sequence exhausted')
        if action_digest(self.actions) != self.original_action_sha256:
            raise EvidenceError('Recorded replay action bytes changed')
        # Copy preserves the original float32 values; no normalization, clamp,
        # interpolation, first-frame replacement or dependence on observations.
        result = torch.from_numpy(self.actions[index:index+1].copy()).to(device)
        self.calls += 1
        return result

    def execution_record(self):
        return dict(version=VERSION, open_loop_replay=True, source_sequence_frames=len(self.actions),
            sequence_calls=self.calls, actor_calls=0, teacher_calls=0,
            inference_mode='disabled_open_loop', action_source='recorded_teacher_executed_action',
            original_action_sha256=self.original_action_sha256,
            runtime_state_feedback=False, runtime_state_write=False, action_override=False,
            autonomous_behavior_accepted=False, formal_ppo_authorized=False)


def diagnostic_metadata():
    """Truthful source identity deliberately rejected by all existing label gates."""
    return dict(artifact_kind=ARTIFACT_KIND, open_loop_replay=True, actor_calls=0, teacher_calls=0,
        actor_inference='disabled_open_loop', actor_unique_action_source=False,
        recorded_sequence_unique_action_source=True, action_override=False,
        online_teacher_control=False, runtime_state_feedback=False, runtime_state_write=False,
        training_labels_allowed=False, autonomous_behavior_accepted=False,
        autonomous_anchor_pass=False, unified_actor_accepted=False, ppo_updates=0,
        formal_ppo_authorized=False)


def prepare(teacher_runs, slope_deg, output):
    """Strictly reread complete same-profile directions; never load an Actor."""
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Use a fresh open-loop diagnostic output directory')
    arrays, bundle = read_teacher_bundle(teacher_runs)
    manifest = dict(teacher_bundle=bundle)
    case, profile = select_case(manifest, slope_deg)
    selected = [index for index, episode in enumerate(bundle['episodes'])
                if episode['case']['slope_deg'] == slope_deg]
    if len(selected) != 1:
        raise EvidenceError('Choose exactly one complete source episode for the selected replay direction')
    episode_id = selected[0]; episode = bundle['episodes'][episode_id]
    source = Path(episode['directory'])
    with np.load(source/'seed43_case00_policy_trace.npz', allow_pickle=False) as trace:
        original = trace['executed_action'].copy()
        times = trace['teacher_time_s'].copy()
    accepted = arrays['action'][arrays['episode_id'] == episode_id]
    if (accepted.dtype != original.dtype or accepted.shape != original.shape
            or accepted.tobytes() != original.tobytes() or episode['samples'] != len(original)):
        raise EvidenceError('Accepted teacher labels differ from the original full executed sequence')
    sequence = RecordedTeacherSequence(original, times)
    clock = check_source_preparation(manifest, episode['command_timing']['actual_move_start_s'], case_slope=slope_deg)
    sources = {str(path): sha256(path) for path in student_worker.dependency_sources()}
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    prepared = dict(execution_kind=VERSION, sequence=sequence, output=output, sources=sources,
        case=case, profile=profile, preparation=clock, source_teacher_manifest=manifest,
        source_episode=episode, source_episode_id=episode_id,
        source_sequence_sha256=sequence.original_action_sha256,
        control_frames=len(original), trial_duration_s=len(original)*.02,
        provenance_checkpoints={},
        diagnostic_metadata=diagnostic_metadata())
    for recorded in bundle['episodes']:
        request = json.loads((Path(recorded['directory'])/'request.json').read_text())
        prepared['provenance_checkpoints'][request['checkpoint']] = request['checkpoint_sha256']
    verify_prepared(prepared)
    return prepared


def verify_prepared(prepared):
    if prepared.get('execution_kind') != VERSION or prepared.get('diagnostic_metadata') != diagnostic_metadata():
        raise EvidenceError('Unknown open-loop diagnostic execution identity')
    sequence = prepared['sequence']
    if (sequence.calls != 0 or prepared['control_frames'] != len(sequence.actions)
            or prepared['trial_duration_s'] != len(sequence.actions)*.02
            or action_digest(sequence.actions) != prepared['source_sequence_sha256']):
        raise EvidenceError('Prepared replay sequence/clock changed before execution')
    if prepared['output'].exists():
        raise EvidenceError('Open-loop output was created before the explicit diagnostic execution')
    verify_source_files(prepared)


def verify_source_files(prepared):
    """Repeat before App and after recording without rewriting historical evidence."""
    if any(sha256(name) != digest for name, digest in prepared['sources'].items()):
        raise EvidenceError('Open-loop execution source changed before application start')
    if action_digest(prepared['sequence'].actions) != prepared['source_sequence_sha256']:
        raise EvidenceError('Recorded sequence bytes changed during diagnostic execution')
    if any(sha256(name) != digest for name,digest in prepared['provenance_checkpoints'].items()):
        raise EvidenceError('Original teacher checkpoint provenance changed')
    for episode in prepared['source_teacher_manifest']['teacher_bundle']['episodes']:
        for name, digest in episode['source_files'].items():
            if sha256(Path(episode['directory'])/name) != digest:
                raise EvidenceError('Complete teacher evidence changed before diagnostic execution')


def run_simulation(prepared):
    verify_prepared(prepared)
    # Until the explicit shared branch has been reviewed/installed, this entry
    # fails before AppLauncher/output; never impersonate StudentActionInterface.
    if getattr(student_worker, 'OPEN_LOOP_DIAGNOSTIC_VERSION', None) != VERSION:
        raise EvidenceError('Shared student worker has no reviewed explicit open-loop diagnostic branch yet')
    return student_worker.run_simulation(prepared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-runs', nargs='+', type=Path, required=True)
    parser.add_argument('--slope-deg', type=float, choices=(25., -20.), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_simulation(prepare(args.teacher_runs, args.slope_deg, args.output))
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        parser.exit(2, 'Open-loop diagnostic rejected: '+str(error)+'\n')
    if not result.get('open_loop_complete_protocol_pass'):
        parser.exit(2, 'Recorded sequence did not reproduce the complete physical protocol.\n')


if __name__ == '__main__':
    main()
