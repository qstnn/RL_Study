"""持续站立独立来源/资产快照；保留v6，禁止旧模型隐式续训。"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
sys.path.insert(0, str(ROUTE / 'smooth_v6'))
from common_smooth import (REPO, read, write, sha, now, sources as old_sources,
    snapshot as old_snapshot, load_actor, actor_network, TEACHER, teacher_evidence, feedback_candidate)
from common import asset_hashes
from two_leg_balance_sustain_v7.spec import contract

SOURCE = REPO / 'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/two_leg_balance_sustain_v7'
ACTOR = ROUTE / 'smooth_v6/runs/2026-10-22_bc_corrected_04/actor.pt'
ACTOR_SHA = 'e986f82bdd9c4832084e4a9a000f2f05274bf914c7e869a4bc94dc8b6af88f65'
BASELINE = ROUTE / 'smooth_v6/runs/2026-10-22_bc_corrected_04_autonomous'
CONTROL = ROUTE / 'smooth_v6/diagnostics/capture_trigger04/runs/2026-10-22_capture_trigger_6'


def sources():
    paths = sorted(SOURCE.glob('*.py')) + sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh'))
    return {**old_sources(), **{str(p.relative_to(REPO)): sha(p) for p in paths}}


def check_legacy():
    for name in ('source_manifest.json', 'comparison_source_manifest.json'):
        for relative, digest in read(CONTROL / name).items():
            if sha(REPO / relative) != digest or sha(CONTROL / 'source_snapshot' / relative) != digest:
                raise ValueError('Frozen legacy source changed: ' + relative)
    if sha(ACTOR) != ACTOR_SHA:
        raise ValueError('Pinned Actor changed')
    load_actor(ACTOR)


def snapshot(run):
    check_legacy()
    old_snapshot(run)
    for relative in sources():
        dest = run / 'source_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)
    write(run / 'source_manifest.json', sources())
    write(run / 'contract.json', contract())
    write(run / 'workflow_contract.json', dict(version='r4a_sustain_v7',
        automatic_evaluation=False, automatic_training=False, ppo_enabled=False,
        old_actor_role='legacy observation adapter, diagnostic prefix until 3.2s only',
        teacher_role='privileged closed-loop diagnostic; not autonomous acceptance'))
    shutil.copy2(BASELINE / 'geometry.json', run / 'geometry.json')


def require(run):
    if read(run / 'source_manifest.json') != sources() or read(run / 'contract.json') != contract():
        raise ValueError('Sustain source/contract mismatch; create a separate run after changes')
    for relative, digest in sources().items():
        if sha(run / 'source_snapshot' / relative) != digest:
            raise ValueError('Archived source mismatch: ' + relative)
    if read(run / 'asset_manifest.json') != asset_hashes():
        raise ValueError('Canonical robot changed')
    for relative, digest in read(run / 'asset_manifest.json').items():
        if sha(run / 'assets' / relative) != digest:
            raise ValueError('Archived robot changed')
    assert read(run / 'workflow_contract.json')['version'] == 'r4a_sustain_v7'


def candidate(blend):
    base = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
    return dict(base, trigger_pitch_deg=-5., blend_s=blend)
