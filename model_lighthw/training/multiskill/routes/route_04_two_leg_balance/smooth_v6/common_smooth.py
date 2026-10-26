"""独立低速漂移/抑振协议；旧模型仅作为显式迁移输入，不跨版本续训。"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
GUIDED = ROUTE / 'guided_v5'
TAKEOVER = GUIDED / 'takeover_v3'
sys.path.insert(0, str(TAKEOVER))
from common_takeover import (REPO, read, write, sha, now, arrays,
    sources as legacy_sources, snapshot as legacy_snapshot,
    load_actor as legacy_actor, dataset_check as legacy_dataset, successful, prefix_check,
    teacher_evidence, feedback_candidate, actor_network, finite)
from common import asset_hashes, SOURCE as OLD_SOURCE
from two_leg_balance_smooth_v6.spec import contract

SOURCE = OLD_SOURCE.parent / 'two_leg_balance_smooth_v6'
LEGACY_ACTOR = TAKEOVER / 'runs/2026-10-22_bc_corrected_03/actor.pt'
LEGACY_DATASET = TAKEOVER / 'runs/2026-10-22_corrected_dataset_03'
LEGACY_BASELINE = TAKEOVER / 'runs/2026-10-22_bc_corrected_03_autonomous'
TEACHER = GUIDED / 'runs/2026-10-22_teacher13_pair_final'
FILES = ('common_smooth.py', 'worker_smooth.py', 'run.py', 'learning_smooth.py', 'verify_smooth.py', 'r4a_smooth.sh')


def sources():
    paths = [HERE / name for name in FILES] + sorted(SOURCE.glob('*.py'))
    return {**legacy_sources(), **{str(p.relative_to(REPO)): sha(p) for p in paths}}


def snapshot(run):
    legacy_snapshot(run)
    for name in sources():
        dest = run / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, dest)
    write(run / 'source_manifest.json', sources())
    write(run / 'contract.json', contract())
    write(run / 'workflow_contract.json', dict(version='r4a_smooth_v6', automatic_evaluation=False,
        ppo_enabled=False, legacy_transfer_explicit=True, protocol_changed_by_user=True))


def require(run):
    run = Path(run)
    if read(run / 'source_manifest.json') != sources() or read(run / 'contract.json') != contract():
        raise ValueError('Smooth v6 source/protocol mismatch')
    if read(run / 'asset_manifest.json') != asset_hashes():
        raise ValueError('Smooth v6 assets changed')
    if read(run / 'workflow_contract.json') != dict(version='r4a_smooth_v6', automatic_evaluation=False,
            ppo_enabled=False, legacy_transfer_explicit=True, protocol_changed_by_user=True):
        raise ValueError('Smooth workflow contract changed')
    if any(sha(run / 'source_snapshot' / name) != digest for name, digest in sources().items()):
        raise ValueError('Archived sources changed')


def load_actor(path):
    import torch
    path = Path(path).resolve()
    require(path.parent)
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('kind') != 'smooth_bc_actor' or value.get('contract') != contract() or value.get('sources') != sources():
        raise ValueError('Actor is not this exact smooth workflow')
    if sha(path) != read(path.parent / 'result.json')['actor_sha256']:
        raise ValueError('Actor hash changed')
    finite(value['actor_state_dict'])
    actor_network().load_state_dict(value['actor_state_dict'], strict=True)
    from learning_smooth import dataset_check
    _, meta = dataset_check(value['dataset'])
    if value['dataset_sha256'] != meta['dataset_sha256']:
        raise ValueError('Actor dataset changed')
    if sha(value['initialization_actor']) != value['initialization_actor_sha256']:
        raise ValueError('Explicit BC initialization actor changed')
    legacy_actor(value['initialization_actor'])
    return value


def input_actor(path):
    path = Path(path).resolve()
    if path == LEGACY_ACTOR.resolve():
        return legacy_actor(path)
    return load_actor(path)
