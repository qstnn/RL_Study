"""R4a示范训练的独立来源合同；旧split_v4保持原样。"""
from pathlib import Path
import hashlib
import shutil
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'split_v4'
sys.path.insert(0, str(BASE / 'scripts'))
from common import REPO, SOURCE, now, read, write, sha, manifest as base_manifest, snapshot as base_snapshot
from two_leg_balance_split_v4.spec import contract as base_contract

PHYSICS_FILES = ('common_guided.py', 'teacher.py', 'physics_worker.py', 'run.py', 'r4a_guided.sh')


def sources(learning=False):
    files = PHYSICS_FILES + (('learning.py',) if learning else ())
    return {**base_manifest(), **{str((HERE / name).relative_to(REPO)): sha(HERE / name) for name in files}}


def contract():
    return {**base_contract(2., 'rear', 'soft_transfer_v2'),
            'version': 'lighthw_r4a_guided_v5', 'initialization': 'verified teacher observations -> actor imitation -> autonomous PPO',
            'teacher_in_actor_path': False, 'actor_observation_dim': 305,
            'teacher_privileged': True, 'formal_ppo_budget': 200,
            'base_leg_exploration_std': .05, 'base_wheel_exploration_std': .03,
            'missed_hold_early_termination': False,
            'automatic_frozen_evaluation': False, 'automatic_promotion': False}


def snapshot(run, learning=False):
    base_snapshot(run)
    hashes = sources(learning)
    for rel in hashes:
        dest = run / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    write(run / 'source_manifest.json', hashes)
    write(run / 'contract.json', contract())


def digest_object(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_sources(folder, learning=False):
    if read(folder / 'source_manifest.json') != sources(learning):
        raise ValueError('Guided sources changed; use the recorded snapshot or repeat validation, never silently mix versions')
    if read(folder / 'contract.json') != contract():
        raise ValueError('Guided workflow contract mismatch')


def require_teacher(folder):
    """目的：成功示范必须来自原协议完整回合；部分保持不能伪装完整示范。"""
    folder = Path(folder).resolve()
    require_sources(folder)
    result = read(folder / 'result.json')
    if result.get('kind') != 'teacher_replay' or not result.get('dataset_eligible'):
        raise ValueError('Teacher has not passed repeated full-protocol verification; collection/pretraining is locked')
    for name, digest in result['trace_hashes'].items():
        if sha(folder / name) != digest:
            raise ValueError('Teacher trace changed')
    return result
