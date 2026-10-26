"""Versioned sources, canonical robot and R4b-only artifacts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = HERE.parents[5]
SOURCE = REPO / 'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw'
URDF = REPO / 'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
for folder in (ROUTE, SOURCE, ROUTE/'sustain_v7/student_feedback_v5/ppo_design'):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
from front_vertical_il_v1.settings import VERSION, contract


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def sources():
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.sh')) + [HERE/'geometry.json']
    for name in ('two_leg_balance_split_v4', 'two_leg_balance_smooth_v6', 'two_leg_balance_sustain_v7'):
        files += list((SOURCE/name).glob('*.py'))
    files += [ROUTE/'sustain_v7/continuing_ppo.py',
              ROUTE/'sustain_v7/student_feedback_v5/ppo_design/continuing_driver.py']
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(set(files))}


def snapshot(run):
    if sha(URDF) != '781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9':
        raise ValueError('Canonical robot changed')
    write(run/'sources.json', sources()); write(run/'contract.json', contract())
    for relative in sources():
        dest = run/'source_snapshot'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/relative, dest)
    assets = URDF.parent.parent
    shutil.copytree(assets, run/'assets', ignore=shutil.ignore_patterns('__pycache__', '*.usd'))
    write(run/'assets.json', {str(p.relative_to(run/'assets')): sha(p)
                              for p in (run/'assets').rglob('*') if p.is_file()})


def require(run):
    run = Path(run).resolve()
    if read(run/'contract.json') != contract() or read(run/'sources.json') != sources():
        raise ValueError('R4b contract/source drift; preserve this version and create a new version')
    for relative, digest in read(run/'sources.json').items():
        if sha(run/'source_snapshot'/relative) != digest: raise ValueError('Archived source mismatch')
    for relative, digest in read(run/'assets.json').items():
        if sha(run/'assets'/relative) != digest or sha(URDF.parent.parent/relative) != digest:
            raise ValueError('Robot asset drift')
    return read(run/'request.json')
