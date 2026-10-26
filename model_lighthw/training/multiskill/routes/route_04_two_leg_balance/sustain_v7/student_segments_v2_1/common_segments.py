"""短段接管独立来源合同；父级v7和student_v1执行源不变。"""
from pathlib import Path
import shutil
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'student_v1'))
import common_student as baseline
from common_student import parent,REPO,read,write,sha,now,actor_network

ACTOR=HERE.parent/'student_v1/runs/2026-10-22_bc_02/actor.pt'
ACTOR_SHA='7cc64a84ee1508451855c475bd9ebb012b4159b398414b623f0f21f6df3c2982'
BASELINE=ACTOR.parent.parent/'2026-10-22_bc_02_autonomous'
LEGACY_RUN=HERE.parent/'student_segments_v2/runs/2026-10-23_short_segments_4'
LEGACY_MANIFEST_SHA='4cf96a5bdfaac381d07bd2b9c953520c60a6f1b22ee72067c3c4533e6e07d14f'
LEGACY_CONTRACT_SHA='ad2a972d9ef6f3750a7feb99f61ecb14cbb8beeb490d56d221777b144a85c96e'
NOMINAL=HERE.parent/'runs/2026-10-22_sustain_30s_blend800'
PREVIOUS=HERE.parent/'student_v1/runs/2026-10-22_bc_01_takeover3200'


def sources():
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))}


def contract():
    return dict(version='r4a_student_segments_v2_1',parent=baseline.contract(),
        observation_unchanged=True,network_unchanged=True,physics_and_reward_unchanged=True,
        intervention='shadow teacher continuous; control switched only; no reset at takeover/release',
        correction_labels='only actual teacher execution in fully successful paired programs',
        student_execution_labels='excluded except frozen nominal hybrid prefix',
        ppo_admission='six teacher-off extra10s AND real PPO update/save/resume')


def snapshot(run):
    assert sha(ACTOR)==ACTOR_SHA
    baseline.snapshot(run)
    for relative in sources():
        target=run/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,target)
    write(run/'segment_sources.json',sources())
    write(run/'segment_contract.json',contract())


def require(run):
    baseline.require(run)
    manifest=read(run/'segment_sources.json')
    if manifest==sources():
        assert read(run/'segment_contract.json')==contract()
    else:
        assert sha(LEGACY_RUN/'segment_sources.json')==LEGACY_MANIFEST_SHA
        assert sha(LEGACY_RUN/'segment_contract.json')==LEGACY_CONTRACT_SHA
        assert manifest==read(LEGACY_RUN/'segment_sources.json')
        assert read(run/'segment_contract.json')==read(LEGACY_RUN/'segment_contract.json')
    for relative,digest in manifest.items():
        assert sha(REPO/relative)==sha(run/'source_snapshot'/relative)==digest,relative


def load_actor(path):
    path=Path(path).resolve()
    if path==ACTOR.resolve():
        assert sha(path)==ACTOR_SHA
        return baseline.load_actor(path)
    require(path.parent)
    value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['kind']=='sustain_segment_bc' and value['contract']==contract()
    assert value['sources']==sources() and value['parent_sources']==parent.sources()
    assert value['baseline_sources']==baseline.sources()
    assert sha(path)==read(path.parent/'result.json')['actor_sha256']
    dataset=Path(value['dataset']);require(dataset)
    assert sha(dataset/'dataset.npz')==value['dataset_sha256']==read(dataset/'dataset.json')['dataset_sha256']
    assert sha(value['initialization'])==value['initialization_sha256']
    assert all(torch.isfinite(t).all() for t in value['actor_state_dict'].values())
    actor=actor_network().eval();actor.load_state_dict(value['actor_state_dict'],strict=True)
    return actor,value
