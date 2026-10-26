"""阶段单因素研究的独立来源合同；所有旧执行文件只读。"""
from pathlib import Path
import sys
import shutil
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'student_segments_v2_1'))
import common_segments as previous
from common_segments import parent,baseline,REPO,read,write,sha,now,actor_network,ACTOR,ACTOR_SHA,BASELINE
from model_phase import PhaseActor

DATASET=previous.HERE/'datasets/2026-10-23_teacher_segments'
DATASET_SHA='0d7e611d43a172a221736006eb78aedde3b6d856490a2aaa555c42c77d06b279'


def sources():
    paths=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))+[HERE/'analysis/phase_information.py']
    return {str(p.relative_to(REPO)):sha(p) for p in paths}


def contract():
    return dict(version='r4a_student_phase_v3',parent=previous.contract(),observation_unchanged=True,
        architecture='shared305-256-256-128ELU; two128-16 heads; both cloned from student02',
        single_factor='uniform average versus pitch<=-5deg observed routing',teacher_input=False,
        gate_state='current observed gravity, commanded entry phase and stand_request; stateless',
        dataset_sha256=DATASET_SHA,initialization_sha256=ACTOR_SHA,ppo_admission='six teacher-off extra10s AND robot PPO update/save/resume')


def snapshot(run):
    previous.snapshot(run)
    for relative in sources():
        target=run/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,target)
    write(run/'phase_sources.json',sources());write(run/'phase_contract.json',contract())


def require(run):
    previous.require(run)
    assert read(run/'phase_sources.json')==sources() and read(run/'phase_contract.json')==contract()
    for relative,digest in sources().items():assert sha(REPO/relative)==sha(run/'source_snapshot'/relative)==digest,relative


def load_actor(path):
    path=Path(path).resolve()
    if path==ACTOR.resolve():return previous.load_actor(path)
    require(path.parent);value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['kind']=='sustain_phase_bc' and value['contract']==contract() and value['sources']==sources()
    assert sha(path)==read(path.parent/'result.json')['actor_sha256']
    assert value['routing']==read(path.parent/'result.json')['routing']
    previous.require(DATASET)
    assert sha(DATASET/'dataset.npz')==value['dataset_sha256']==DATASET_SHA
    assert sha(ACTOR)==value['initialization_sha256']==ACTOR_SHA
    assert all(torch.isfinite(v).all() for v in value['actor_state_dict'].values())
    model=PhaseActor(actor_network(),value['routing']).eval()
    model.load_state_dict(value['actor_state_dict'],strict=True)
    return model,value
