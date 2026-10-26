"""捕获记忆对照的独立来源与观测合同。"""
from pathlib import Path
import sys
import shutil
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'student_phase_v3'))
import common_phase as previous
from common_phase import parent,REPO,read,write,sha,now
from model_memory import MemoryActor

INITIAL=previous.HERE/'runs/2026-10-23_observed/actor.pt'
INITIAL_SHA='f5f829134614e6181ce268395c9257f8942bf24180f4c36580dc10bc884388a4'
RAW_DATASET=previous.DATASET
RAW_SHA=previous.DATASET_SHA
DATASET=HERE/'datasets/2026-10-23_memory'


def sources():
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))}


def contract():
    return dict(version='r4a_student_memory_v4',parent=previous.contract(),
        observation='raw305 unchanged +2 causal memory; 307 policy input; env/critic remain305/88',
        feature_semantics=['pitch-trigger latch','smoothstep(min(control_steps_since_trigger/40,1))'],
        reset='explicit env reset OR commanded entry OR stand_request=false',
        counter='50Hz; one update per policy input; no absolute episode age or evaluation horizon',
        routing='unchanged current-observation pitch<=-5deg gate; memory is not a second routing change',
        single_factor='memory channels zero versus exposed; same307 network and initial weights',
        initialization_sha256=INITIAL_SHA,raw_dataset_sha256=RAW_SHA,teacher_input=False,
        ppo_admission='six teacher-off extra10s AND robot PPO update/save/resume')


def snapshot(run):
    previous.snapshot(run)
    for rel in sources():
        target=run/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,target)
    write(run/'memory_sources.json',sources());write(run/'memory_contract.json',contract())


def require(run):
    previous.require(run)
    assert read(run/'memory_sources.json')==sources() and read(run/'memory_contract.json')==contract()
    for rel,digest in sources().items():assert sha(REPO/rel)==sha(run/'source_snapshot'/rel)==digest,rel


def initial_actor():
    assert sha(INITIAL)==INITIAL_SHA
    return previous.load_actor(INITIAL)[0]


def load_actor(path):
    path=Path(path).resolve();require(path.parent)
    value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['kind']=='sustain_memory_bc' and value['contract']==contract() and value['sources']==sources()
    r=read(path.parent/'result.json');assert sha(path)==r['actor_sha256'] and value['memory_mode']==r['memory_mode']
    require(DATASET);assert sha(DATASET/'dataset.npz')==value['dataset_sha256']==read(DATASET/'dataset.json')['dataset_sha256']
    assert value['initialization_sha256']==sha(INITIAL)==INITIAL_SHA
    assert all(torch.isfinite(v).all() for v in value['actor_state_dict'].values())
    model=MemoryActor(initial_actor(),value['memory_mode']).eval()
    model.load_state_dict(value['actor_state_dict'],strict=True)
    return model,value
