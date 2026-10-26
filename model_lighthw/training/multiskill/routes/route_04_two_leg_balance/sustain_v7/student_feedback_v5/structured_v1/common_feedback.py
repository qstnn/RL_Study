"""结构化本体反馈学生：隔离来源、原物理及模型合同。"""
from pathlib import Path
import sys
import shutil
import torch
HERE=Path(__file__).resolve().parent
V5=HERE.parent
sys.path.insert(0,str(V5.parent/'student_memory_v4'))
import common_memory as previous
from common_memory import parent,REPO,read,write,sha,now
sys.path.insert(0,str(V5/'kinematics'))
from body_estimator import TorchBodyEstimator,BodyEstimator
from capture_memory import CaptureMemory
PREFIX=previous.HERE/'runs/2026-10-23_exposed/actor.pt'
PREFIX_SHA='95bc6e9b381bbe6180f931dd4a291544a739e0604754bd08d3ecee0c8655c5ee'
DATASET=previous.DATASET
URDF=REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'


def sources():
    files=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))+[V5/'kinematics/body_estimator.py']
    return {str(p.relative_to(REPO)):sha(p) for p in files}


def contract():
    return dict(version='r4a_structured_feedback_v1',parent=previous.contract(),actor_dim=325,critic_dim=88,action_dim=16,
        observation='raw305 + latch/progress2 + trigger-previous-executed-action16 + URDF proprioceptive gap/rolling velocity2',
        memory='50Hz causal trigger latch, 40-step blend, reset/entry/return clear; no episode horizon',
        feedback='URDF kinematics from raw305 only; rolling estimate uses full wheel angular velocity; no privileged input',
        model='frozen learned rise prefix, learned masked affine capture head, smooth blend from latched previous executed action',
        prefix_sha256=PREFIX_SHA,environment_reward_physics_unchanged=True,
        admission='six autonomous extra10s plus real robot PPO update/save/resume and terminal-value checks')


def prefix_actor():
    assert sha(PREFIX)==PREFIX_SHA
    return previous.load_actor(PREFIX)[0]


def snapshot(run):
    previous.snapshot(run)
    for rel in sources():
        dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,dest)
    write(run/'feedback_sources.json',sources());write(run/'feedback_contract.json',contract())


def require(run):
    previous.require(run)
    assert read(run/'feedback_sources.json')==sources() and read(run/'feedback_contract.json')==contract()
    for rel,digest in sources().items():assert sha(run/'source_snapshot'/rel)==digest,rel


def load_actor(path):
    from model_feedback import FeedbackActor
    path=Path(path).resolve();require(path.parent)
    value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['kind']=='structured_feedback_bc' and value['contract']==contract()
    assert value['sources']==sources() and value['prefix_sha256']==PREFIX_SHA
    assert sha(path)==read(path.parent/'result.json')['actor_sha256']
    assert sha(DATASET/'dataset.npz')==value['dataset_sha256']
    model=FeedbackActor(prefix_actor(),value['variant']).eval()
    model.load_state_dict(value['actor_state_dict'],strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    return model,value
