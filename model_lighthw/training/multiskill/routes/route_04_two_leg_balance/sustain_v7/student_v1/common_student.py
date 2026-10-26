"""v7学生独立来源合同；诊断执行源和旧模型只读。"""
from pathlib import Path
import sys
import shutil
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common_sustain as parent
from common_sustain import read, write, now, sha, REPO, actor_network


def sources():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh'))}


def contract():
    return dict(version='r4a_sustain_student_v1', parent=parent.contract(),
        observation='v7 policy 5x61; stand_request, no absolute episode age',
        action='original 12 leg targets + 4 wheel velocities; original limits',
        architecture='305-256-256-128-16 ELU; raw output then environment clamp',
        teacher_input=False, previous_actor_implicit_resume=False,
        ppo_updates=0, admission='six autonomous extra 10s AND real PPO update/save/resume')


def snapshot(run):
    parent.snapshot(run)
    for relative in sources():
        target = run/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/relative, target)
    write(run/'student_sources.json', sources())
    write(run/'student_contract.json', contract())


def require(run):
    parent.require(run)
    assert read(run/'student_sources.json') == sources(), 'Student source changed; create a new version'
    assert read(run/'student_contract.json') == contract()
    for relative, digest in sources().items():
        assert sha(run/'source_snapshot'/relative) == digest, relative


def load_actor(path):
    path = Path(path).resolve()
    require(path.parent)
    value = torch.load(path, map_location='cpu', weights_only=False)
    assert value['kind'] == 'sustain_student_bc' and value['contract'] == contract()
    assert value['sources'] == sources() and value['parent_sources'] == parent.sources()
    assert sha(path) == read(path.parent/'result.json')['actor_sha256']
    assert sha(Path(value['dataset'])/'dataset.npz') == value['dataset_sha256']
    assert all(torch.isfinite(t).all() for t in value['actor_state_dict'].values())
    model = actor_network().eval()
    model.load_state_dict(value['actor_state_dict'], strict=True)
    return model, value
