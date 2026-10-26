"""Synthetic CPU algebra/provenance checks; no real model artifact or simulation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1 import constrained_refit as fit,constrained_student as student,constrained_student_worker as worker
from migration_v1.bc import _mlp
from migration_v1.contract import CONTRACT,install_actor_mask,sha256
from migration_v1.dataset import EvidenceError
from migration_v1.policy import FrozenBaseTowingResidualActor,tensor_fingerprint


@pytest.fixture
def synthetic():
    torch.manual_seed(91)
    base=_mlp();install_actor_mask(base)
    parent=FrozenBaseTowingResidualActor(base).eval()
    rng=np.random.default_rng(92)
    x=rng.normal(size=(32,300)).astype(np.float32);x[:,298]=0.;x[:,299]=1.
    desired=deepcopy(parent)
    with torch.no_grad():
        desired.residual_actor[-1].bias.add_(torch.linspace(-.1,.15,16))
        desired.residual_actor[-1].weight.add_(torch.from_numpy(rng.normal(scale=.005,size=(16,128)).astype(np.float32)))
        target=torch.cat([desired(torch.from_numpy(row)[None]) for row in x]).numpy()
    arrays=dict(observation=x,action=target,group_id=np.repeat(np.arange(4),8),
        episode_id=np.repeat([0,1],16),successful=np.ones(32,bool))
    return parent,arrays


def test_cpu_algebra_exact_first_frames_changes_only_last_residual_linear(synthetic):
    parent,arrays=synthetic
    before=[tensor_fingerprint(parent.base_actor),tensor_fingerprint(parent.residual_actor)]
    candidate,report=fit.fit_last_layer(parent,arrays)
    assert report['gradient_updates']==0 and report['optimizer_steps']==0
    assert report['offline_fit_checks_pass'] and report['float32_anchor_raw_max']<1.e-5
    assert report['anchor_indices']==[0,16]
    assert before==[tensor_fingerprint(parent.base_actor),tensor_fingerprint(parent.residual_actor)]
    for key,value in parent.state_dict().items():
        if isinstance(value,torch.Tensor) and key not in fit.HEAD_KEYS:
            assert torch.equal(value,candidate.state_dict()[key])
    protected=torch.from_numpy(arrays['observation']).clone();protected[:,299]=0.;protected[::2,298]=1.
    with torch.inference_mode():assert torch.equal(parent(protected),candidate(protected))
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('ridge',[0.,-1.,float('nan'),float('inf')])
def test_ridge_must_be_finite_positive(synthetic,ridge):
    with pytest.raises(ValueError,match='finite and positive'):
        fit.fit_last_layer(*synthetic,ridge=ridge)


def test_duplicate_first_observations_are_rejected_instead_of_pseudoinverse_fake_fit(synthetic):
    parent,arrays=synthetic
    arrays['observation'][16]=arrays['observation'][0]
    with pytest.raises(EvidenceError,match='not independent at float32'):
        fit.fit_last_layer(parent,arrays)


def test_nonfinite_or_unsuccessful_labels_cannot_enter_kernel(synthetic):
    parent,arrays=synthetic
    arrays['successful'][0]=False
    with pytest.raises(EvidenceError,match='executed successful labels'):
        fit.fit_last_layer(parent,arrays)
    arrays['successful'][0]=True;arrays['action'][0,0]=np.nan
    with pytest.raises(EvidenceError,match='executed successful labels'):
        fit.fit_last_layer(parent,arrays)


@pytest.fixture
def artifact(tmp_path,monkeypatch,synthetic):
    """Synthetic-only isolated artifact roundtrip; real provenance loader is mocked."""
    parent,arrays=synthetic
    parent_dir=tmp_path/'synthetic_parent';parent_dir.mkdir()
    cp=parent_dir/'student.pt';cp.write_bytes(b'synthetic parent identity, not a real robot model')
    (parent_dir/'manifest.json').write_text('{}')
    teacher=dict(episodes=[dict(directory=str(tmp_path),source_files={}),dict(directory=str(tmp_path),source_files={})])
    metadata=dict(schema='synthetic_old_schema',teacher_bundle=teacher,baseline=dict(synthetic=True))
    def load(*args,**kwargs):return SimpleNamespace(actor=deepcopy(parent)),deepcopy(metadata)
    monkeypatch.setattr(fit,'load_parent',load);monkeypatch.setattr(student,'load_parent',load)
    monkeypatch.setattr(fit,'read_teacher_bundle',lambda paths:(arrays,teacher))
    monkeypatch.setattr(fit,'prepare_sampling',lambda *args,**kwargs:(arrays,dict(synthetic=True)))
    out=tmp_path/'synthetic_constrained';report=fit.fit_from_checkpoint(cp,out)
    return out,cp,parent,report


def test_new_schema_roundtrip_does_not_claim_behavior_or_ppo(artifact):
    out,_,_,report=artifact
    interface,manifest=student.load_student(out/'student.pt')
    assert manifest['schema']==fit.SCHEMA
    assert not manifest['formal_ppo_authorized'] and not manifest['autonomous_behavior_accepted']
    assert not manifest['runtime_first_frame_override']
    assert interface.teacher_calls==0
    assert report['student_checkpoint_sha256']==sha256(out/'student.pt')


@pytest.mark.parametrize('change',['checkpoint','parent','prefix','schema','source','behavior_claim'])
def test_new_loader_refuses_tampering_or_schema_confusion(artifact,change):
    out,cp,_,_=artifact
    path=out/'student.pt';manifest_path=out/'manifest.json'
    if change=='checkpoint':path.write_bytes(b'corrupt')
    elif change=='parent':cp.write_bytes(b'changed parent')
    else:
        payload=torch.load(path,weights_only=True);manifest=json.loads(manifest_path.read_text())
        if change=='prefix':payload['actor_state_dict']['residual_actor.2.weight'][0,0]+=.01
        elif change=='schema':
            payload['schema']='recovery_unknown';payload['metadata']['schema']='recovery_unknown';manifest['schema']='recovery_unknown'
        elif change=='behavior_claim':
            payload['metadata']['autonomous_behavior_accepted']=True;manifest['autonomous_behavior_accepted']=True
        else:
            key=next(iter(payload['metadata']['sources']));payload['metadata']['sources'][key]='0'*64;manifest['sources'][key]='0'*64
        torch.save(payload,path);manifest['student_checkpoint_sha256']=sha256(path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError,RuntimeError,KeyError)):
        student.load_student(path)


def test_worker_prepare_loads_explicit_new_schema_before_any_environment(monkeypatch,tmp_path):
    def reject(*args,**kwargs):raise EvidenceError('unknown constrained artifact')
    monkeypatch.setattr(worker,'load_student',reject)
    out=tmp_path/'must_not_exist'
    with pytest.raises(EvidenceError,match='unknown constrained'):
        worker.prepare(tmp_path/'missing.pt',25.,out)
    assert not out.exists()


def test_worker_execution_function_is_original_without_runtime_action_override():
    from migration_v1.student_worker import run_simulation
    assert worker.run_simulation is run_simulation


def test_fit_rejects_existing_destination_before_loading_parent(tmp_path,monkeypatch):
    out=tmp_path/'existing';out.mkdir()
    monkeypatch.setattr(fit,'load_parent',lambda *args,**kwargs:pytest.fail('should reject destination first'))
    with pytest.raises(EvidenceError,match='fresh'):
        fit.fit_from_checkpoint(tmp_path/'missing.pt',out)


def test_new_source_inventory_explicitly_covers_mapping_urdf():
    robot=fit.REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    assert fit.source_signatures()[str(robot.resolve())]==sha256(robot)
