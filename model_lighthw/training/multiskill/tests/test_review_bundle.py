"""Filesystem/subprocess delivery checks; no simulator or behavior claims."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'offline'))
import review_bundle as bundle
import station


def fixture_run(tmp_path, route='06'):
    project=tmp_path/'project with spaces'; project.mkdir()
    run=project/'runs'/'trial'; run.mkdir(parents=True)
    bundle.write(run/'request.json',{'mode':'train','route':route,'payload_kg':0})
    bundle.write(run/'run.json',{'status':'completed','mode':'train'})
    bundle.write(run/'runtime_check.json',{'dimensions':[58,237,16]})
    bundle.write(run/'result.json',{'status':'completed','mode':'train'})
    snapshot=run/'source_snapshot'/'source'/'config.py';snapshot.parent.mkdir(parents=True);snapshot.write_text('value=42\n')
    bundle.write(run/'source_manifest.json',{'source/config.py':bundle.digest(snapshot)})
    asset=project/'assets'/'robot.stl';asset.parent.mkdir();asset.write_bytes(b'fixture mesh bytes')
    bundle.write(run/'asset_manifest.json',{'meshes':{'assets/robot.stl':bundle.digest(asset)}})
    for name in ('model_0.pt','model_20.pt','model_99.pt'):
        p=run/'training'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode())
    (run/'console.log').write_text('complete log\n')
    (run/'asset_cache').mkdir();(run/'asset_cache'/'ignore.usd').write_text('generated')
    return project,run


def test_complete_return_is_portable_and_contains_every_saved_model(tmp_path):
    project,run=fixture_run(tmp_path)
    folder=bundle.collect([run],'06',project=project)
    report=bundle.read(folder/'review.json')
    assert len(report['runs'][0]['checkpoints'])==3
    assert not list(folder.rglob('ignore.usd'))
    assert len(list((folder/'assets').rglob('robot.stl')))==1
    assert bundle.verify(folder)['status']=='passed'
    moved=tmp_path/'USB copied folder';folder.rename(moved)
    result=subprocess.run([sys.executable,str(moved/'verify_review.py'),str(moved)],capture_output=True,text=True)
    assert result.returncode==0, result.stdout+result.stderr
    assert (run/'training/model_0.pt').exists()


def test_corrupted_return_detected_without_touching_original(tmp_path):
    project,run=fixture_run(tmp_path)
    folder=bundle.collect([run],'06',project=project)
    next(folder.rglob('model_20.pt')).write_bytes(b'USB corruption')
    with pytest.raises(ValueError,match='checksum mismatch'):bundle.verify(folder)
    assert (run/'training/model_20.pt').read_bytes()==b'model_20.pt'


def test_failed_job_keeps_diagnostics_and_never_claims_completion(tmp_path):
    project,run=fixture_run(tmp_path)
    bundle.write(run/'run.json',{'status':'failed'})
    bundle.write(run/'worker_error.json',{'error':'original backend error'})
    (run/'result.json').unlink()
    folder=bundle.collect([run],'06',project=project)
    report=bundle.read(folder/'review.json')
    assert report['runs'][0]['run_status']=='failed'
    assert report['warnings']
    assert next(folder.rglob('worker_error.json')).is_file()
    assert report['behavior_acceptance']=='not_evaluated'


def test_snapshot_mismatch_is_not_silently_replaced_with_current_code(tmp_path):
    project,run=fixture_run(tmp_path)
    (run/'source_snapshot/source/config.py').write_text('changed\n')
    with pytest.raises(ValueError,match='snapshot mismatch'):bundle.collect([run],'06',project=project)
    assert not [p for p in (project/'usb_review').iterdir() if not p.name.endswith('.collecting')]


def test_symlink_escape_and_recursive_collection_are_rejected(tmp_path):
    project,run=fixture_run(tmp_path)
    with pytest.raises(ValueError,match='outside'):bundle.collect([run],'06',project=project,output_root=run/'returns')
    (run/'linked_secret').symlink_to(project/'assets/robot.stl')
    with pytest.raises(ValueError,match='linked'):bundle.collect([run],'06',project=project)


def test_generic_route_keeps_its_own_label_and_source_contract(tmp_path):
    project,run=fixture_run(tmp_path,route='stairs_future')
    folder=bundle.collect([run],'stairs_future',project=project)
    report=bundle.read(folder/'review.json')
    assert report['route']=='stairs_future'
    assert bundle.read(next(folder.rglob('request.json')))['route']=='stairs_future'


def test_subprocess_failure_does_not_hide_exit_status(tmp_path):
    session=tmp_path/'session';session.mkdir()
    state={'route':'06','stages':[]}
    with pytest.raises(RuntimeError,match='code 7'):
        station.run_stage(session,state,'failure',[sys.executable,'-c','print("original failure", flush=True); raise SystemExit(7)'])
    result=bundle.read(session/'session.json')
    assert result['stages'][0]['status']=='failed' and result['stages'][0]['returncode']==7
    assert 'original failure' in (session/'failure.log').read_text()


def test_session_lock_prevents_parallel_mutation(tmp_path):
    session=tmp_path/'session';session.mkdir()
    with station.session_lock(session):
        with pytest.raises(RuntimeError,match='in use'):
            with station.session_lock(session): pass


def test_unknown_route_is_rejected_before_training():
    with pytest.raises(SystemExit) as error: station.main(['train','--route','05b'])
    assert error.value.code==2


def test_actuator_limit_diagnostic_does_not_use_large_physx_cap():
    import torch
    from types import SimpleNamespace
    scripts=ROOT/'model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts'
    sys.path.insert(0,str(scripts))
    from review_limits import actuator_effort_limits
    sim=torch.tensor([[1.e9,1.e9,10.]])
    robot=SimpleNamespace(data=SimpleNamespace(joint_effort_limits=sim),actuators={
        'legs':SimpleNamespace(joint_indices=[0],effort_limit=torch.tensor([[48.]])),
        'wheels':SimpleNamespace(joint_indices=[1,2],effort_limit=torch.tensor([[24.,24.]]))})
    result=actuator_effort_limits(robot)
    torch.testing.assert_close(result,torch.tensor([[48.,24.,10.]]))
    assert sim[0,0]==1.e9  # read-only diagnostic
    actual=torch.tensor([[48.,12.,10.]])
    torch.testing.assert_close((actual/result>=.99).float().mean(),torch.tensor(2/3))


def test_active_run_is_not_exported_as_a_finished_snapshot(tmp_path):
    project,run=fixture_run(tmp_path)
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)',str(run)])
    try:
        bundle.write(run/'worker_status.json',{'pid':process.pid})
        with pytest.raises(RuntimeError,match='still active'):bundle.collect([run],'06',project=project)
    finally:
        process.terminate();process.wait()


def test_runtime_snapshot_guard_reports_actual_source_mismatch(tmp_path):
    scripts=ROOT/'model_lighthw/training/multiskill/routes/route_06_heavy_payload/scripts'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('_r6_snapshot_guard',scripts/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    relative='offline/routes.json'
    bundle.write(tmp_path/'source_manifest.json',{relative:bundle.digest(ROOT/relative)})
    module.verify_run_sources(tmp_path,'test')
    assert bundle.read(tmp_path/'source_consistency.json')['status']=='passed'
    bundle.write(tmp_path/'source_manifest.json',{relative:'0'*64})
    with pytest.raises(RuntimeError,match='source changed'):module.verify_run_sources(tmp_path,'test')
    assert bundle.read(tmp_path/'source_consistency.json')['changed']==[relative]


@pytest.mark.parametrize('video,physics',[(True,False),(False,True)])
def test_required_review_artifacts_cannot_be_missing(tmp_path,video,physics):
    import launch
    bundle.write(tmp_path/'runtime_check.json',{'dimensions':[58,237,16]})
    bundle.write(tmp_path/'result.json',{'mode':'play','status':'completed','cases':[{'seed':42,'episode':0}]})
    bundle.write(tmp_path/'request.json',{'record_video':video,'physics_trace':physics})
    with pytest.raises(RuntimeError,match='trace metadata missing'):launch.verify_completion(tmp_path,'play')
    bundle.write(tmp_path/'seed42_episode0_trace.json',{'physics_samples':19,'policy_samples':1,'video_frames':1})
    with pytest.raises(RuntimeError,match='video missing|Incomplete 1kHz'):launch.verify_completion(tmp_path,'play')
