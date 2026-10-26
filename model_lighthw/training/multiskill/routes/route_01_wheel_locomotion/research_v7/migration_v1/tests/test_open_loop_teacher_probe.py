"""CPU sequence/source boundary tests; no real replay or optimization."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1 import open_loop_teacher_probe as probe, student_worker, dataset, recovery_dataset, admission
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError


def test_sequence_preserves_every_original_float32_action_and_uses_no_observation():
    raw=np.arange(48,dtype=np.float32).reshape(3,16)/7.
    raw[0,0]=-0.;raw[0,1]=np.nextafter(np.float32(0.),np.float32(1.))
    source=raw.copy()
    sequence=probe.RecordedTeacherSequence(raw,np.arange(3)*.02)
    raw[:]=0.  # Own immutable copy, no alias to caller memory.
    for index in range(3):
        actual=sequence.next_action(index,index*.02,'cpu')
        assert actual.dtype==torch.float32 and actual.numpy().tobytes()==source[index:index+1].tobytes()
    record=sequence.execution_record()
    assert record['sequence_calls']==3 and record['actor_calls']==record['teacher_calls']==0
    assert not hasattr(sequence,'actor') and not record['autonomous_behavior_accepted']
    with pytest.raises(EvidenceError,match='exhausted'):
        sequence.next_action(3,.06,'cpu')


@pytest.mark.parametrize('change',['float64','nonfinite','shape','clock_shift','missing_time'])
def test_sequence_rejects_conversion_or_time_alignment_shortcuts(change):
    raw=np.zeros((3,16),dtype=np.float32);times=np.arange(3)*.02
    if change=='float64':raw=raw.astype(np.float64)
    elif change=='nonfinite':raw[0,0]=np.nan
    elif change=='shape':raw=raw[:,:15]
    elif change=='clock_shift':times+=.02
    else:times=times[:-1]
    with pytest.raises(EvidenceError):probe.RecordedTeacherSequence(raw,times)


def test_sequence_rejects_skipped_reset_and_repeated_frames():
    sequence=probe.RecordedTeacherSequence(np.zeros((3,16),dtype=np.float32),np.arange(3)*.02)
    with pytest.raises(EvidenceError):sequence.next_action(1,.02,'cpu')
    with pytest.raises(EvidenceError):sequence.next_action(0,.02,'cpu')
    assert sequence.calls==0
    sequence.next_action(0,0.,'cpu')
    with pytest.raises(EvidenceError):sequence.next_action(0,0.,'cpu')
    assert sequence.calls==1


def test_open_loop_identity_is_rejected_by_existing_teacher_recovery_and_autonomy_gates():
    request=probe.diagnostic_metadata()
    request.update(initial_state_mode='ordinary',execution_physics={'device':'cuda:0'},seed=43,
        slope_deg=25.,physical_cart_mass_kg=40.,noise=False,physical_trailer_connected=True)
    initial=dict(mode='ordinary',cuda_available=True,robot_tensor_device='cuda:0')
    with pytest.raises(EvidenceError,match='clean executed teacher'):
        dataset.validate_physics(request,{},initial,{}, {})
    with pytest.raises(EvidenceError,match='explicit actor inference'):
        recovery_dataset.validate_recovery_physics(request,{},initial,{}, {})
    with pytest.raises(EvidenceError,match='cannot prove autonomous'):
        admission._verify_identity(request,None,None,None,False)
    assert request['actor_calls']==request['teacher_calls']==0
    assert request['open_loop_replay'] and not request['training_labels_allowed']


@pytest.fixture
def synthetic_prepared(monkeypatch,tmp_path):
    """Synthetic source IO stub only, explicitly not physical success evidence."""
    baseline=tmp_path/'synthetic_source.pt';baseline.write_text('synthetic provenance only')
    actions=np.arange(6*16,dtype=np.float32).reshape(6,16)/20.
    episodes=[]
    for index,(slope,start) in enumerate([(25.,3.),(-20.,7.48)]):
        run=tmp_path/f'synthetic_episode_{index}';run.mkdir()
        trace=run/'seed43_case00_policy_trace.npz'
        np.savez(trace,executed_action=actions[index*3:index*3+3],teacher_time_s=np.arange(3)*.02)
        request=run/'request.json';request.write_text(json.dumps(dict(checkpoint=str(baseline),checkpoint_sha256=sha256(baseline))))
        episodes.append(dict(directory=str(run),case=dict(slope_deg=slope,payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.2),
            physics_profile={'synthetic_interface_only':True},command_timing=dict(actual_move_start_s=start),samples=3,
            source_files={p.name:sha256(p) for p in (trace,request)}))
    arrays=dict(action=actions,episode_id=np.repeat([0,1],3))
    monkeypatch.setattr(probe,'read_teacher_bundle',lambda _:(arrays,dict(episodes=episodes)))
    return probe.prepare(['synthetic_up','synthetic_down'],25.,tmp_path/'not_created')


def test_prepare_has_no_actor_and_freezes_shared_execution_sources(synthetic_prepared):
    prepared=synthetic_prepared
    assert 'policy' not in prepared and 'checkpoint' not in prepared
    assert prepared['preparation']['preparation_seconds']==3.
    assert prepared['sequence'].calls==0 and not prepared['output'].exists()
    assert str(Path(probe.__file__).resolve()) in prepared['sources']
    assert str(Path(student_worker.__file__).resolve()) in prepared['sources']
    assert not any('ordinary_entry_controller.py' in name for name in prepared['sources'])


def test_missing_shared_diagnostic_support_rejects_before_app_or_output(monkeypatch,synthetic_prepared):
    monkeypatch.setattr(student_worker,'OPEN_LOOP_DIAGNOSTIC_VERSION','not_supported')
    monkeypatch.setattr(student_worker,'run_simulation',lambda _:pytest.fail('Unsupported diagnostic reached App path'))
    with pytest.raises(EvidenceError,match='no reviewed explicit'):
        probe.run_simulation(synthetic_prepared)
    assert not synthetic_prepared['output'].exists()


def test_known_probe_reuses_same_physical_worker_without_fake_actor(monkeypatch,synthetic_prepared):
    calls=[]
    monkeypatch.setattr(student_worker,'run_simulation',lambda value:calls.append(value) or {'synthetic_no_sim':True})
    assert probe.run_simulation(synthetic_prepared)=={'synthetic_no_sim':True}
    assert calls==[synthetic_prepared] and 'policy' not in calls[0]


def test_changed_original_teacher_bytes_rejected_before_execution(synthetic_prepared):
    source=Path(synthetic_prepared['source_episode']['directory'])/'seed43_case00_policy_trace.npz'
    with source.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(EvidenceError,match='Complete teacher evidence changed'):
        probe.verify_prepared(synthetic_prepared)
    assert not synthetic_prepared['output'].exists()


def test_unknown_shared_source_fails_before_application_or_output(tmp_path):
    with pytest.raises(EvidenceError,match='Unknown shared physical execution'):
        student_worker.run_simulation(dict(execution_kind='unreviewed',output=tmp_path/'none'))
    assert not (tmp_path/'none').exists()


def test_optional_probe_import_only_occurs_in_explicit_diagnostic_branch():
    tree=ast.parse(Path(student_worker.__file__).read_text())
    selected=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='diagnostic':
            selected.extend(x for x in node.body if isinstance(x,ast.ImportFrom)
                and x.module=='migration_v1.open_loop_teacher_probe')
    all_imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)
                 and node.module=='migration_v1.open_loop_teacher_probe']
    assert len(selected)==len(all_imports)==1
    assert Path(probe.__file__).resolve() not in student_worker.dependency_sources()
