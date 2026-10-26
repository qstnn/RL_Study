"""Synthetic recovery interfaces only; no robot simulation or real-data fitting."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1 import bc, student, recovery_bc, recovery_dataset, recovery_student, recovery_student_worker
from migration_v1.contract import SPEC_PATH, _load, install_actor_mask, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.policy import FrozenBaseTowingResidualActor, tensor_fingerprint
from migration_v1.runtime_checks import check_source_preparation


@pytest.fixture
def prefix_case():
    n,k=12,2
    parent=dict(teacher_bundle=dict(episodes=[dict(case=dict(slope_deg=slope),
        command_timing=dict(actual_move_start_s=start)) for slope,start in [(25.,3.),(-20.,7.48)]]))
    provenance=dict(path='/tmp/synthetic_student.pt',checkpoint_sha256='a'*64,manifest_sha256='b'*64)
    metadata=dict(version=recovery_dataset.PREFIX_VERSION,inference_mode='row',prefix_steps=k,prefix_seconds=.04,
        student_checkpoint=provenance['path'],student_checkpoint_sha256=provenance['checkpoint_sha256'],
        student_manifest_sha256=provenance['manifest_sha256'],prefix_teacher_actions_are_shadow_only=True,
        student_prefix_is_teacher_label=False,teacher_executed_actions_require_complete_episode_pass=True,
        autonomous_behavior_accepted=False,formal_ppo_authorized=False,
        fixed_command_clock=check_source_preparation(parent,3.,case_slope=25.))
    teacher=np.ones((n,16),dtype=np.float32);prediction=np.zeros((n,16),dtype=np.float32)
    actual=teacher.copy();actual[:k]=prediction[:k]
    policy=dict(observation=np.zeros((n,300),dtype=np.float32),executed_action=actual,
        student_prefix_active=(np.arange(n)<k).astype(float),teacher_executed=(np.arange(n)>=k).astype(float),
        student_prefix_frame_index=np.arange(n,dtype=float),student_prefix_takeover_frame=(np.arange(n)==k).astype(float),
        teacher_time_s=np.arange(n)*.02,student_prefix_student_raw=prediction,student_prefix_teacher_raw=teacher)
    execution=dict(metadata,actual_student_frames=k,actual_teacher_frames=n-k,complete_success_verified=False)
    interface=lambda x:torch.zeros((len(x),16),dtype=torch.float32)
    return dict(policy=policy,metadata=metadata,execution=execution,interface=interface,student_manifest=parent,
        provenance=provenance,report=dict(move_window_s=[3.,20.]),request=dict(move_start_s=3.,slope_deg=25.))


def test_exact_prefix_suffix_source_filter_never_selects_shadow(prefix_case):
    rows,report=recovery_dataset.validate_prefix_trace(**prefix_case)
    assert np.array_equal(rows,np.arange(2,12))
    assert not report['shadow_labels_used'] and not report['student_prefix_labels_used']
    # Counterfactual shadow content can differ arbitrarily during prefix.
    prefix_case['policy']['student_prefix_teacher_raw'][:2]=1234.
    assert np.array_equal(recovery_dataset.validate_prefix_trace(**prefix_case)[0],rows)


@pytest.mark.parametrize('change',['overlap','release','takeover','time','index','actual_prefix','actual_suffix',
    'student_after_switch','prediction','accounting','checkpoint','clock','float_k','nonfinite','shadow_label'])
def test_invalid_recovery_prefix_contract_rejected(prefix_case,change):
    p=prefix_case['policy'];m=prefix_case['metadata'];e=prefix_case['execution']
    if change=='overlap':p['teacher_executed'][0]=1.
    elif change=='release':p['student_prefix_active'][-1]=1.
    elif change=='takeover':p['student_prefix_takeover_frame'][3]=1.
    elif change=='time':p['teacher_time_s'][4]+=.02
    elif change=='index':p['student_prefix_frame_index'][4]=3.
    elif change=='actual_prefix':p['executed_action'][0,0]=1.
    elif change=='actual_suffix':p['executed_action'][4,0]=0.
    elif change=='student_after_switch':p['student_prefix_student_raw'][4,0]=1.
    elif change=='prediction':p['student_prefix_student_raw'][0,0]=p['executed_action'][0,0]=.01
    elif change=='accounting':e['actual_teacher_frames']-=1
    elif change=='checkpoint':m['student_checkpoint_sha256']='c'*64
    elif change=='clock':prefix_case['report']['move_window_s'][0]=3.02
    elif change=='float_k':m['prefix_steps']=2.
    elif change=='nonfinite':p['executed_action'][8,0]=np.nan
    else:m['student_prefix_is_teacher_label']=True
    with pytest.raises(EvidenceError):recovery_dataset.validate_prefix_trace(**prefix_case)


def test_recovery_physics_never_relabels_actor_flag_to_pass_teacher_gate():
    request=dict(actor_inference=False,fixed_command_clock=True,student_prefix={})
    original=deepcopy(request)
    with pytest.raises(EvidenceError,match='explicit actor inference'):
        recovery_dataset.validate_recovery_physics(request,{}, {},{}, {})
    assert request==original


def test_missing_recovery_prevents_real_entry_optimization_and_output(monkeypatch,tmp_path):
    monkeypatch.setattr(recovery_bc,'read_recovery_bundle',lambda *a,**k:(_ for _ in ()).throw(EvidenceError('No complete executed recovery')))
    monkeypatch.setattr(recovery_bc,'source_student_provenance',lambda *_:pytest.fail('Warm start before recovery gate'))
    output=tmp_path/'not_created'
    with pytest.raises(EvidenceError,match='No complete executed recovery'):
        recovery_bc.fit_from_recovery(['nominal'],['failed'],'absent.pt',output)
    assert not output.exists()


def test_recovery_worker_refuses_before_app_and_output(monkeypatch,tmp_path):
    monkeypatch.setattr(recovery_student_worker,'load_recovery_student',lambda *a,**k:(_ for _ in ()).throw(EvidenceError('Recovery evidence absent')))
    monkeypatch.setattr(recovery_student_worker,'run_simulation',lambda _:pytest.fail('App path reached'))
    output=tmp_path/'not_created'
    with pytest.raises(SystemExit) as error:
        recovery_student_worker.main(['--checkpoint',str(tmp_path/'absent.pt'),'--slope-deg','25','--output',str(output)])
    assert error.value.code==2 and not output.exists() and 'isaaclab.app' not in sys.modules


@pytest.fixture
def synthetic_bundle(tmp_path):
    torch.manual_seed(43);base=bc._mlp();install_actor_mask(base)
    actor=FrozenBaseTowingResidualActor(base)
    with torch.no_grad():actor.residual_actor[-1].bias.fill_(.125)
    rng=np.random.default_rng(45);x=rng.normal(size=(12,300)).astype(np.float32);x[:,298]=0.;x[:,299]=1.
    with torch.no_grad():y=actor(torch.from_numpy(x)).numpy()+.1
    arrays=dict(observation=x,action=y,stage=np.tile(['prepare','move','brake','hold'],3),
        episode_id=np.repeat([0,1,2],4),group_id=np.repeat([0,1,2],4),case_key=np.repeat(['up','down','up'],4),
        successful=np.ones(12,dtype=bool),source_frame_index=np.r_[np.arange(4),np.arange(4),np.arange(2,6)])
    episodes=[]
    for index in range(3):
        directory=tmp_path/f'synthetic_episode_{index}';directory.mkdir()
        trace=directory/'seed43_case00_policy_trace.npz';trace.write_text('synthetic provenance '+str(index))
        episodes.append(dict(directory=str(directory),source_files={trace.name:sha256(trace)}))
    warm=tmp_path/'synthetic_old_student.pt';warm.write_text('synthetic stub; no real model')
    warm.with_name('manifest.json').write_text('{"synthetic_interface_test_only":true}')
    provenance=dict(path=str(warm),checkpoint_sha256=sha256(warm),manifest_sha256=sha256(warm.with_name('manifest.json')),schema=bc.SCHEMA)
    parent=dict(schema=bc.SCHEMA,teacher_bundle=dict(schema='synthetic_interface_test_only',episodes=episodes[:2]),
        baseline=dict(synthetic_interface_test_only=True,frozen_base_tensor_sha256=tensor_fingerprint(base)),
        fitting=dict(split=dict(train_episode_ids=[0,1])))
    bundle=dict(schema='synthetic_interface_test_only',teacher_bundle=parent['teacher_bundle'],recovery_episodes=[episodes[2]],
        episodes=episodes,source_student=provenance,balance_preparation=True,sampling='synthetic_three_groups',groups=[])
    def loader(_):return student.StudentActionInterface(deepcopy(actor)),deepcopy(parent),deepcopy(provenance)
    return arrays,bundle,parent,provenance,loader


def test_holdout_cannot_hide_prior_warm_start_training(synthetic_bundle):
    arrays,bundle,parent,_,_=synthetic_bundle
    with pytest.raises(EvidenceError,match='already trained'):
        recovery_bc.validate_holdout_lineage(arrays,bundle,parent,[0])
    result=recovery_bc.validate_holdout_lineage(arrays,bundle,parent,[2])
    assert result['validation_kind']=='whole_independent_episodes'
    assert len(result['ancestor_training_episode_sha256'])==2
    assert len(result['cumulative_training_episode_sha256'])==2


def test_constrained_parent_marks_all_nominal_episodes_trained(synthetic_bundle):
    arrays,bundle,parent,_,_=synthetic_bundle
    parent['schema']=recovery_dataset.CONSTRAINED_STUDENT_SCHEMA
    # Even if an older BC stage had withheld episode 1, the constrained refit
    # objective subsequently consumed both whole nominal episodes.
    parent['fitting']=dict(independent_holdout_available=False)
    assert recovery_bc.prior_training_identities(parent)==sorted(
        recovery_dataset.episode_identity(ep) for ep in parent['teacher_bundle']['episodes'])
    for index in (0,1):
        # Keep a second synthetic episode of the held case so the prior-data
        # rejection, rather than missing-direction rejection, is exercised.
        local=dict(arrays,case_key=arrays['case_key'].copy())
        local['case_key'][arrays['episode_id']==2]=arrays['case_key'][arrays['episode_id']==index][0]
        with pytest.raises(EvidenceError,match='already trained'):
            recovery_bc.validate_holdout_lineage(local,bundle,parent,[index])
    assert recovery_bc.validate_holdout_lineage(arrays,bundle,parent,[2])['validation_kind']=='whole_independent_episodes'


@pytest.mark.parametrize('constrained',[False,True])
def test_recovery_parent_schema_dispatch_is_explicit(monkeypatch,tmp_path,constrained):
    from migration_v1 import constrained_student
    checkpoint=tmp_path/'student.pt';checkpoint.write_text('synthetic parent routing only')
    schema=recovery_dataset.CONSTRAINED_STUDENT_SCHEMA if constrained else bc.SCHEMA
    checkpoint.with_name('manifest.json').write_text(json.dumps(dict(schema=schema)))
    calls=[];interface=object();manifest=dict(schema=schema)
    def load(path,**kwargs):calls.append((path,kwargs));return interface,manifest
    if constrained:
        monkeypatch.setattr(constrained_student,'load_student',load)
        monkeypatch.setattr(recovery_dataset,'load_student',lambda *a,**k:pytest.fail('Old loader used for constrained source'))
    else:
        monkeypatch.setattr(recovery_dataset,'load_student',load)
        monkeypatch.setattr(constrained_student,'load_student',lambda *a,**k:pytest.fail('Constrained loader used for old source'))
    actual,returned,provenance=recovery_dataset.source_student_provenance(checkpoint)
    assert actual is interface and returned is manifest and provenance['schema']==schema
    assert calls==[(checkpoint,dict(device='cpu',inference_mode='row'))]


def test_constrained_prefix_requires_explicit_strict_loader_declaration(prefix_case):
    schema=recovery_dataset.CONSTRAINED_STUDENT_SCHEMA
    prefix_case['student_manifest']['schema']=schema
    with pytest.raises(EvidenceError,match='strict schema loader'):
        recovery_dataset.validate_prefix_trace(**prefix_case)
    declaration=dict(student_schema=schema,student_loader='migration_v1.constrained_student.load_student')
    prefix_case['metadata'].update(declaration);prefix_case['execution'].update(declaration)
    assert np.array_equal(recovery_dataset.validate_prefix_trace(**prefix_case)[0],np.arange(2,12))


def test_recovery_signatures_include_recursive_parent_and_runtime_source():
    from migration_v1.constrained_refit import source_signatures
    sources=recovery_bc.source_signatures()
    parent=source_signatures()
    assert len(sources)==19 and set(parent)<set(sources)
    assert all(sources[name]==digest for name,digest in parent.items())
    assert Path(recovery_dataset.__file__).with_name('runtime_checks.py').resolve().as_posix() in sources


def test_synthetic_new_schema_roundtrip_retains_base_and_rejects_old_loader(synthetic_bundle,monkeypatch,tmp_path):
    arrays,bundle,parent,provenance,loader=synthetic_bundle
    for module in (recovery_bc,recovery_student):
        monkeypatch.setattr(module,'read_recovery_bundle',lambda *a,**k:(arrays,deepcopy(bundle)))
        monkeypatch.setattr(module,'source_student_provenance',loader)
    output=tmp_path/'synthetic_recovery_candidate'
    manifest=recovery_bc.fit_from_recovery(['synthetic_up','synthetic_down'],['synthetic_recovery'],
        provenance['path'],output,steps=1,batch_size=12,learning_rate=.001,validation_episode_ids=[2])
    assert manifest['schema']!=bc.SCHEMA and manifest['fitting']['optimizer_steps']==1
    assert manifest['fitting']['warm_start_student']==provenance and not manifest['fitting']['optimizer_restored']
    assert not manifest['autonomous_behavior_accepted'] and not manifest['formal_ppo_authorized']
    candidate,metadata=recovery_student.load_recovery_student(output/'student.pt')
    assert tensor_fingerprint(candidate.actor.base_actor)==parent['baseline']['frozen_base_tensor_sha256']
    candidate.actor.assert_invariants()
    with pytest.raises(EvidenceError,match='manifest and embedded provenance'):
        student.load_student(output/'student.pt')
    with (output/'student.pt').open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(EvidenceError,match='SHA'):
        recovery_student.load_recovery_student(output/'student.pt')


def test_bundle_preserves_complete_episode_ids_and_original_suffix_frames(synthetic_bundle,monkeypatch):
    arrays,bundle,parent,provenance,loader=synthetic_bundle
    profile={'synthetic_interface_test_only':True}
    parent['baseline']['checkpoint_sha256']='d'*64
    for index,episode in enumerate(bundle['episodes']):
        slope=-20. if index==1 else 25.
        episode.update(case=dict(slope_deg=slope,payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.2),
            physics_profile=profile,command_timing=dict(actual_move_start_s=7.48 if slope<0 else 3.))
        (Path(episode['directory'])/'request.json').write_text(json.dumps(dict(checkpoint_sha256='d'*64)))
    ordinary={key:value[:8] for key,value in arrays.items()}
    recovered={key:value[8:] for key,value in arrays.items() if key in ('observation','action','stage','source_frame_index')}
    recovered.update(sampling_stage=recovered['stage'],manifest=bundle['episodes'][2])
    monkeypatch.setattr(recovery_dataset,'source_student_provenance',loader)
    monkeypatch.setattr(recovery_dataset,'read_teacher_bundle',lambda _:(ordinary,parent['teacher_bundle']))
    monkeypatch.setattr(recovery_dataset,'_read_recovery_episode',lambda *a,**k:recovered)
    packed,report=recovery_dataset.read_recovery_bundle(['synthetic'],[bundle['episodes'][2]['directory']],provenance['path'],balance_preparation=False)
    assert np.array_equal(packed['observation'],arrays['observation'])
    assert np.array_equal(packed['action'],arrays['action'])
    assert np.array_equal(packed['episode_id'],arrays['episode_id'])
    assert np.array_equal(packed['source_frame_index'][-4:],[2,3,4,5])
    assert np.all(packed['origin'][-4:]=='executed_recovery')
    assert not report['student_prefix_labels_used'] and not report['shadow_labels_used']
    assert report['complete_episodes_verified_before_filtering']
    monkeypatch.setattr(recovery_dataset,'_read_recovery_episode',lambda *a,**k:(_ for _ in ()).throw(EvidenceError('failed full hold')))
    with pytest.raises(EvidenceError,match='failed full hold'):
        recovery_dataset.read_recovery_bundle(['synthetic'],[bundle['episodes'][2]['directory']],provenance['path'],balance_preparation=False)


def test_reader_checks_full_trace_before_teacher_suffix_selection(prefix_case,monkeypatch,tmp_path):
    """Synthetic I/O ordering test; physical success predicate is explicitly stubbed."""
    d=recovery_dataset;run=tmp_path/'synthetic_interface_episode';run.mkdir()
    baseline=tmp_path/'synthetic_baseline';baseline.write_text('synthetic')
    cart=run/'assets/cart.urdf';cart.parent.mkdir();cart.write_text('synthetic cart')
    profile={'synthetic_interface_test_only':True,'robot_urdf_sha256':'e'*64,'cart_urdf_sha256':sha256(cart)}
    parent=prefix_case['student_manifest'];parent['sources']={};parent['baseline']={'checkpoint_sha256':sha256(baseline)}
    for episode in parent['teacher_bundle']['episodes']:episode['physics_profile']=profile
    source_paths=[d.V7/'scripts/student_prefix.py',d.V7/'migration_v1/runtime_checks.py']
    prefix_case['metadata']['source_sha256']={str(path):sha256(path) for path in source_paths}
    prefix_case['execution'].update(prefix_case['metadata'])
    for source in source_paths:
        copied=run/'source_snapshot'/source.relative_to(d.REPO);copied.parent.mkdir(parents=True,exist_ok=True)
        copied.write_bytes(source.read_bytes())
        if source.name=='student_prefix.py':
            copied.write_bytes(copied.read_bytes()+b'\n# synthetic historical archived helper; do not re-sign\n')
            prefix_case['metadata']['source_sha256'][str(source)]=sha256(copied)
            assert sha256(copied)!=sha256(source)
    case=dict(slope_deg=25.,payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.2)
    request=dict(prefix_case['request'],student_prefix=prefix_case['metadata'],checkpoint=str(baseline),
        checkpoint_sha256=sha256(baseline),base_network_contract={'urdf_sha256':'e'*64},cases=[case])
    for name,value in [('request.json',request),('student_prefix_execution.json',prefix_case['execution'])]:
        (run/name).write_text(json.dumps(value))
    for name in ('runtime_check.json','initial_state_runtime.json','source_manifest.json','research_source_manifest.json',
                 'solver_runtime_usd.json','rounded_cart_geometry.json'):(run/name).write_text('{}')
    (run/'seed43_case00.csv').write_text('synthetic transition placeholder')
    np.savez(run/'seed43_case00_policy_trace.npz',**prefix_case['policy'])
    n=len(prefix_case['policy']['observation'])
    np.savez(run/'seed43_case00_physics_trace.npz',joint_pos=np.zeros((20*n,16)),joint_velocity=np.zeros((20*n,16)),
        body_quat_w=np.zeros((20*n,4)),body_omega_b=np.zeros((20*n,3)))
    np.savez(run/'seed43_case00_motor_trace.npz',synthetic=np.zeros(1))
    monkeypatch.setattr(d,'validate_recovery_physics',lambda *a:{'synthetic_interface_test_only':True})
    monkeypatch.setattr(d,'verify_snapshot',lambda *a,**k:{'model_lighthw/lighthw_urdf/urdf/lighthw.urdf':'e'*64})
    monkeypatch.setattr(d,'source_student_provenance',lambda _:(prefix_case['interface'],parent,prefix_case['provenance']))
    checked=[]
    def full_observation(policy,physics,initial):
        checked.append(len(policy['observation']))
        assert len(physics['joint_pos'])==20*n
        return {'synthetic_interface_test_only':0.}
    monkeypatch.setattr(d,'validate_observation_trace',full_observation)
    monkeypatch.setattr(d,'stage_labels',lambda *a:np.full(n,'prepare'))
    report=dict(prefix_case['report'],move_brake_hold_pass=True,command_zero_time_s=20.)
    monkeypatch.setattr(d,'_behavior_audit',lambda _:report)
    result=d.read_recovery_episode(run,'synthetic_stub',balance_preparation=False)
    assert checked==[n] and len(result['action'])==n-2
    assert result['manifest']['label_frame_indices']==list(range(2,n))
    assert result['manifest']['label_times_s'][0]==.04
    assert result['manifest']['full_episode_samples']==n and not result['manifest']['actor_success']
    checked.clear();report['move_brake_hold_pass']=False
    with pytest.raises(EvidenceError,match='no suffix labels'):
        d.read_recovery_episode(run,'synthetic_stub',balance_preparation=False)
    assert not checked
