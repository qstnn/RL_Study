"""CPU admission rejection/replay tests. Synthetic evidence never proves robot behavior.

No simulator, BC optimization, or PPO update is invoked by this file.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1 import admission as a
from migration_v1.contract import CONTRACT
from migration_v1.dataset import EvidenceError


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def trace(frames=3,payload=0.,towed=0.):
    q = np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.],np.float32)
    observation = np.zeros((frames,300),np.float32)
    for name,value in dict(projected_gravity=[0.,0.,-1.],task_context=[payload/15.,towed/40.]).items():
        begin,end = CONTRACT['layout'][name]['columns']
        observation[:,begin:end] = np.tile(value,5)
    policy = dict(observation=observation,actor_action=np.zeros((frames,16),np.float32),
        executed_action=np.zeros((frames,16),np.float32),leg_target=np.tile(q[:12],(frames,1)),
        wheel_target=np.zeros((frames,4),np.float32),pre_action_time_s=np.arange(frames)*.02,
        phase=np.zeros(frames,int),command_vx=np.zeros(frames))
    physics = dict(joint_pos=np.tile(q,(frames*20,1)),joint_velocity=np.zeros((frames*20,16)),
        body_quat_w=np.tile([1.,0.,0.,0.],(frames*20,1)),body_omega_b=np.zeros((frames*20,3)))
    initial = dict(joint_position_action_order=q.tolist(),joint_velocity_action_order=[0.]*16,
        robot_root_state=[0.,0.,.416223,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    motor = dict(requested=np.zeros((frames*20,16)),applied=np.zeros((frames*20,16)),
        velocity=np.zeros((frames*20,16)),limits=np.r_[np.full(12,96.),np.full(4,48.)],
        joint_names=np.asarray(CONTRACT['action_order']),physics_dt=np.array(.001))
    case = dict(slope_deg=0.,payload_kg=payload,towed_kg=towed,speed=0.,goal_distance_m=1.)
    rows = [dict(t=(i+1)*.02,command_vx=0.,eval_phase=0.,slope_deg=0.,payload_kg=payload,towed_kg=towed)
            for i in range(frames)]
    return policy,physics,motor,rows,initial,case


def test_rebuilds_observation_for_both_payload_and_towing_without_context_relabeling():
    for payload,towed in [(0.,0.),(15.,0.),(0.,40.)]:
        policy,physics,_,_,initial,case = trace(payload=payload,towed=towed)
        assert max(a._verify_observation(policy,physics,initial,case).values()) == 0.
        policy['observation'][1,298] += .01
        with pytest.raises(EvidenceError,match='observation'):
            a._verify_observation(policy,physics,initial,case)


@pytest.mark.parametrize('change',['time','command','phase','case','motor','physics','override'])
def test_raw_traces_reject_missing_steps_or_modified_execution(change):
    policy,physics,motor,rows,_,case = trace()
    assert a._verify_traces(policy,physics,motor,rows,case) == 3
    if change == 'time': policy['pre_action_time_s'][1] += .02
    elif change == 'command': rows[1]['command_vx'] = .2
    elif change == 'phase': rows[0]['eval_phase'] = 1.
    elif change == 'case': rows[0]['towed_kg'] = 40.
    elif change == 'motor': motor['applied'][0,12] = 48.1
    elif change == 'physics': physics['joint_pos'] = physics['joint_pos'][:-1]
    else: policy['action_override'] = np.array([False,True,False])
    with pytest.raises(EvidenceError):
        a._verify_traces(policy,physics,motor,rows,case)


def test_cpu_row_actor_replay_and_native_mapping_both_required():
    policy,_,_,_,initial,_ = trace()
    def zero_student(observation): return torch.zeros((len(observation),16))
    result = a._verify_actor_and_mapping(zero_student,policy,initial,a.ROBOT)
    assert result['replayed_frames'] == 3
    assert result['cpu_row_inference_max_absolute_error'] == 0.
    policy['actor_action'][1,0] = .01
    with pytest.raises(EvidenceError,match='candidate Actor'):
        a._verify_actor_and_mapping(zero_student,policy,initial,a.ROBOT)
    policy['actor_action'][1,0] = 0.
    policy['executed_action'][1,0] = .01
    with pytest.raises(EvidenceError,match='manager action'):
        a._verify_actor_and_mapping(zero_student,policy,initial,a.ROBOT)
    policy['executed_action'][1,0] = 0.
    policy['wheel_target'][1,0] = .01
    with pytest.raises(EvidenceError,match='mapping'):
        a._verify_actor_and_mapping(zero_student,policy,initial,a.ROBOT)


def test_selected_direction_clock_need_not_equal_the_other_direction():
    manifest = dict(teacher_bundle=dict(episodes=[dict(case=dict(slope_deg=25.),
        command_timing=dict(actual_move_start_s=3.)),dict(case=dict(slope_deg=-20.),
        command_timing=dict(actual_move_start_s=7.48))]))
    frames = 376
    policy = dict(pre_action_time_s=np.arange(frames)*.02,phase=np.r_[np.zeros(374),np.ones(2)],
        command_vx=np.r_[np.zeros(374),.001,.002])
    request = dict(fixed_preparation_s=7.48,move_start_s=7.48,
        preparation_protocol=a.runtime_checks.check_source_preparation(manifest,7.48,case_slope=-20.))
    result = a._verify_clock(policy,[{}]*frames,manifest,dict(slope_deg=-20.),request,False)
    assert result['actual_pre_action_move_start_s'] == 7.48
    policy['command_vx'][373] = .01
    with pytest.raises(EvidenceError,match='before'):
        a._verify_clock(policy,[{}]*frames,manifest,dict(slope_deg=-20.),request,False)


def test_source_manifest_requires_archived_bytes_and_current_execution_identity(tmp_path,monkeypatch):
    repo = tmp_path/'repo'; source = repo/'source.py'; source.parent.mkdir(); source.write_text('original\n')
    monkeypatch.setattr(a,'REPO',repo)
    monkeypatch.setattr(a,'dependency_sources',lambda:[source])
    run = tmp_path/'run'; snapshot = a._snapshot_path(run,source)
    snapshot.parent.mkdir(parents=True); snapshot.write_bytes(source.read_bytes())
    write_json(run/'student_source_manifest.json',{str(source):a.sha256(source)})
    assert a._verify_sources(run,False)['source_files'] == 1
    snapshot.write_text('changed\n')
    with pytest.raises(EvidenceError,match='snapshot changed'):
        a._verify_sources(run,False)
    snapshot.write_bytes(source.read_bytes()); source.write_text('changed\n')
    with pytest.raises(EvidenceError,match='Current execution source'):
        a._verify_sources(run,False)


@pytest.mark.parametrize('change',['teacher','checkpoint','actor','override','manifest'])
def test_run_identity_cannot_mix_teachers_or_candidates(tmp_path,change):
    checkpoint = tmp_path/'student.pt'; checkpoint.write_bytes(b'synthetic-not-a-model')
    digest = a.sha256(checkpoint); manifest = {'identity':'synthetic'}
    request = dict(artifact_kind='autonomous_student_execution',checkpoint=str(checkpoint),
        checkpoint_sha256=digest,source_student_manifest=manifest,teacher_calls=0,
        actor_unique_action_source=True,action_override=False,actor_inference='row',
        initial_state_mode='ordinary',seed=43,noise=False,ppo_updates=0)
    a._verify_identity(request,checkpoint,digest,manifest,False)
    if change == 'teacher': request['artifact_kind'] = 'teacher_execution'
    elif change == 'checkpoint': request['checkpoint_sha256'] = '0'*64
    elif change == 'actor': request['actor_unique_action_source'] = False
    elif change == 'override': request['action_override'] = True
    else: request['source_student_manifest'] = {'identity':'another'}
    with pytest.raises(EvidenceError): a._verify_identity(request,checkpoint,digest,manifest,False)


def test_actual_no_towing_force_is_required_even_when_nominal_case_is_zero():
    physics = {name:np.zeros((40,3)) for name in ('real_cart_robot_force_w_N','real_cart_robot_torque_w_Nm',
               'real_cart_cart_forces_w_N','real_cart_cart_torques_w_Nm')}
    rows = [dict(towed_kg=0.,drawbar_force_N=0.)]
    assert a.retention_protocol.check_no_towing(physics,rows)['no_towing_force_verified']
    physics['real_cart_robot_force_w_N'][0,0] = 1.e-7
    with pytest.raises(EvidenceError,match='Unexpected external force'):
        a.retention_protocol.check_no_towing(physics,rows)


def test_retention_reason_rebuilt_from_raw_terminal_and_complete_window():
    case = a.retention_protocol.select_case(0)
    rows = [dict(t=(i+1)*.02,terminated=0.,time_out=0.,tilt_deg=0.,drawbar_force_N=0.,body_contact=0.,
        x=0.,y=0.,z=.416,vx=0.,velocity_error=0.,heading_rad=0.,height=.416,speed=0.,yaw_rate_abs=0.,
        four_wheel=1.,torque_abs_max=0.,command_vx=0.,target_vx=0.) for i in range(1000)]
    assert a._retention_behavior(rows,case)['success']
    rows[-1]['terminated'] = 1.
    assert a._retention_behavior(rows,case)['failure_reason'] == 'terminated'
    assert not a._retention_behavior(rows,case)['success']
    rows[-1]['terminated'] = 0.
    assert not a._retention_behavior(rows[:-1],case)['success']


@pytest.fixture
def orchestration(tmp_path,monkeypatch):
    """Only API orchestration is synthetic; this fixture is not runtime evidence."""
    checkpoint = tmp_path/'synthetic.pt'; checkpoint.write_bytes(b'not a real checkpoint')
    digest = a.sha256(checkpoint)
    anchors = [tmp_path/'up',tmp_path/'down']; retention = [tmp_path/('case'+str(i)) for i in range(12)]
    for path,slope in zip(anchors,[25.,-20.]):
        write_json(path/'request.json',dict(checkpoint_sha256=digest,slope_deg=slope))
    for index,path in enumerate(retention):
        write_json(path/'request.json',dict(checkpoint_sha256=digest,case_index=index))
    monkeypatch.setattr(a,'load_student',lambda *args,**kwargs:(object(),{}))
    monkeypatch.setattr(a,'_verify_run',lambda *args,**kwargs:dict(passed=True))
    return checkpoint,anchors,retention


def test_successful_behavior_gate_still_cannot_authorize_formal_ppo(orchestration):
    result = a.verify_behavior_admission(*orchestration)
    assert result['pre_ppo_behavior_ready']  # Mocked evidence only: exercises boundary, not actual acceptance.
    assert not result['formal_ppo_authorized']
    assert not result['full64_gpu_update_resume_verified']
    assert result['real_ppo_updates_performed'] == 0
    assert len(result['anchors']) == 2 and len(result['retention']) == 12
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('change',['missing','duplicate','case','mixed','direction','strict_load'])
def test_public_gate_closes_on_incomplete_mixed_or_unloadable_evidence(orchestration,monkeypatch,change):
    checkpoint,anchors,retention = orchestration
    if change == 'missing': retention.pop()
    elif change == 'duplicate': retention[-1] = retention[0]
    elif change == 'strict_load':
        def reject(*args,**kwargs): raise EvidenceError('strict model source differs')
        monkeypatch.setattr(a,'load_student',reject)
    else:
        path = retention[-1] if change == 'case' else anchors[-1]
        request = a._json(path/'request.json')
        if change == 'case': request['case_index'] = 0
        elif change == 'mixed': request['checkpoint_sha256'] = '0'*64
        else: request['slope_deg'] = 25.
        write_json(path/'request.json',request)
    result = a.verify_behavior_admission(checkpoint,anchors,retention)
    assert not result['pre_ppo_behavior_ready'] and result['errors']


def test_public_gate_uses_recomputed_failure_and_does_not_trust_saved_pass(orchestration,monkeypatch):
    checkpoint,anchors,retention = orchestration
    for run in anchors+retention:
        write_json(run/'saved_pass.json',dict(success=True,pre_ppo_behavior_ready=True))
    def recompute(run,*args,**kwargs):
        return dict(passed=Path(run) != anchors[0],behavior=dict(recomputed_from_raw=True))
    monkeypatch.setattr(a,'_verify_run',recompute)
    result = a.verify_behavior_admission(checkpoint,anchors,retention)
    assert not result['pre_ppo_behavior_ready']
    assert len(result['anchors']) == 2 and len(result['retention']) == 12
    assert 'recomputed raw behavior failed' in result['errors'][0]


def test_teacher_call_record_checked_separately_from_saved_success():
    review = dict(success=True,teacher_calls=0,actor_unique_action_source=True,action_override=False,
        student_execution=dict(teacher_calls=0,actor_calls=3,policy_dim=300,action_dim=16,
            inference_mode='row',actor_input='deployed_observation_only'))
    a._check_execution_record(review,3)
    review['student_execution']['teacher_calls'] = 1
    with pytest.raises(EvidenceError,match='Actor call count'):
        a._check_execution_record(review,3)


def test_run_recomputes_raw_audit_even_if_saved_worker_review_claims_pass(tmp_path,monkeypatch):
    """Short artificial data exercise wiring; mocked runtime/audit is not physical evidence."""
    run = tmp_path/'run'; run.mkdir()
    checkpoint = tmp_path/'student.pt'; checkpoint.write_bytes(b'synthetic')
    policy,physics,motor,rows,initial,case = trace(towed=40.)
    case.update(slope_deg=25.,speed=.3)
    for row in rows: row['slope_deg'] = 25.
    manifest = dict(teacher_bundle=dict(episodes=[dict(case=case,physics_profile={},
        command_timing=dict(actual_move_start_s=3.)),dict(case=dict(case,slope_deg=-20.),
        physics_profile={},command_timing=dict(actual_move_start_s=7.48))]))
    preparation = a.runtime_checks.check_source_preparation(manifest,3.,case_slope=25.)
    request = dict(artifact_kind='autonomous_student_execution',checkpoint=str(checkpoint),
        checkpoint_sha256=a.sha256(checkpoint),source_student_manifest=manifest,teacher_calls=0,
        actor_unique_action_source=True,action_override=False,actor_inference='row',initial_state_mode='ordinary',
        seed=43,noise=False,ppo_updates=0,slope_deg=25.,physical_trailer_connected=True,
        physical_cart_mass_kg=40.,trial_duration_s=40.,rounded_cart_wheels=True,records_drawbar_pre_step=True,
        execution_physics=dict(device='cuda:0'),cases=[case],fixed_preparation_s=3.,move_start_s=3.,
        preparation_protocol=preparation)
    review = dict(success=True,autonomous_anchor_pass=True,teacher_calls=0,actor_unique_action_source=True,
        action_override=False,student_execution=dict(teacher_calls=0,actor_calls=3,policy_dim=300,action_dim=16,
        inference_mode='row',actor_input='deployed_observation_only'))
    for name,value in [('request.json',request),('initial_state_runtime.json',initial),
        ('student_behavior_review.json',review),('physics_recorder.json',dict(sample_hz=1000)),
        ('runtime_check.json',{}),('solver_runtime_usd.json',{}),('friction_runtime.json',{}),
        ('student_source_manifest.json',{}),(a.PREFIX+'.json',dict(success=True,complete=True))]:
        write_json(run/name,value)
    write_json(run/'assets/cart.urdf',{})
    robot = a._snapshot_path(run,a.ROBOT); robot.parent.mkdir(parents=True); robot.write_bytes(a.ROBOT.read_bytes())
    with (run/(a.PREFIX+'.csv')).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name,values in [('policy',policy),('physics',physics),('motor',motor)]:
        np.savez(run/(a.PREFIX+'_'+name+'_trace.npz'),**values)
    monkeypatch.setattr(a,'_verify_sources',lambda *args:dict(synthetic_inventory_only=True))
    monkeypatch.setattr(a.runtime_checks,'check_runtime',lambda *args:dict(synthetic_readback_only=True))
    calls = []
    def raw_audit(path):
        calls.append(Path(path))
        return dict(move_brake_hold_pass=False,duration_s=.06,checks=dict(full_declared_window=False))
    monkeypatch.setattr(a,'_behavior_audit',raw_audit)
    zero_student = lambda observation:torch.zeros((len(observation),16))
    result = a._verify_run(run,checkpoint,a.sha256(checkpoint),manifest,zero_student,False)
    assert calls == [run]
    assert not result['passed'] and not result['behavior']['move_brake_hold_pass']
    assert result['actor_replay']['replayed_frames'] == 3
    assert result['evidence_errors'] == ['Actual command clock did not reach its manifest/protocol movement start']
    assert a._json(run/'student_behavior_review.json')['success'] is True  # Read-only; original retained.
