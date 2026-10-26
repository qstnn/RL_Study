"""Read-only autonomous behavior gate. No simulation, optimizer, or training entry.

Only raw traces from the same strictly loaded student can satisfy this gate.
Even success is *pre_ppo_behavior_ready*, never formal PPO authorization: the
separate full-64 GPU update/save/resume evidence is outside this module.
"""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .contract import CONTRACT, REPO, sha256
from .dataset import EvidenceError, _behavior_audit, validate_observation_trace
from . import retention_protocol, runtime_checks
from .student import load_student
from .student_worker import dependency_sources


PREFIX = 'seed43_case00'
ROBOT = REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
# Row inference is repeated on CPU, whereas the recorded actor ran on GPU.
# These are numerical comparison tolerances, not physical acceptance limits.
ACTOR_ATOL = 1.e-4
ACTOR_RTOL = 2.e-6


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _json(path):
    return json.loads(Path(path).read_text())


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _snapshot_path(run, source):
    source = Path(source)
    _require(source.is_absolute() and source.is_relative_to(REPO.parent),
             'Execution source is outside the reviewed workspace: '+str(source))
    relative = (source.relative_to(REPO) if source.is_relative_to(REPO)
                else Path('_dependencies')/source.relative_to(REPO.parent))
    return Path(run)/'source_snapshot'/relative


def _verify_sources(run, retention):
    """Reuse the execution dependency inventory; neither execute snapshot code nor accept saved flags."""
    name = 'retention_source_manifest.json' if retention else 'student_source_manifest.json'
    manifest = _json(run/name)
    required = set(dependency_sources())
    if retention:
        required |= {retention_protocol.PROTOCOL_SOURCE, Path(retention_protocol.__file__).resolve(),
                     Path(__file__).with_name('retention_worker.py')}
    _require(set(manifest) == {str(path) for path in required},
             'Execution source inventory is incomplete or contains an unreviewed dependency')
    for source, digest in manifest.items():
        _require(sha256(source) == digest, 'Current execution source changed: '+source)
        _require(sha256(_snapshot_path(run, source)) == digest, 'Execution snapshot changed: '+source)
    return dict(source_files=len(manifest), manifest_sha256=sha256(run/name),
                current_sources_and_archived_bytes_verified=True)


def _verify_identity(request, checkpoint, digest, manifest, retention):
    expected = 'autonomous_retention_execution' if retention else 'autonomous_student_execution'
    _require(request.get('artifact_kind') == expected, 'Teacher/non-student artifact cannot prove autonomous behavior')
    _require(request.get('checkpoint_sha256') == digest and
             sha256(request['checkpoint']) == digest, 'Executed checkpoint SHA differs from the candidate')
    _require(request.get('source_student_manifest') == manifest, 'Executed student provenance differs from candidate')
    _require(request.get('teacher_calls') == 0 and request.get('actor_unique_action_source') is True
             and request.get('action_override') is False and request.get('actor_inference') == 'row',
             'Execution is not exclusively the deployed row-inference student')
    _require(request.get('initial_state_mode') == 'ordinary' and request.get('seed') == 43
             and request.get('noise') is False and request.get('ppo_updates') == 0,
             'Execution reset/seed/noise/PPO declaration differs from this behavior protocol')


def _check_execution_record(review, frames):
    execution = review['student_execution']
    _require(review.get('teacher_calls') == 0 and review.get('actor_unique_action_source') is True
             and review.get('action_override') is False, 'Recorded execution used teacher or action override')
    _require(execution.get('teacher_calls') == 0 and execution.get('actor_calls') == frames
             and execution.get('policy_dim') == 300 and execution.get('action_dim') == 16
             and execution.get('inference_mode') == 'row'
             and execution.get('actor_input') == 'deployed_observation_only',
             'Actor call count/input contract does not match actual recorded control frames')
    # No success/pass/runtime-verified field from this saved review is consumed.


def _verify_traces(policy, physics, motor, rows, case):
    frames = len(rows)
    _require(frames >= 2, 'At least two actual transitions are required')
    shapes = dict(observation=(frames,300), actor_action=(frames,16), executed_action=(frames,16),
                  leg_target=(frames,12), wheel_target=(frames,4), pre_action_time_s=(frames,),
                  phase=(frames,), command_vx=(frames,))
    for name, shape in shapes.items():
        value = np.asarray(policy[name])
        _require(value.shape == shape and np.isfinite(value).all(), 'Malformed policy trace: '+name)
    _require(np.isin(policy['phase'], [0,1,2,3]).all() and (np.diff(policy['phase']) >= 0).all(),
             'Command phases contain reset/unknown transitions')
    _require(np.allclose(policy['pre_action_time_s'], np.arange(frames)*.02, atol=1.e-8, rtol=0.),
             'Pre-action clock is not continuous from the ordinary reset')
    _require(all(np.isfinite(list(row.values())).all() for row in rows), 'Nonfinite raw CSV measurement')
    _require(np.allclose([r['t'] for r in rows], (np.arange(frames)+1)*.02, atol=1.e-8, rtol=0.),
             'CSV clock differs from executed transition count')
    _require(np.allclose([r['command_vx'] for r in rows], policy['command_vx'], atol=1.e-7, rtol=0.),
             'CSV actual transition commands differ from Actor pre-action commands')
    _require(np.array_equal([r['eval_phase'] for r in rows[:-1]], policy['phase'][1:]),
             'Post-transition CSV phases do not match the next Actor pre-action phase')
    for key in ('slope_deg','payload_kg','towed_kg'):
        _require(np.allclose([r[key] for r in rows], case[key], atol=1.e-5, rtol=0.),
                 'CSV actual case differs: '+key)
    for name, value in physics.items():
        _require(value.ndim > 0 and len(value) == frames*20 and np.isfinite(value).all(),
                 'Incomplete/nonfinite 1kHz physical trace: '+name)
    _require(physics['joint_pos'].shape == (frames*20,16)
             and physics['joint_velocity'].shape == (frames*20,16), '1kHz joint ordering/shape differs')
    for name in ('requested','applied','velocity'):
        _require(motor[name].shape == (frames*20,16) and np.isfinite(motor[name]).all(),
                 'Incomplete/nonfinite 1kHz motor trace: '+name)
    limits = np.r_[np.full(12,96.),np.full(4,48.)]
    _require(np.array_equal(motor['limits'], limits)
             and motor['joint_names'].tolist() == CONTRACT['action_order']
             and float(motor['physics_dt']) == .001, 'Motor trace limits/order/timestep differ')
    _require(np.max(np.abs(motor['applied'])-limits) <= 1.e-5, 'Actual motor estimate exceeds original software limits')
    _require(np.allclose(motor['applied'], np.clip(motor['requested'],-limits,limits), atol=1.e-4, rtol=0.),
             'Actual motor record does not match original software clipping')
    if 'action_override' in policy:
        _require(not np.asarray(policy['action_override']).any(), 'Recorded policy action override')
    return frames


def _verify_observation(policy, physics, initial, case):
    """The frozen dataset helper assumes tow40; generalize only its context for retention.

    Retention is reconstructed independently with the same exact formulas, not
    by changing recorded contexts into artificial towing observations.
    """
    x = policy['observation']; frames = len(x)
    if case['towed_kg'] == 40.:
        errors = validate_observation_trace(policy, physics, initial)
    else:
        q = np.vstack((initial['joint_position_action_order'],physics['joint_pos'][19::20][:-1]))
        qd = np.vstack((initial['joint_velocity_action_order'],physics['joint_velocity'][19::20][:-1]))
        quat = np.vstack((initial['robot_root_state'][3:7],physics['body_quat_w'][19::20][:-1]))
        omega = np.vstack((np.zeros(3),physics['body_omega_b'][19::20][:-1]))
        rotation = Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        order = CONTRACT['observation_to_action_indices']
        relative = (q-np.asarray(initial['joint_position_action_order']))[:,order]
        relative[:,CONTRACT['wheel_position_channels_in_frame']] = 0.
        expected = dict(base_ang_vel=np.clip(omega,-100.,100.)*.5, projected_gravity=-rotation[:,2,:],
            joint_pos=relative, joint_vel=np.clip(qd[:,order],-100.,100.)*.05,
            velocity_commands=np.column_stack((policy['command_vx'],np.zeros((frames,2)))),
            heading_error=np.arctan2(rotation[:,1,0],rotation[:,0,0])[:,None],
            task_context=np.tile([case['payload_kg']/15.,0.],(frames,1)))
        errors = {}
        for name, value in expected.items():
            begin,end = CONTRACT['layout'][name]['latest']
            errors[name] = float(np.abs(x[:,begin:end]-value).max())
        begin,end = CONTRACT['layout']['actions']['latest']
        errors['actual_previous_action'] = float(np.abs(x[1:,begin:end]-np.clip(policy['executed_action'][:-1],-100.,100.)).max())
    for name, layout in CONTRACT['layout'].items():
        begin,end = layout['columns']; history = x[:,begin:end].reshape(frames,5,layout['width'])
        errors['history_'+name] = float(np.abs(history[1:,:4]-history[:-1,1:]).max())
        errors['initial_history_'+name] = float(np.abs(history[0]-history[0,-1]).max())
    _require(all(np.isfinite(value) and value <= 1.e-5 for value in errors.values()),
             'Actual deployed observation reconstruction differs: '+repr(errors))
    return errors


def _verify_actor_and_mapping(student, policy, initial, robot):
    raw = policy['actor_action']; observed = policy['observation']
    maximum = 0.
    with torch.inference_mode():
        for begin in range(0,len(raw),128):
            predicted = student(torch.from_numpy(observed[begin:begin+128]).float()).cpu().numpy()
            actual = raw[begin:begin+128]
            maximum = max(maximum,float(np.abs(predicted-actual).max()))
            _require(np.isfinite(predicted).all() and np.allclose(predicted,actual,atol=ACTOR_ATOL,rtol=ACTOR_RTOL),
                     'Recorded raw action is not reproduced by the candidate Actor; max error='+str(maximum))
    joints = {joint.get('name'):joint for joint in ET.parse(robot).getroot().findall('joint')}
    limits = np.array([[float(joints[name].find('limit').get(key)) for key in ('lower','upper')]
                       for name in CONTRACT['action_order'][:12]],dtype=raw.dtype)
    default = np.asarray(initial['joint_position_action_order'][:12],dtype=raw.dtype)
    mapping = runtime_checks.check_action_mapping(raw,policy['executed_action'],policy['leg_target'],
        policy['wheel_target'],np.broadcast_to(default,(len(raw),12)),np.broadcast_to(limits,(len(raw),12,2)))
    return dict(replayed_frames=len(raw),cpu_row_inference_max_absolute_error=maximum,
                numerical_atol=ACTOR_ATOL,numerical_rtol=ACTOR_RTOL,**mapping)


def _verify_clock(policy, rows, manifest, case, request, retention):
    if retention:
        for i in range(len(rows)):
            retention_protocol.check_pre_action(policy['observation'][i:i+1],policy['pre_action_time_s'][i],
                policy['phase'][i],policy['command_vx'][i],case)
        start = 3. if case['speed'] > 0. else None
    else:
        start = request['fixed_preparation_s']
        alignment = runtime_checks.check_source_preparation(manifest,start,case_slope=case['slope_deg'])
        _require(request['move_start_s'] == start and request['preparation_protocol'] == alignment,
                 'Declared command preparation differs from source manifest')
        for t,phase,command in zip(policy['pre_action_time_s'],policy['phase'],policy['command_vx']):
            runtime_checks.check_pre_action_preparation(t,phase,command,start)
    moving = np.flatnonzero(policy['phase'] >= 1)
    actual = float(policy['pre_action_time_s'][moving[0]]) if len(moving) else None
    if start is not None:
        _require(actual is not None and abs(actual-start) <= 1.e-8,
                 'Actual command clock did not reach its manifest/protocol movement start')
    else:
        _require(actual is None, 'Static retention unexpectedly entered movement')
    return dict(actual_pre_action_move_start_s=actual,expected_pre_action_move_start_s=start,
                runtime_ready_or_teacher_gate=False)


def _retention_behavior(rows, case):
    # Reconstruct reason from actual transitions, not saved success/failure_reason.
    terminated = any(row['terminated'] != 0. or row['time_out'] != 0. for row in rows)
    reason = 'terminated' if terminated else (None if retention_protocol.original.finished(rows,case)
                                             else 'incomplete_raw_protocol')
    _require(len(rows) <= retention_protocol.original.max_steps(case), 'Retention raw duration exceeds original budget')
    return retention_protocol.original.summarize(rows,case,reason)


def _verify_run(run, checkpoint, digest, manifest, student, retention):
    run = Path(run).resolve(strict=True)
    _require(not list(run.glob('*worker_error.json')), 'Execution error artifact exists: '+str(run))
    review_name = 'retention_behavior_review.json' if retention else 'student_behavior_review.json'
    source_name = 'retention_source_manifest.json' if retention else 'student_source_manifest.json'
    names = ['request.json','runtime_check.json','initial_state_runtime.json','solver_runtime_usd.json',
             'friction_runtime.json','physics_recorder.json',source_name,review_name,PREFIX+'.json',
             PREFIX+'.csv',PREFIX+'_policy_trace.npz',PREFIX+'_physics_trace.npz',PREFIX+'_motor_trace.npz',
             'assets/cart.urdf']
    if retention:
        names.append('payload_physical_readback.json')
    hashes = {name:sha256(run/name) for name in names}
    request = _json(run/'request.json')
    _verify_identity(request,checkpoint,digest,manifest,retention)
    sources = _verify_sources(run,retention)
    if retention:
        index = request['case_index']; case = retention_protocol.select_case(index)
        profile = manifest['teacher_bundle']['episodes'][0]['physics_profile']
        _require(request['source_towing_physics_profile'] == profile and
                 request['source_protocol_sha256'] == sha256(retention_protocol.PROTOCOL_SOURCE),
                 'Retention physics/source protocol differs from the candidate')
        _require(request.get('physical_trailer_connected') is False and request.get('nominal_towed_kg') == 0.,
                 'Retention must be nominally untowed and physically disconnected')
    else:
        index = None
        case,profile = runtime_checks.select_case(manifest,request['slope_deg'])
        _require(request.get('physical_trailer_connected') is True and request.get('physical_cart_mass_kg') == 40.
                 and request.get('trial_duration_s') == 40. and request.get('rounded_cart_wheels') is True
                 and request.get('records_drawbar_pre_step') is True,
                 'Heavy anchor cart/duration/raw drawbar contract differs')
        _require(request['execution_physics'] == dict(profile,device='cuda:0'), 'Anchor physical profile differs')
    _require(request['cases'] == [case], 'Run case does not match its original protocol/source case')
    runtime,initial,solver,materials = [_json(run/name) for name in
        ('runtime_check.json','initial_state_runtime.json','solver_runtime_usd.json','friction_runtime.json')]
    robot = _snapshot_path(run,ROBOT)
    assets = dict(robot=robot,cart=run/'assets/cart.urdf')
    if retention:
        runtime_review = retention_protocol.check_runtime(runtime,initial,solver,materials,assets,profile,case,
            _json(run/'payload_physical_readback.json'))
    else:
        runtime_review = runtime_checks.check_runtime(runtime,initial,solver,materials,assets,profile,case)
    _require(_json(run/'physics_recorder.json')['sample_hz'] == 1000, 'Native recorder sample rate differs')
    with (run/(PREFIX+'.csv')).open(newline='') as stream:
        rows = [{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    policy,physics,motor = [_npz(run/(PREFIX+'_'+name+'_trace.npz')) for name in ('policy','physics','motor')]
    frames = _verify_traces(policy,physics,motor,rows,case)
    _check_execution_record(_json(run/review_name),frames)
    observation = _verify_observation(policy,physics,initial,case)
    actor = _verify_actor_and_mapping(student,policy,initial,robot)
    no_towing = retention_protocol.check_no_towing(physics,rows) if retention else None
    behavior = _retention_behavior(rows,case) if retention else _behavior_audit(run)
    # A terminated entry may never reach movement. Still recompute and retain
    # its raw physical failure, rather than replacing it with a clock-only error.
    evidence_errors = []
    try:
        clock = _verify_clock(policy,rows,manifest,case,request,retention)
    except EvidenceError as error:
        evidence_errors.append(str(error))
        clock = dict(verified=False,error=str(error))
    passed = bool((behavior['success'] if retention else behavior['move_brake_hold_pass'])
                  and not evidence_errors)
    _require(hashes == {name:sha256(run/name) for name in names}, 'Raw evidence changed during admission review')
    _require(_verify_sources(run,retention) == sources, 'Execution sources changed during admission review')
    _require(sha256(checkpoint) == digest, 'Candidate checkpoint changed during admission review')
    return dict(directory=str(run),case_index=index,case=case,passed=passed,source_review=sources,
        runtime_review=runtime_review,raw_files_sha256=hashes,observation_reconstruction=observation,
        actor_replay=actor,command_clock=clock,no_towing=no_towing,behavior=behavior,evidence_errors=evidence_errors)


def verify_behavior_admission(checkpoint, anchor_runs, retention_runs):
    """Return a fail-closed, JSON-serializable report for exactly two anchors and 12 cases.

    Evidence errors are returned in ``errors``; this function never writes runs,
    creates missing evidence, starts an environment, or performs PPO updates.
    The caller may save the report independently. Existing saved pass flags are
    not inputs to the acceptance decision.
    """
    report = dict(schema='r1_migration_v1_pre_ppo_behavior_admission',checkpoint=str(checkpoint),
        pre_ppo_behavior_ready=False,formal_ppo_authorized=False,real_ppo_updates_performed=0,
        full64_gpu_update_resume_verified=False,anchors=[],retention=[],errors=[],
        remaining_formal_requirement='Separate real full-64 first-batch GPU PPO update/save/resume validation')
    try:
        checkpoint = Path(checkpoint).resolve(strict=True)
        student,manifest = load_student(checkpoint,device='cpu',inference_mode='row')
        digest = sha256(checkpoint); report['checkpoint_sha256'] = digest
        anchors = [Path(path).resolve() for path in anchor_runs]
        retention = [Path(path).resolve() for path in retention_runs]
        _require(len(anchors) == 2 and len(retention) == 12,
                 'Require exactly two directional anchors and all original 12 retention cases')
        _require(len(set(anchors+retention)) == 14, 'Duplicate/mixed run directories cannot provide case coverage')
        identities = [_json(run/'request.json') for run in anchors+retention]
        for request in identities:
            _require(request.get('checkpoint_sha256') == digest, 'Mixed candidate checkpoint evidence')
        _require({request.get('slope_deg') for request in identities[:2]} == {-20.,25.},
                 'Heavy anchors must cover +25 and -20 degrees')
        indices = [request.get('case_index') for request in identities[2:]]
        _require(all(type(index) is int for index in indices) and sorted(indices) == list(range(12)),
                 'Retention must cover each original case index exactly once')
        for kind, runs, is_retention in [('anchors',anchors,False),('retention',retention,True)]:
            for run in runs:
                try:
                    result = _verify_run(run,checkpoint,digest,manifest,student,is_retention)
                    report[kind].append(result)
                    if not result['passed']:
                        report['errors'].append(str(run)+': recomputed raw behavior failed')
                except (OSError,ValueError,KeyError,TypeError,RuntimeError,IndexError) as error:
                    report[kind].append(dict(directory=str(run),passed=False,error=str(error)))
                    report['errors'].append(str(run)+': '+str(error))
        report['pre_ppo_behavior_ready'] = not report['errors']
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,IndexError) as error:
        report['errors'].append(str(error))
    return report
