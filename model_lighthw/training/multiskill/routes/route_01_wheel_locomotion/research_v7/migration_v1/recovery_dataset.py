"""Executed recovery labels: a real student prefix, then a successful teacher suffix.

Full raw episodes are verified before filtering. Shadow actions and student
prefix actions are never labels. This module does not authorize autonomy/PPO.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .contract import CONTRACT, REPO, V7, masked_observations, sha256, validate_runtime
from .dataset import (EvidenceError, _json, _behavior_audit, verify_snapshot,
    required_execution_sources, validate_observation_trace, stage_labels, read_teacher_bundle)
from .preparation_sampling import prepare_sampling, preparation_stages
from .runtime_checks import check_source_preparation
from .student import load_student
from .bc import SCHEMA as NOMINAL_STUDENT_SCHEMA

SCHEMA = 'r1_executed_recovery_labels_v1'
PREFIX_VERSION = 'r1_executed_student_prefix_then_teacher_v1'
CONSTRAINED_STUDENT_SCHEMA = 'r1_migration_v1_constrained_head_candidate_v1'


def episode_identity(episode):
    """Identity is the immutable complete recorded trajectory, not its directory."""
    return episode['source_files']['seed43_case00_policy_trace.npz']


def source_student_provenance(checkpoint):
    path = Path(checkpoint).resolve(strict=True)
    schema = _json(path.with_name('manifest.json')).get('schema')
    if schema == NOMINAL_STUDENT_SCHEMA:
        interface, manifest = load_student(path, device='cpu', inference_mode='row')
    elif schema == CONSTRAINED_STUDENT_SCHEMA:
        from .constrained_student import load_student as load_constrained_student
        interface, manifest = load_constrained_student(path, device='cpu', inference_mode='row')
    else:
        raise EvidenceError('Unsupported recovery parent student schema')
    if manifest.get('schema') != schema:
        raise EvidenceError('Recovery strict parent loader returned a different schema')
    return interface, manifest, dict(path=str(path), checkpoint_sha256=sha256(path),
        manifest_sha256=sha256(path.with_name('manifest.json')), schema=manifest['schema'])


def validate_recovery_physics(request, runtime, initial, solver, geometry):
    """Explicit prefix-aware contract; never falsify actor_inference for old gates.

    The physical checks intentionally mirror dataset.validate_physics while the
    control-source check is different. The original ten signed files stay intact.
    """
    physics = request.get('execution_physics', {})
    if request.get('actor_inference') is not True or request.get('fixed_command_clock') is not True or not isinstance(request.get('student_prefix'), dict):
        raise EvidenceError('Recovery requires explicit actor inference, fixed clock and student-prefix provenance')
    if request.get('initial_state_mode') != 'ordinary' or initial.get('mode') != 'ordinary':
        raise EvidenceError('Recovery requires an actual ordinary reset')
    if not str(physics.get('device', '')).startswith('cuda') or initial.get('cuda_available') is not True or not str(initial.get('robot_tensor_device', '')).startswith('cuda'):
        raise EvidenceError('Recovery requires confirmed GPU execution')
    if request.get('seed') != 43 or request.get('slope_deg') not in (25., -20.) or request.get('physical_cart_mass_kg') != 40.:
        raise EvidenceError('Recovery format requires seed43 +25/-20 degree 40kg anchors')
    if request.get('noise') is not False or not request.get('physical_trailer_connected') or request.get('ppo_updates') != 0:
        raise EvidenceError('Recovery must use clean physical towing without PPO')
    expected = dict(hitch_height_m=.04, robot_hitch_offset_m=.2, ground_friction=3.,
        rolling_coefficient=.02, base_com_shift_m=0., robot_mass_variant=False)
    if any(physics.get(key) != value for key,value in expected.items()) or request.get('ground_friction') != 3.:
        raise EvidenceError('Recovery physical assumptions differ from the accepted mu3 line')
    if (not request.get('rounded_cart_wheels') or geometry.get('variant') != 'spherical_tread_passive_wheels'
            or geometry.get('total_mass_kg') != 40. or geometry.get('wheel_radius_m') != .14):
        raise EvidenceError('Recovery passive rounded cart asset differs')
    if [request.get('position_iterations'), request.get('velocity_iterations')] != [16,4] or request.get('cart_solver_iterations') != [16,4]:
        raise EvidenceError('Recovery requires robot and cart GPU16/4')
    for suffix in ('/Robot/Base_link','/Trailer/cart_base'):
        entries = [value for key,value in solver.items() if key.endswith(suffix)]
        if len(entries) != 1 or any(entries[0].get('physxArticulation:solver'+name+'IterationCount') != value for name,value in [('Position',16),('Velocity',4)]):
            raise EvidenceError('Recovery actual USD solver differs: '+suffix)
    scene = solver.get('/physicsScene', {})
    if scene.get('physxScene:maxPositionIterationCount') != 16 or scene.get('physxScene:maxVelocityIterationCount') != 4:
        raise EvidenceError('Recovery scene solver differs')
    angle = np.deg2rad(request['slope_deg']); normal = np.array([-np.sin(angle),0.,np.cos(angle)])
    default = np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])
    root,cart,origin,q = [np.asarray(initial.get(key, []),dtype=float) for key in
        ('robot_root_state','cart_root_state','env_origin','joint_position_action_order')]
    quat = np.array([np.cos(angle/2.),0.,-np.sin(angle/2.),0.])
    if (root.shape != (13,) or cart.shape != (13,) or origin.shape != (3,) or q.shape != (16,)
            or not all(np.isfinite(value).all() for value in (root,cart,origin,q))
            or initial.get('episode_step') != 0 or initial.get('cart_mass_kg') != 40.
            or not np.allclose(q,default,atol=1.e-6,rtol=0.)
            or not np.allclose(root[3:7],quat,atol=1.e-6,rtol=0.) or not np.allclose(cart[3:7],quat,atol=1.e-6,rtol=0.)
            or not np.allclose(root[7:],0.,atol=1.e-6,rtol=0.) or not np.allclose(cart[7:],0.,atol=1.e-6,rtol=0.)
            or abs((root[:3]-origin)@normal-.416223) > 1.e-5):
        raise EvidenceError('Recovery ordinary reset content does not reproduce')
    validate_runtime(runtime,request['base_network_contract'])
    points = dict(robot_hitch_local_m=[-.38,0.,-.176223],cart_hitch_local_m=[.35,0.,-.26])
    for key,point in points.items():
        if not np.allclose(runtime.get(key),point,atol=1.e-6,rtol=0.):
            raise EvidenceError('Recovery actual hitch differs: '+key)
    result = dict(physics); result.pop('device')
    result.update(robot_solver=[16,4],cart_solver=[16,4],mass_kg=40.,geometry=geometry,**points)
    return result


def validate_prefix_trace(policy, metadata, execution, interface, student_manifest, provenance, report, request):
    """Verify one contiguous prefix/takeover and both raw action sources."""
    if metadata.get('version') != PREFIX_VERSION or metadata.get('inference_mode') != 'row':
        raise EvidenceError('Unknown recovery prefix execution contract')
    schema = student_manifest.get('schema')
    if (schema == CONSTRAINED_STUDENT_SCHEMA and (
            metadata.get('student_schema') != schema
            or metadata.get('student_loader') != 'migration_v1.constrained_student.load_student')):
        raise EvidenceError('Constrained recovery prefix must declare its strict schema loader')
    if 'student_schema' in metadata and metadata['student_schema'] != schema:
        raise EvidenceError('Recovery prefix declared schema differs from its strict parent')
    required = dict(prefix_teacher_actions_are_shadow_only=True,student_prefix_is_teacher_label=False,
        teacher_executed_actions_require_complete_episode_pass=True,autonomous_behavior_accepted=False,formal_ppo_authorized=False)
    if any(metadata.get(key) is not value for key,value in required.items()):
        raise EvidenceError('Recovery prefix/shadow label declaration is invalid')
    if (metadata.get('student_checkpoint_sha256') != provenance['checkpoint_sha256']
            or metadata.get('student_manifest_sha256') != provenance['manifest_sha256']
            or Path(metadata['student_checkpoint']).resolve() != Path(provenance['path'])):
        raise EvidenceError('Recovery prefix student provenance differs')
    k = metadata.get('prefix_steps'); seconds = metadata.get('prefix_seconds')
    if (not isinstance(k,int) or isinstance(k,bool) or not 1 <= k <= 50
            or not isinstance(seconds,(int,float)) or not np.isfinite(seconds) or abs(seconds-k*.02)>1.e-8):
        raise EvidenceError('Invalid declared recovery prefix clock')
    n = len(policy['observation'])
    if n <= k or any(execution.get(key) != value for key,value in metadata.items()):
        raise EvidenceError('Recovery execution record/provenance differs or lacks teacher suffix')
    if execution.get('actual_student_frames') != k or execution.get('actual_teacher_frames') != n-k or execution.get('complete_success_verified') is not False:
        raise EvidenceError('Recovery executed frame accounting differs')
    keys = ('student_prefix_active','teacher_executed','student_prefix_frame_index','student_prefix_takeover_frame','teacher_time_s')
    values = {key:np.asarray(policy.get(key,[])) for key in keys}
    if any(value.shape != (n,) or not np.isfinite(value).all() for value in values.values()):
        raise EvidenceError('Recovery source masks/time lengths or finiteness differ')
    before = np.arange(n)<k
    if (not np.array_equal(values['student_prefix_active'],before.astype(float))
            or not np.array_equal(values['teacher_executed'],(~before).astype(float))
            or not np.array_equal(values['student_prefix_frame_index'],np.arange(n))
            or not np.array_equal(values['student_prefix_takeover_frame'],(np.arange(n)==k).astype(float))
            or not np.allclose(values['teacher_time_s'],np.arange(n)*.02,atol=1.e-7,rtol=0.)):
        raise EvidenceError('Recovery must have exactly one uninterrupted student-to-teacher switch')
    student,teacher,actual = [np.asarray(policy.get(key,[])) for key in
        ('student_prefix_student_raw','student_prefix_teacher_raw','executed_action')]
    if any(value.shape != (n,16) or not np.isfinite(value).all() for value in (student,teacher,actual)):
        raise EvidenceError('Recovery raw/actual action arrays differ')
    if (not np.array_equal(actual[before],student[before]) or not np.array_equal(actual[~before],teacher[~before])
            or np.any(student[~before] != 0.)):
        raise EvidenceError('Recovery executed action differs from its declared sole source')
    with torch.inference_mode():
        prediction = interface(torch.as_tensor(policy['observation'][:k],dtype=torch.float32)).cpu().numpy()
    error = float(abs(prediction-student[:k]).max())
    # Explicit CPU-vs-GPU row inference tolerance, not action blending allowance.
    if not np.allclose(prediction,student[:k],atol=3.e-5,rtol=2.e-6):
        raise EvidenceError('Recorded prefix does not reproduce the declared student row inference')
    clock = check_source_preparation(student_manifest,request['move_start_s'],case_slope=request['slope_deg'])
    if metadata.get('fixed_command_clock') != clock or abs(report['move_window_s'][0]-clock['preparation_seconds'])>1.e-8:
        raise EvidenceError('Recovery actual command clock differs from fixed source task clock')
    return np.flatnonzero(~before),dict(prefix_frames=k,teacher_frames=n-k,
        prefix_inference_max_abs_error=error,prefix_inference_atol=3.e-5,prefix_inference_rtol=2.e-6,
        source_selection_exact=True,shadow_labels_used=False,student_prefix_labels_used=False,command_clock=clock)


def read_recovery_episode(directory, source_student, *, balance_preparation=True):
    return _read_recovery_episode(directory,source_student,balance_preparation=balance_preparation)


def _read_recovery_episode(directory, source_student, *, balance_preparation=True, _loaded=None):
    run = Path(directory).resolve(strict=True)
    request,runtime,initial = [_json(run/name) for name in ('request.json','runtime_check.json','initial_state_runtime.json')]
    profile = validate_recovery_physics(request,runtime,initial,_json(run/'solver_runtime_usd.json'),_json(run/'rounded_cart_geometry.json'))
    interface,parent,provenance = _loaded or source_student_provenance(source_student)
    sources = verify_snapshot(run,required=required_execution_sources())
    urdf = 'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    if sources.get(urdf) != request['base_network_contract']['urdf_sha256']:
        raise EvidenceError('Recovery robot asset differs from its source contract')
    profile.update(robot_urdf_sha256=sources[urdf],cart_urdf_sha256=sha256(run/'assets/cart.urdf'))
    if any(profile != episode['physics_profile'] for episode in parent['teacher_bundle']['episodes']):
        raise EvidenceError('Recovery and source student teacher physics differ')
    if request.get('checkpoint_sha256') != parent['baseline']['checkpoint_sha256'] or sha256(Path(request['checkpoint'])) != request['checkpoint_sha256']:
        raise EvidenceError('Recovery baseline provenance differs')
    metadata = request['student_prefix']; declared = metadata.get('source_sha256',{})
    required = set(parent['sources']) | {str(V7/'scripts/student_prefix.py'),str(V7/'migration_v1/runtime_checks.py')}
    if set(declared) != required or any(declared.get(key) != value for key,value in parent['sources'].items()):
        raise EvidenceError('Recovery prefix execution source set differs')
    for name,digest in declared.items():
        original = Path(name)
        if not original.is_absolute() or not original.is_relative_to(REPO):
            raise EvidenceError('Invalid recovery prefix source path')
        if sha256(run/'source_snapshot'/original.relative_to(REPO)) != digest:
            raise EvidenceError('Recovery prefix archived source SHA differs')
    files = ('request.json','runtime_check.json','initial_state_runtime.json','source_manifest.json','research_source_manifest.json',
        'solver_runtime_usd.json','rounded_cart_geometry.json','assets/cart.urdf','student_prefix_execution.json',
        'seed43_case00.csv','seed43_case00_physics_trace.npz','seed43_case00_policy_trace.npz','seed43_case00_motor_trace.npz')
    before = {name:sha256(run/name) for name in files}
    report = _behavior_audit(run)
    if not report.get('move_brake_hold_pass'):
        raise EvidenceError('Recovery failed complete physical episode; no suffix labels: '+repr([key for key,value in report.get('checks',{}).items() if not value]))
    with np.load(run/'seed43_case00_policy_trace.npz',allow_pickle=False) as data: policy = dict(data)
    with np.load(run/'seed43_case00_physics_trace.npz',allow_pickle=False) as data:
        physics = {key:data[key] for key in ('joint_pos','joint_velocity','body_quat_w','body_omega_b')}
    observation_errors = validate_observation_trace(policy,physics,initial)  # Entire N frames before suffix selection.
    rows,selection = validate_prefix_trace(policy,metadata,_json(run/'student_prefix_execution.json'),interface,parent,provenance,report,request)
    cases = request['cases']
    if len(cases)!=1 or cases[0]['payload_kg']!=0. or cases[0]['towed_kg']!=40. or cases[0]['slope_deg']!=request['slope_deg']:
        raise EvidenceError('Recovery nominal case differs from actual physical request')
    stage = stage_labels(policy,report,cases[0]['speed']); sampling_stage = stage
    if balance_preparation:
        import csv
        with (run/'seed43_case00.csv').open() as stream: csv_rows = list(csv.DictReader(stream))
        columns = {key:np.array([float(row[key]) for row in csv_rows]) for key in ('t','eval_phase','command_vx')}
        sampling_stage = preparation_stages(policy,stage,request['entry_duration_s'],columns['t'],columns['eval_phase'],columns['command_vx'])
    if any(sha256(run/name)!=digest for name,digest in before.items()):
        raise EvidenceError('Recovery source changed during reading')
    manifest = dict(directory=str(run),case=cases[0],seed=43,slot=0,physics_profile=profile,source_files=before,
        prefix_source_files=declared,source_student=provenance,prefix=selection,
        full_episode_samples=len(stage),samples=len(rows),label_frame_indices=rows.tolist(),
        label_times_s=(rows*.02).tolist(),complete_recovery_pass=True,successful_labels_are_executed=True,
        actor_success=False,student_prefix_success_labels=False,shadow_labels_used=False,
        observation_reconstruction_max_errors=observation_errors,
        command_timing=dict(actual_move_start_s=report['move_window_s'][0],command_zero_time_s=report['command_zero_time_s']))
    return dict(observation=policy['observation'][rows],action=policy['executed_action'][rows],stage=stage[rows],
        sampling_stage=sampling_stage[rows],source_frame_index=rows,manifest=manifest)


def read_recovery_bundle(nominal_runs, recovery_runs, source_student, *, balance_preparation=True):
    runs = [Path(path).resolve(strict=True) for path in recovery_runs]
    if not runs or len(runs)!=len(set(runs)):
        raise EvidenceError('Recovery fit requires at least one nonduplicate complete executed recovery')
    loaded = source_student_provenance(source_student)
    ordinary,teachers = read_teacher_bundle(nominal_runs)
    sampled,configuration = prepare_sampling(ordinary,teachers,enabled=balance_preparation)
    recovery = [_read_recovery_episode(path,source_student,balance_preparation=balance_preparation,_loaded=loaded) for path in runs]
    episodes = teachers['episodes']+[item['manifest'] for item in recovery]
    if len({episode_identity(episode) for episode in episodes})!=len(episodes):
        raise EvidenceError('Duplicate full trajectory under different directories')
    if any(episode['physics_profile'] != episodes[0]['physics_profile'] for episode in episodes):
        raise EvidenceError('Nominal and recovery physical profiles differ')
    if any(_json(Path(ep['directory'])/'request.json')['checkpoint_sha256'] != loaded[1]['baseline']['checkpoint_sha256'] for ep in teachers['episodes']):
        raise EvidenceError('Nominal and recovery baseline provenance differs')
    for slope in (25.,-20.):
        expected = check_source_preparation(loaded[1],next(e['command_timing']['actual_move_start_s'] for e in loaded[1]['teacher_bundle']['episodes'] if e['case']['slope_deg']==slope),case_slope=slope)['preparation_seconds']
        if any(abs(e['command_timing']['actual_move_start_s']-expected)>1.e-8 for e in episodes if e['case']['slope_deg']==slope):
            raise EvidenceError('Nominal/recovery selected-task command clocks differ')
    groups={}; packed={key:[] for key in ('observation','action','stage','sampling_stage','episode_id','group_id','case_key','source_frame_index','origin')}
    for index,episode in enumerate(episodes):
        nominal=index<len(teachers['episodes']); origin='nominal_teacher' if nominal else 'executed_recovery'
        if nominal:
            mask=ordinary['episode_id']==index
            data={key:ordinary[key][mask] for key in ('observation','action','stage')}
            data['source_frame_index']=np.arange(mask.sum())
            if balance_preparation:
                names={record['group_id']:record['stage'] for record in configuration['groups']}
                data['sampling_stage']=np.array([names[int(group)] for group in sampled['group_id'][mask]])
            else:data['sampling_stage']=data['stage']
        else:data=recovery[index-len(teachers['episodes'])]
        case=episode['case'];key=f"slope{case['slope_deg']:g}_payload0_tow40_speed{case['speed']:g}";n=len(data['action'])
        group=np.array([groups.setdefault((origin,key,str(stage)),len(groups)) for stage in data['sampling_stage']],dtype=np.int64)
        for name in ('observation','action','stage','sampling_stage','source_frame_index'):packed[name].append(data[name])
        for name,value in dict(episode_id=np.full(n,index),group_id=group,case_key=np.full(n,key),origin=np.full(n,origin)).items():packed[name].append(value)
    arrays={key:np.concatenate(value) for key,value in packed.items()}
    arrays.update(observation_masked=masked_observations(arrays['observation']),successful=np.ones(len(arrays['action']),dtype=bool))
    records=[]
    for (origin,case,stage),group in groups.items():
        rows=np.flatnonzero(arrays['group_id']==group)
        records.append(dict(group_id=group,origin=origin,case_key=case,stage=stage,count=len(rows),
            members=[dict(episode_id=int(ep),frame_indices=arrays['source_frame_index'][rows[arrays['episode_id'][rows]==ep]].tolist(),
                teacher_times_s=(arrays['source_frame_index'][rows[arrays['episode_id'][rows]==ep]]*.02).tolist()) for ep in np.unique(arrays['episode_id'][rows])]))
    manifest=dict(schema=SCHEMA,teacher_bundle=teachers,recovery_episodes=[item['manifest'] for item in recovery],
        episodes=episodes,source_student=loaded[2],balance_preparation=bool(balance_preparation),groups=records,
        samples=len(arrays['action']),label_kind='actual_executed_teacher_actions_only',
        sampling='uniform_origin_case_stage_then_uniform_frame',student_prefix_labels_used=False,shadow_labels_used=False,
        complete_episodes_verified_before_filtering=True,autonomous_behavior_accepted=False,formal_ppo_authorized=False)
    return arrays,manifest
