"""学生worker纯CPU接线/拒绝门检查；人工读回字典不作为物理证据。"""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1 import student_worker, runtime_checks
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError


@pytest.fixture
def fake_runtime(tmp_path):
    fixture = Path(__file__).parent/'fixtures/ordinary_observation_excerpt.json'
    data = json.loads(fixture.read_text())
    runtime = deepcopy(data['runtime']); initial = deepcopy(data['initial'])
    initial.update(robot_tensor_device='cuda:0', simulation_device='cuda:0', command_vx=0.,
        joint_velocity_action_order=[0.]*16, cart_joint_velocity=[0.]*4)
    physics = dict(data['request_metadata']['execution_physics'])
    physics.update(robot_solver=[16,4], cart_solver=[16,4],
        robot_hitch_local_m=deepcopy(runtime['robot_hitch_local_m']), cart_hitch_local_m=deepcopy(runtime['cart_hitch_local_m']))
    assets = {name: tmp_path/(name+'.urdf') for name in ('robot','cart')}
    for name, path in assets.items():
        path.write_text('synthetic asset hash check only: '+name)
        physics[name+'_urdf_sha256'] = sha256(path)
    solver = {'/physicsScene': {'physxScene:maxPositionIterationCount':16, 'physxScene:maxVelocityIterationCount':4}}
    for name in ('Robot/Base_link','Trailer/cart_base'):
        solver['/World/envs/env_0/'+name] = {'physxArticulation:solverPositionIterationCount':16,
            'physxArticulation:solverVelocityIterationCount':4}
    materials = dict(assets={name: [[[3.,3.]]] for name in ('robot','trailer')},
        terrain={'synthetic':dict(static=3.,dynamic=3.,combine='min')})
    case = dict(slope_deg=25.,payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.2)
    return dict(runtime=runtime, initial=initial, solver=solver, materials=materials, assets=assets, profile=physics, case=case)


def test_runtime_checker_accepts_consistent_synthetic_readbacks_without_acceptance(fake_runtime):
    result = runtime_checks.check_runtime(**fake_runtime)
    assert result['runtime_profile_verified']
    assert result['student_behavior_accepted'] is False
    assert result['formal_ppo_authorized'] is False


@pytest.mark.parametrize('change', ['cpu','solver','material','hitch','preload','joint_speed','noise','case','acceleration','inertia'])
def test_runtime_readbacks_cannot_silently_drift(fake_runtime, change):
    if change == 'cpu':
        fake_runtime['initial']['robot_tensor_device'] = 'cpu'
    elif change == 'solver':
        fake_runtime['solver']['/physicsScene']['physxScene:maxPositionIterationCount'] = 4
    elif change == 'material':
        fake_runtime['materials']['assets']['robot'][0][0][0] = 4.
    elif change == 'hitch':
        fake_runtime['runtime']['robot_hitch_local_m'][2] += .01
    elif change == 'preload':
        fake_runtime['initial']['cart_root_state'][0] -= .1
    elif change == 'joint_speed':
        fake_runtime['initial']['joint_velocity_action_order'][12] = 1.
    elif change == 'noise':
        fake_runtime['runtime']['policy_enable_corruption'] = True
    elif change == 'case':
        fake_runtime['runtime']['slopes_deg'] = [-20.]
    elif change == 'acceleration':
        fake_runtime['runtime']['loaded_acceleration_mps2'] = .1
    else:
        fake_runtime['runtime']['physical_com_inertia_verified'] = False
    with pytest.raises(ValueError):
        runtime_checks.check_runtime(**fake_runtime)


def test_native_runtime_tuple_dimensions_are_normalized_at_worker_boundary(fake_runtime):
    fake_runtime['runtime']['dimensions'] = tuple(fake_runtime['runtime']['dimensions'])
    fake_runtime['runtime'] = student_worker.serial(fake_runtime['runtime'])
    assert runtime_checks.check_runtime(**fake_runtime)['runtime_profile_verified']


def test_case_uses_declared_12m_command_not_1m_acceptance(fake_runtime):
    up = dict(case=fake_runtime['case'], physics_profile=fake_runtime['profile'])
    down = deepcopy(up); down['case']['slope_deg'] = -20.
    manifest = dict(teacher_bundle=dict(episodes=[up,down]))
    case, profile = runtime_checks.select_case(manifest, -20.)
    assert case['goal_distance_m'] == 1.2 and profile == fake_runtime['profile']
    down['physics_profile']['ground_friction'] = 4.
    with pytest.raises(EvidenceError, match='different physics'):
        runtime_checks.select_case(manifest, -20.)


def test_native_mapping_accepts_only_original_clips_without_controller_increment():
    raw = np.linspace(-12,12,16,dtype=np.float32)[None]
    clip = 9.
    actual = np.clip(raw,-clip,clip)
    default = np.zeros((1,12),dtype=np.float32)
    limits = np.broadcast_to(np.array([-1.,1.],dtype=np.float32),(1,12,2))
    leg = np.clip(actual[:,:12]*np.array([.08,.14,.14]*4,dtype=np.float32),-1,1)
    wheel = np.clip(actual[:,12:]*5.,-50.,50.)
    result = runtime_checks.check_action_mapping(raw,actual,leg,wheel,default,limits,clip)
    assert result['action_override'] is False
    modified = actual.copy(); modified[0,12] += .01
    with pytest.raises(EvidenceError,match='differs from Actor'):
        runtime_checks.check_action_mapping(raw,modified,leg,wheel,default,limits,clip)
    wheel[0,0] += .2
    with pytest.raises(EvidenceError,match='mapping differs'):
        runtime_checks.check_action_mapping(raw,actual,leg,wheel,default,limits,clip)


def test_missing_evidence_rejected_before_app_or_output(monkeypatch,tmp_path):
    def missing(*args,**kwargs):
        raise EvidenceError('No complete downhill teacher/student evidence')
    monkeypatch.setattr(student_worker,'load_student',missing)
    monkeypatch.setattr(student_worker,'run_simulation',lambda _: pytest.fail('App path reached before strict source gate'))
    output=tmp_path/'never_created'
    with pytest.raises(SystemExit) as raised:
        student_worker.main(['--checkpoint',str(tmp_path/'absent.pt'),'--slope-deg','25','--output',str(output)])
    assert raised.value.code == 2 and not output.exists()


def test_dependency_snapshot_freezes_native_physics_and_recorders_without_importing_sim():
    files = student_worker.dependency_sources()
    assert all(path.is_file() for path in files)
    names = {path.name for path in files}
    assert {'cart.py','coordinated_control.py','contact_state.py','actuator_pd.py','observation_manager.py','vecenv_wrapper.py'} <= names
    assert any(path.suffix.lower()=='.stl' for path in files)
    assert 'isaaclab.app' not in sys.modules
    tree=ast.parse(Path(student_worker.__file__).read_text())
    top_imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
    assert not any(isinstance(node,ast.ImportFrom) and node.module=='isaaclab.app' for node in top_imports)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert node.module not in ('loaded_cart_worker','worker','controller','ordinary_entry_controller')


def test_student_dependency_scope_covers_local_imports_without_parallel_ppo_sources():
    directory=Path(student_worker.__file__).parent
    files=set(student_worker.dependency_sources())
    selected={path.name for path in files if path.parent==directory}
    expected={'__init__.py','contract.py','dataset.py','policy.py','bc.py','preparation_sampling.py',
        'student.py','runtime_checks.py','student_recording.py','student_worker.py'}
    assert selected==expected
    assert {'velocity_command.py','command_manager.py','articulation_data.py'} <= {path.name for path in files}
    assert not {'ppo_policy.py','command_adapter.py','reward_adapter.py','retention_worker.py','retention_protocol.py','schedule.py'}&selected
    for filename in selected:
        for node in ast.walk(ast.parse((directory/filename).read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                local=node.module.split('.')[-1] if node.level==1 or node.module.startswith('migration_v1.') else None
                if local is not None:
                    if filename=='student_worker.py' and local=='open_loop_teacher_probe':
                        # Explicit optional diagnostic branch; the ordinary Actor
                        # path never imports this module or freezes its source.
                        continue
                    assert local+'.py' in selected,(filename,local)


def test_changed_actual_execution_source_rejected_before_app_constructor(monkeypatch,tmp_path):
    # 只修改临时合成源；不触碰正在执行的真实迁移/物理文件。
    source=tmp_path/'student_recording.py'
    source.write_text('synthetic original recorder source')
    expected=sha256(source)
    source.write_text('synthetic changed recorder source')
    monkeypatch.setattr(student_worker,'REPO',tmp_path)
    monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(
        AppLauncher=lambda **kwargs:pytest.fail('App constructed before changed-source rejection')))
    prepared=dict(output=tmp_path/'synthetic_execution',sources={str(source):expected})
    with pytest.raises(EvidenceError,match='dependency changed before application start'):
        student_worker.run_simulation(prepared)


@pytest.mark.parametrize('seconds,offset', [(3.,0),(7.48,224),(8.,250)])
def test_fixed_preparation_uses_original_three_second_clock_offset(seconds,offset):
    assert runtime_checks.preparation_step_offset(seconds,.02) == offset
    # 原命令的elapsed_steps >= round(3/dt) 首次满足时刻，须等于显式准备时刻。
    first = next(step for step in range(1000) if step-offset >= round(3./.02))
    assert first*.02 == seconds


@pytest.mark.parametrize('seconds', [float('nan'),float('inf'),2.98,8.01])
def test_invalid_or_rounded_preparation_clock_rejected(seconds):
    with pytest.raises(EvidenceError):
        runtime_checks.preparation_step_offset(seconds)


def clock_manifest(starts):
    return dict(teacher_bundle=dict(episodes=[dict(case=dict(slope_deg=slope),
        command_timing=dict(actual_move_start_s=start)) for slope,start in zip((25.,-20.),starts)]))


def test_only_actual_matched_teacher_clocks_allow_selected_student_clock():
    result=runtime_checks.check_source_preparation(clock_manifest([8.,8.]),8.)
    assert result['teacher_clock_matched'] and not result['student_clock_runtime_verified']
    for starts, selected in [([3.,7.48],3.),([3.,7.48],8.),([8.,None],8.),([8.,8.],3.)]:
        with pytest.raises(EvidenceError,match='actual preparation clocks'):
            runtime_checks.check_source_preparation(clock_manifest(starts),selected)


@pytest.mark.parametrize('slope,seconds,episode', [(25.,3.,0),(-20.,7.48,1)])
def test_selected_task_uses_its_verified_clock_without_cross_task_same_start(slope,seconds,episode):
    result=runtime_checks.check_source_preparation(clock_manifest([3.,7.48]),seconds,case_slope=slope)
    assert result['teacher_clock_matched'] and result['matched_source_episode_ids']==[episode]
    assert result['alignment_scope']=='selected_task_direction' and result['selected_case_slope_deg']==slope
    assert result['teacher_actual_move_starts_s']==[seconds]
    assert result['fixed_before_reset'] and not result['runtime_ready_or_contact_gate']
    assert not result['student_clock_runtime_verified']


@pytest.mark.parametrize('slope,seconds', [(25.,7.48),(-20.,3.),(-20.,8.),(0.,3.),(float('nan'),3.)])
def test_selected_task_cannot_use_other_or_missing_direction_clock(slope,seconds):
    with pytest.raises(EvidenceError):
        runtime_checks.check_source_preparation(clock_manifest([3.,7.48]),seconds,case_slope=slope)


def test_selected_task_still_requires_complete_bundle_and_all_its_episode_clocks():
    manifest=clock_manifest([3.,7.48])
    extra=deepcopy(manifest['teacher_bundle']['episodes'][1]);extra['command_timing']['actual_move_start_s']=8.
    manifest['teacher_bundle']['episodes'].append(extra)
    with pytest.raises(EvidenceError,match='actual preparation clocks'):
        runtime_checks.check_source_preparation(manifest,7.48,case_slope=-20.)
    manifest['teacher_bundle']['episodes']=manifest['teacher_bundle']['episodes'][1:]
    with pytest.raises(EvidenceError,match='both complete teacher directions'):
        runtime_checks.check_source_preparation(manifest,7.48,case_slope=-20.)


def test_pre_action_phase_and_command_must_follow_selected_clock():
    for sample in [(0.,0,0.),(7.98,0,0.),(8.,1,.001),(9.,1,.051),(20.,2,.1),(25.,3,0.)]:
        runtime_checks.check_pre_action_preparation(*sample,8.)
    for sample in [(3.,1,.001),(7.98,0,.001),(8.,0,0.),(8.,2,.001),(8.,1,0.)]:
        with pytest.raises(EvidenceError):
            runtime_checks.check_pre_action_preparation(*sample,8.)


def test_mixed_teacher_clocks_rejected_in_prepare_before_dependency_or_output(monkeypatch,tmp_path,fake_runtime):
    manifest=clock_manifest([3.,7.48])
    for episode in manifest['teacher_bundle']['episodes']:
        episode['case'].update(payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.2)
        episode['physics_profile']=deepcopy(fake_runtime['profile'])
    monkeypatch.setattr(student_worker,'load_student',lambda *args,**kwargs: (object(),manifest))
    monkeypatch.setattr(student_worker,'dependency_sources',lambda: pytest.fail('Reached source snapshot before clock gate'))
    output=tmp_path/'never_created_mixed_clock'
    with pytest.raises(EvidenceError,match='actual preparation clocks'):
        student_worker.prepare(tmp_path/'synthetic.pt',25.,output,8.)
    assert not output.exists()


@pytest.mark.parametrize('slope,seconds', [(25.,3.),(-20.,7.48)])
def test_prepare_passes_selected_task_clock_without_app_or_output(monkeypatch,tmp_path,fake_runtime,slope,seconds):
    manifest=clock_manifest([3.,7.48])
    for episode in manifest['teacher_bundle']['episodes']:
        episode['case'].update(payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.2)
        episode['physics_profile']=deepcopy(fake_runtime['profile'])
    actor=object()
    monkeypatch.setattr(student_worker,'load_student',lambda *args,**kwargs:(actor,manifest))
    monkeypatch.setattr(student_worker,'dependency_sources',lambda:[])
    output=tmp_path/'not_created_until_run'
    prepared=student_worker.prepare(tmp_path/'synthetic.pt',slope,output,seconds)
    assert prepared['policy'] is actor and prepared['case']['slope_deg']==slope
    assert prepared['preparation']['selected_case_slope_deg']==slope
    assert prepared['preparation']['preparation_seconds']==seconds
    assert not output.exists() and 'isaaclab.app' not in sys.modules
