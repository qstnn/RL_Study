"""Original source methods on CPU snapshots; no Isaac instance or PPO updates."""
import ast
import __future__
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1.contract import CONTRACT, POSITION_SOURCE, REPO, SOURCE, _load
from migration_v1 import terminal_capture as terminal
from migration_v1.dataset import EvidenceError


NATIVE = REPO.parent/'Isaaclab/source/isaaclab/isaaclab'


def execute_source_nodes(path, names, namespace, class_name=None, base=None):
    """Compile unchanged method bodies from current source; no algorithm replica."""
    tree = ast.parse(Path(path).read_text())
    parent = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), tree)
    nodes = [deepcopy(n) for n in parent.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = [d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == 'property']
    if class_name:
        nodes = [ast.ClassDef(name=class_name, bases=[ast.Name(id=base, ctx=ast.Load())] if base else [],
            keywords=[], body=nodes, decorator_list=[])]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(Path(path).resolve()), 'exec', flags=__future__.annotations.compiler_flag), namespace)


@pytest.fixture(scope='module')
def native_methods():
    namespace = dict(torch=torch, __name__='cpu_terminal_native_fixture')
    core = _load(SOURCE/'wheel_unified_v6/core.py', '_terminal_preview_math')
    execute_source_nodes(NATIVE/'utils/math.py', ['quat_apply', 'quat_apply_inverse'], namespace)
    namespace['core'] = core
    execute_source_nodes(SOURCE/'wheel_unified_v6/mdp.py', ['heading_error', 'privileged_context', 'state'], namespace)
    namespace['mdp'] = SimpleNamespace(state=namespace['state'])
    execute_source_nodes(NATIVE/'managers/command_manager.py', ['compute'], namespace, 'CommandTerm')
    base = namespace['CommandTerm']
    base.num_envs = property(lambda self: self._env.num_envs)
    base.device = property(lambda self: self._env.device)
    execute_source_nodes(NATIVE/'envs/mdp/commands/velocity_command.py', ['command'], namespace,
        'UniformVelocityCommand', 'CommandTerm')
    execute_source_nodes(SOURCE/'wheel_unified_v6/commands.py',
        ['_update_command', '_update_metrics', 'phase_elapsed'], namespace,
        'UnifiedVelocityCommand', 'UniformVelocityCommand')
    execute_source_nodes(Path(terminal.__file__).with_name('command_adapter.py'), ['_update_command'], namespace,
        'MigrationVelocityCommand', 'UnifiedVelocityCommand')
    namespace['noise'] = SimpleNamespace(NoiseCfg=type('NoiseCfg', (), {}), NoiseModelCfg=type('NoiseModelCfg', (), {}))
    execute_source_nodes(NATIVE/'managers/observation_manager.py', ['compute_group'], namespace, 'ObservationManager')
    manager = namespace['ObservationManager']
    manager.active_terms = property(lambda self: self._group_obs_term_names)
    manager.group_obs_dim = property(lambda self: {'critic': (252,)})
    namespace['SceneEntityCfg'] = lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)
    execute_source_nodes(NATIVE/'envs/mdp/observations.py',
        ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'generated_commands',
         'joint_vel_rel', 'last_action', 'height_scan'], namespace)
    namespace['joint_positions_v1'] = _load(POSITION_SOURCE, '_terminal_preview_position').joint_positions_v1
    return namespace


@pytest.fixture
def env(native_methods):
    n = 2
    cfg = SimpleNamespace(fixed_speed_mps=torch.tensor([.3, .3]), loaded_acceleration=torch.tensor([.05, .05]),
        unloaded_acceleration=1., eval_distance_m=torch.tensor([1.2, 1.2]), command_profile='slew')
    joint_names = list(CONTRACT['observation_joint_order'])
    robot = SimpleNamespace(joint_names=joint_names, data=SimpleNamespace(
        root_pos_w=torch.tensor([[.2, .01, .3], [.4, -.02, .3]]),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]*n), root_lin_vel_b=torch.ones(n, 3)*.1,
        root_ang_vel_b=torch.zeros(n, 3), root_lin_vel_w=torch.ones(n, 3)*.1,
        root_ang_vel_w=torch.zeros(n, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]*n),
        joint_pos=torch.arange(n*16).reshape(n, 16).float()*.01,
        default_joint_pos=torch.zeros(n, 16), joint_vel=torch.ones(n, 16)*.2,
        default_joint_vel=torch.zeros(n, 16)))
    scanner = SimpleNamespace(data=SimpleNamespace(pos_w=torch.tensor([[0., 0., .6]]*n),
        ray_hits_w=torch.zeros(n, 176, 3)))
    result = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, cfg=cfg,
        episode_length_buf=torch.tensor([150, 374]),
        r1_axis_x=torch.tensor([[1., 0., 0.]]*n), r1_axis_z=torch.tensor([[0., 0., 1.]]*n),
        r1_normal=torch.tensor([[0., 0., 1.]]*n), r1_tangent=torch.tensor([[1., 0., 0.]]*n),
        r1_payload_kg=torch.zeros(n), r1_towed_kg=torch.full((n,), 40.),
        r1_eval_phase=torch.zeros(n, dtype=torch.long), r1_eval_origin=torch.zeros(n, 3),
        r1_eval_progress=torch.zeros(n), r1_eval_stop_time=torch.full((n,), -1.),
        action_manager=SimpleNamespace(action=torch.arange(n*16).reshape(n, 16).float()*.001),
        event_manager=SimpleNamespace(available_modes=['startup', 'reset']),
        reset_terminated=torch.zeros(n, dtype=torch.bool), reset_time_outs=torch.ones(n, dtype=torch.bool))
    result.scene = terminal._SceneSnapshot(robot, scanner)
    term = object.__new__(native_methods['MigrationVelocityCommand'])
    term._env, term.robot = result, robot
    term.cfg = SimpleNamespace(heading_command=False, debug_vis=False)
    term._debug_vis_handle = None
    term.time_left = torch.full((n,), 1.e9)
    term.command_counter = torch.ones(n, dtype=torch.long)
    term.vel_command_b = torch.zeros(n, 3)
    term.requested_vx = torch.full((n,), .3)
    term.declared_preparation_steps = torch.tensor([150, 374])
    term.declared_cycle_offsets = torch.tensor([0, 224])
    term.metrics = dict(error_vel_xy=torch.zeros(n), error_vel_yaw=torch.zeros(n))
    for name in ('metric_steps', 'target_vx', 'hold_elapsed'):
        setattr(term, name, torch.zeros(n))
    for name in ('hold_steps', 'cycle_start_step', 'phase_start_step', 'cycles_completed',
                 'hold_windows_completed', 'static_cycles_completed'):
        setattr(term, name, torch.zeros(n, dtype=torch.long))
    term.hold_origin = torch.zeros(n, 3)
    for name in ('holding_active', 'cycle_completed', 'completed_cycle_this_step',
                 'completed_hold_this_step', 'distance_reached_this_step'):
        setattr(term, name, torch.zeros(n, dtype=torch.bool))
    result.command_manager = terminal._CommandSnapshot(term)
    manager = object.__new__(native_methods['ObservationManager'])
    manager._env = result
    manager._resolve_terms_handle = None
    manager._group_obs_term_names = {'critic': [item[0] for item in terminal.CRITIC_TERMS]}
    manager._group_obs_term_dim = {'critic': [(item[1],) for item in terminal.CRITIC_TERMS]}
    manager._group_obs_concatenate = {'critic': True}
    manager._group_obs_concatenate_dim = {'critic': -1}
    configs = []
    for name, _, function, scale in terminal.CRITIC_TERMS:
        params = {}
        if name in ('joint_pos', 'joint_vel'):
            params['asset_cfg'] = SimpleNamespace(name='robot', joint_names=joint_names, joint_ids=list(range(16)))
        if name == 'joint_pos':
            params['wheel_joint_names'] = [x for x in joint_names if 'wheel' in x]
        if name == 'velocity_commands':
            params['command_name'] = 'base_velocity'
        if name == 'height_scan':
            params['sensor_cfg'] = SimpleNamespace(name='height_scanner')
        configs.append(SimpleNamespace(history_length=0, noise=None, modifiers=None, func=native_methods[function],
            params=params, scale=scale,
            clip=None if name == 'context' else (-1., 1.) if name == 'height_scan' else (-100., 100.)))
    manager._group_obs_term_cfgs = {'critic': configs}
    manager._group_obs_term_history_buffer = {'policy': {'fake_history': SimpleNamespace(
        _buffer=torch.full((5, n, 3), .123), _pointer=2, _num_pushes=torch.tensor([6, 9]))}, 'critic': {}}
    result.observation_manager = manager
    return result


def set_boundary(env, name):
    term = env.command_manager.term
    if name == 'prepare_before':
        env.episode_length_buf -= 1
    elif name == 'prepare_end':
        pass
    else:
        env.episode_length_buf[:] = 500
        env.r1_eval_phase[:] = 1
        term.phase_start_step[:] = 150
        term.vel_command_b[:, 0] = .3
        env.r1_eval_origin.copy_(env.scene['robot'].data.root_pos_w)
        if name == 'distance_brake':
            env.r1_eval_origin[:, 0] -= 1.2
        elif name in ('zero_to_hold', 'holding', 'settled_anchor', 'timeout_hold'):
            env.r1_eval_phase[:] = 2 if name == 'zero_to_hold' else 3
            term.vel_command_b[:, 0] = .001 if name == 'zero_to_hold' else 0.
            term.holding_active[:] = name != 'zero_to_hold'
            term.hold_steps[:] = {'zero_to_hold': 0, 'holding': 19, 'settled_anchor': 99, 'timeout_hold': 349}[name]
            term.hold_elapsed[:] = term.hold_steps*.02
            if name == 'timeout_hold':
                env.episode_length_buf[:] = 3250
                term.phase_start_step[:] = 2900
        elif name != 'moving':
            raise AssertionError(name)


@pytest.mark.parametrize('boundary', ['prepare_before', 'prepare_end', 'moving', 'distance_brake',
    'zero_to_hold', 'holding', 'settled_anchor', 'timeout_hold'])
def test_preview_equals_unreset_native_next_critic_without_mutating_original(env, boundary):
    set_boundary(env, boundary)
    original = terminal._clone_tree(terminal._snapshot_inputs(env, env.command_manager.term))
    # Independent copy of the whole synthetic native object graph, not preview's proxy builder.
    unreset = deepcopy(env)
    assert unreset.command_manager.term._env is unreset
    held = unreset.observation_manager.compute_group('critic', update_history=False).clone()
    unreset.command_manager.term.compute(.02)
    expected = unreset.observation_manager.compute_group('critic', update_history=False).clone()
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    result = terminal.preview_terminal_critic(env)
    assert torch.equal(result.held_command_critic, held)
    assert torch.equal(result.post_command_critic, expected)
    assert terminal._equal_tree(original, terminal._snapshot_inputs(env, env.command_manager.term))
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == python_rng
    assert terminal._equal_tree(numpy_rng, np.random.get_state())
    for name, value in result.command_state.items():
        assert terminal._equal_tree(value, getattr(unreset.command_manager.term, name)), name
    assert result.review['rng_unchanged'] and not result.review['real_gpu_timeout_validation']
    if boundary == 'prepare_end':
        assert result.task_state['r1_eval_phase'].tolist() == [1, 1]
        assert torch.equal(expected[:, 9], torch.full((2,), .001))
    if boundary == 'distance_brake':
        assert result.task_state['r1_eval_phase'].tolist() == [2, 2]
    if boundary == 'zero_to_hold':
        assert result.task_state['r1_eval_phase'].tolist() == [3, 3]
        assert expected[:, 244].eq(0).all()
    if boundary == 'settled_anchor':
        assert expected[:, 242:244].eq(0).all()
    if boundary == 'timeout_hold':
        assert result.command_state['completed_hold_this_step'].all()


def test_capture_returns_projected_timeout_values_and_keeps_true_terminal_no_bootstrap(env):
    set_boundary(env, 'prepare_end')
    capture = terminal.PreResetCriticCapture(preview_post_command=True)
    capture.begin(env)
    capture.before_reset(env, torch.tensor([0, 1]))
    expected = capture.projected.clone()
    # Simulate only the later reset mutation; no actual simulator is involved.
    env.scene['robot'].data.joint_pos.zero_()
    env.command_manager.term.vel_command_b.zero_()
    result = capture.finish(env.reset_terminated, env.reset_time_outs)
    assert torch.equal(result.for_timeout_bootstrap(), expected)
    assert result.held_command_critic[:, 9].eq(0).all()
    assert result.post_command_critic[:, 9].gt(0).all()


@pytest.mark.parametrize('unsafe', ['resample', 'not_fixed', 'interval', 'heading', 'noise', 'stateful', 'extra_command'])
def test_unsupported_preview_is_rejected_before_any_original_change(env, unsafe):
    if unsafe == 'resample': env.command_manager.term.time_left[:] = .02
    if unsafe == 'not_fixed': env.cfg.fixed_speed_mps = None
    if unsafe == 'interval': env.event_manager.available_modes.append('interval')
    if unsafe == 'heading': env.command_manager.term.cfg.heading_command = True
    if unsafe == 'noise': env.observation_manager._group_obs_term_cfgs['critic'][0].noise = object()
    if unsafe == 'stateful': env.observation_manager._group_obs_term_cfgs['critic'][0].func = SimpleNamespace()
    if unsafe == 'extra_command': env.command_manager.active_terms = ['base_velocity', 'other']
    before = terminal._clone_tree(terminal._snapshot_inputs(env, env.command_manager.term))
    with pytest.raises(EvidenceError):
        terminal.preview_terminal_critic(env)
    assert terminal._equal_tree(before, terminal._snapshot_inputs(env, env.command_manager.term))


def test_all_mutable_command_values_are_cloned_and_handles_are_not_shared(env):
    term = env.command_manager.term
    snapshot = terminal._clone_tree(terminal._snapshot_inputs(env, term))
    proxy, copied, manager = terminal._copy_preview_objects(env, term, snapshot)
    assert copied._debug_vis_handle is manager._resolve_terms_handle is None
    for name, value in vars(term).items():
        if isinstance(value, torch.Tensor):
            assert value.data_ptr() != getattr(copied, name).data_ptr()
    copied.metrics['error_vel_xy'] += 100
    proxy.r1_eval_phase[:] = 3
    manager._group_obs_term_cfgs['critic'][0].scale = 99
    assert terminal._equal_tree(snapshot, terminal._snapshot_inputs(env, term))
    with pytest.raises(EvidenceError, match='resampling'):
        copied._resample(torch.tensor([0]))


def test_rng_consumption_is_restored_but_rejected_not_claimed_pure(env, monkeypatch):
    original = terminal._copy_preview_objects
    def with_bad_reader(*args):
        proxy, term, manager = original(*args)
        ordinary = manager.compute_group
        def stochastic(*args, **kwargs):
            torch.rand(1); random.random(); np.random.rand()
            return ordinary(*args, **kwargs)
        manager.compute_group = stochastic
        return proxy, term, manager
    monkeypatch.setattr(terminal, '_copy_preview_objects', with_bad_reader)
    rng, py, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    with pytest.raises(EvidenceError, match='consumed RNG'):
        terminal.preview_terminal_critic(env)
    assert torch.equal(rng, torch.get_rng_state()) and py == random.getstate()
    assert terminal._equal_tree(np_rng, np.random.get_state())


def test_preview_reads_current_terminal_joint_feedback_not_a_pre_action_cache(env):
    first = terminal.preview_terminal_critic(env)
    env.scene['robot'].data.joint_pos[:, 0] += .123
    env.scene['robot'].data.joint_vel[:, 1] += 2.
    second = terminal.preview_terminal_critic(env)
    torch.testing.assert_close(second.post_command_critic[:, 12]-first.post_command_critic[:, 12],
        torch.full((2,), .123))
    torch.testing.assert_close(second.post_command_critic[:, 29]-first.post_command_critic[:, 29],
        torch.full((2,), .1))
    assert torch.equal(second.held_command_critic[:, 12:44], second.post_command_critic[:, 12:44])


def test_exception_during_projection_restores_rng_without_returning_a_preview(env, monkeypatch):
    original = terminal._copy_preview_objects
    def with_failure(*args):
        proxy, term, manager = original(*args)
        def failed(*args, **kwargs):
            torch.rand(1); random.random(); np.random.rand()
            raise RuntimeError('injected reader failure')
        manager.compute_group = failed
        return proxy, term, manager
    monkeypatch.setattr(terminal, '_copy_preview_objects', with_failure)
    rng, py, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError, match='injected reader'):
        terminal.preview_terminal_critic(env)
    assert torch.equal(rng, torch.get_rng_state()) and py == random.getstate()
    assert terminal._equal_tree(np_rng, np.random.get_state())


def test_physical_column_failure_keeps_exact_values_and_reader_side_effect_stage(env, monkeypatch):
    original = terminal._copy_preview_objects
    def with_mutating_reader(*args):
        proxy, term, manager = original(*args)
        compute = manager.compute_group
        calls = 0
        def mutate(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = compute(*args, **kwargs)
            if calls == 1:
                proxy.scene['robot'].data.root_lin_vel_b[0, 0] += .125
            return result
        manager.compute_group = mutate
        return proxy, term, manager
    monkeypatch.setattr(terminal, '_copy_preview_objects', with_mutating_reader)
    before = terminal._clone_tree(terminal._snapshot_inputs(env, env.command_manager.term))
    with pytest.raises(EvidenceError, match='physical Critic columns') as caught:
        terminal.preview_terminal_critic(env)
    diagnostic = caught.value.terminal_preview_diagnostic
    assert diagnostic['changed_columns'] == [0] and diagnostic['changed_rows'] == [0]
    assert diagnostic['max_abs_error'] == pytest.approx(.25)
    assert diagnostic['proxy_physical_inputs_unchanged'] == {
        'before_held__to__after_held': False,
        'after_held__to__after_command': True,
        'after_command__to__after_projected': True}
    assert terminal._equal_tree(before, terminal._snapshot_inputs(env, env.command_manager.term))
    assert torch.equal(caught.value.terminal_preview_arrays['held_critic'][0, :1], torch.tensor([.2]))


def test_actual_gpu_v2_failure_is_only_float32_normal_rotation_roundoff():
    path = Path(terminal.__file__).parents[1]/'results/2026-10-23_admission_continuation/terminal_preview_short_timeout_v2_gpu/terminal_preview_failed_projection.npz'
    assert terminal.sha256(path)=='866b07667f7522a623538a2dadc3b558be1cff9dde0f9d3a4c2254bddf2fced5'
    with np.load(path,allow_pickle=False) as data:
        held=torch.from_numpy(data['projection/held_critic'].copy())
        projected=torch.from_numpy(data['projection/projected_critic'].copy())
        q=data['projection/original_input_snapshot/robot/root_quat_w'].astype(np.float64)
        normal=data['projection/original_input_snapshot/env/r1_normal'].astype(np.float64)
        snapshots={stage: {key.split('/'+stage+'/')[1]: torch.from_numpy(data[key].copy())
            for key in data.files if key.startswith('projection/snapshots/'+stage+'/')}
            for stage in ('before_held','after_held','after_command','after_projected')}
    vector=q[:,1:]; scalar=q[:,:1]
    expected=(1.-2.*(vector*vector).sum(1,keepdims=True))*normal + 2.*vector*(vector*normal).sum(1,keepdims=True)-2.*scalar*np.cross(vector,normal)
    assert np.max(abs(held[:,236:239].numpy()-expected))<9.e-8
    assert np.max(abs(projected[:,236:239].numpy()-expected))<9.e-8
    review=terminal._validate_physical_columns(held,projected,snapshots,{})
    assert review['normal_max_abs_difference']==2.**-23
    assert not review['all_physical_columns_exact']
    assert review['normal_recompute_atol']==2.e-7
    changed=projected.clone();changed[0,240]+=1.e-8
    with pytest.raises(EvidenceError): terminal._validate_physical_columns(held,changed,snapshots,{})
    changed=held.clone();changed[0,236]+=3.e-7
    with pytest.raises(EvidenceError): terminal._validate_physical_columns(held,changed,snapshots,{})
    snapshots['after_command']['robot/root_quat_w'][0,0]+=1.e-5
    with pytest.raises(EvidenceError): terminal._validate_physical_columns(held,projected,snapshots,{})
