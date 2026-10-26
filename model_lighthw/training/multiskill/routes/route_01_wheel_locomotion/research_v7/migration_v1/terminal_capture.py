"""Local physical terminal capture and isolated fixed-command next-Critic preview.

The native step advances commands after resetting ended slots. The captured
252-vector consequently contains the physical terminal state and the command
held during the transition. It is not the counterfactual next observation of
an unreset environment. The optional preview advances cloned command/task
state using native functions, then computes the full next-Critic on snapshots.
The held-command vector alone is always refused for timeout bootstrap.

Only a local subclass uses the mixin. No global patch, copied native step,
policy observation computation, original command advance, or physical write occurs.
"""
from dataclasses import dataclass
from copy import deepcopy
import inspect
from pathlib import Path
import random
from types import SimpleNamespace, MethodType

import numpy as np
import torch

from .contract import CONTRACT, POSITION_SOURCE, REPO, SOURCE, sha256
from .dataset import EvidenceError
from .ppo_policy import critic_contract_252


# GPU v2 captured identical inputs but native quaternion-derived normal differed
# by 1.1920929e-7; both were within 8.35e-8 of independent float64 rotation.
# Restrict tolerance to surface_normal_b, never raw physical inputs or other columns.
NORMAL_RECOMPUTE_ATOL = 2.e-7
NORMAL_COLUMNS = slice(236, 239)


CRITIC_TERMS = (
    ('base_lin_vel', 3, 'base_lin_vel', 2.),
    ('base_ang_vel', 3, 'base_ang_vel', .5),
    ('projected_gravity', 3, 'projected_gravity', 1.),
    ('velocity_commands', 3, 'generated_commands', 1.),
    ('joint_pos', 16, 'joint_positions_v1', 1.),
    ('joint_vel', 16, 'joint_vel_rel', .05),
    ('actions', 16, 'last_action', 1.),
    ('height_scan', 176, 'height_scan', 1.),
    ('context', 16, 'privileged_context', 1.),
)


def terminal_semantics():
    fields = {item['name']: item['columns'] for item in critic_contract_252()['fields']}
    names = ('velocity_commands', 'hold_route_displacement', 'hold_lateral_displacement',
             'hold_elapsed', 'requested_vx', 'phase', 'phase_elapsed', 'remaining_distance')
    return dict(version='r1_pre_reset_held_command_critic_v1', critic_dimension=252,
        physical_state='after physics, termination and reward; before native reset',
        command_state='held transition command; before native command_manager.compute',
        command_dependent_columns={name: fields[name] for name in names},
        actor_history_advanced=False, command_advanced=False, physical_state_written=False,
        post_command_next_critic_verified=False, timeout_bootstrap_ready=False,
        fixed_command_preview_implemented=True, real_gpu_timeout_validation=False,
        missing_validation='Actual GPU pre-reset capture and projected Critic readback')


def validate_capture_manager(manager):
    """Fail before stepping if an additional Critic read could mutate history.

Function bodies and the native manager must additionally be source-frozen by
the caller. These checks are a narrow runtime contract, not a proof that an
arbitrary same-named Python function is pure.
"""
    names = [item[0] for item in CRITIC_TERMS]
    if (list(manager.active_terms.get('critic', ())) != names
            or tuple(manager.group_obs_dim.get('critic', ())) != (252,)
            or not manager._group_obs_concatenate.get('critic', False)):
        raise EvidenceError('Terminal capture requires the named concatenated 252-column Critic')
    configs = manager._group_obs_term_cfgs['critic']
    dims = manager._group_obs_term_dim['critic']
    if len(configs) != len(names) or list(map(tuple, dims)) != [(x[1],) for x in CRITIC_TERMS]:
        raise EvidenceError('Terminal Critic term widths differ from the declared semantics')
    for config, (name, _, function, scale) in zip(configs, CRITIC_TERMS):
        if (config.history_length != 0 or config.noise is not None
                or config.modifiers not in (None, [])):
            raise EvidenceError('Terminal Critic must have no history/noise/modifiers: '+name)
        if not inspect.isfunction(config.func) or config.func.__name__ != function:
            raise EvidenceError('Terminal Critic function differs from the reviewed plain reader: '+name)
        actual_scale = 1. if config.scale is None else config.scale
        scale_tensor = torch.as_tensor(actual_scale).float()
        if not bool(torch.allclose(scale_tensor, scale_tensor.new_tensor(scale), atol=1.e-7, rtol=0.)):
            raise EvidenceError('Terminal Critic scale differs: '+name)
        expected_clip = None if name == 'context' else (-1., 1.) if name == 'height_scan' else (-100., 100.)
        if config.clip != expected_clip:
            raise EvidenceError('Terminal Critic clip differs: '+name)
    return terminal_semantics()


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_tree(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, slice, torch.device, torch.dtype)):
        return value
    raise EvidenceError('Unsupported live object in terminal command state: '+type(value).__name__)


def _equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def _source_file(function, expected):
    actual = inspect.getsourcefile(function)
    if actual is None or Path(actual).resolve() != Path(expected).resolve():
        raise EvidenceError('Unreviewed terminal-preview function source: '+repr(actual))
    return str(Path(actual).resolve())


def _preview_contract(env):
    validate_capture_manager(env.observation_manager)
    manager = env.command_manager
    if list(manager.active_terms) != ['base_velocity']:
        raise EvidenceError('Terminal command preview supports only the single fixed base_velocity term')
    if 'interval' in env.event_manager.available_modes:
        raise EvidenceError('Terminal preview does not support interval events after command updates')
    term = manager.get_term('base_velocity')
    if (env.cfg.fixed_speed_mps is None or env.cfg.command_profile != 'slew'
            or getattr(term.cfg, 'heading_command', False)
            or getattr(term.cfg, 'debug_vis', False) or getattr(term, '_debug_vis_handle', None) is not None):
        raise EvidenceError('Terminal preview requires fixed slew commands without heading/debug callbacks')
    if (not bool(torch.isfinite(term.time_left).all())
            or not bool((term.time_left - env.step_dt > 0).all())):
        raise EvidenceError('Terminal preview would resample; no stochastic command branch is supported')
    if (not hasattr(term, 'declared_cycle_offsets') or not hasattr(term, 'declared_preparation_steps')
            or type(term).__name__ != 'MigrationVelocityCommand'):
        raise EvidenceError('Terminal preview requires the reviewed migration fixed-clock command class')
    native = REPO.parent/'Isaaclab/source/isaaclab/isaaclab'
    files = {_source_file(type(env.observation_manager).compute_group, native/'managers/observation_manager.py'),
        _source_file(type(term).compute, native/'managers/command_manager.py'),
        _source_file(type(term)._update_command, Path(__file__).with_name('command_adapter.py')),
        _source_file(type(term)._update_metrics, SOURCE/'wheel_unified_v6/commands.py')}
    parents = [cls for cls in type(term).__mro__ if cls.__name__ == 'UnifiedVelocityCommand']
    if len(parents) != 1:
        raise EvidenceError('Unexpected native command inheritance')
    files.add(_source_file(parents[0]._update_command, SOURCE/'wheel_unified_v6/commands.py'))
    files.update(str(path.resolve()) for path in (SOURCE/'wheel_unified_v6/core.py',
                 native/'utils/math.py', Path(__file__)))
    for config, (name, _, _, _) in zip(env.observation_manager._group_obs_term_cfgs['critic'], CRITIC_TERMS):
        expected = (POSITION_SOURCE if name == 'joint_pos' else SOURCE/'wheel_unified_v6/mdp.py'
                    if name == 'context' else native/'envs/mdp/observations.py')
        files.add(_source_file(config.func, expected))
        if name == 'actions' and config.params.get('action_name') is not None:
            raise EvidenceError('Terminal preview supports the original complete action observation only')
        if name == 'velocity_commands' and config.params.get('command_name') != 'base_velocity':
            raise EvidenceError('Unexpected Critic command source')
        if name in ('joint_pos', 'joint_vel'):
            if tuple(config.params['asset_cfg'].joint_names) != tuple(CONTRACT['observation_joint_order']):
                raise EvidenceError('Terminal Critic named joint order changed')
        if name == 'height_scan' and config.params['sensor_cfg'].name != 'height_scanner':
            raise EvidenceError('Unexpected terminal height sensor')
    return term, {name: sha256(name) for name in sorted(files)}


class _SceneSnapshot(dict):
    def __init__(self, robot, scanner):
        super().__init__(robot=robot, height_scanner=scanner)
        self.sensors = {'height_scanner': scanner}


class _CommandSnapshot:
    active_terms = ['base_velocity']
    def __init__(self, term):
        self.term = term
    def get_term(self, name):
        if name != 'base_velocity':
            raise EvidenceError('Unknown command requested from terminal snapshot')
        return self.term
    def get_command(self, name):
        return self.get_term(name).command


ROBOT_READERS = ('root_pos_w', 'root_quat_w', 'root_lin_vel_b', 'root_ang_vel_b',
    'root_lin_vel_w', 'root_ang_vel_w', 'projected_gravity_b', 'joint_pos',
    'default_joint_pos', 'joint_vel', 'default_joint_vel')
CONFIG_READERS = ('fixed_speed_mps', 'command_profile', 'loaded_acceleration',
                  'unloaded_acceleration', 'eval_distance_m')


def _snapshot_inputs(env, term):
    robot = env.scene['robot']
    scanner = env.scene.sensors['height_scanner']
    env_values = {name: value for name, value in vars(env).items()
                  if name.startswith('r1_') and isinstance(value, torch.Tensor)}
    env_values['episode_length_buf'] = env.episode_length_buf
    command_values = {name: value for name, value in vars(term).items()
                      if name not in ('_env', 'robot', 'cfg', '_debug_vis_handle')}
    histories = {group: {name: vars(buffer) for name, buffer in terms.items()}
                 for group, terms in env.observation_manager._group_obs_term_history_buffer.items()}
    return dict(env=env_values, command=command_values,
        robot={name: getattr(robot.data, name) for name in ROBOT_READERS},
        scanner={name: getattr(scanner.data, name) for name in ('pos_w', 'ray_hits_w')},
        action=env.action_manager.action,
        cfg={name: getattr(env.cfg, name) for name in CONFIG_READERS}, history=histories)


def _copy_preview_objects(env, term, snapshot):
    # No live scene/asset/command/observation manager is reachable from the proxy.
    proxy = SimpleNamespace(num_envs=env.num_envs, device=env.device, step_dt=env.step_dt,
        cfg=SimpleNamespace(**_clone_tree(snapshot['cfg'])), **_clone_tree(snapshot['env']))
    robot = SimpleNamespace(data=SimpleNamespace(**_clone_tree(snapshot['robot'])),
        joint_names=list(env.scene['robot'].joint_names))
    scanner = SimpleNamespace(data=SimpleNamespace(**_clone_tree(snapshot['scanner'])))
    proxy.scene = _SceneSnapshot(robot, scanner)
    proxy.action_manager = SimpleNamespace(action=_clone_tree(snapshot['action']))
    copied = object.__new__(type(term))
    # Native destructors must never unsubscribe the production subscriptions.
    copied._debug_vis_handle = None
    copied.__dict__.update(_clone_tree(snapshot['command']))
    copied.cfg = deepcopy(term.cfg)
    copied._env, copied.robot = proxy, robot
    def forbidden_resample(self, *args, **kwargs):
        raise EvidenceError('Terminal preview attempted command resampling')
    copied._resample = MethodType(forbidden_resample, copied)
    proxy.command_manager = _CommandSnapshot(copied)
    manager = object.__new__(type(env.observation_manager))
    manager._resolve_terms_handle = None
    manager._env = proxy
    for name in ('_group_obs_term_names', '_group_obs_term_cfgs', '_group_obs_term_dim',
                 '_group_obs_concatenate', '_group_obs_concatenate_dim'):
        setattr(manager, name, {'critic': deepcopy(getattr(env.observation_manager, name)['critic'])})
    manager._group_obs_term_history_buffer = {'critic': {}}
    return proxy, copied, manager


def _physical_preview_inputs(proxy):
    """Read-only fingerprints of the proxy inputs not advanced by commands."""
    return _clone_tree(dict(robot=vars(proxy.scene['robot'].data),
        scanner=vars(proxy.scene['height_scanner'].data), action=proxy.action_manager.action,
        axes={name: getattr(proxy, name) for name in
            ('r1_axis_x', 'r1_axis_z', 'r1_normal', 'r1_tangent', 'r1_payload_kg', 'r1_towed_kg')}))


def _physical_column_error(held, projected, physical, snapshots, original):
    changed = (held != projected) & physical.unsqueeze(0)
    locations = changed.nonzero()
    examples = [dict(row=int(row), column=int(column), held=float(held[row, column]),
        projected=float(projected[row, column]),
        difference=float(projected[row, column]-held[row, column]))
        for row, column in locations[:32].tolist()]
    names = list(snapshots)
    unchanged = {names[i-1]+'__to__'+names[i]: _equal_tree(snapshots[names[i-1]], snapshots[names[i]])
        for i in range(1, len(names))}
    report = dict(changed_columns=changed.any(0).nonzero().flatten().tolist(),
        changed_rows=changed.any(1).nonzero().flatten().tolist(), changed_entries=int(changed.sum()),
        max_abs_error=float((projected-held).abs()[changed].max()), examples=examples,
        proxy_physical_inputs_unchanged=unchanged, strict_equality_required=True)
    error = EvidenceError('Command-only preview changed physical Critic columns: '
        +str(report['changed_columns'])+'; max_abs_error='+str(report['max_abs_error']))
    error.terminal_preview_diagnostic = report
    error.terminal_preview_arrays = dict(held_critic=held.detach().clone(),
        projected_critic=projected.detach().clone(), physical_column_mask=physical.detach().clone(),
        snapshots=snapshots, original_input_snapshot=original)
    return error


def _validate_physical_columns(held, projected, snapshots, original):
    physical = torch.ones(252, dtype=torch.bool, device=projected.device)
    physical[9:12] = False
    physical[242:252] = False
    exact = physical.clone()
    exact[NORMAL_COLUMNS] = False
    names = list(snapshots)
    raw_unchanged = all(_equal_tree(snapshots[names[i-1]], snapshots[names[i]])
        for i in range(1, len(names)))
    exact_equal = torch.equal(held[:, exact], projected[:, exact])
    normal_difference = float((held[:, NORMAL_COLUMNS]-projected[:, NORMAL_COLUMNS]).abs().max())
    if not raw_unchanged or not exact_equal or normal_difference > NORMAL_RECOMPUTE_ATOL:
        if not torch.equal(held[:, physical], projected[:, physical]):
            error = _physical_column_error(held, projected, physical, snapshots, original)
            error.terminal_preview_diagnostic.update(normal_only_atol=NORMAL_RECOMPUTE_ATOL,
                normal_columns=[236,237,238], other_physical_columns_require_exact=True)
            raise error
        raise EvidenceError('Terminal preview reader/command changed proxy physical inputs')
    return dict(raw_proxy_physical_inputs_exact=True, other_physical_columns_exact=True,
        all_physical_columns_exact=torch.equal(held[:, physical], projected[:, physical]),
        normal_columns=[236,237,238], normal_max_abs_difference=normal_difference,
        normal_recompute_atol=NORMAL_RECOMPUTE_ATOL, normal_recompute_rtol=0.)


@dataclass(frozen=True)
class TerminalPreview:
    held_command_critic: torch.Tensor
    post_command_critic: torch.Tensor
    command_state: dict
    task_state: dict
    review: dict
    input_snapshot: dict


@torch.no_grad()
def preview_terminal_critic(env):
    """Preview normal next-Critic on the ACTUAL pre-reset physical state.

Supported fixed commands are advanced by the original CommandTerm.compute on
an isolated object. All inputs are clones; resampling is both excluded by the
timer gate and replaced with a throwing method on the local copy. The original
single-frame ObservationManager computes both full 252-vectors on a local
manager. No native history or command is advanced. This is not GPU validation.
"""
    term, sources = _preview_contract(env)
    before = _clone_tree(_snapshot_inputs(env, term))
    proxy, copied, manager = _copy_preview_objects(env, term, before)
    device = torch.device(env.device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = {index: torch.cuda.get_rng_state(index).clone() for index in cuda_devices}
    changed_rng = False
    diagnostic_inputs = {'before_held': _physical_preview_inputs(proxy)}
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            try:
                held = manager.compute_group('critic', update_history=False).detach().clone()
                diagnostic_inputs['after_held'] = _physical_preview_inputs(proxy)
                copied.compute(dt=env.step_dt)  # Native metrics/time-left/update, on copies only.
                diagnostic_inputs['after_command'] = _physical_preview_inputs(proxy)
                projected = manager.compute_group('critic', update_history=False).detach().clone()
                diagnostic_inputs['after_projected'] = _physical_preview_inputs(proxy)
            finally:
                changed_rng = (not torch.equal(cpu_rng, torch.random.get_rng_state())
                    or any(not torch.equal(value, torch.cuda.get_rng_state(index)) for index, value in cuda_rng.items())
                    or random.getstate() != python_rng or not _equal_tree(numpy_rng, np.random.get_state()))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if changed_rng:
        raise EvidenceError('Terminal preview consumed RNG; restored state but rejected the projection')
    if not _equal_tree(before, _snapshot_inputs(env, term)):
        raise EvidenceError('Terminal preview changed original env/command/physical inputs/observation history')
    if any(sha256(name) != digest for name, digest in sources.items()):
        raise EvidenceError('Terminal preview source changed during projection')
    for name, value in [('held', held), ('post-command', projected)]:
        if value.shape != (env.num_envs, 252) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise EvidenceError('Malformed/nonfinite '+name+' terminal Critic projection')
    physical_review = _validate_physical_columns(held, projected, diagnostic_inputs, before)
    state = {name: _clone_tree(value) for name, value in vars(copied).items()
             if name not in ('_env', 'robot', 'cfg', '_debug_vis_handle', '_resample')}
    task = {name: _clone_tree(value) for name, value in vars(proxy).items()
            if isinstance(value, torch.Tensor)}
    return TerminalPreview(held, projected, state, task,
        dict(version='r1_fixed_command_terminal_preview_v1', sources=sources,
            checked_original_task_command_history_unchanged=True, rng_unchanged=True,
            projection='native CommandTerm.compute on cloned fixed-task state; no reset',
            command_resampling=False, actual_terminal_physics_read=True,
            physical_critic_consistency=physical_review, post_command_projection_computed=True,
            real_gpu_timeout_validation=False,
            read_cache_note='Native sensor/articulation getters may materialize read caches; '
                'projection has no live scene or physical state-write interface.',
            critic_has_direct_contact_channels=False), before)


@dataclass(frozen=True)
class CapturedTransition:
    held_command_critic: torch.Tensor
    captured: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terminal_episode_steps: torch.Tensor
    step_number: int
    post_command_critic: torch.Tensor | None = None
    previewed: torch.Tensor | None = None
    preview_reviews: tuple = ()
    preview_inputs: tuple = ()

    def for_timeout_bootstrap(self):
        """Never label a pre-command state as the normal post-command next state."""
        timeout = self.truncated & ~self.terminated
        if bool(timeout.any()):
            if (self.post_command_critic is None or self.previewed is None
                    or not bool(self.previewed[timeout].all())):
                raise EvidenceError('Timeout bootstrap unavailable: captured Critic has the held command; '
                    'native-equivalent post-command terminal preview was not captured')
            return self.post_command_critic.detach().clone()
        return None  # True task termination has no timeout value bootstrap.


class PreResetCriticCapture:
    def __init__(self, *, preview_post_command=False):
        if type(preview_post_command) is not bool:
            raise EvidenceError('preview_post_command must be an explicit boolean')
        self.preview_post_command = preview_post_command
        self.active = False
        self.step_number = 0
        self.latest = None

    def begin(self, env):
        if self.active:
            raise EvidenceError('Nested terminal capture step')
        validate_capture_manager(env.observation_manager)
        if self.preview_post_command:
            _preview_contract(env)
        n, device = env.num_envs, env.device
        self.values = torch.zeros(n, 252, device=device, dtype=torch.float32)
        self.captured = torch.zeros(n, device=device, dtype=torch.bool)
        self.terminated = torch.zeros_like(self.captured)
        self.truncated = torch.zeros_like(self.captured)
        self.episode_steps = torch.full((n,), -1, device=device, dtype=torch.long)
        self.projected = torch.zeros_like(self.values)
        self.previewed = torch.zeros_like(self.captured)
        self.preview_reviews = []
        self.preview_inputs = []
        self.latest = None
        self.active = True
        self.step_number += 1

    def before_reset(self, env, env_ids):
        if not self.active:
            return  # Construction, explicit reset and wrapper reset are not transitions.
        if (not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1
                or env_ids.dtype != torch.long or env_ids.device != self.values.device
                or len(env_ids) == 0 or len(torch.unique(env_ids)) != len(env_ids)
                or bool(((env_ids < 0) | (env_ids >= env.num_envs)).any())):
            raise EvidenceError('Invalid terminal reset indices')
        if bool(self.captured[env_ids].any()):
            raise EvidenceError('A terminal slot was reset twice in one physics transition')
        if not bool((env.reset_terminated[env_ids] | env.reset_time_outs[env_ids]).all()):
            raise EvidenceError('A reset inside step lacks a native termination/timeout signal')
        # Single-frame Critic only; never compute() or compute_group('policy').
        if self.preview_post_command:
            preview = preview_terminal_critic(env)
            values = preview.held_command_critic
            self.projected[env_ids] = preview.post_command_critic[env_ids].clone()
            self.previewed[env_ids] = True
            self.preview_reviews.append(preview.review)
            self.preview_inputs.append(dict(env_ids=env_ids.detach().clone(), inputs=preview.input_snapshot))
        else:
            values = env.observation_manager.compute_group('critic', update_history=False)
        if (not isinstance(values, torch.Tensor) or values.shape != self.values.shape
                or values.dtype != torch.float32 or values.device != self.values.device
                or not bool(torch.isfinite(values).all())):
            raise EvidenceError('Nonfinite or malformed pre-reset Critic')
        self.values[env_ids] = values[env_ids].detach().clone()
        self.captured[env_ids] = True
        self.terminated[env_ids] = env.reset_terminated[env_ids].clone()
        self.truncated[env_ids] = env.reset_time_outs[env_ids].clone()
        self.episode_steps[env_ids] = env.episode_length_buf[env_ids].clone()

    def finish(self, terminated, truncated):
        if not self.active:
            raise EvidenceError('Terminal capture finish without an active step')
        for value in (terminated, truncated):
            if (not isinstance(value, torch.Tensor) or value.shape != self.captured.shape
                    or value.dtype != torch.bool or value.device != self.captured.device):
                raise EvidenceError('Invalid native terminal flags')
        if (not torch.equal(self.captured, terminated | truncated)
                or not torch.equal(self.terminated, terminated)
                or not torch.equal(self.truncated, truncated)):
            raise EvidenceError('Native returned terminal flags disagree with pre-reset capture')
        self.latest = CapturedTransition(self.values.clone(), self.captured.clone(),
            terminated.clone(), truncated.clone(), self.episode_steps.clone(), self.step_number,
            self.projected.clone() if self.preview_post_command else None,
            self.previewed.clone() if self.preview_post_command else None, tuple(self.preview_reviews),
            tuple(self.preview_inputs))
        self.active = False
        return self.latest

    def abort(self):
        self.active = False
        self.latest = None


class PreResetCriticCaptureMixin:
    """Use only as first base of a NEW local MigrationEnv subclass, after App."""
    def step(self, action):
        if not hasattr(self, 'r1_terminal_capture'):
            self.r1_terminal_capture = PreResetCriticCapture(
                preview_post_command=getattr(self, 'r1_terminal_preview_enabled', False))
        capture = self.r1_terminal_capture
        capture.begin(self)  # Contract failure occurs before any physics step.
        try:
            result = super().step(action)  # Original production implementation, once.
            capture.finish(result[2], result[3])
            return result
        except BaseException:
            capture.abort()
            raise

    def _reset_idx(self, env_ids):
        capture = getattr(self, 'r1_terminal_capture', None)
        if capture is not None:
            capture.before_reset(self, env_ids)
        return super()._reset_idx(env_ids)
