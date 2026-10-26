"""Independent two-step terminal-preview diagnostic; no optimizer or PPO entry.

Run short_timeout and uninterrupted in SEPARATE applications, then compare.
Only the short diagnostic episode horizon differs; ordinary reset counters and
native task/command methods are unchanged. Neither run is behavior acceptance.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration_v1 import environment_probe, env_adapter, command_adapter, reward_adapter
from migration_v1.contract import REPO, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.runtime_checks import check_action_mapping
from migration_v1.student_worker import write_json, serial
from migration_v1.terminal_capture import PreResetCriticCaptureMixin, ROBOT_READERS


KIND = 'diagnostic_terminal_preview_two_step'
MODES = ('short_timeout', 'uninterrupted')
STEPS = 2
# Fixed before either run; no relaxing thresholds after looking at a failed pair.
PAIR_ATOL = dict(observation=1.e-5, action=1.e-6, physical=1.e-5, critic=1.e-5)
PAIR_RTOL = 1.e-6


def prepare(checkpoint, output, mode):
    if mode not in MODES:
        raise EvidenceError('Choose short_timeout or uninterrupted diagnostic mode')
    prepared = environment_probe.prepare(checkpoint, output, steps=STEPS)
    native = REPO.parent/'Isaaclab/source/isaaclab/isaaclab'
    files = [Path(__file__), Path(__file__).with_name('terminal_capture.py'),
        Path(__file__).with_name('ppo_policy.py'), native/'envs/mdp/observations.py',
        native/'utils/math.py', native/'managers/termination_manager.py']
    prepared['sources'].update({str(path.resolve()): sha256(path) for path in files})
    prepared.update(mode=mode, artifact_kind=KIND, num_envs=64,
        diagnostic_episode_horizon_override=mode == 'short_timeout',
        source_episode_length_s=65., actual_episode_length_s=.04 if mode == 'short_timeout' else 65.)
    return prepared


def _relative(path):
    path = Path(path)
    return path.relative_to(REPO) if path.is_relative_to(REPO) else Path('_dependencies')/path.relative_to(REPO.parent)


def _array(value):
    return value.detach().cpu().numpy().copy()


def _physical(base):
    robot = base.scene['robot'].data
    scanner = base.scene.sensors['height_scanner'].data
    return {**{'robot_'+name: _array(getattr(robot, name)) for name in ROBOT_READERS},
        'scanner_pos_w': _array(scanner.pos_w), 'scanner_ray_hits_w': _array(scanner.ray_hits_w)}


def _terminal_physical(base, captured):
    values = _physical(base)  # Native returned state for non-reset slots.
    for snapshot in captured.preview_inputs:
        ids, inputs = _array(snapshot['env_ids']), snapshot['inputs']
        for group in ('robot', 'scanner'):
            for name, tensor in inputs[group].items():
                values[group+'_'+name][ids] = _array(tensor)[ids]
    return values


def _validate_observation(value, width):
    result = _array(value)
    if result.shape != (64, width) or not np.isfinite(result).all():
        raise EvidenceError('Malformed/nonfinite diagnostic observation')
    return result


def run_simulation(prepared):
    """Explicit diagnostics only; called by root/user, never by CPU tests."""
    environment_probe.verify_sources(prepared)
    out = prepared['output']
    out.mkdir(parents=True, exist_ok=False)
    for name, digest in prepared['sources'].items():
        target = out/'source_snapshot'/_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
        if sha256(target) != digest or sha256(name) != digest:
            raise EvidenceError('Terminal-preview execution source changed during archive')
    write_json(out/'terminal_preview_source_manifest.json', prepared['sources'])
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    env = None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml

        cfg, agent, configuration = env_adapter.configure_environment(out, 64, seed=43)
        if float(cfg.episode_length_s) != prepared['source_episode_length_s']:
            raise EvidenceError('The original diagnostic comparison horizon changed')
        if configuration['schedule'] != prepared['schedule']:
            raise EvidenceError('Actual diagnostic layout changed the 64-case schedule')
        if any(configuration[key] != prepared['profile'][key]
               for key in ('robot_urdf_sha256', 'cart_urdf_sha256')):
            raise EvidenceError('Diagnostic physical assets differ from the strict loaded profile')
        commands = command_adapter.apply_command_overrides(cfg, prepared['command_parameters'])
        rewards = reward_adapter.apply_reward_overrides(cfg)
        cfg.episode_length_s = prepared['actual_episode_length_s']
        if cfg.sim.dt*cfg.decimation != .02:
            raise EvidenceError('The diagnostic requires the original .02 second control step')
        dump_yaml(str(out/'environment_config.yaml'), cfg)
        dump_yaml(str(out/'agent_config.yaml'), agent)

        class CaptureEnv(PreResetCriticCaptureMixin, env_adapter._runtime_types()[1]):
            r1_terminal_preview_enabled = True

        env = RslRlVecEnvWrapper(CaptureEnv(cfg), clip_actions=agent.clip_actions)
        base = env.unwrapped
        policy = prepared['policy']; policy.actor.to('cuda:0').eval().assert_invariants()
        env.seed(43)
        observation, extras = env.reset()
        initial = _validate_observation(observation, 300)
        initial_critic = _validate_observation(extras['observations']['critic'], 252)
        initial_physical = _physical(base)
        pre_action = env_adapter.verify_pre_action(env, observation, out)
        if base.max_episode_length != round(prepared['actual_episode_length_s']/base.step_dt):
            raise EvidenceError('Actual native timeout length differs from the declared diagnostic horizon')
        term = base.command_manager.get_term('base_velocity')
        if not np.array_equal(_array(term.declared_preparation_steps),
                np.rint(np.asarray(commands['preparation_seconds'])/base.step_dt)):
            raise EvidenceError('Actual fixed preparation command clocks changed')
        for name in rewards['changed_terms']:
            actual = base.reward_manager.get_term_cfg(name)
            function = 'tracking' if name in ('track_broad', 'track_fine') else name
            if actual.func is not getattr(reward_adapter, function) or actual.weight != rewards['term_weights'][name]:
                raise EvidenceError('Actual diagnostic reward adapter differs')
        write_json(out/'request.json', dict(artifact_kind=KIND, mode=prepared['mode'],
            checkpoint=prepared['checkpoint'], checkpoint_sha256=prepared['checkpoint_sha256'],
            checkpoint_manifest_sha256=prepared['checkpoint_manifest_sha256'], seed=43,
            num_envs=64, requested_control_steps=STEPS, source_episode_length_s=65.,
            actual_episode_length_s=prepared['actual_episode_length_s'],
            diagnostic_episode_horizon_override=prepared['diagnostic_episode_horizon_override'],
            episode_counter_override=False, reset_physical_state_override=False,
            sample_actions=False, actor_inference='row', action_override=False, teacher_calls=0,
            observation_noise=False, physical_randomization=False, device='cuda:0',
            installed_commands=commands, installed_rewards=rewards, configuration=configuration,
            original_physics_profile=prepared['profile'], original_command_methods=True,
            ppo_updates=0, optimizer_steps=0, behavior_accepted=False,
            autonomous_retention_accepted=False, formal_ppo_authorized=False))
        latest, mapping, frames = {}, {}, []
        native_process = base.action_manager.process_action
        def record_action(action):
            native_process(action)
            legs, wheels = base.action_manager.get_term('joint_pos'), base.action_manager.get_term('joint_vel')
            mapping.clear()
            mapping.update(actor_raw=_array(latest['raw']), manager_raw=_array(action),
                leg_target=_array(legs.processed_actions), wheel_target=_array(wheels.processed_actions),
                default_legs=_array(base.scene['robot'].data.default_joint_pos[:, legs._joint_ids]),
                leg_clip=_array(legs._clip))
            mapping['review'] = check_action_mapping(**mapping, wrapper_clip=env.clip_actions)
        base.action_manager.process_action = record_action
        with torch.inference_mode():
            for step in range(STEPS):
                if not app.is_running():
                    raise EvidenceError('Application closed before the two-step diagnostic ended')
                previous = _validate_observation(observation, 300)
                pre_steps = _array(base.episode_length_buf)
                action = policy(observation)
                latest['raw'] = action.detach().clone()
                observation, reward, done, extras = env.step(action)
                capture = base.r1_terminal_capture.latest
                if capture is None:
                    raise EvidenceError('Native step returned without capture bookkeeping')
                physical = _terminal_physical(base, capture)
                frames.append(dict(pre_observation=previous, pre_episode_steps=pre_steps,
                    actor_raw=_array(action), manager_raw=mapping['manager_raw'],
                    leg_target=mapping['leg_target'], wheel_target=mapping['wheel_target'],
                    returned_observation=_validate_observation(observation, 300),
                    returned_critic=_validate_observation(extras['observations']['critic'], 252),
                    held_critic=_array(capture.held_command_critic), projected_critic=_array(capture.post_command_critic),
                    captured=_array(capture.captured), previewed=_array(capture.previewed),
                    terminated=_array(capture.terminated), truncated=_array(capture.truncated),
                    reward=_array(reward), done=_array(done),
                    terminal_episode_steps=_array(capture.terminal_episode_steps),
                    post_episode_steps=_array(base.episode_length_buf), physical=physical,
                    mapping_review=mapping['review'], preview_reviews=capture.preview_reviews))
                if bool(capture.terminated.any()):
                    break  # Natural physical failures are preserved, never counted as timeout validation.
        environment_probe.verify_sources(prepared)
        policy.actor.assert_invariants()
        if policy.teacher_calls != 0 or policy.actor_calls != len(frames):
            raise EvidenceError('Diagnostic actor call accounting differs')
        arrays = dict(initial_observation=initial, initial_critic=initial_critic,
            **{'initial_'+name: value for name, value in initial_physical.items()})
        arrays.update({name: np.stack([frame[name] for frame in frames]) for name in frames[0]
            if name not in ('physical', 'mapping_review', 'preview_reviews')})
        arrays.update({name: np.stack([frame['physical'][name] for frame in frames]) for name in initial_physical})
        trace = out/'terminal_preview_trace.npz'
        np.savez_compressed(trace, **arrays)
        review = dict(artifact_kind=KIND, mode=prepared['mode'], trace_sha256=sha256(trace),
            requested_control_steps=STEPS, executed_control_steps=len(frames),
            runtime_first_batch_verified=pre_action['runtime_pre_action_verified'],
            actual_conditions_count=pre_action['coverage']['unique_conditions'],
            native_true_termination_slots=np.flatnonzero(arrays['terminated'].any(0)).tolist(),
            native_time_out_slots=np.flatnonzero(arrays['truncated'].any(0)).tolist(),
            eligible_timeout_slots=np.flatnonzero((arrays['truncated'] & ~arrays['terminated']).any(0)).tolist(),
            preview_reviews=[frame['preview_reviews'] for frame in frames],
            action_mapping=[frame['mapping_review'] for frame in frames],
            actor_execution=policy.execution_record(), sources_unchanged=True,
            paired_uninterrupted_next_critic_verified=False, behavior_accepted=False,
            ppo_updates=0, optimizer_steps=0, formal_ppo_authorized=False)
        write_json(out/'terminal_preview_review.json', review)
        return review
    except Exception as error:
        save_error_diagnostic(out, error)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def save_error_diagnostic(output, error):
    """Preserve exact failed projection arrays without pickle or a success review."""
    report = dict(error=str(error), traceback=traceback.format_exc(), artifact_kind=KIND,
        ppo_updates=0, optimizer_steps=0, formal_ppo_authorized=False)
    diagnostic = getattr(error, 'terminal_preview_diagnostic', None)
    if diagnostic is not None:
        report['projection_diagnostic'] = diagnostic
        arrays = {}
        def flatten(value, name):
            if hasattr(value, 'detach'):
                arrays[name] = _array(value)
            elif isinstance(value, np.ndarray):
                arrays[name] = value.copy()
            elif isinstance(value, dict):
                for key, child in value.items(): flatten(child, name+'/'+str(key))
            elif isinstance(value, (tuple, list)):
                for key, child in enumerate(value): flatten(child, name+'/'+str(key))
        flatten(error.terminal_preview_arrays, 'projection')
        path = Path(output)/'terminal_preview_failed_projection.npz'
        np.savez_compressed(path, **arrays)
        report['failed_projection_trace'] = dict(file=path.name, sha256=sha256(path),
            array_count=len(arrays), successful_preview=False)
    write_json(Path(output)/'terminal_preview_error.json', report)
    return report


def _read_run(directory):
    root = Path(directory).resolve()
    request = json.loads((root/'request.json').read_text())
    review = json.loads((root/'terminal_preview_review.json').read_text())
    sources = json.loads((root/'terminal_preview_source_manifest.json').read_text())
    if request.get('artifact_kind') != KIND or review.get('artifact_kind') != KIND:
        raise EvidenceError('Only independent terminal-preview diagnostics may be compared')
    for name, digest in sources.items():
        if sha256(root/'source_snapshot'/_relative(name)) != digest:
            raise EvidenceError('Diagnostic archived source SHA differs')
    trace = root/'terminal_preview_trace.npz'
    if sha256(trace) != review['trace_sha256']:
        raise EvidenceError('Diagnostic raw trace SHA differs')
    with np.load(trace, allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files}
    return request, review, sources, arrays


def compare_arrays(timeout, reference):
    """Pure numerical pair check. A failed pair is retained, never relaxed to pass."""
    errors, differences = [], {}
    def compare(name, a, b, kind):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            errors.append(name+': shape/nonfinite'); return
        difference = float(np.max(np.abs(a-b))) if a.size else 0.
        passed = bool(np.allclose(a, b, atol=PAIR_ATOL[kind], rtol=PAIR_RTOL))
        differences[name] = dict(max_abs_error=difference, atol=PAIR_ATOL[kind], rtol=PAIR_RTOL, passed=passed)
        if not passed:
            errors.append(name+': exceeds declared pair tolerance')
    for arrays, name in ((timeout, 'timeout'), (reference, 'reference')):
        if arrays['terminated'].shape != (2, 64) or arrays['truncated'].shape != (2, 64):
            raise EvidenceError(name+': require exactly two real 64-slot transitions')
        if np.any(arrays['terminated']): errors.append(name+': natural physical termination occurred')
    eligible = timeout['truncated'] & ~timeout['terminated']
    expected = np.zeros((2, 64), dtype=bool); expected[1] = True
    if not np.array_equal(eligible, expected): errors.append('Expected only final-step true timeouts for all64')
    if np.any(reference['truncated']): errors.append('Uninterrupted reference also timed out')
    if not np.array_equal(timeout['captured'], expected) or not np.array_equal(timeout['previewed'], expected):
        errors.append('Not all and only actual timeout slots were captured/projected')
    if not np.array_equal(timeout['pre_episode_steps'], np.broadcast_to([[0], [1]], (2, 64))):
        errors.append('Timeout run pre-action counters did not start at ordinary reset')
    if not np.array_equal(reference['post_episode_steps'], np.broadcast_to([[1], [2]], (2, 64))):
        errors.append('Reference was reset or its native episode counter was overridden')
    if not np.all(timeout['post_episode_steps'][-1] == 0): errors.append('Native timeout reset was not observed')
    for name in ('initial_observation', 'pre_observation'):
        compare(name, timeout[name], reference[name], 'observation')
    for name in ('actor_raw', 'manager_raw', 'leg_target', 'wheel_target'):
        compare(name, timeout[name], reference[name], 'action')
    for name in sorted(timeout):
        if name.startswith(('robot_', 'scanner_', 'initial_robot_', 'initial_scanner_')):
            compare(name, timeout[name], reference[name], 'physical')
    compare('projected_vs_native_unreset_next_critic', timeout['projected_critic'][-1],
        reference['returned_critic'][-1], 'critic')
    reset_difference = float(np.max(abs(timeout['projected_critic'][-1]-timeout['returned_critic'][-1])))
    if reset_difference <= PAIR_ATOL['critic']:
        errors.append('Diagnostic did not demonstrate a distinct native reset observation')
    return dict(terminal_preview_gpu_pair_verified=not errors, differences=differences, errors=errors,
        projected_vs_returned_reset_max_abs_error=reset_difference, actual_eligible_timeout_count=int(eligible.sum()),
        comparison_scope='Separate-app native next-Critic; physical/action agreement is required within fixed tolerances',
        autonomous_behavior_accepted=False, pre_ppo_behavior_ready=False,
        ppo_updates=0, optimizer_steps=0, formal_ppo_authorized=False)


def compare_runs(timeout_run, reference_run):
    a, ar, sources, av = _read_run(timeout_run)
    b, br, other_sources, bv = _read_run(reference_run)
    if a['mode'] != 'short_timeout' or b['mode'] != 'uninterrupted' or sources != other_sources:
        raise EvidenceError('Need two independent diagnostic modes from identical execution sources')
    for path in (Path(__file__).resolve(), Path(__file__).with_name('terminal_capture.py').resolve()):
        if sources.get(str(path)) != sha256(path):
            raise EvidenceError('Diagnostic comparison implementation changed after execution')
    for key in ('checkpoint_sha256', 'checkpoint_manifest_sha256', 'seed', 'num_envs',
                'installed_commands', 'installed_rewards', 'original_physics_profile'):
        if a[key] != b[key]: raise EvidenceError('Diagnostic pair differs: '+key)
    for request in (a, b):
        if (request.get('episode_counter_override') is not False or request.get('reset_physical_state_override') is not False
                or request.get('sample_actions') is not False or request.get('action_override') is not False
                or request.get('teacher_calls') != 0 or request.get('ppo_updates') != 0
                or request.get('optimizer_steps') != 0 or request.get('device') != 'cuda:0'
                or request.get('source_episode_length_s') != 65. or request.get('seed') != 43
                or request.get('num_envs') != 64 or request.get('requested_control_steps') != 2
                or request.get('original_command_methods') is not True
                or request.get('actor_inference') != 'row' or request.get('observation_noise') is not False
                or request.get('physical_randomization') is not False):
            raise EvidenceError('Diagnostic pair contains an unreviewed override or non-GPU execution')
    if a['actual_episode_length_s'] != .04 or b['actual_episode_length_s'] != 65.:
        raise EvidenceError('Diagnostic comparison horizons differ')
    if a.get('diagnostic_episode_horizon_override') is not True or b.get('diagnostic_episode_horizon_override') is not False:
        raise EvidenceError('Episode-horizon diagnostic override is not explicit')
    if any(review.get('actual_conditions_count') != 64 or review.get('runtime_first_batch_verified') is not True
           for review in (ar, br)):
        raise EvidenceError('Actual64 pre-action readback is missing')
    result = compare_arrays(av, bv)
    result.update(artifact_kind='diagnostic_terminal_preview_pair_review', timeout_run=str(Path(timeout_run).resolve()),
        reference_run=str(Path(reference_run).resolve()), checkpoint_sha256=a['checkpoint_sha256'],
        timeout_trace_sha256=ar['trace_sha256'], reference_trace_sha256=br['trace_sha256'], source_count=len(sources))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run'); run.add_argument('--checkpoint', required=True)
    run.add_argument('--output', required=True); run.add_argument('--mode', choices=MODES, required=True)
    compare = commands.add_parser('compare'); compare.add_argument('--timeout-run', required=True)
    compare.add_argument('--reference-run', required=True); compare.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'run':
        result = run_simulation(prepare(args.checkpoint, args.output, args.mode))
    else:
        out = Path(args.output)
        if out.exists(): parser.error('Use a new diagnostic comparison report path')
        result = compare_runs(args.timeout_run, args.reference_run)
        write_json(out, result)
    print('R1_TERMINAL_PREVIEW_DIAGNOSTIC', serial(result), flush=True)


if __name__ == '__main__':
    main()
