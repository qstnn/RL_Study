"""Explicit 64-slot GPU wiring probe: reset and at most two mean-action steps.

Strict current student provenance is checked before AppLauncher/output creation.
This is neither a behavior test nor a PPO update or admission decision.
"""
import argparse
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import CONTRACT, REPO, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.student import load_student, environment_reconstruction_contract
from migration_v1.student_worker import dependency_sources, serial, write_json
from migration_v1.runtime_checks import check_action_mapping
from migration_v1.schedule import make_schedule
from migration_v1 import command_adapter, env_adapter, reward_adapter


def probe_sources(schedule):
    files = set(dependency_sources())
    files |= {Path(__file__).resolve(), *(Path(module.__file__).resolve()
        for module in (command_adapter, env_adapter, reward_adapter))}
    files.add(Path(__file__).with_name('schedule.py'))
    files |= {Path(name) for name in schedule['catalog_source_sha256']}
    native = REPO.parent/'Isaaclab/source'
    files |= {native/name for name in (
        'isaaclab/isaaclab/managers/reward_manager.py',
        'isaaclab/isaaclab/utils/io/yaml.py',
        'isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py')}
    return {str(path.resolve()): sha256(path) for path in sorted(files)}


def prepare(checkpoint, output, steps=2):
    """CPU source gate only; no output, App, simulation, or optimizer."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps not in (1, 2):
        raise EvidenceError('The wiring probe permits exactly one or two control steps')
    out = Path(output).resolve()
    if out.exists():
        raise EvidenceError('Use a fresh environment-probe output directory')
    policy, manifest = load_student(checkpoint, device='cpu', inference_mode='row')
    profile = environment_reconstruction_contract(manifest)['physics_profile']
    expected = dict(ground_friction=3., rolling_coefficient=.02, hitch_height_m=.04,
        robot_hitch_offset_m=.20, robot_solver=[16, 4], cart_solver=[16, 4])
    if any(profile.get(key) != value for key, value in expected.items()):
        raise EvidenceError('Current student physical profile does not match the isolated 64-slot adapter')
    schedule = make_schedule(64)
    parameters = command_adapter.command_parameters(schedule['conditions'], manifest)
    checkpoint = Path(checkpoint).resolve(strict=True)
    sources = probe_sources(schedule)
    return dict(policy=policy, manifest=manifest, profile=profile, output=out, steps=steps,
        schedule=schedule, command_parameters=parameters, sources=sources,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        checkpoint_manifest_sha256=sha256(checkpoint.with_name('manifest.json')))


def verify_sources(prepared):
    if any(sha256(name) != digest for name, digest in prepared['sources'].items()):
        raise EvidenceError('Environment-probe execution source changed')
    checkpoint = Path(prepared['checkpoint'])
    if (sha256(checkpoint) != prepared['checkpoint_sha256']
            or sha256(checkpoint.with_name('manifest.json')) != prepared['checkpoint_manifest_sha256']):
        raise EvidenceError('Environment-probe student artifact changed after strict loading')


def _cpu(value):
    return value.detach().cpu().numpy().copy()


def _finite(value, name, shape):
    result = _cpu(value)
    if result.shape != shape or not np.isfinite(result).all():
        raise EvidenceError('Nonfinite or malformed probe '+name)
    return result


def run_simulation(prepared):
    verify_sources(prepared)
    out = prepared['output']
    out.mkdir(parents=True, exist_ok=False)
    for name, digest in prepared['sources'].items():
        source = Path(name)
        relative = (source.relative_to(REPO) if source.is_relative_to(REPO)
                    else Path('_dependencies')/source.relative_to(REPO.parent))
        target = out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest or sha256(source) != digest:
            raise EvidenceError('Environment-probe source changed during archive creation')
    write_json(out/'probe_source_manifest.json', prepared['sources'])
    # Imports/constructor are deliberately below the evidence and archive checks.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    env = None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml

        cfg, agent, configuration = env_adapter.configure_environment(out, 64, seed=43)
        if configuration['schedule'] != prepared['schedule']:
            raise EvidenceError('Actual probe configuration changed the verified 64-slot schedule')
        if any(configuration[key] != prepared['profile'][key]
               for key in ('robot_urdf_sha256', 'cart_urdf_sha256')):
            raise EvidenceError('Generated probe robot/cart assets differ from the strict teacher physical profile')
        commands = command_adapter.apply_command_overrides(cfg, prepared['command_parameters'])
        rewards = reward_adapter.apply_reward_overrides(cfg)
        dump_yaml(str(out/'environment_config.yaml'), cfg)
        dump_yaml(str(out/'agent_config.yaml'), agent)
        write_json(out/'probe_configuration.json', dict(environment=configuration,
            installed_commands=commands, installed_rewards=rewards,
            environment_config_sha256=sha256(out/'environment_config.yaml'),
            agent_config_sha256=sha256(out/'agent_config.yaml'),
            configuration_scope='wiring only; command/reward overrides applied after base configuration',
            behavior_accepted=False, ppo_updates=0, formal_ppo_authorized=False))
        env = RslRlVecEnvWrapper(env_adapter.make_environment(cfg), clip_actions=agent.clip_actions)
        base = env.unwrapped
        policy = prepared['policy']; policy.actor.to('cuda:0').eval().assert_invariants()
        env.seed(43)
        observation, reset_extras = env.reset()
        initial_observation = _finite(observation, 'reset observation', (64, 300))
        critic_dim = CONTRACT['teacher_critic_dim']
        initial_critic = _finite(reset_extras['observations']['critic'], 'reset critic observation', (64, critic_dim))
        pre_action = env_adapter.verify_pre_action(env, observation, out)
        term = base.command_manager.get_term('base_velocity')
        actual_command_steps = _cpu(term.declared_preparation_steps)
        if not np.array_equal(actual_command_steps, np.rint(np.asarray(commands['preparation_seconds'])/base.step_dt)):
            raise EvidenceError('Actual command preparation steps differ from declared fixed clocks')
        reward_terms = list(base.reward_manager.active_terms)
        for name in rewards['changed_terms']:
            actual = base.reward_manager.get_term_cfg(name)
            function_name = 'tracking' if name in ('track_broad', 'track_fine') else name
            if actual.func is not getattr(reward_adapter, function_name) or actual.weight != rewards['term_weights'][name]:
                raise EvidenceError('Actual reward manager did not retain the configured isolated adapters')
        write_json(out/'request.json', dict(artifact_kind='migration64_environment_wiring_probe',
            checkpoint=prepared['checkpoint'], checkpoint_sha256=prepared['checkpoint_sha256'],
            checkpoint_manifest_sha256=prepared['checkpoint_manifest_sha256'],
            num_envs=64, requested_control_steps=prepared['steps'], seed=43, device='cuda:0',
            actor_inference='row', actor_unique_action_source=True, action_override=False,
            teacher_calls=0, observation_noise=False, physical_randomization=False,
            sample_actions=False, ppo_updates=0, optimizer_steps=0,
            behavior_accepted=False, autonomous_retention_accepted=False, formal_ppo_authorized=False,
            native_cop_behavior_audit_performed=False,
            command_parameters=commands, actual_preparation_steps=actual_command_steps))

        frames = []; mapping = {}; latest = {}
        native_process = base.action_manager.process_action
        robot = base.scene['robot']
        def record_action(action):
            native_process(action)
            legs = base.action_manager.get_term('joint_pos')
            wheels = base.action_manager.get_term('joint_vel')
            mapping.clear()
            mapping.update(actor_raw=_cpu(latest['raw']), manager_raw=_cpu(action),
                leg_target=_cpu(legs.processed_actions), wheel_target=_cpu(wheels.processed_actions),
                default_legs=_cpu(robot.data.default_joint_pos[:, legs._joint_ids]), leg_clip=_cpu(legs._clip))
            mapping['review'] = check_action_mapping(**mapping, wrapper_clip=env.clip_actions)
        base.action_manager.process_action = record_action
        with torch.inference_mode():
            for step in range(prepared['steps']):
                if not app.is_running():
                    raise EvidenceError('Application closed before the requested short wiring probe')
                previous = _finite(observation, 'pre-action observation', (64, 300))
                pre_steps = _cpu(base.episode_length_buf)
                pre_phase = _cpu(base.r1_eval_phase)
                raw = policy(observation)
                raw_array = _finite(raw, 'mean action', (64, 16))
                latest['raw'] = raw.clone()
                observation, reward, done, extras = env.step(raw)
                post = _finite(observation, 'returned observation', (64, 300))
                critic = _finite(extras['observations']['critic'], 'returned critic observation', (64, critic_dim))
                reward_array = _finite(reward, 'reward', (64,))
                reward_rates = _finite(base.reward_manager._step_reward, 'reward term rates', (64, len(reward_terms)))
                done_array = _finite(done, 'done', (64,))
                metrics = {name: _finite(value, 'physical metric '+name, (64,))
                           for name, value in base.r1_last_metrics.items()}
                # UnifiedEnv saves terminal metrics before automatic reset; the
                # returned observation may instead be the next episode reset.
                frames.append(dict(pre_observation=previous, returned_observation=post,
                    pre_episode_steps=pre_steps, pre_phase=pre_phase, actor_raw=raw_array,
                    manager_raw=mapping['manager_raw'], leg_target=mapping['leg_target'],
                    wheel_target=mapping['wheel_target'], reward=reward_array, reward_term_rates=reward_rates,
                    returned_critic_observation=critic, done=done_array,
                    action_mapping=mapping['review'], terminal_preserving_metrics=metrics))
                if np.any(done_array):
                    break
        policy.actor.assert_invariants()
        verify_sources(prepared)
        if policy.teacher_calls != 0 or policy.actor_calls != len(frames):
            raise EvidenceError('Probe action-call accounting differs')
        arrays = {'initial_observation': initial_observation, 'initial_critic_observation': initial_critic,
                  'reward_term_names': np.asarray(reward_terms)}
        keys = ('pre_observation', 'returned_observation', 'pre_episode_steps', 'pre_phase',
                'actor_raw', 'manager_raw', 'leg_target', 'wheel_target', 'reward', 'reward_term_rates',
                'returned_critic_observation', 'done')
        arrays.update({key: np.stack([frame[key] for frame in frames]) for key in keys})
        arrays.update({'metric_'+key: np.stack([frame['terminal_preserving_metrics'][key] for frame in frames])
                       for key in frames[0]['terminal_preserving_metrics']})
        trace = out/'environment_probe_trace.npz'
        np.savez_compressed(trace, **arrays)
        terminated_slots = sorted(set(np.flatnonzero(arrays['done'].any(axis=0)).tolist()))
        review = dict(artifact_kind='migration64_environment_wiring_probe',
            actual_pre_action_verified=pre_action['runtime_pre_action_verified'],
            actual_conditions_count=pre_action['coverage']['unique_conditions'],
            requested_control_steps=prepared['steps'], executed_control_steps=len(frames),
            complete_requested_steps=len(frames) == prepared['steps'],
            control_dt_s=base.step_dt, recorded_duration_s=len(frames)*base.step_dt,
            finite_observations_actions_rewards_metrics=True, terminated_slots=terminated_slots,
            reward_term_names=reward_terms, reward_term_rates_are_weighted_pre_dt=True,
            critic_observation_dim=critic_dim, critic_network_or_optimizer_created=False,
            termination_metrics=[{key: value for key, value in frame['terminal_preserving_metrics'].items()
                if key.startswith('termination_') or key in ('terminated', 'time_out')}
                for frame in frames],
            actor_execution=policy.execution_record(),
            action_mapping=[frame['action_mapping'] for frame in frames],
            trace_sha256=sha256(trace), sources_unchanged=True,
            returned_observation_can_be_auto_reset=True, physical_metrics_preserve_terminal_state=True,
            behavior_accepted=False, autonomous_retention_accepted=False,
            native_cop_behavior_audit_performed=False, ppo_updates=0, optimizer_steps=0,
            formal_ppo_authorized=False)
        write_json(out/'environment_probe_review.json', review)
        return review
    except Exception as error:
        write_json(out/'environment_probe_error.json', dict(error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), behavior_accepted=False,
            ppo_updates=0, formal_ppo_authorized=False))
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.checkpoint, args.output, args.steps)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    review = run_simulation(prepared)
    print('R1_ENVIRONMENT_PROBE', serial(review), flush=True)


if __name__ == '__main__':
    main()
