"""Deterministic R1 physical environment and causal deployment-clock bridge.

This exploratory PPO configuration inherits the measured mu3 / GPU16-4
physical line. It does not require a successful BC policy or claim behavior
acceptance. No environment noise, randomized initial state, teacher action or
post-reset physical state write is introduced.
"""
from functools import lru_cache
from pathlib import Path
import inspect
import math

import numpy as np
import torch

from migration_v1 import command_adapter, env_adapter, reward_adapter
from migration_v1.dataset import EvidenceError
from migration_v1.environment_probe import probe_sources
from migration_v1.schedule import make_schedule
from migration_v1.terminal_capture import PreResetCriticCaptureMixin

from .contract import (ACTOR_DIM, BASE_OBS_DIM, RESET_TIME_SCALE_S,
                       PREPARATION_SCALE_S)


def declared_commands(schedule):
    cases = np.asarray(schedule['conditions'], dtype=float)
    towing = cases[:, 2] > 0.
    preparation = np.where(towing & (cases[:, 0] < 0.), 7.48, 3.)
    return dict(version='exploration_v1_fixed_task_clocks',
        preparation_seconds=preparation.tolist(),
        loaded_acceleration_mps2=np.where(towing, .05, .3).tolist(),
        goal_distance_m=np.where(towing, 1.2, 1.).tolist(),
        task_clock_only=True, live_teacher_or_contact_ready=False,
        teacher_runtime_required=False,
        source_note='Declared task clocks retain prior nominal physical experiments; '
                    'they do not assert candidate Actor success.',
        distance_event_note='Original command uses robot net distance for braking; '
                    'the event is a training milestone, not joint robot/cart behavior acceptance.',
        non_anchor_behavior_verified=False)


def configure(output, num_envs=64, seed=43, *, episode_seconds=65.):
    """After AppLauncher; generate config/assets but do not instantiate the env."""
    if (isinstance(episode_seconds, bool) or not math.isfinite(episode_seconds)
            or episode_seconds < 40.):
        raise EvidenceError('Exploration episodes must allow at least the 40-second physical protocol')
    cfg, agent, physical = env_adapter.configure_environment(output, num_envs, seed)
    cfg.episode_length_s = float(episode_seconds)
    commands = declared_commands(physical['schedule'])
    command_adapter.apply_command_overrides(cfg, commands)
    rewards = reward_adapter.apply_reward_overrides(cfg)
    manifest = dict(artifact_kind='exploration_v1_environment_configuration',
        physical=physical, commands=commands, rewards=rewards,
        episode_seconds=float(episode_seconds), policy_dimension=ACTOR_DIM,
        base_observation_dimension=BASE_OBS_DIM, critic_dimension=252,
        causal_clock_sources=['episode_length_buf * step_dt',
                              'base_velocity.declared_preparation_steps * step_dt'],
        clock_scales_s=[RESET_TIME_SCALE_S, PREPARATION_SCALE_S],
        teacher_calls=0, observation_noise=False, physical_randomization=False,
        training_runtime_ready=False, final_behavior_pass=False,
        hold_completion_note='Command clock completion is not a physical hold success label.')
    cfg.r1_exploration_configuration = manifest
    return cfg, agent, manifest


@lru_cache(maxsize=1)
def _captured_environment_type():
    parent = env_adapter._runtime_types()[1]
    class ExplorationCapturedEnv(PreResetCriticCaptureMixin, parent):
        pass
    return ExplorationCapturedEnv


def make_environment(cfg, *, terminal_preview=True, **kwargs):
    if terminal_preview is not True or not hasattr(cfg, 'r1_exploration_configuration'):
        raise EvidenceError('Exploration requires its explicit configuration and terminal preview')
    base = _captured_environment_type()(cfg, **kwargs)
    base.r1_terminal_preview_enabled = True
    return base


def observe(base, observation):
    """Append clocks once to an existing observation; never advance its history."""
    if (not isinstance(observation, torch.Tensor) or observation.dtype != torch.float32
            or observation.shape != (base.num_envs, BASE_OBS_DIM)
            or not torch.isfinite(observation).all()):
        raise EvidenceError('Expected the original finite float32 [N,300] observation')
    term = base.command_manager.get_term('base_velocity')
    elapsed = base.episode_length_buf
    preparation = term.declared_preparation_steps
    if (elapsed.shape != (base.num_envs,) or preparation.shape != elapsed.shape
            or elapsed.dtype != torch.long or preparation.dtype != torch.long
            or elapsed.device != observation.device or preparation.device != observation.device
            or (elapsed < 0).any() or (preparation <= 0).any()
            or not math.isclose(base.step_dt, .02, abs_tol=1.e-12, rel_tol=0.)):
        raise EvidenceError('Causal task clock shape, device, reset or control dt differs')
    clocks = torch.stack((elapsed.to(torch.float32) * base.step_dt / RESET_TIME_SCALE_S,
                          preparation.to(torch.float32) * base.step_dt / PREPARATION_SCALE_S), dim=-1)
    result = torch.cat((observation, clocks.clamp(0., 1.)), dim=-1)
    if result.shape != (base.num_envs, ACTOR_DIM) or not torch.isfinite(result).all():
        raise EvidenceError('Malformed causal exploration Actor observation')
    return result


def pre_action_conditions(base):
    return torch.stack((base.r1_slope_rad * (180. / math.pi), base.r1_payload_kg,
                        base.r1_towed_kg, base.r1_eval_phase.to(torch.float32)), dim=-1)


def verify_commands(base):
    """Inspect the actual command object/config, not only a stored declaration."""
    schedule = base.cfg.migration_schedule
    declared = declared_commands(schedule)
    term = base.command_manager.get_term('base_velocity')
    function_source = inspect.getsourcefile(type(term)._update_command)
    if (type(term).__name__ != 'MigrationVelocityCommand' or function_source is None
            or Path(function_source).resolve() != Path(command_adapter.__file__).resolve()
            or base.cfg.command_profile != 'slew'
            or not math.isclose(float(base.cfg.unloaded_acceleration), 1., abs_tol=1.e-9, rel_tol=0.)):
        raise EvidenceError('Runtime command implementation or slew configuration differs')
    cpu = lambda value: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    preparation = np.asarray(declared['preparation_seconds'])
    conditions = np.asarray(schedule['conditions'])
    expected_speed = np.where(conditions[:, 1:].sum(-1) == 0., 3.1, .3)
    comparisons = (
        (term.declared_preparation_steps, np.rint(preparation / base.step_dt)),
        (term.declared_cycle_offsets, np.rint((preparation - 3.) / base.step_dt)),
        (base.cfg.loaded_acceleration, np.asarray(declared['loaded_acceleration_mps2'])),
        (base.cfg.eval_distance_m, np.asarray(declared['goal_distance_m'])),
        (base.cfg.fixed_speed_mps, expected_speed),
        (term.requested_vx, expected_speed),
    )
    if any(cpu(actual).shape != expected.shape
           or not np.allclose(cpu(actual), expected, atol=1.e-6, rtol=0.)
           for actual, expected in comparisons):
        raise EvidenceError('Runtime preparation, acceleration, distance or requested-speed vector differs')
    return declared


def verify_pre_action(env, observation, output=None):
    """Native readback of all64, physical assets and the two added task clocks."""
    base = env.unwrapped
    if observation.shape != (base.num_envs, ACTOR_DIM):
        raise EvidenceError('Exploration initial Actor input must have 302 columns')
    original = observation[:, :BASE_OBS_DIM]
    review = env_adapter.verify_pre_action(env, original, output)
    if not torch.equal(observation, observe(base, original)) or not torch.all(observation[:, 300] == 0):
        raise EvidenceError('Initial Actor clocks differ from actual ordinary reset')
    verify_commands(base)
    if getattr(base, 'r1_terminal_preview_enabled', False) is not True:
        raise EvidenceError('Terminal pre-reset preview was not enabled')
    for name in base.cfg.r1_exploration_configuration['rewards']['changed_terms']:
        term_cfg = base.reward_manager.get_term_cfg(name)
        expected_name = 'tracking' if name in ('track_broad', 'track_fine') else name
        if (term_cfg.func is not getattr(reward_adapter, expected_name)
                or term_cfg.weight != base.cfg.r1_exploration_configuration['rewards']['term_weights'][name]):
            raise EvidenceError('Runtime exploration reward override differs: ' + name)
    return dict(review, exploration_actor_dimension=ACTOR_DIM, actual_clocks_verified=True,
                terminal_preview_enabled=True, actual_reward_functions_verified=True,
                final_behavior_pass=False)


def source_signature(num_envs=64):
    from migration_v1.contract import sha256
    sources = probe_sources(make_schedule(num_envs))
    sources.update({str(path.resolve()): sha256(path) for path in
        (Path(__file__), Path(__file__).with_name('contract.py'),
         Path(__file__).parents[1] / 'migration_v1/terminal_capture.py')})
    return sources
