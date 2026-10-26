"""Deterministic per-task command clocks for a single shared deployment Actor.

Preparation times come from audited command protocols, never live teacher
readiness. Only the command changes over time; no new Actor inputs are added.
"""
import math

import numpy as np

from .dataset import EvidenceError
from .runtime_checks import check_source_preparation
from .schedule import permitted_catalog


def command_parameters(conditions, manifest):
    cases = np.asarray(conditions, dtype=float)
    if cases.ndim != 2 or cases.shape[1] != 3 or not len(cases) or not np.isfinite(cases).all():
        raise ValueError('Expected finite nominal (slope,payload,tow) conditions')
    if not set(map(tuple, cases)) <= set(permitted_catalog()['cases']):
        raise ValueError('Commands contain forbidden R1 nominal conditions')
    episodes = manifest['teacher_bundle']['episodes']
    starts, goals, accelerations = {}, {}, {}
    for slope in (25., -20.):
        selected = [e for e in episodes if e['case']['slope_deg'] == slope]
        if not selected:
            raise EvidenceError('Both complete teacher anchors are required')
        start_values = {float(e['command_timing']['actual_move_start_s']) for e in selected}
        goal_values = {float(e['case']['goal_distance_m']) for e in selected}
        acceleration_values = {float(e['physics_profile']['loaded_acceleration_mps2']) for e in selected}
        if len(start_values) != 1 or len(goal_values) != 1 or len(acceleration_values) != 1:
            raise EvidenceError('Source episodes disagree on the selected command protocol')
        starts[slope] = start_values.pop(); goals[slope] = goal_values.pop()
        accelerations[slope] = acceleration_values.pop()
        check_source_preparation(manifest, starts[slope], case_slope=slope)
    if goals[25.] != goals[-20.] or accelerations[25.] != accelerations[-20.]:
        raise EvidenceError('Anchor distance/acceleration protocols differ')
    if not math.isfinite(goals[25.]) or goals[25.] < 1. or accelerations[25.] <= 0:
        raise EvidenceError('Invalid audited towing command protocol')
    towing = cases[:, 2] > 0
    # Other slopes/masses are unvalidated training cases, not new teacher evidence.
    preparation = np.where(towing, np.where(cases[:, 0] < 0, starts[-20.], starts[25.]), 3.)
    acceleration = np.where(towing, accelerations[25.], .3)
    distance = np.where(towing, goals[25.], 1.)
    return dict(version='migration_v1_declared_case_clock', preparation_seconds=preparation.tolist(),
        loaded_acceleration_mps2=acceleration.tolist(), goal_distance_m=distance.tolist(),
        source_anchor_preparation_seconds={str(key): value for key, value in starts.items()},
        task_clock_only=True, live_teacher_or_contact_ready=False,
        non_anchor_protocol='candidate: negative towing slopes use downhill clock, others uphill clock',
        non_anchor_behavior_verified=False)


def apply_command_overrides(cfg, parameters):
    """After AppLauncher; fixed-speed environment uses exactly one task per reset."""
    import torch
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.commands import UnifiedVelocityCommand
    n = cfg.scene.num_envs
    preparation = np.asarray(parameters['preparation_seconds'], dtype=float)
    acceleration = np.asarray(parameters['loaded_acceleration_mps2'], dtype=float)
    distance = np.asarray(parameters['goal_distance_m'], dtype=float)
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in (preparation, acceleration, distance)):
        raise ValueError('Command parameter arrays must match the actual environment count')
    dt = cfg.sim.dt*cfg.decimation
    if (preparation < 3.).any() or np.max(abs(preparation/dt-np.rint(preparation/dt))) > 1.e-5:
        raise ValueError('Preparation must be at least 3 seconds and on the control-step grid')
    if (acceleration <= 0).any() or (distance < 1.).any() or cfg.fixed_speed_mps is None:
        raise ValueError('Require positive acceleration, real distance and explicit fixed task commands')
    offsets = np.rint((preparation-3.)/dt).astype(np.int64).tolist()
    preparation_steps = np.rint(preparation/dt).astype(np.int64).tolist()

    class MigrationVelocityCommand(UnifiedVelocityCommand):
        def __init__(self, command_cfg, env):
            super().__init__(command_cfg, env)
            self.declared_preparation_steps = torch.tensor(preparation_steps, device=self.device)
            self.declared_cycle_offsets = torch.tensor(offsets, device=self.device)

        @property
        def towing_preparation_complete(self):
            return self._env.episode_length_buf >= self.declared_preparation_steps

        def _update_command(self):
            self.cycle_start_step[:] = self.declared_cycle_offsets
            super()._update_command()

    cfg.loaded_acceleration = torch.tensor(acceleration, device=cfg.sim.device, dtype=torch.float32)
    cfg.eval_distance_m = torch.tensor(distance, device=cfg.sim.device, dtype=torch.float32)
    cfg.commands.base_velocity.class_type = MigrationVelocityCommand
    return parameters
