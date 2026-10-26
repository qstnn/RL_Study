"""Progressive straight-line commands for the LightHW flat baseline.

The previous high-speed route sampled the final ``3.2 m/s`` envelope from the
first update.  This command term exposes the same zero-lateral/zero-yaw task
while increasing the absolute forward-speed cap in five fixed lessons.  The
lesson is selected from the global control-step counter, so every environment
in a PPO batch sees the same command curriculum.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .hierarchical_commands import HierarchicalLightHWVelocityCommandCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import UniformThresholdVelocityCommand


class ProgressiveLightHWVelocityCommand(UniformThresholdVelocityCommand):
    """Sample straight commands from 1.0 to 3.2 m/s in five lessons."""

    cfg: "ProgressiveLightHWVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics["speed_curriculum_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["speed_curriculum_max"] = torch.zeros(self.num_envs, device=self.device)

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _lesson(self) -> tuple[int, float]:
        boundaries = self.cfg.stage_start_steps
        speeds = self.cfg.stage_max_speeds
        if len(boundaries) != len(speeds) or not boundaries or boundaries[0] != 0:
            raise ValueError("stage_start_steps and stage_max_speeds must have equal length and start at 0")
        step = int(getattr(self._env, "common_step_counter", 0))
        stage = sum(step >= int(start) for start in boundaries) - 1
        stage = max(0, min(stage, len(speeds) - 1))
        return stage, float(speeds[stage])

    def _resample_command(self, env_ids: Sequence[int]):
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return

        stage, speed_max = self._lesson()
        count = ids.numel()
        draw = torch.rand(count, device=self.device)
        standing = draw < float(self.cfg.zero_command_probability)
        active = ~standing

        # Keep a small nonzero lower bound so the early lessons teach rolling,
        # while the anchor samples make each lesson's upper limit observable.
        minimum = min(float(self.cfg.minimum_abs_speed), speed_max)
        magnitude = torch.empty(count, device=self.device).uniform_(minimum, speed_max)
        anchor = torch.rand(count, device=self.device) < float(self.cfg.anchor_probability)
        magnitude = torch.where(anchor, torch.full_like(magnitude, speed_max), magnitude)
        sign = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)

        self.vel_command_b[ids, :] = 0.0
        self.vel_command_b[ids[active], 0] = sign[active] * magnitude[active]
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = standing
        self.metrics["speed_curriculum_stage"][ids] = float(stage)
        self.metrics["speed_curriculum_max"][ids] = speed_max


@configclass
class ProgressiveLightHWVelocityCommandCfg(HierarchicalLightHWVelocityCommandCfg):
    """Configuration for the staged LightHW speed curriculum."""

    class_type: type = ProgressiveLightHWVelocityCommand
    stage_start_steps: tuple[int, ...] = (0, 28_800, 57_600, 86_400, 115_200)
    stage_max_speeds: tuple[float, ...] = (1.0, 1.5, 2.0, 2.6, 3.2)
    zero_command_probability: float = 0.02
    minimum_abs_speed: float = 0.20
    anchor_probability: float = 0.40

