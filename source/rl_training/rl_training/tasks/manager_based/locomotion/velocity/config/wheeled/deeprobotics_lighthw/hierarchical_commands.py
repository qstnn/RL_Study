"""分层速度命令，用于在保留 3.5 m/s 样本的同时训练静止与中低速稳定性。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import (
    UniformThresholdVelocityCommand,
    UniformThresholdVelocityCommandCfg,
)


class HierarchicalLightHWVelocityCommand(UniformThresholdVelocityCommand):
    """按 standing/low/mid/high 四个层级采样命令。

    高速层始终采样 2.5--3.5 m/s，并保留一部分精确 3.5 m/s 直行锚点，
    因此分层训练不会因为增加静止样本而删除原有高速能力。
    """

    cfg: "HierarchicalLightHWVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Logged on episode reset; this does not enter the policy observation.
        self.metrics["hierarchy_stage"] = torch.zeros(self.num_envs, device=self.device)

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids: Sequence[int]):
        # Initialize all fields and metrics exactly as the existing command term does.
        super()._resample_command(env_ids)
        ids = self._env_ids_tensor(env_ids)
        count = ids.numel()
        if count == 0:
            return

        probabilities = torch.tensor(self.cfg.stage_probabilities, device=self.device, dtype=torch.float32)
        probabilities = probabilities / torch.sum(probabilities)
        cumulative = torch.cumsum(probabilities, dim=0)
        draw = torch.rand(count, device=self.device)
        stage = torch.bucketize(draw, cumulative[:-1])

        # Each non-standing stage keeps the full lateral/yaw command envelope.
        local = torch.empty(count, device=self.device)
        self.vel_command_b[ids, 1] = local.uniform_(*self.cfg.lateral_range)
        self.vel_command_b[ids, 2] = local.uniform_(*self.cfg.yaw_range)

        standing = stage == 0
        low = stage == 1
        mid = stage == 2
        high = stage == 3

        self.vel_command_b[ids[low], 0] = local[low].uniform_(*self.cfg.low_speed_range)
        self.vel_command_b[ids[mid], 0] = local[mid].uniform_(*self.cfg.mid_speed_range)

        if torch.any(high):
            high_ids = ids[high]
            high_count = int(high.sum().item())
            sign = torch.where(torch.rand(high_count, device=self.device) < 0.5, -1.0, 1.0)
            anchor = torch.rand(high_count, device=self.device) < self.cfg.high_speed_anchor_ratio
            magnitude = local[high].uniform_(self.cfg.high_speed_min, self.cfg.high_speed_max)
            magnitude = torch.where(anchor, torch.full_like(magnitude, self.cfg.high_speed_max), magnitude)
            self.vel_command_b[high_ids, 0] = sign * magnitude
            # High-speed anchors are straight rolling commands; do not overwrite
            # their zero yaw with the heading controller on the next update.
            self.vel_command_b[high_ids, 2] = 0.0
            self.is_heading_env[high_ids] = False

        # A standing stage is explicitly zero in all three command dimensions.
        self.vel_command_b[ids[standing], :] = 0.0
        # The hierarchy, rather than the parent random gate, owns standing
        # assignment; this prevents the 20% standing layer from being doubled.
        self.is_standing_env[ids] = standing
        self.is_heading_env[ids[standing]] = False

        # Keep stage labels available for future diagnostics without changing the
        # observation or action interface.
        if not hasattr(self, "hierarchical_stage"):
            self.hierarchical_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hierarchical_stage[ids] = stage
        self.metrics["hierarchy_stage"][ids] = stage.float()


@configclass
class HierarchicalLightHWVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for the LightHW command hierarchy."""

    class_type: type = HierarchicalLightHWVelocityCommand
    # standing, low, mid, high; the high-speed layer keeps 40% of samples.
    stage_probabilities: tuple[float, float, float, float] = (0.20, 0.20, 0.20, 0.40)
    lateral_range: tuple[float, float] = (-0.15, 0.15)
    yaw_range: tuple[float, float] = (-0.30, 0.30)
    low_speed_range: tuple[float, float] = (-0.80, 0.80)
    mid_speed_range: tuple[float, float] = (-1.80, 1.80)
    high_speed_min: float = 2.50
    high_speed_max: float = 3.50
    high_speed_anchor_ratio: float = 0.25
