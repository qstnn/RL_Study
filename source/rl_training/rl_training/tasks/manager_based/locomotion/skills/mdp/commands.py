# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# command 是任务意图，不是策略动作。例如技能编号告诉同一个策略“当前做哪种技能”，
# phase 告诉策略动作进行到哪个时刻。


class SkillCommand(CommandTerm):
    """离散技能命令，以及归一化的阶段、目标位置和支撑对参数。

    命令向量布局：
        [skill_one_hot..., phase_sin, phase_cos, target_x, target_y, target_z, support_pair]
    """

    cfg: "SkillCommandCfg"

    def __init__(self, cfg: "SkillCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._skill_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, cfg.num_skills + 6, device=self.device)
        self.metrics["skill_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def __str__(self) -> str:
        return (
            "SkillCommand:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tSkill ids: {self.cfg.skill_ids}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        )

    def _update_metrics(self):
        self.metrics["skill_id"] = self._skill_ids.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        skill_choices = torch.tensor(self.cfg.skill_ids, dtype=torch.long, device=self.device)
        sampled = torch.randint(len(self.cfg.skill_ids), (len(env_ids),), device=self.device)
        self._skill_ids[env_ids] = skill_choices[sampled]

        self._command[env_ids] = 0.0
        self._command[env_ids, self._skill_ids[env_ids]] = 1.0

        low = torch.tensor(self.cfg.target_pos_range[0], device=self.device)
        high = torch.tensor(self.cfg.target_pos_range[1], device=self.device)
        target = low + torch.rand((len(env_ids), 3), device=self.device) * (high - low)
        self._command[env_ids, self.cfg.num_skills + 2 : self.cfg.num_skills + 5] = target

        support_pairs = torch.tensor(self.cfg.support_pairs, dtype=torch.float32, device=self.device)
        sampled_support = torch.randint(len(self.cfg.support_pairs), (len(env_ids),), device=self.device)
        self._command[env_ids, self.cfg.num_skills + 5] = support_pairs[sampled_support]

    def _update_command(self):
        phase = self._env.episode_length_buf.float() * self._env.step_dt / self.cfg.cycle_time
        self._command[:, self.cfg.num_skills] = torch.sin(2.0 * torch.pi * phase)
        self._command[:, self.cfg.num_skills + 1] = torch.cos(2.0 * torch.pi * phase)


@configclass
class SkillCommandCfg(CommandTermCfg):
    """LightHW 技能命令配置；修改字段会改变策略观测到的任务条件。"""

    class_type: type = SkillCommand

    num_skills: int = 5
    skill_ids: list[int] = [0]
    cycle_time: float = 2.0

    target_pos_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    support_pairs: list[int] = [0]
