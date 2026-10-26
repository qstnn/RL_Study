"""Independent PPO-only forward-speed curriculum with explicit stop evidence."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import torch
from isaaclab.utils import configclass

from .ppo_action_curriculum import LightHWLegPositionAction, LightHWLegPositionActionCfg
from .ppo_forward_speed_core import (
    FORWARD_SPEEDS, GATE_PARAMETERS, PROFILE_ID, forward_quality_gate, validate_route10_transfer_state,
)
from .progressive_commands import ProgressiveLightHWVelocityCommand, ProgressiveLightHWVelocityCommandCfg


class LightHWForwardSpeedLegAction(LightHWLegPositionAction):
    """保持原生 PPO 动作映射和阶段 2 腿包络，仅逐步扩大正向速度。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._transfer_source = None
        # 仅用于观测，不参与升级判定、动作映射或 checkpoint 签名。
        # 与质量窗口共用样本掩码，记录命令 vx、实际 vx、实际倾角的和。
        self._monitor_window = torch.zeros(3, 3, device=self.device, dtype=torch.float64)
        self.verbose_curriculum = True
        if set(cfg.gate_parameters) != set(GATE_PARAMETERS):
            raise ValueError("Forward-speed gate parameters are incomplete or contain unknown fields")
        if any(not math.isfinite(value) or value <= 0 for value in cfg.gate_parameters.values()):
            raise ValueError("Forward-speed gate parameters must be positive and finite")

    def _signature_payload(self) -> dict:
        payload = super()._signature_payload()
        command = self._env.command_manager.get_term("base_velocity")
        payload.update(
            profile_id=PROFILE_ID, gate_parameters=deepcopy(self.cfg.gate_parameters),
            observation_action_dimensions={"policy": 60, "critic": 236, "action": 16},
            command_distribution={
                "forward_only": True, "zero_command_probability": command.cfg.zero_command_probability,
                "anchor_probability": command.cfg.anchor_probability,
                "minimum_abs_speed": command.cfg.minimum_abs_speed,
            },
        )
        return payload

    def export_curriculum_state(self) -> dict:
        state = super().export_curriculum_state()
        if self._transfer_source is not None:
            state["transfer_source"] = deepcopy(self._transfer_source)
        return state

    def validate_curriculum_state(self, state: dict):
        """预检完整新路线 checkpoint；不修改动作、命令或任何课程计数。"""
        if not isinstance(state, dict):
            raise ValueError("Forward-speed resume requires complete curriculum metadata")
        expected = self.export_curriculum_state()
        saved_hash = hashlib.sha256(json.dumps(state.get("action_mapping"), sort_keys=True).encode()).hexdigest()
        if (state.get("version") != 1 or state.get("signature") != expected["signature"]
                or saved_hash != state.get("signature")):
            raise ValueError("Checkpoint does not match this forward-speed profile, gates or action mapping")
        restored = deepcopy(self.schedule)
        try:
            restored.load_state_dict(state["schedule"], reset_window=False)
        except (KeyError, TypeError) as exc:
            raise ValueError("Forward-speed checkpoint has incomplete curriculum counters") from exc
        if (state.get("hip_limit_rad") != restored.hip_limits[restored.stage]
                or state.get("leg_limit_rad") != restored.leg_limits[restored.stage]
                or state.get("max_speed_m_s") != restored.max_speeds[restored.stage]):
            raise ValueError("Checkpoint stage and forward-speed action limits disagree")
        steps = state.get("control_steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid forward-speed checkpoint control step count")
        return restored

    def _apply_command_stage(self, *, resample: bool):
        command = self._env.command_manager.get_term("base_velocity")
        command.cfg.ranges.lin_vel_x = (0.0, self.curriculum_max_speed)
        if resample:
            command._resample_command(torch.arange(self.num_envs, device=self.device))

    def _clear_quality_window(self):
        self._window.zero_()
        self._monitor_window.zero_()
        self._episode_stats.zero_()
        self._command_age.zero_()

    def restore_curriculum_state(self, state: dict, training: bool):
        restored = self.validate_curriculum_state(state)
        restored.load_state_dict(state["schedule"], reset_window=training)
        self.schedule = restored
        self._control_steps = state["control_steps"]
        self._transfer_source = deepcopy(state.get("transfer_source"))
        self._set_limits()
        self._training_curriculum = bool(training)
        self.last_quality_metrics = dict(state.get("last_quality_metrics", {}))
        self._clear_quality_window()
        self._apply_command_stage(resample=training)
        print(f"[LightHW forward PPO] restored stage={self.curriculum_stage}, "
              f"vx=0..{self.curriculum_max_speed:.2f} m/s, training={training}", flush=True)

    def validate_route10_transfer(self, state: dict) -> dict:
        return validate_route10_transfer_state(state, self._signature_payload())

    def initialize_route10_transfer(self, state: dict):
        """显式权重迁移后重新开始正向课程；不继承旧门槛、计数或评估窗口。"""
        source = self.validate_route10_transfer(state)
        self.schedule.load_state_dict({
            "stage": 0, "stage_updates": 0, "total_updates": 0,
            "consecutive_passes": 0, "window_updates": 0,
        }, reset_window=True)
        self._transfer_source = source
        self._control_steps = 0
        self._set_limits()
        self._training_curriculum = True
        self.last_quality_metrics = {}
        self._clear_quality_window()
        self._apply_command_stage(resample=True)
        print("[LightHW forward PPO] route 10 stage-2 weights transferred; "
              "new curriculum starts at stage=0, vx=0..0.20 m/s, counters=0", flush=True)

    def record_quality(self):
        if not self._training_curriculum:
            return
        self._control_steps += 1
        env, gate = self._env, self.cfg.gate_parameters
        sensor = env.scene.sensors["contact_forces"]
        if self._wheel_body_ids is None:
            self._wheel_body_ids = [i for i, name in enumerate(sensor.body_names) if name.endswith("_wheel_link")]
            self._nonwheel_body_ids = [i for i, name in enumerate(sensor.body_names) if not name.endswith("_wheel_link")]
            if len(self._wheel_body_ids) != 4:
                raise ValueError("Forward-speed curriculum requires exactly four wheel contact bodies")
        command = env.command_manager.get_command("base_velocity")
        changed = (command - self._last_command).abs().amax(dim=1) > 1.0e-5
        # 普通持久缓冲必须保留：采样在 inference_mode 内，升阶/恢复在外。
        self._command_age.copy_(torch.where(
            changed, torch.zeros_like(self._command_age), self._command_age + env.step_dt,
        ))
        self._last_command.copy_(command)
        settled = self._command_age >= gate["settle_seconds"]
        velocity = self._asset.data.root_lin_vel_b
        yaw_rate = self._asset.data.root_ang_vel_b[:, 2].abs()
        gravity = self._asset.data.projected_gravity_b
        height = self._asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        forces = sensor.data.net_forces_w
        support = (forces[:, self._wheel_body_ids, 2] > gate["wheel_support_force_n"]).all(dim=1)
        body_contact = forces[:, self._nonwheel_body_ids].norm(dim=-1).amax(dim=1) > gate["body_contact_force_n"]
        standing = command[:, 0].abs() < gate["standing_command_threshold"]
        speed_error = torch.where(standing, velocity[:, :2].norm(dim=1), (velocity[:, 0] - command[:, 0]).abs())
        tilt_bad = torch.where(
            standing, gravity[:, 2] > -math.cos(math.radians(gate["stop_tilt_degrees"])),
            gravity[:, 2] > -math.cos(math.radians(gate["moving_tilt_degrees"])),
        )
        # 第五列仅记录实际高度；从奖励移除高度后，也不以高度作为升阶门槛。
        features = torch.stack((torch.ones_like(speed_error), speed_error, support.float(), tilt_bad.float(),
                                height, velocity[:, 1].abs(), yaw_rate, body_contact.float()), dim=1)
        finite = torch.isfinite(velocity).all(dim=1) & torch.isfinite(gravity).all(dim=1)
        finite &= torch.isfinite(height) & torch.isfinite(forces).flatten(1).all(dim=1) & torch.isfinite(yaw_rate)
        finite &= torch.isfinite(self._asset.data.joint_pos).all(dim=1) & torch.isfinite(self._raw_actions).all(dim=1)
        finite &= torch.isfinite(self._asset.data.joint_vel).all(dim=1) & torch.isfinite(command).all(dim=1)
        finite &= torch.isfinite(env.action_manager.get_term("joint_vel").raw_actions).all(dim=1)
        forward = command[:, 0] > gate["standing_command_threshold"]
        fast = forward & (command[:, 0] >= gate["fast_command_fraction"] * self.curriculum_max_speed)
        # fast 是 forward 的子集：上限附近的差表现不可被大量较慢命令掩盖。
        masks = torch.stack((forward, fast, standing), dim=1) & settled[:, None]
        self._window += (masks.float().T @ torch.nan_to_num(features, nan=1e6, posinf=1e6, neginf=1e6)).double()
        tilt_degrees = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0)))
        monitor = torch.stack((command[:, 0], velocity[:, 0], tilt_degrees), dim=1)
        self._monitor_window += (
            masks.float().T @ torch.nan_to_num(monitor, nan=1e6, posinf=1e6, neginf=1e6)
        ).double()
        self._episode_stats += torch.stack((env.reset_buf.sum(), env.reset_terminated.sum(), (~finite).sum())).double()

    def quality_snapshot(self) -> dict[str, float]:
        """读取当前窗口，不清零、不推进课程；无样本组由 samples=0 标识。

        前进和 fast 的速度误差是 vx 的 MAE，停车速度是平面速度模长；
        actual_vx 则始终为有符号的实际纵向速度，不能由命令减 MAE 推算。
        所有组沿用 50 Hz、命令稳定后的质量采样，不代表独立行为验收。
        """
        metrics = {}
        rows = self._window.detach().cpu().tolist()
        monitors = self._monitor_window.detach().cpu().tolist()
        for group, row, monitor in zip(("forward", "fast", "stop"), rows, monitors):
            metrics[f"{group}/samples"] = row[0]
            for key, value in zip(("speed_error", "four_wheel_support", "tilt_bad_fraction", "base_height",
                                   "lateral_speed", "yaw_rate", "body_contact_fraction"), row[1:]):
                metrics[f"{group}/{key}"] = value / max(row[0], 1.0)
            for key, value in zip(("command_vx", "actual_vx", "tilt_deg"), monitor):
                metrics[f"{group}/{key}"] = value / max(row[0], 1.0)
        completed, failed, nonfinite = self._episode_stats.detach().cpu().tolist()
        metrics.update(completed_episodes=completed, failed_episode_fraction=failed / max(completed, 1),
                       nonfinite_samples=nonfinite)
        return metrics

    def advance_curriculum(self) -> dict[str, float]:
        if not self._training_curriculum:
            return {}
        metrics, passed = {}, None
        due = self.schedule.window_updates + 1 >= self.schedule.evaluation_interval
        if due:
            metrics = self.quality_snapshot()
            metrics.update(evaluated_stage=float(self.curriculum_stage),
                           evaluated_max_speed_m_s=self.curriculum_max_speed)
            gate = self.cfg.gate_parameters
            passed = forward_quality_gate(metrics, max_speed=self.curriculum_max_speed,
                                          min_samples=max(gate["min_samples"], self.num_envs * gate["samples_per_env"]),
                                          parameters=gate)
            metrics["window_passed"] = float(passed)
            self.last_quality_metrics = metrics.copy()
            self._window.zero_()
            self._monitor_window.zero_()
            self._episode_stats.zero_()
        old_stage = self.curriculum_stage
        promoted = self.schedule.update(passed)
        if promoted:
            self._set_limits()
            self._apply_command_stage(resample=True)
            self._command_age.zero_()
        metrics.update(stage=float(self.curriculum_stage), total_updates=float(self.schedule.total_updates),
                       stage_updates=float(self.schedule.stage_updates), consecutive_passes=float(self.schedule.consecutive_passes),
                       hip_limit_rad=self.schedule.hip_limits[self.curriculum_stage],
                       leg_limit_rad=self.schedule.leg_limits[self.curriculum_stage], max_speed_m_s=self.curriculum_max_speed)
        if due and self.verbose_curriculum:
            print(f"[LightHW forward PPO] evaluated_stage={old_stage}, promoted={promoted}, "
                  + json.dumps(metrics, sort_keys=True), flush=True)
        return metrics


@configclass
class LightHWForwardSpeedLegActionCfg(LightHWLegPositionActionCfg):
    class_type: type = LightHWForwardSpeedLegAction
    stage_hip_limits: tuple[float, ...] = (0.04,) * len(FORWARD_SPEEDS)
    stage_leg_limits: tuple[float, ...] = (0.10,) * len(FORWARD_SPEEDS)
    stage_max_speeds: tuple[float, ...] = FORWARD_SPEEDS
    min_stage_updates: int = 200
    evaluation_interval: int = 100
    required_passes: int = 2
    gate_parameters: dict = deepcopy(GATE_PARAMETERS)


class LightHWForwardSpeedVelocityCommand(ProgressiveLightHWVelocityCommand):
    """75%正向、25%停车；移动样本一半锚定当前上限，不生成负向命令。"""

    def _lesson(self) -> tuple[int, float]:
        term = self._env.action_manager.get_term("joint_pos")
        return term.curriculum_stage, term.curriculum_max_speed

    def _resample_command(self, env_ids):
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return
        stage, speed_max = self._lesson()
        standing = torch.rand(ids.numel(), device=self.device) < self.cfg.zero_command_probability
        minimum = min(self.cfg.minimum_abs_speed, speed_max)
        magnitude = torch.empty(ids.numel(), device=self.device).uniform_(minimum, speed_max)
        anchor = torch.rand(ids.numel(), device=self.device) < self.cfg.anchor_probability
        magnitude = torch.where(anchor, torch.full_like(magnitude, speed_max), magnitude)
        self.cfg.ranges.lin_vel_x = (0.0, speed_max)
        self.vel_command_b[ids, :] = 0.0
        self.vel_command_b[ids[~standing], 0] = magnitude[~standing]
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = standing
        self.metrics["speed_curriculum_stage"][ids] = float(stage)
        self.metrics["speed_curriculum_max"][ids] = speed_max


@configclass
class LightHWForwardSpeedVelocityCommandCfg(ProgressiveLightHWVelocityCommandCfg):
    class_type: type = LightHWForwardSpeedVelocityCommand
    stage_max_speeds: tuple[float, ...] = FORWARD_SPEEDS
    zero_command_probability: float = 0.25
    minimum_abs_speed: float = 0.08
    anchor_probability: float = 0.50
