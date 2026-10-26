"""Bounded leg targets and performance-gated curriculum, trained only by PPO.

All 16 actions always come from the actor.  There is no demonstration loss,
wheel-speed feedforward or frozen leg controller in this route.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .ppo_curriculum_core import LegCurriculumSchedule, quality_gate
from .progressive_commands import ProgressiveLightHWVelocityCommand, ProgressiveLightHWVelocityCommandCfg


class LightHWLegPositionAction(JointPositionAction):
    """Apply ``q_default + limit(stage) * tanh(actor_action)`` to all 12 legs.

    The default posture is a coordinate offset, not a trajectory generator.
    The actor controls every target from the first rollout.  Promotion only
    happens between PPO updates, so each rollout has a fixed action mapping.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not cfg.use_default_offset:
            raise ValueError("LightHW curriculum requires default-pose-relative targets")
        self.schedule = LegCurriculumSchedule(
            tuple(cfg.stage_hip_limits), tuple(cfg.stage_leg_limits), tuple(cfg.stage_max_speeds),
            cfg.min_stage_updates, cfg.evaluation_interval, cfg.required_passes,
        )
        self._training_curriculum = True
        self._control_steps = 0
        self._hip_mask = torch.tensor([name.endswith("_hip_joint") for name in self._joint_names], device=self.device)
        self._limits = torch.empty(self.action_dim, device=self.device)
        self._set_limits()
        self._processed_actions = self._offset.clone()
        self._previous_targets = self._processed_actions.clone()
        self._previous_wheel_targets = torch.zeros(self.num_envs, 4, device=self.device)
        self._rate_cache = torch.zeros(self.num_envs, device=self.device)
        self._rate_step = -1
        self._command_age = torch.zeros(self.num_envs, device=self.device)
        self._last_command = torch.zeros(self.num_envs, 3, device=self.device)
        # columns: count, speed MAE, all-wheel support, bad tilt, bad height,
        # lateral speed, absolute yaw rate, non-wheel contact.
        self._window = torch.zeros(3, 8, device=self.device, dtype=torch.float64)
        self._episode_stats = torch.zeros(3, device=self.device, dtype=torch.float64)
        self._wheel_body_ids = None
        self._nonwheel_body_ids = None
        self.last_quality_metrics = {}

    @property
    def curriculum_stage(self) -> int:
        return self.schedule.stage

    @property
    def curriculum_max_speed(self) -> float:
        return self.schedule.max_speeds[self.curriculum_stage]

    def _set_limits(self):
        self._limits[:] = torch.where(
            self._hip_mask, self.schedule.hip_limits[self.curriculum_stage],
            self.schedule.leg_limits[self.curriculum_stage],
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._previous_targets.copy_(self._processed_actions)
        targets = self._offset + torch.tanh(actions) * self._limits
        if self.cfg.clip is not None:
            targets = targets.clamp(self._clip[:, :, 0], self._clip[:, :, 1])
        # RSL rollout 在 inference_mode 内执行；保留初始化时的普通持久缓冲，
        # 以便 rollout 外的 reset 也能写入。变换数值和动作映射保持不变。
        self._processed_actions.copy_(targets)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        # Reset only this episode's physical-action history.  Never reset the
        # global curriculum or its accumulated evaluation window here.
        if hasattr(self, "_previous_targets"):
            self._processed_actions[ids] = self._offset[ids]
            self._previous_targets[ids] = self._offset[ids]
            self._previous_wheel_targets[ids] = 0
            self._command_age[ids] = 0

    def validate_action_layout(self):
        manager = self._env.action_manager
        wheel = manager.get_term("joint_vel")
        if list(manager.active_terms) != ["joint_pos", "joint_vel"] or self.action_dim != 12 or wheel.action_dim != 4:
            raise ValueError("LightHW PPO curriculum requires 12 position actions followed by 4 wheel velocities")
        if sum(self._hip_mask.tolist()) != 4 or any("wheel" in name for name in self._joint_names):
            raise ValueError("Unexpected LightHW leg action joint selection")
        if any(not name.endswith("_wheel_joint") for name in wheel._joint_names):
            raise ValueError("Unexpected LightHW wheel action joint selection")

    def curriculum_initial_noise_std(self) -> list[float]:
        self.validate_action_layout()
        return [self.cfg.leg_noise_std] * 12 + [self.cfg.wheel_noise_std] * 4

    def set_curriculum_training(self, training: bool):
        self._training_curriculum = bool(training)

    def _signature_payload(self) -> dict:
        wheel = self._env.action_manager.get_term("joint_vel")
        return {
            "mapping": "default_plus_tanh_limits_v1", "leg_joints": list(self._joint_names),
            "wheel_joints": list(wheel._joint_names),
            "default_positions": [round(x, 6) for x in self._offset[0].detach().cpu().tolist()],
            "absolute_leg_clip": self.cfg.clip, "wheel_scale": wheel.cfg.scale, "wheel_clip": wheel.cfg.clip,
            "hip_limits": list(self.schedule.hip_limits), "leg_limits": list(self.schedule.leg_limits),
            "max_speeds": list(self.schedule.max_speeds), "physics_dt": self._env.physics_dt,
            "control_dt": self._env.step_dt, "min_stage_updates": self.schedule.min_stage_updates,
            "evaluation_interval": self.schedule.evaluation_interval, "required_passes": self.schedule.required_passes,
        }

    def export_curriculum_state(self) -> dict:
        payload = self._signature_payload()
        return {
            "version": 1, "signature": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "action_mapping": payload, "schedule": self.schedule.state_dict(), "control_steps": self._control_steps,
            "hip_limit_rad": self.schedule.hip_limits[self.curriculum_stage],
            "leg_limit_rad": self.schedule.leg_limits[self.curriculum_stage],
            "max_speed_m_s": self.curriculum_max_speed, "last_quality_metrics": self.last_quality_metrics,
        }

    def restore_curriculum_state(self, state: dict, training: bool):
        expected = self.export_curriculum_state()
        saved_hash = hashlib.sha256(json.dumps(state.get("action_mapping"), sort_keys=True).encode()).hexdigest()
        if (state.get("version") != 1 or state.get("signature") != expected["signature"]
                or saved_hash != state.get("signature")):
            raise ValueError("Checkpoint does not match this LightHW PPO action mapping/curriculum")
        restored = deepcopy(self.schedule)
        restored.load_state_dict(state["schedule"], reset_window=training)
        if (
            state.get("hip_limit_rad") != restored.hip_limits[restored.stage]
            or state.get("leg_limit_rad") != restored.leg_limits[restored.stage]
            or state.get("max_speed_m_s") != restored.max_speeds[restored.stage]
        ):
            raise ValueError("Checkpoint stage and physical action limits disagree")
        steps = state["control_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid checkpoint control step counter")
        self.schedule = restored
        self._control_steps = steps
        self._set_limits()
        self._training_curriculum = bool(training)
        self.last_quality_metrics = dict(state.get("last_quality_metrics", {}))
        self._window.zero_()
        self._episode_stats.zero_()
        self._command_age.zero_()
        command = self._env.command_manager.get_term("base_velocity")
        command.cfg.ranges.lin_vel_x = (-self.curriculum_max_speed, self.curriculum_max_speed)
        if training:
            # The initial reset happened before checkpoint load.  Replace its
            # stage-0 commands before collecting any resumed rollout.
            command._resample_command(torch.arange(self.num_envs, device=self.device))
        print(
            f"[LightHW PPO curriculum] restored stage={self.curriculum_stage}, "
            f"hip=+/-{state['hip_limit_rad']:.3f} rad, thigh/calf=+/-{state['leg_limit_rad']:.3f} rad, "
            f"vx=+/-{self.curriculum_max_speed:.2f}, training={training}", flush=True,
        )

    def record_quality(self):
        if not self._training_curriculum:
            return
        self._control_steps += 1
        env = self._env
        sensor = env.scene.sensors["contact_forces"]
        if self._wheel_body_ids is None:
            self._wheel_body_ids = [i for i, name in enumerate(sensor.body_names) if name.endswith("_wheel_link")]
            self._nonwheel_body_ids = [i for i, name in enumerate(sensor.body_names) if not name.endswith("_wheel_link")]
            if len(self._wheel_body_ids) != 4:
                raise ValueError("Curriculum requires exactly four LightHW wheel contact bodies")
        command = env.command_manager.get_command("base_velocity")
        changed = (command - self._last_command).abs().amax(dim=1) > 1.0e-5
        # 不把持久缓冲替换成 inference tensor：PPO 更新之后，课程升级会
        # 在 inference_mode 外对它 zero_()。copy_ 保留原来的普通张量。
        self._command_age.copy_(
            torch.where(changed, torch.zeros_like(self._command_age), self._command_age + env.step_dt)
        )
        self._last_command.copy_(command)
        settled = self._command_age >= 2.0
        velocity = self._asset.data.root_lin_vel_b
        yaw_rate = self._asset.data.root_ang_vel_b[:, 2].abs()
        gravity = self._asset.data.projected_gravity_b
        height = self._asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        forces = sensor.data.net_forces_w
        support = (forces[:, self._wheel_body_ids, 2] > 5.0).all(dim=1)
        body_contact = forces[:, self._nonwheel_body_ids].norm(dim=-1).amax(dim=1) > 1.0
        standing = command[:, 0].abs() < 0.05
        speed_error = torch.where(standing, velocity[:, :2].norm(dim=1), (velocity[:, 0] - command[:, 0]).abs())
        features = torch.stack((
            torch.ones_like(speed_error), speed_error, support.float(),
            (gravity[:, 2] > -math.cos(math.radians(5.0))).float(),
            ((height < 0.385) | (height > 0.440)).float(), velocity[:, 1].abs(), yaw_rate, body_contact.float(),
        ), dim=1)
        finite = torch.isfinite(velocity).all(dim=1) & torch.isfinite(gravity).all(dim=1)
        finite &= torch.isfinite(height) & torch.isfinite(forces).flatten(1).all(dim=1) & torch.isfinite(yaw_rate)
        finite &= torch.isfinite(self._asset.data.joint_pos).all(dim=1) & torch.isfinite(self._raw_actions).all(dim=1)
        finite &= torch.isfinite(self._asset.data.joint_vel).all(dim=1) & torch.isfinite(command).all(dim=1)
        finite &= torch.isfinite(env.action_manager.get_term("joint_vel").raw_actions).all(dim=1)
        masks = torch.stack((command[:, 0] > 0.05, command[:, 0] < -0.05, standing), dim=1) & settled[:, None]
        # Collect before automatic reset, so falls cannot disappear behind a
        # freshly reset upright robot.  Nonfinite samples always fail a window.
        self._window += (masks.float().T @ torch.nan_to_num(features, nan=1e6, posinf=1e6, neginf=1e6)).double()
        self._episode_stats += torch.stack((
            env.reset_buf.sum(), env.reset_terminated.sum(), (~finite).sum(),
        )).double()

    def advance_curriculum(self) -> dict[str, float]:
        if not self._training_curriculum:
            return {}
        metrics = {}
        due = self.schedule.window_updates + 1 >= self.schedule.evaluation_interval
        passed = None
        if due:
            values = self._window.detach().cpu().tolist()
            for group, row in zip(("forward", "reverse", "stop"), values):
                metrics[f"{group}/samples"] = row[0]
                for key, value in zip(("speed_error", "four_wheel_support", "tilt_bad_fraction", "height_bad_fraction",
                                       "lateral_speed", "yaw_rate", "body_contact_fraction"), row[1:]):
                    metrics[f"{group}/{key}"] = value / max(row[0], 1.0)
            completed, failed, nonfinite = self._episode_stats.detach().cpu().tolist()
            metrics.update(completed_episodes=completed, failed_episode_fraction=failed / max(completed, 1),
                           nonfinite_samples=nonfinite)
            passed = quality_gate(metrics, min_samples=max(100, self.num_envs * 10))
            metrics["window_passed"] = float(passed)
            self.last_quality_metrics = metrics.copy()
            self._window.zero_()
            self._episode_stats.zero_()
        old_stage = self.curriculum_stage
        promoted = self.schedule.update(passed)
        if promoted:
            self._set_limits()
            command = self._env.command_manager.get_term("base_velocity")
            command.cfg.ranges.lin_vel_x = (-self.curriculum_max_speed, self.curriculum_max_speed)
            # A new assessment window measures only the new stage.  Existing
            # velocity commands change on their normal 8 s resampling boundary.
            self._command_age.zero_()
        metrics.update(stage=float(self.curriculum_stage), total_updates=float(self.schedule.total_updates),
                       stage_updates=float(self.schedule.stage_updates), consecutive_passes=float(self.schedule.consecutive_passes),
                       hip_limit_rad=self.schedule.hip_limits[self.curriculum_stage],
                       leg_limit_rad=self.schedule.leg_limits[self.curriculum_stage], max_speed_m_s=self.curriculum_max_speed)
        if due:
            print(f"[LightHW PPO curriculum] evaluated_stage={old_stage}, promoted={promoted}, "
                  + json.dumps(metrics, sort_keys=True), flush=True)
        return metrics


@configclass
class LightHWLegPositionActionCfg(JointPositionActionCfg):
    class_type: type = LightHWLegPositionAction
    stage_hip_limits: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.08, 0.08)
    stage_leg_limits: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 0.20, 0.20)
    stage_max_speeds: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.40, 0.60)
    min_stage_updates: int = 300
    evaluation_interval: int = 100
    required_passes: int = 3
    leg_noise_std: float = 0.30
    wheel_noise_std: float = 0.20


class LightHWActionCurriculumVelocityCommand(ProgressiveLightHWVelocityCommand):
    def _lesson(self) -> tuple[int, float]:
        term = self._env.action_manager.get_term("joint_pos")
        return term.curriculum_stage, term.curriculum_max_speed


@configclass
class LightHWActionCurriculumVelocityCommandCfg(ProgressiveLightHWVelocityCommandCfg):
    class_type: type = LightHWActionCurriculumVelocityCommand


def record_ppo_curriculum_metrics(env) -> torch.Tensor:
    """每控制步记录课程验收数据；返回全零，不参与 PPO 奖励塑形。

    在 termination 已计算、自动 reset 尚未发生时调用，避免遗漏摔倒。
    配置权重需为 1，防止 Isaac Lab 跳过零权重项。
    """
    env.action_manager.get_term("joint_pos").record_quality()
    return torch.zeros(env.num_envs, device=env.device)


def bounded_action_rate_l2(env) -> torch.Tensor:
    """每控制步惩罚实际目标变化，返回无量纲平方和；腿按0.14rad、轮按10rad/s归一。

    使用经过 tanh/限幅后的目标，避免动作包络改变时原始 action 差分失去物理含义。
    reset 时清理每个环境历史，重复计算同一步不会二次更新历史。
    """
    leg = env.action_manager.get_term("joint_pos")
    wheel_targets = env.action_manager.get_term("joint_vel").processed_actions
    if leg._rate_step != env.common_step_counter:
        leg._rate_cache.copy_(((leg.processed_actions - leg._previous_targets) / 0.14).square().sum(dim=1))
        leg._rate_cache += ((wheel_targets - leg._previous_wheel_targets) / 10.0).square().sum(dim=1)
        leg._previous_wheel_targets.copy_(wheel_targets)
        leg._rate_step = env.common_step_counter
    return leg._rate_cache


def weakest_wheel_support(env, sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    """每控制步奖励最弱轮支撑；返回最小法向力/阈值，裁剪至[0,1]。

    四轮都承载才获得满分；超过阈值后不追求等载，允许起停的前后载荷转移。
    """
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (force.amin(dim=1) / threshold).clamp(0.0, 1.0)


def base_height_band_l2(env, minimum: float = 0.395, maximum: float = 0.425,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """平地每控制步约束受力后的机身高度；容许带内为0，带外返回距离平方(m²)。

    保留原默认姿态；容许PD受力下沉，不要求每一步精确达到名义0.416m。
    """
    height = env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height - height.clamp(minimum, maximum)).square()
