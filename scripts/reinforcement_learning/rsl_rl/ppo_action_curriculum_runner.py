# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""LightHW 原生 PPO 动作课程的日志和检查点适配，不改动 PPO 损失或采样。"""

from __future__ import annotations

import math
import json
import statistics
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from episode_metrics import CompleteEpisodeMetricsRunner
from lighthw_training_monitor import compact_native_summary, format_training_summary, quality_checks


def curriculum_command_bounds(env) -> torch.Tensor:
    """读取恢复后的命令范围，避免回放沿用加载 checkpoint 前的配置副本。"""
    ranges = env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges
    return torch.tensor(
        [ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z],
        device=env.unwrapped.device,
        dtype=torch.float32,
    )


def restore_curriculum_keyboard(controller, env) -> torch.Tensor:
    """重建全部运动按键的幅度；保留 payload 等附加回调，清除加载期间的旧按键状态。"""
    bounds = curriculum_command_bounds(env)
    limits = bounds[:, 1].detach().cpu().tolist()
    controller.v_x_sensitivity, controller.v_y_sensitivity, controller.omega_z_sensitivity = limits
    controller._create_key_bindings()
    controller._INPUT_KEY_MAPPING.update({
        "W": np.asarray([limits[0], 0.0, 0.0]),
        "S": np.asarray([-limits[0], 0.0, 0.0]),
        "A": np.asarray([0.0, limits[1], 0.0]),
        "D": np.asarray([0.0, -limits[1], 0.0]),
        "Q": np.asarray([0.0, 0.0, limits[2]]),
        "E": np.asarray([0.0, 0.0, -limits[2]]),
    })
    controller.reset()
    print(f"[LightHW PPO curriculum] keyboard bounds restored: {bounds.detach().cpu().tolist()}", flush=True)
    return bounds


class LightHWPPOActionCurriculumRunner(CompleteEpisodeMetricsRunner):
    """每次 PPO 更新后推进腿部动作课程，并保存训练/回放共用的动作语义。"""

    _CHECKPOINT_KEY = "lighthw_action_curriculum"

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", *, curriculum_training=True):
        if not train_cfg.get("lighthw_ppo_action_curriculum", False):
            raise ValueError("LightHW action curriculum runner requires its explicit task configuration flag.")
        if train_cfg["algorithm"]["class_name"] != "PPO":
            raise ValueError("LightHW action curriculum supports native PPO only.")
        if curriculum_training and log_dir is None:
            raise ValueError("LightHW action curriculum training requires logging for the update hook.")

        self._curriculum_training = bool(curriculum_training)
        self._lighthw_forward_speed = bool(train_cfg.get("lighthw_forward_speed", False))
        self._compact_console = (self._lighthw_forward_speed and self._curriculum_training
                                 and bool(train_cfg.get("lighthw_compact_console", True)))
        self._console_interval = train_cfg.get("lighthw_console_interval", 10)
        if (isinstance(self._console_interval, bool) or not isinstance(self._console_interval, int)
                or self._console_interval < 1):
            raise ValueError("lighthw_console_interval must be a positive integer (PPO updates).")
        self._curriculum_initialized = not self._lighthw_forward_speed
        self._route10_transfer_source = None
        self._curriculum_term = env.unwrapped.action_manager.get_term("joint_pos")
        self._curriculum_term.validate_action_layout()
        self._curriculum_term.set_curriculum_training(self._curriculum_training)
        if self._lighthw_forward_speed:
            self._curriculum_term.verbose_curriculum = not self._compact_console

        # 检查实例化后的真实张量，不依赖配置文件里预估的观测维度。
        policy_obs, extras = env.get_observations()
        critic_obs = extras.get("observations", {}).get("critic")
        dimensions = (
            int(policy_obs.shape[-1]),
            int(critic_obs.shape[-1]) if critic_obs is not None else None,
            int(env.num_actions),
        )
        if dimensions != (60, 236, 16):
            raise ValueError(f"LightHW PPO action curriculum requires policy=60, critic=236, action=16; got {dimensions}.")

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if self.is_distributed or self.disable_logs:
            raise ValueError("LightHW action curriculum currently requires a single process with logging enabled.")
        if not hasattr(self.alg.policy, "log_std"):
            raise ValueError("LightHW action curriculum requires policy.noise_std_type='log'.")

        # RSL-RL 3.0.1 不接受列表形式的 init_noise_std。仅在新建网络时按关节顺序
        # 写入逐维初值；之后 load() 将完整恢复已学习的 log_std，不再次重置。
        noise = torch.as_tensor(
            self._curriculum_term.curriculum_initial_noise_std(),
            device=self.device,
            dtype=self.alg.policy.log_std.dtype,
        )
        if noise.shape != self.alg.policy.log_std.shape or not torch.all(torch.isfinite(noise) & (noise > 0)):
            raise ValueError("LightHW initial action noise must contain 16 finite positive values in action order.")
        with torch.no_grad():
            self.alg.policy.log_std.copy_(torch.log(noise))
        print("[LightHW PPO curriculum] policy=60, critic=236, action=16", flush=True)
        print(
            f"[LightHW PPO curriculum] training={self._curriculum_training}; "
            f"initial_action_std={noise.detach().cpu().tolist()}",
            flush=True,
        )
        if self._compact_console:
            print(f"[LightHW monitor] 每 {self._console_interval} 次 PPO 更新显示窗口摘要；"
                  "完整评估及首末次更新立即显示。完整奖励/损失保留在 TensorBoard。", flush=True)

    def _require_initialized_curriculum(self):
        if not self._curriculum_initialized:
            raise ValueError(
                "LightHW forward-speed policy has not been initialized: use the checked route 10 "
                "warm start for training, or load a checkpoint from the same forward-speed task."
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """禁止新任务绕过来源检查从随机网络开始采样。"""
        self._require_initialized_curriculum()
        return super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

    def get_inference_policy(self, device=None):
        self._require_initialized_curriculum()
        return super().get_inference_policy(device=device)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """一次更新只推进一次；仍由原生 runner 完成 rollout、PPO 更新和保存时序。"""
        compact = getattr(self, "_compact_console", False)
        if compact:
            with compact_native_summary(width):
                super().log(locs, width=width, pad=pad)
        else:
            super().log(locs, width=width, pad=pad)
        if not self._curriculum_training:
            return
        context, snapshot = None, None
        if compact:
            term, schedule = self._curriculum_term, self._curriculum_term.schedule
            due = schedule.window_updates + 1 >= schedule.evaluation_interval
            show = (due or (schedule.total_updates + 1) % self._console_interval == 0
                    or locs["it"] == locs["start_iter"] or locs["it"] == locs["tot_iter"] - 1)
            if show:
                # 当前 rollout 已采完，但课程尚未计数/清窗；诊断始终属于此旧阶段。
                stage, gate = schedule.stage, term.cfg.gate_parameters
                context = dict(stage=stage, max_speed=term.curriculum_max_speed, due=due,
                    window_updates=schedule.window_updates + 1, evaluation_interval=schedule.evaluation_interval,
                    min_stage_updates=schedule.min_stage_updates, required_passes=schedule.required_passes,
                    last_stage=len(schedule.max_speeds) - 1,
                    next_speed=schedule.max_speeds[min(stage + 1, len(schedule.max_speeds) - 1)],
                    gate_parameters=gate, min_samples=max(gate["min_samples"], term.num_envs * gate["samples_per_env"]))
                # 完整窗口使用 advance 返回的同一份快照，避免重复 CPU/GPU 同步。
                if not due:
                    snapshot = term.quality_snapshot()
        metrics = self._curriculum_term.advance_curriculum()
        values = {key: float(value) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"LightHW action curriculum returned non-finite metrics: {values}")
        for key, value in values.items():
            self.writer.add_scalar(f"Curriculum/{key}", value, locs["it"])
        if compact:
            if context is not None:
                self._log_forward_summary(values if context["due"] else snapshot, context, values, locs)
            return
        print(
            "[LightHW PPO curriculum] " + ", ".join(f"{key}={value:.6g}" for key, value in values.items()),
            flush=True,
        )

    def _log_forward_summary(self, snapshot, context, values, locs):
        """定期展示当前窗口；新 Monitor 标签与每轮原生标量同时保存。"""
        for key, value in snapshot.items():
            if math.isfinite(value):
                self.writer.add_scalar(f"Monitor/window/{key}", value, locs["it"])
        checks = quality_checks(snapshot, max_speed=context["max_speed"],
            parameters=context["gate_parameters"], min_samples=context["min_samples"])
        for check in checks:
            self.writer.add_scalar(f"Monitor/limit/{check.key}", check.limit, locs["it"])
            # 无样本的均值不能解释为真实达标余量。
            has_samples = ('/' not in check.key or check.key.endswith('/samples')
                           or snapshot.get(check.key.split('/')[0] + '/samples', 0) > 0)
            if check.valid and has_samples:
                self.writer.add_scalar(f"Monitor/gap/{check.key}", check.gap, locs["it"])
        for name in ("stage", "max_speed", "window_updates"):
            self.writer.add_scalar(f"Monitor/window_context/{name}", context[name], locs["it"])
        # 腿 tanh 映射非线性，raw std 不能直接声称为关节角噪声；轮速标注裁剪前。
        std = self.alg.policy.log_std.detach().exp()
        leg_std = std[:12].mean().item()
        wheel = self.env.unwrapped.action_manager.get_term("joint_vel")
        scale = torch.as_tensor(wheel._scale, device=std.device, dtype=std.dtype)
        wheel_std = (std[12:] * scale.abs()).mean().item()
        self.writer.add_scalar("Monitor/noise/leg_raw_std", leg_std, locs["it"])
        self.writer.add_scalar("Monitor/noise/wheel_target_std_before_clip_rad_s", wheel_std, locs["it"])
        completed_here = locs["it"] - locs["start_iter"] + 1
        remaining = max(0, locs["tot_iter"] - locs["it"] - 1)
        eta_seconds = self.tot_time / completed_here * remaining
        eta_hours, eta_remainder = divmod(int(eta_seconds), 3600)
        eta_minutes, eta_seconds = divmod(eta_remainder, 60)
        training = dict(iteration=locs["it"], last_iteration=locs["tot_iter"] - 1,
            reward=f"{statistics.mean(locs['rewbuffer']):.2f}" if locs["rewbuffer"] else "--",
            leg_std=leg_std, wheel_std=wheel_std, learning_rate=self.alg.learning_rate,
            fps=int(self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
                    / (locs["collection_time"] + locs["learn_time"])),
            eta=f"{eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}",
            grouped_noise=getattr(self, "_lighthw_grouped_noise", False))
        progress = {key: int(values[key]) for key in ("stage", "total_updates", "stage_updates", "consecutive_passes")}
        progress["max_speed"] = values["max_speed_m_s"]
        print(format_training_summary(metrics=snapshot, context=context, progress=progress, training=training), flush=True)

    def save(self, path: str, infos=None):
        """将课程与网络保存在同一 checkpoint，避免断点续训和回放动作范围漂移。"""
        self._require_initialized_curriculum()
        if infos is not None and not isinstance(infos, Mapping):
            raise TypeError("LightHW curriculum checkpoint infos must be a mapping or None.")
        checkpoint_infos = dict(infos or {})
        checkpoint_infos[self._CHECKPOINT_KEY] = self._curriculum_term.export_curriculum_state()
        if self._route10_transfer_source is not None:
            checkpoint_infos["lighthw_route10_transfer"] = dict(self._route10_transfer_source)
        return super().save(path, infos=checkpoint_infos)

    def load(self, path: str, load_optimizer: bool = True):
        """拒绝旧任务 checkpoint；训练继续课程，回放固定在保存时的动作范围。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        state = infos.get(self._CHECKPOINT_KEY) if isinstance(infos, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError(
                "Checkpoint has no LightHW PPO action curriculum state. "
                "Start this task from scratch or resume a checkpoint produced by the same curriculum task."
            )
        # 在权重写入前拒绝其他课程；完整状态与计数校验仍由 action term 负责。
        expected = self._curriculum_term.export_curriculum_state()
        if state.get("version") != expected["version"] or state.get("signature") != expected["signature"]:
            raise ValueError("Checkpoint does not match this LightHW PPO action mapping/curriculum.")
        if self._lighthw_forward_speed:
            self._curriculum_term.validate_curriculum_state(dict(state))
        del checkpoint
        loaded_infos = super().load(path, load_optimizer=load_optimizer)
        if self._lighthw_forward_speed and load_optimizer:
            # RSL-RL 3.0.1 恢复 Adam 的 lr，但未同步 PPO 自己的 adaptive lr。
            # 同任务续训同步二者，避免下次更新从配置初值重新调节学习率。
            rates = {float(group["lr"]) for group in self.alg.optimizer.param_groups}
            if len(rates) != 1 or not all(math.isfinite(rate) and rate > 0 for rate in rates):
                raise ValueError("Forward-speed resume requires one finite positive optimizer learning rate.")
            self.alg.learning_rate = rates.pop()
        self._curriculum_term.restore_curriculum_state(dict(state), training=self._curriculum_training)
        self._route10_transfer_source = infos.get("lighthw_route10_transfer")
        self._curriculum_initialized = True
        print(
            "[LightHW PPO curriculum] restored checkpoint action ranges; "
            + ("training progression enabled." if self._curriculum_training else "playback progression frozen."),
            flush=True,
        )
        return loaded_infos

    def load_route10_checkpoint(self, path: str):
        """只供 route 11 首次完整迁移：检查原阶段/映射，继承网络并重置速度课程。"""
        if not self._lighthw_forward_speed or not self._curriculum_training:
            raise ValueError("Route 10 transfer is only available to forward-speed PPO training.")
        if self._curriculum_initialized:
            raise ValueError("Route 10 transfer requires an uninitialized forward-speed runner.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        state = infos.get(self._CHECKPOINT_KEY) if isinstance(infos, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("Route 10 transfer requires a checkpoint containing its complete curriculum state.")
        self._curriculum_term.validate_route10_transfer(dict(state))
        source_iteration = int(checkpoint["iter"])
        del checkpoint
        # 跳过本类普通 load 的同课程限制，仍由原生 RSL-RL 完整加载网络和 normalizer。
        loaded_infos = super().load(path, load_optimizer=False)
        self._curriculum_term.initialize_route10_transfer(dict(state))
        self._route10_transfer_source = {
            "checkpoint": str(Path(path).resolve()),
            "source_iteration": source_iteration,
            "source_curriculum_updates": state["schedule"]["total_updates"],
        }
        self._curriculum_initialized = True
        print(
            "[LightHW forward-speed] checked route 10 transfer; actor/critic/noise/normalizers preserved; "
            "new forward-speed curriculum initialized at stage 0.",
            flush=True,
        )
        return loaded_infos

    def export_action_mapping(self, directory: str, *, checkpoint_path: str):
        """随 ONNX/JIT 导出实际动作变换；网络输出本身仍是16维原始 action。"""
        self._require_initialized_curriculum()
        leg = self._curriculum_term
        wheel = self.env.unwrapped.action_manager.get_term("joint_vel")

        def values(value):
            if isinstance(value, torch.Tensor):
                # JointAction 的比例/偏移/限幅可能按环境保存，部署只需一个环境的数值。
                return (value[0] if value.ndim >= 2 else value).detach().cpu().tolist()
            return value

        metadata = {
            "format_version": 1,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "policy_observation_dim": 60,
            "critic_observation_dim": 236,
            "raw_action_dim": 16,
            "network_output": "raw actions; apply the transforms below before sending actuator targets",
            "leg_transform": "clip(default_positions + stage_limits * tanh(raw_action[0:12]), absolute_clip)",
            "wheel_transform": "clip(scale * raw_action[12:16] + offset, absolute_clip)",
            "leg_joints": list(leg._joint_names),
            "wheel_joints": list(wheel._joint_names),
            "leg_default_positions_rad": values(leg._offset),
            "leg_stage_limits_rad": values(leg._limits),
            "leg_absolute_clip_rad": values(leg._clip) if leg.cfg.clip is not None else None,
            "wheel_scale_rad_s": values(wheel._scale),
            "wheel_offset_rad_s": values(wheel._offset),
            "wheel_absolute_clip_rad_s": values(wheel._clip) if wheel.cfg.clip is not None else None,
            "curriculum": leg.export_curriculum_state(),
        }
        target = Path(directory) / "action_mapping.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[LightHW PPO curriculum] exported required action transforms to: {target}", flush=True)
