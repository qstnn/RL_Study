# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""route 12 细化速度课程的受检迁移；完整保留已学 PPO 状态。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch

from episode_metrics import CompleteEpisodeMetricsRunner
from ppo_action_curriculum_runner import LightHWPPOActionCurriculumRunner


class LightHWPPOSpeedBridgeRunner(LightHWPPOActionCurriculumRunner):
    """从 route 11 的 0.8 m/s 模型建立独立课程，同任务恢复保持原生语义。"""

    _TRANSFER_KEY = "lighthw_route11_transfer"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", *, curriculum_training=True):
        if not train_cfg.get("lighthw_speed_bridge", False) or not train_cfg.get("lighthw_forward_speed", False):
            raise ValueError("LightHW speed bridge requires its explicit task configuration flags.")
        self._lighthw_speed_bridge = True
        self._route11_transfer_source = None
        super().__init__(env, train_cfg, log_dir, device, curriculum_training=curriculum_training)

    def _require_initialized_curriculum(self):
        if not self._curriculum_initialized:
            raise ValueError(
                "LightHW speed-bridge policy has not been initialized: use the checked route 11 "
                "0.8 m/s warm start, or load a checkpoint from the same speed-bridge task."
            )

    @staticmethod
    def _validate_tensor_state(name, source, target):
        """先核对全部键、形状、类型与有限性，避免原生 load 部分写入后才报错。"""
        if not isinstance(source, Mapping) or set(source) != set(target):
            raise ValueError(f"Speed-bridge {name} has incompatible state keys.")
        for key, expected in target.items():
            value = source[key]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype):
                raise ValueError(f"Speed-bridge {name}/{key} has incompatible tensor shape or dtype.")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Speed-bridge {name}/{key} contains NaN or Inf.")

    def _validate_training_checkpoint(self, checkpoint):
        self._validate_tensor_state("model", checkpoint.get("model_state_dict"), self.alg.policy.state_dict())
        normalizers = ("obs_norm_state_dict", "privileged_obs_norm_state_dict")
        if any(key in checkpoint for key in normalizers) != self.empirical_normalization:
            raise ValueError("Speed-bridge transfer requires matching empirical_normalization settings.")
        if self.empirical_normalization:
            self._validate_tensor_state("actor normalizer", checkpoint.get(normalizers[0]),
                                        self.obs_normalizer.state_dict())
            self._validate_tensor_state("critic normalizer", checkpoint.get(normalizers[1]),
                                        self.privileged_obs_normalizer.state_dict())
        iteration = checkpoint.get("iter")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Speed-bridge checkpoint requires a nonnegative integer iteration.")

        # Adam 使用参数编号而非名字保存状态。按原生加载相同的组内顺序验证每个参数，
        # 完整继承一阶/二阶矩、步数、学习率和其他组设置；不得悄悄退化为新优化器。
        optimizer = checkpoint.get("optimizer_state_dict")
        if not isinstance(optimizer, Mapping):
            raise ValueError("Speed-bridge checkpoint requires the complete Adam state.")
        groups, states = optimizer.get("param_groups"), optimizer.get("state")
        if not isinstance(groups, list) or not isinstance(states, Mapping) or len(groups) != len(self.alg.optimizer.param_groups):
            raise ValueError("Speed-bridge Adam parameter groups are incompatible.")
        used_ids, rates = [], set()
        for source_group, target_group in zip(groups, self.alg.optimizer.param_groups):
            if not isinstance(source_group, Mapping) or not isinstance(source_group.get("params"), list):
                raise ValueError("Speed-bridge Adam parameter group is invalid.")
            source_ids, parameters = source_group["params"], target_group["params"]
            if len(source_ids) != len(parameters):
                raise ValueError("Speed-bridge Adam parameter count is incompatible.")
            rate = source_group.get("lr")
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise ValueError("Speed-bridge Adam requires a finite positive learning rate.")
            betas = source_group.get("betas")
            if (not isinstance(betas, (tuple, list)) or len(betas) != 2
                    or any(not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta < 1
                           for beta in betas)):
                raise ValueError("Speed-bridge Adam betas are invalid.")
            for name in ("eps", "weight_decay"):
                value = source_group.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Speed-bridge Adam {name} is invalid.")
            rates.add(float(rate))
            for source_id, parameter in zip(source_ids, parameters):
                used_ids.append(source_id)
                state = states.get(source_id)
                required = {"step", "exp_avg", "exp_avg_sq"}
                if source_group.get("amsgrad", False):
                    required.add("max_exp_avg_sq")
                if not isinstance(state, Mapping) or set(state) != required:
                    raise ValueError("Speed-bridge Adam requires complete per-parameter moments and step.")
                for name, tensor in state.items():
                    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
                        raise ValueError(f"Speed-bridge Adam/{name} contains invalid tensors or NaN/Inf.")
                    if name == "step":
                        if tensor.numel() != 1 or tensor.item() < 0 or tensor.item() != int(tensor.item()):
                            raise ValueError("Speed-bridge Adam step is invalid.")
                    elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                        raise ValueError(f"Speed-bridge Adam/{name} has incompatible shape or dtype.")
                    elif name.endswith("sq") and torch.any(tensor < 0).item():
                        raise ValueError("Speed-bridge Adam second moments cannot be negative.")
        if len(set(used_ids)) != len(used_ids) or set(used_ids) != set(states) or len(rates) != 1:
            raise ValueError("Speed-bridge Adam parameter identifiers or learning rates are inconsistent.")
        return rates.pop()

    def load_route10_checkpoint(self, path):
        raise ValueError("Speed bridge accepts only a checked route 11 0.8 m/s source, not route 10.")

    def load_route11_checkpoint(self, path):
        """只重置新实验计数；Actor/Critic/std/归一化/Adam/adaptive LR 全部继承。"""
        if not self._curriculum_training or self._curriculum_initialized:
            raise ValueError("Route 11 transfer is only available to uninitialized speed-bridge training.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        state = infos.get(self._CHECKPOINT_KEY) if isinstance(infos, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("Route 11 transfer requires the complete source curriculum state.")
        self._curriculum_term.validate_route11_transfer(dict(state))
        learning_rate = self._validate_training_checkpoint(checkpoint)
        source_iteration = checkpoint["iter"]
        # 普通 load 只允许同任务；显式迁移经过全部预检后，直接调用原生完整加载。
        loaded_infos = CompleteEpisodeMetricsRunner.load(self, path, load_optimizer=True)
        self.alg.learning_rate = learning_rate
        self._curriculum_term.initialize_route11_transfer(dict(state))
        self.current_learning_iteration = 0
        self._route11_transfer_source = {
            "checkpoint": str(Path(path).resolve()),
            "source_iteration": source_iteration,
            "source_curriculum_updates": state["schedule"]["total_updates"],
            "source_max_speed_m_s": 0.8,
            "optimizer_preserved": True,
            "initial_learning_rate": learning_rate,
        }
        self._curriculum_initialized = True
        print(
            "[LightHW speed bridge] checked route 11 transfer; actor/critic/noise/normalizers/Adam "
            f"preserved; adaptive learning_rate={learning_rate:g}; new iteration=0; "
            "new curriculum=0.8 -> 1.0 -> 1.2 m/s.", flush=True,
        )
        return loaded_infos

    def save(self, path, infos=None):
        if infos is not None and not isinstance(infos, Mapping):
            raise TypeError("LightHW curriculum checkpoint infos must be a mapping or None.")
        checkpoint_infos = dict(infos or {})
        if self._route11_transfer_source is not None:
            checkpoint_infos[self._TRANSFER_KEY] = dict(self._route11_transfer_source)
        return super().save(path, infos=checkpoint_infos)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        state = infos.get(self._CHECKPOINT_KEY) if isinstance(infos, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("Speed-bridge resume requires a checkpoint from the same task.")
        self._curriculum_term.validate_curriculum_state(dict(state))
        self._validate_training_checkpoint(checkpoint)
        provenance = infos.get(self._TRANSFER_KEY)
        if not isinstance(provenance, Mapping) or not provenance.get("optimizer_preserved"):
            raise ValueError("Speed-bridge checkpoint is missing its checked route 11 transfer provenance.")
        loaded_infos = super().load(path, load_optimizer=load_optimizer)
        self._route11_transfer_source = dict(provenance)
        return loaded_infos
