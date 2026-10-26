"""Unconnected PPO driver for a future R4a Actor contract.

This module is deliberately independent of Isaac Sim and of any student model.
It accepts a vector environment, an arbitrary ``nn.Module`` Actor and a causal
``feature_builder(raw_observation, done_mask)``. CPU tests here validate software
semantics only; they are not robot behavior evidence or PPO admission.
"""

from __future__ import annotations

import os
import random
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from rsl_rl.modules import ActorCritic

SUSTAIN_DIR = Path(__file__).resolve().parents[2]
if str(SUSTAIN_DIR) not in sys.path:
    sys.path.insert(0, str(SUSTAIN_DIR))
from continuing_ppo import ContinuingPPO  # noqa: E402


def _cpu_copy(value):
    """Write portable CPU copies of model/Adam tensors without changing live state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, OrderedDict):
        result = OrderedDict((key, _cpu_copy(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = _cpu_copy(value._metadata)
        return result
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


@dataclass(frozen=True)
class DriverConfig:
    actor_obs_dim: int
    critic_obs_dim: int = 88
    num_actions: int = 16
    num_steps_per_env: int = 32
    initial_std: float = 0.05
    learning_rate: float = 3e-4
    num_learning_epochs: int = 2
    num_mini_batches: int = 1
    gamma: float = 0.995
    lam: float = 0.95
    clip_param: float = 0.2
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in ("actor_obs_dim", "critic_obs_dim", "num_actions", "num_steps_per_env",
                     "num_learning_epochs", "num_mini_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 < self.initial_std <= 1.0):
            raise ValueError("initial_std must be explicit and in (0, 1]")
        if not (0.0 < self.gamma < 1.0 and 0.0 <= self.lam <= 1.0):
            raise ValueError("Invalid GAE discount")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer parameters")


class ContinuingDriver:
    """Run real RSL PPO updates without resetting a live vector env per iteration.

    Required env interface matches Isaac Lab's RSL wrapper: ``num_envs``,
    ``num_actions``, ``device``, ``get_observations()`` and ``step(actions)``.
    The builder is called once on the initial raw observation with all slots
    marked reset, then exactly once after each env step with that step's done
    mask. It must use deployable signals only and return [N, actor_obs_dim].

    A checkpoint resumes model/Adam/RNG/counts, but a new process starts a new
    physics episode. It never claims continuous standing across a restart.
    """

    def __init__(
        self,
        env,
        actor: nn.Module,
        feature_builder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: DriverConfig,
        *,
        observation_contract: str,
        policy_contract: str,
        device: str | torch.device = "cpu",
    ):
        if not observation_contract or not policy_contract:
            raise ValueError("Explicit observation and policy contracts are required")
        if not isinstance(actor, nn.Module):
            raise TypeError("actor must be an nn.Module")
        if not callable(feature_builder):
            raise TypeError("feature_builder must be callable")
        self.env = env
        self.feature_builder = feature_builder
        self.config = config
        self.observation_contract = observation_contract
        self.policy_contract = policy_contract
        self.device = torch.device(device)
        if torch.device(env.device) != self.device:
            raise ValueError("PPO, vector env and terminal Critic observations must use the same device")
        self.num_envs = int(env.num_envs)
        if self.num_envs <= 0 or int(env.num_actions) != config.num_actions:
            raise ValueError("Vector environment action/count mismatch")
        samples = self.num_envs * config.num_steps_per_env
        if samples < 2 or config.num_mini_batches > samples:
            raise ValueError("PPO batch must contain at least two samples and one sample per mini-batch")
        self.policy = ActorCritic(
            config.actor_obs_dim, config.critic_obs_dim, config.num_actions,
            actor_hidden_dims=[16], critic_hidden_dims=[128, 128],
            init_noise_std=config.initial_std, noise_std_type="log",
        )
        self.policy.actor = actor
        self.policy.to(self.device)
        # A BC loader may return actor.eval(); PPO collection/update must not
        # silently keep it in that mode after module replacement.
        self.policy.train()
        self.alg = ContinuingPPO(
            self.policy, num_learning_epochs=config.num_learning_epochs,
            num_mini_batches=config.num_mini_batches, clip_param=config.clip_param,
            gamma=config.gamma, lam=config.lam, learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm, schedule="fixed", device=self.device,
        )
        self.alg.init_storage(
            "rl", self.num_envs, config.num_steps_per_env,
            [config.actor_obs_dim], [config.critic_obs_dim], [config.num_actions],
        )
        # A RslRlVecEnvWrapper resets in its constructor. This driver reads its
        # current observation; it never calls env.reset() at a batch boundary.
        with torch.no_grad():
            raw, info = env.get_observations()
            self.obs = self._build(raw, torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
            self.critic_obs = self._critic(info)
        self.completed_updates = 0
        self.control_steps = 0
        self.transition_samples = 0
        self.optimizer_steps = 0
        self.bootstrap_events = 0
        self.physical_terminations = 0
        self.feature_build_calls = 1
        self.initial_observation_reads = 1
        self.driver_reset_calls = 0
        self._rollout_ready = False

    def _build(self, raw, dones):
        if raw.device != self.device:
            raise ValueError("Raw Actor observation is on the wrong device")
        result = self.feature_builder(raw, dones.to(self.device).bool())
        if not isinstance(result, torch.Tensor) or result.device != self.device:
            raise ValueError("Feature builder must return a tensor on the PPO device")
        if result.shape != (self.num_envs, self.config.actor_obs_dim):
            raise ValueError(f"Actor observation shape {result.shape} is not "
                             f"({self.num_envs}, {self.config.actor_obs_dim})")
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite Actor observation")
        return result.detach()

    def _critic(self, info):
        result = info["observations"]["critic"]
        if result.device != self.device:
            raise ValueError("Critic observation is on the wrong device")
        if result.shape != (self.num_envs, self.config.critic_obs_dim):
            raise ValueError("Critic observation shape mismatch")
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite Critic observation")
        return result.detach()

    def collect_rollout(self):
        """Collect one RSL storage batch; keep the live env and memory intact."""
        if self._rollout_ready or self.alg.storage.step != 0:
            raise RuntimeError("Previous rollout must be updated first")
        self.policy.train()
        timeouts = failures = 0
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                action = self.alg.act(self.obs, self.critic_obs)
                if action.shape != (self.num_envs, self.config.num_actions) or not torch.isfinite(action).all():
                    raise ValueError("Invalid sampled action")
                raw, reward, done, info = self.env.step(action.to(self.env.device))
                if reward.device != self.device or done.device != self.device:
                    raise ValueError("Reward/done is on the wrong device")
                reward = reward.reshape(-1)
                done = done.reshape(-1).bool()
                if reward.shape != (self.num_envs,) or done.shape != (self.num_envs,):
                    raise ValueError("Reward/done vector shape mismatch")
                if not torch.isfinite(reward).all():
                    raise ValueError("Non-finite reward")
                truncated = info["time_outs"].to(self.device).reshape(-1).bool()
                terminated = info["physical_terminated"].to(self.device).reshape(-1).bool()
                if truncated.shape != done.shape or terminated.shape != done.shape:
                    raise ValueError("Terminal vector shape mismatch")
                if not torch.equal(done, truncated | terminated) or (truncated & terminated).any():
                    raise ValueError("Physical termination and truncation disagree with done")
                if truncated.any():
                    terminal = info["terminal_critic_observation"]
                    if terminal.device != self.device or terminal.shape != (self.num_envs, self.config.critic_obs_dim):
                        raise ValueError("Pre-reset terminal Critic observation shape/device mismatch")
                self.alg.process_env_step(reward, done.long(), info)
                # PPO has now consumed the pre-reset terminal Critic observation.
                # The raw observation returned by Isaac Lab belongs to the
                # reset episode for done slots, and builder clears only them.
                self.obs = self._build(raw, done)
                self.critic_obs = self._critic(info)
                timeouts += int(truncated.sum())
                failures += int(terminated.sum())
                self.control_steps += 1
                self.transition_samples += self.num_envs
                self.feature_build_calls += 1
        self.bootstrap_events += timeouts
        self.physical_terminations += failures
        self._rollout_ready = True
        return {"timeouts": timeouts, "physical_terminations": failures,
                "storage_steps": self.alg.storage.step}

    def update_from_rollout(self):
        """Consume one full rollout and perform actual RSL optimizer steps."""
        if not self._rollout_ready or self.alg.storage.step != self.config.num_steps_per_env:
            raise RuntimeError("No full rollout to update")
        with torch.inference_mode():
            self.alg.compute_returns(self.critic_obs)
        losses = self.alg.update()
        if self.alg.storage.step != 0 or not all(np.isfinite(v) for v in losses.values()):
            raise RuntimeError("PPO update/storage did not finish cleanly")
        if not all(torch.isfinite(value).all() for value in self.policy.state_dict().values()):
            raise RuntimeError("Non-finite policy parameter after PPO update")
        self._rollout_ready = False
        self.completed_updates += 1
        self.optimizer_steps += self.config.num_learning_epochs * self.config.num_mini_batches
        return {"completed_updates": self.completed_updates,
                "optimizer_steps": self.optimizer_steps,
                "control_steps": self.control_steps,
                "transition_samples": self.transition_samples,
                "bootstrap_events": self.bootstrap_events,
                "driver_reset_calls": self.driver_reset_calls,
                "feature_build_calls": self.feature_build_calls,
                "losses": losses}

    def run_iteration(self):
        collection = self.collect_rollout()
        return dict(collection=collection, **self.update_from_rollout())

    def save_checkpoint(self, path: str | Path):
        """Save only at an optimizer boundary; never overwrite an existing run."""
        path = Path(path)
        if self._rollout_ready or self.alg.storage.step != 0:
            raise RuntimeError("Checkpoint requires a completed PPO update")
        if not self.completed_updates:
            raise RuntimeError("Checkpoint requires at least one actual PPO update")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema": "r4a_unconnected_ppo_design_v1",
            "config": asdict(self.config),
            "observation_contract": self.observation_contract,
            "policy_contract": self.policy_contract,
            "num_envs": self.num_envs,
            "device_type": self.device.type,
            "model_state_dict": _cpu_copy(self.policy.state_dict()),
            "optimizer_state_dict": _cpu_copy(self.alg.optimizer.state_dict()),
            "learning_rate": self.alg.learning_rate,
            "counters": {name: getattr(self, name) for name in (
                "completed_updates", "control_steps", "transition_samples",
                "optimizer_steps", "bootstrap_events", "physical_terminations")},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": _cpu_copy(torch.cuda.get_rng_state_all()) if self.device.type == "cuda" else [],
            },
            "physics_state_saved": False,
            "new_physics_episode_on_process_resume": True,
        }
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            torch.save(state, temporary)
            # A hard link fails if another process created the final file after
            # our pre-check, preserving an existing training artifact.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path):
        """Restore PPO state into a *fresh* environment, not a physics snapshot."""
        if self._rollout_ready or self.alg.storage.step != 0 or self.completed_updates:
            raise RuntimeError("Resume must happen before collecting new transitions")
        # Load to CPU first; load_state_dict/Adam then move parameters and
        # moments to this learner's device. CUDA RNG byte tensors stay on CPU.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        expected = ("r4a_unconnected_ppo_design_v1", asdict(self.config),
                    self.observation_contract, self.policy_contract, self.num_envs, self.device.type)
        actual = (saved["schema"], saved["config"], saved["observation_contract"],
                  saved["policy_contract"], saved["num_envs"], saved["device_type"])
        if actual != expected or saved["physics_state_saved"]:
            raise ValueError("Checkpoint/Actor/environment contract mismatch")
        if saved["rng"]["torch_cuda"]:
            if not torch.cuda.is_available() or len(saved["rng"]["torch_cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG device count changed")
        self.policy.load_state_dict(saved["model_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
        self.alg.learning_rate = float(saved["learning_rate"])
        for group in self.alg.optimizer.param_groups:
            if abs(float(group["lr"]) - self.alg.learning_rate) > 1e-12:
                raise ValueError("Adam and algorithm learning rates disagree")
        for name, value in saved["counters"].items():
            setattr(self, name, int(value))
        rng = saved["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if rng["torch_cuda"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
        return {"completed_updates": self.completed_updates,
                "new_physics_episode": True,
                "standing_continuity_across_processes": False,
                "feature_state": "freshly initialized from new environment observation"}
