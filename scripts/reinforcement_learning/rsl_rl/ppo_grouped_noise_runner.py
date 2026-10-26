"""Route 13: checked route-12 transfer with command-dependent PPO exploration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import json
import math

import torch
import rsl_rl.runners.on_policy_runner as native_runner

from episode_metrics import CompleteEpisodeMetricsRunner
from lighthw_grouped_noise_policy import LightHWGroupedNoiseActorCritic
from lighthw_training_monitor import format_grouped_exploration
from ppo_action_curriculum_runner import LightHWPPOActionCurriculumRunner
from ppo_speed_bridge_runner import LightHWPPOSpeedBridgeRunner


class LightHWGroupedNoiseRunner(LightHWPPOSpeedBridgeRunner):
    """复用完整检查点预检；只在首次迁移改变噪声，普通恢复保留全部新状态。

    Actor/Critic、其Adam及adaptive LR完整继承。16维log_std显式重设分组初值，
    仅该参数的Adam矩与step清零。动作均值和物理映射不因迁移发生变化。
    """

    _TRANSFER_KEY = "lighthw_route12_grouped_noise_transfer"
    _NOISE_KEY = "lighthw_grouped_noise"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", *, curriculum_training=True):
        if (not train_cfg.get("lighthw_grouped_noise", False)
                or not train_cfg.get("lighthw_forward_speed", False)):
            raise ValueError("Grouped-noise PPO requires its independent task flags.")
        if train_cfg.get("empirical_normalization", False):
            raise ValueError("Grouped-noise PPO requires unnormalized command observations.")
        if any(train_cfg["algorithm"].get(key) is not None for key in ("symmetry_cfg", "rnd_cfg")):
            raise ValueError("Grouped-noise PPO does not support symmetry or RND in this experiment.")
        if train_cfg["policy"]["class_name"] != "LightHWGroupedNoiseActorCritic":
            raise ValueError("Grouped-noise task requires the matching command-conditioned policy.")
        self._validate_observation_layout(env)
        self._lighthw_grouped_noise = True
        self._lighthw_speed_bridge = False
        self._grouped_noise_transfer_source = None
        # RSL-RL 3.0.1 resolves policy class names in its module namespace. Register
        # only the new name; native ActorCritic and all old task dispatch remain intact.
        native_runner.LightHWGroupedNoiseActorCritic = LightHWGroupedNoiseActorCritic
        LightHWPPOActionCurriculumRunner.__init__(
            self, env, deepcopy(train_cfg), log_dir, device, curriculum_training=curriculum_training
        )
        profile = self.alg.policy.noise_profile()
        mapping = self._curriculum_term.export_curriculum_state()["action_mapping"]
        if (mapping["leg_joints"] != profile["leg_action_order"]
                or mapping["wheel_joints"] != profile["wheel_action_order"]):
            raise ValueError("Grouped noise requires the recorded leg/wheel action order.")
        if (profile["command_scale"] != [1.0, 1.0, 1.0]
                or profile["stop_threshold"] != mapping["gate_parameters"]["standing_command_threshold"]
                or mapping["hip_limits"] != [profile["hip_limit"]]
                or mapping["leg_limits"] != [profile["leg_limit"]]
                or profile["wheel_scale"] != mapping["wheel_scale"]):
            raise ValueError("Grouped-noise profile does not match the actual commands or physical action mapping.")
        self._noise_projection_hook = self.alg.optimizer.register_step_post_hook(
            lambda optimizer, args, kwargs: self.alg.policy.project_grouped_noise_()
        )
        print(f"[LightHW grouped noise] native PPO; leg raw std {profile['leg_std_init']:g} "
              f"[{profile['leg_std_min']:g},{profile['leg_std_max']:g}]; "
              f"wheel moving target std {profile['wheel_std_init'] * profile['wheel_scale']:g} rad/s; "
              f"stop multiplier={profile['stop_wheel_multiplier']:g}; cap held at 1.0 m/s.", flush=True)

    @staticmethod
    def _validate_observation_layout(env):
        manager = env.unwrapped.observation_manager
        names = manager.active_terms["policy"]
        dims = manager.group_obs_term_dim["policy"]
        expected = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                    "joint_pos", "joint_vel", "actions"]
        if names != expected or list(map(tuple, dims)) != [(3,), (3,), (3,), (3,), (16,), (16,), (16,)]:
            raise ValueError("Grouped-noise command must occupy unstacked policy observations 9:12.")
        command = manager._group_obs_term_cfgs["policy"][3]
        scale = torch.as_tensor(1.0 if command.scale is None else command.scale)
        if (not torch.all(scale == 1).item() or command.noise is not None
                or command.modifiers or command.history_length not in (0, None)):
            raise ValueError("Grouped-noise command observations require scale=1, no noise/modifiers/history.")
        if command.clip is not None and (command.clip[0] > -1.0 or command.clip[1] < 1.0):
            raise ValueError("Grouped-noise command observation clipping changes command semantics.")

    def _require_initialized_curriculum(self):
        if not self._curriculum_initialized:
            raise ValueError("Grouped-noise policy has not been initialized: use checked route12 warm start "
                             "or a checkpoint from the same grouped-noise task.")

    def load_route10_checkpoint(self, path):
        raise ValueError("Grouped-noise transfer requires route12 at 1.0 m/s, not route10.")

    def load_route11_checkpoint(self, path):
        raise ValueError("Grouped-noise transfer requires route12 at 1.0 m/s, not route11.")

    def load_route12_checkpoint(self, path):
        if not self._curriculum_training or self._curriculum_initialized:
            raise ValueError("Route12 transfer is only available to uninitialized grouped-noise training.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        state = infos.get(self._CHECKPOINT_KEY) if isinstance(infos, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("Route12 transfer requires complete curriculum metadata.")
        self._curriculum_term.validate_route12_transfer(dict(state))
        rate = self._validate_training_checkpoint(checkpoint)
        source_chain = infos.get("lighthw_route11_transfer")
        if not isinstance(source_chain, Mapping) or source_chain.get("optimizer_preserved") is not True:
            raise ValueError("Route12 source is missing its checked route11 transfer provenance.")
        source_std = checkpoint["model_state_dict"]["log_std"].exp().tolist()
        loaded_infos = CompleteEpisodeMetricsRunner.load(self, path, load_optimizer=True)
        self.alg.learning_rate = rate
        self.alg.policy.reset_grouped_noise()
        noise_state = self.alg.optimizer.state[self.alg.policy.log_std]
        # 原参数对象/组顺序不变；仅noise参数重置，保留全部其他参数的Adam连续性。
        for tensor in noise_state.values():
            tensor.zero_()
        self._curriculum_term.initialize_route12_transfer(dict(state))
        self.current_learning_iteration = 0
        self._grouped_noise_transfer_source = {
            "checkpoint": str(Path(path).resolve()), "source_iteration": checkpoint["iter"],
            "source_curriculum_updates": state["schedule"]["total_updates"],
            "source_max_speed_m_s": 1.0, "source_noise_std": source_std,
            "source_route11_transfer": deepcopy(dict(source_chain)),
            "network_optimizer_preserved": True, "noise_optimizer_reset": True,
            "noise_parameter_reset": True, "initial_learning_rate": rate,
        }
        self._curriculum_initialized = True
        print("[LightHW grouped noise] checked route12 transfer; actor/critic and their Adam preserved; "
              f"adaptive LR={rate:g}; only log_std and its Adam reset; local updates=0, cap=1.0.", flush=True)
        return loaded_infos

    def save(self, path, infos=None):
        self._require_initialized_curriculum()
        if infos is not None and not isinstance(infos, Mapping):
            raise TypeError("Grouped-noise checkpoint infos must be a mapping or None.")
        checkpoint_infos = dict(infos or {})
        checkpoint_infos[self._NOISE_KEY] = self.alg.policy.noise_profile()
        checkpoint_infos[self._TRANSFER_KEY] = deepcopy(self._grouped_noise_transfer_source)
        return LightHWPPOActionCurriculumRunner.save(self, path, infos=checkpoint_infos)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        state = infos.get(self._CHECKPOINT_KEY) if isinstance(infos, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("Grouped-noise resume requires a checkpoint from the same task.")
        self._curriculum_term.validate_curriculum_state(dict(state))
        self._validate_training_checkpoint(checkpoint)
        if infos.get(self._NOISE_KEY) != self.alg.policy.noise_profile():
            raise ValueError("Grouped-noise checkpoint exploration profile does not match this task.")
        profile = self.alg.policy.noise_profile()
        noise = checkpoint["model_state_dict"]["log_std"]
        lower = noise.new_tensor([math.log(profile["leg_std_min"])] * 12
                                 + [math.log(profile["wheel_std_min"])] * 4)
        upper = noise.new_tensor([math.log(profile["leg_std_max"])] * 12
                                 + [math.log(profile["wheel_std_max"])] * 4)
        if torch.any((noise < lower) | (noise > upper)).item():
            raise ValueError("Grouped-noise checkpoint log_std is outside the saved noise bounds.")
        source = infos.get(self._TRANSFER_KEY)
        if (not isinstance(source, Mapping) or source.get("network_optimizer_preserved") is not True
                or source.get("noise_optimizer_reset") is not True):
            raise ValueError("Grouped-noise checkpoint is missing its checked route12 transfer provenance.")
        loaded_infos = LightHWPPOActionCurriculumRunner.load(self, path, load_optimizer=load_optimizer)
        self._grouped_noise_transfer_source = deepcopy(dict(source))
        print("[LightHW grouped noise] same-task checkpoint restored; learned noise and optimizer retained.", flush=True)
        return loaded_infos

    def _log_forward_summary(self, snapshot, context, values, locs):
        super()._log_forward_summary(snapshot, context, values, locs)
        # 固定当前观测估计分布的角度std，不能用跨状态目标波动冒充探索噪声。
        diagnostics = self.alg.policy.diagnostics(locs["obs"][:256])
        keys = list(diagnostics)
        numbers = torch.stack([diagnostics[key] for key in keys]).detach().cpu().tolist()
        data = dict(zip(keys, numbers))
        for key, value in data.items():
            self.writer.add_scalar(f"Exploration/{key}", value, locs["it"])
        print(format_grouped_exploration(data), flush=True)

    def export_action_mapping(self, directory, *, checkpoint_path):
        super().export_action_mapping(directory, checkpoint_path=checkpoint_path)
        target = Path(directory) / "exploration_profile.json"
        target.write_text(json.dumps(self.alg.policy.noise_profile(), indent=2) + "\n")
