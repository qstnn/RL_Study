"""LightHW 的原生 PPO 高斯策略：腿低噪声、移动轮高噪声、停车轮低噪声。

噪声只在策略的 Normal 分布内定义；采样、旧 sigma、log probability、entropy
和自适应 KL 都由 RSL-RL 使用同一个分布。没有环境侧附加噪声或额外控制器。
确定性 actor 与原模型完全相同；噪声配置由 runner 存在 checkpoint metadata，
不增加 state_dict 项，从而能受检迁移原 actor/critic/log_std。
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

import numpy as np
import torch
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic


LEG_ACTION_NAMES = tuple(f"{leg}_{joint}" for leg in ("FR", "FL", "RR", "RL")
                         for joint in ("hip", "thigh", "calf"))
DEFAULT_GROUPED_NOISE_CONFIG = {
    "policy_obs_dim": 60,
    "critic_obs_dim": 236,
    "actions": 16,
    "command_start": 9,
    "command_size": 3,
    "command_scale": [1.0, 1.0, 1.0],
    "stop_threshold": 0.05,
    "leg_std_init": 0.12,
    "leg_std_min": 0.08,
    "leg_std_max": 0.15,
    "wheel_std_init": 0.25,
    "wheel_std_min": 0.20,
    "wheel_std_max": 0.35,
    "stop_wheel_multiplier": 0.4,
    "hip_limit": 0.04,
    "leg_limit": 0.10,
    "wheel_scale": 5.0,
}


def canonical_grouped_noise_config(config=None):
    """校验并复制配置；显式拒绝拼错的键、非有限数和错误的动作/观测布局。"""
    if config is not None and not isinstance(config, dict):
        raise ValueError("grouped_noise_config must be a dictionary")
    config = deepcopy({} if config is None else config)
    unknown = set(config) - set(DEFAULT_GROUPED_NOISE_CONFIG)
    if unknown:
        raise ValueError(f"Unknown grouped noise configuration keys: {sorted(unknown)}")
    result = deepcopy(DEFAULT_GROUPED_NOISE_CONFIG)
    result.update(config)
    for key in ("policy_obs_dim", "critic_obs_dim", "actions", "command_start", "command_size"):
        if type(result[key]) is not int or result[key] != DEFAULT_GROUPED_NOISE_CONFIG[key]:
            raise ValueError(f"Unsupported LightHW observation/action layout: {key}={result[key]!r}")
    scale = result["command_scale"]
    if not isinstance(scale, (tuple, list)) or len(scale) != 3:
        raise ValueError("command_scale must contain three finite positive values")
    scalar_keys = set(result) - {"policy_obs_dim", "critic_obs_dim", "actions", "command_start",
                                 "command_size", "command_scale"}
    for key in scalar_keys:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(value)
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) or value <= 0 for value in scale):
        raise ValueError("command_scale must contain three finite positive values")
    result["command_scale"] = [float(value) for value in scale]
    for group in ("leg", "wheel"):
        if not result[f"{group}_std_min"] <= result[f"{group}_std_init"] <= result[f"{group}_std_max"]:
            raise ValueError(f"{group} noise requires min <= init <= max")
    if result["stop_wheel_multiplier"] > 1:
        raise ValueError("stop_wheel_multiplier must be in (0, 1]")
    return result


class LightHWGroupedNoiseActorCritic(ActorCritic):
    """保持均值网络的 60/236/16 策略，仅改变条件标准差。

    observations 必须保留未做经验归一化的 command 切片。静止判定使用真实命令
    三维向量范数 <= 0.05，与 PPO 中同一观测重放严格一致。effective std 通过
    log_std 的有界映射计算，构造分布时不修改参数；runner在每个Adam更新后
    独立投影log_std，rollout与update始终使用相同的条件分布定义。
    """

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims=(256, 256, 256),
                 critic_hidden_dims=(256, 256, 256), activation="elu", init_noise_std=1.0,
                 noise_std_type="log", grouped_noise_config=None, **kwargs):
        if kwargs:
            raise ValueError(f"Unknown grouped noise policy arguments: {sorted(kwargs)}")
        profile = canonical_grouped_noise_config(grouped_noise_config)
        if (num_actor_obs, num_critic_obs, num_actions) != (60, 236, 16):
            raise ValueError("LightHW grouped noise requires policy=60, critic=236, action=16")
        if noise_std_type != "log":
            raise ValueError("LightHW grouped noise requires noise_std_type='log'")
        if (isinstance(init_noise_std, bool) or not isinstance(init_noise_std, Real)
                or not math.isfinite(init_noise_std) or init_noise_std <= 0):
            raise ValueError("init_noise_std must be finite and positive")
        super().__init__(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims, activation=activation,
                         init_noise_std=init_noise_std, noise_std_type=noise_std_type)
        self._grouped_noise_config = profile
        # 普通属性而非注册 buffer；不能改变旧 checkpoint 的 state_dict 键或导出 actor。
        self._noise_tensor_cache = {}
        self.reset_grouped_noise()

    def noise_profile(self):
        return {
            "version": 1,
            "leg_action_order": [name + "_joint" for name in LEG_ACTION_NAMES],
            "wheel_action_order": [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")],
            **deepcopy(self._grouped_noise_config),
        }

    @torch.no_grad()
    def project_grouped_noise_(self):
        """每次 Adam 更新后投影 log_std，避免硬界外参数持续得到零梯度。

        runner 应通过 optimizer step post-hook 在每个 minibatch 后调用；投影
        不改变 rollout 内分布定义，不增加损失或另一个优化器。
        """
        self._assert_finite(self.log_std, "log_std")
        constants = self._constants(self.log_std)
        self.log_std.clamp_(min=constants["lower"], max=constants["upper"])

    @torch.no_grad()
    def reset_grouped_noise(self):
        """仅显式跨路线迁移时使用；普通 resume 必须恢复保存的 log_std。"""
        self.log_std[:12].fill_(math.log(self._grouped_noise_config["leg_std_init"]))
        self.log_std[12:].fill_(math.log(self._grouped_noise_config["wheel_std_init"]))
        self.distribution = None

    def _constants(self, reference):
        key = (reference.device, reference.dtype)
        if key not in self._noise_tensor_cache:
            cfg = self._grouped_noise_config
            # 首次可能发生在 rollout 的 inference_mode 中；缓存必须为普通 tensor，
            # 后续 PPO autograd 才能安全保存乘数/界限用于反向传播。
            with torch.inference_mode(False):
                tensor = lambda values: torch.tensor(values, device=reference.device, dtype=reference.dtype)
                self._noise_tensor_cache[key] = {
                    "lower": tensor([math.log(cfg["leg_std_min"])] * 12
                                    + [math.log(cfg["wheel_std_min"])] * 4),
                    "upper": tensor([math.log(cfg["leg_std_max"])] * 12
                                    + [math.log(cfg["wheel_std_max"])] * 4),
                    "command_scale": tensor(cfg["command_scale"]),
                    "limits": tensor([cfg["hip_limit"] if index % 3 == 0 else cfg["leg_limit"]
                                      for index in range(12)]),
                }
        return self._noise_tensor_cache[key]

    @staticmethod
    def _assert_finite(value, label):
        # GPU async assert 避免每个 rollout/minibatch 为 .item() 同步到 CPU。
        torch._assert_async(torch.isfinite(value).all(), f"Non-finite {label} in LightHW grouped noise policy")

    def _validate_observations(self, observations):
        if not isinstance(observations, torch.Tensor) or observations.ndim != 2 or observations.shape[1] != 60:
            raise ValueError("Grouped noise observations must have shape [batch, 60]")
        if observations.shape[0] < 1 or not observations.is_floating_point():
            raise ValueError("Grouped noise observations require a nonempty floating point batch")
        self._assert_finite(observations, "observations")

    def _base_std(self):
        self._assert_finite(self.log_std, "log_std")
        constants = self._constants(self.log_std)
        return torch.exp(torch.clamp(self.log_std, min=constants["lower"], max=constants["upper"]))

    def _stop_mask(self, observations):
        cfg = self._grouped_noise_config
        command = observations[:, cfg["command_start"]:cfg["command_start"] + cfg["command_size"]]
        physical_command = command / self._constants(observations)["command_scale"]
        return torch.linalg.vector_norm(physical_command, dim=-1) <= cfg["stop_threshold"]

    def update_distribution(self, observations):
        self._validate_observations(observations)
        mean = self.actor(observations)
        self._assert_finite(mean, "actor mean")
        base = self._base_std()
        stop_multiplier = torch.where(self._stop_mask(observations),
                                      self._grouped_noise_config["stop_wheel_multiplier"], 1.0)
        std = torch.cat((base[:12].expand(mean.shape[0], -1),
                         base[12:] * stop_multiplier[:, None]), dim=-1)
        self.distribution = Normal(mean, std)

    @torch.no_grad()
    def diagnostics(self, observations):
        """返回设备端 scalar tensors，不修改分布、模型、全局 RNG 或课程。

        raw Gaussian 经腿部 limit*tanh 后的目标噪声，用 24 点 Gauss-Hermite
        求每个观测/关节的标准差，再对观测取均值；不是仅在零均值处的近似。
        饱和定义为确定性目标占用动作包络 >=95%，覆盖全部 12 个腿动作。
        轮目标标准差为 scale 后、环境 clip 前数值，不能误称实际轮速波动。
        """
        self._validate_observations(observations)
        means = self.actor(observations)
        self._assert_finite(means, "actor mean")
        base = self._base_std()
        cfg = self._grouped_noise_config
        # 用 float64 避免已饱和 tanh 目标的小方差发生严重消减误差。
        means64, base64 = means[:, :12].double(), base[:12].double()
        constants = self._constants(means64)
        if "gh_nodes" not in constants:
            nodes, weights = np.polynomial.hermite.hermgauss(24)
            with torch.inference_mode(False):
                constants["gh_nodes"] = torch.tensor(nodes * math.sqrt(2), device=means.device, dtype=torch.float64)
                constants["gh_weights"] = torch.tensor(weights / math.sqrt(math.pi), device=means.device,
                                                       dtype=torch.float64)
        targets = (means64[..., None] + base64[None, :, None] * constants["gh_nodes"]).tanh()
        weights = constants["gh_weights"]
        target_means = (targets * weights).sum(-1, keepdim=True)
        target_std = (((targets - target_means).square() * weights).sum(-1).clamp_min(0).sqrt()
                      * constants["limits"])
        occupancy = means64.tanh().abs()
        saturated = (occupancy >= 0.95).double()
        stop_multiplier = cfg["stop_wheel_multiplier"]
        result = {
            "leg_raw_std_mean": base[:12].mean(),
            "leg_raw_std_min": base[:12].min(),
            "leg_raw_std_max": base[:12].max(),
            "leg_target_std_rad_mean": target_std.mean(),
            "leg_mean_envelope_occupancy": occupancy.mean(),
            "leg_saturation_fraction": saturated.mean(),
            "wheel_moving_raw_std_mean": base[12:].mean(),
            "wheel_stop_raw_std_mean": base[12:].mean() * stop_multiplier,
            "wheel_moving_target_std_rad_s": base[12:].mean() * cfg["wheel_scale"],
            "wheel_stop_target_std_rad_s": base[12:].mean() * stop_multiplier * cfg["wheel_scale"],
            "command_stop_fraction": self._stop_mask(observations).float().mean(),
        }
        for index, name in enumerate(LEG_ACTION_NAMES):
            result[f"joint/{name}/target_std_rad"] = target_std[:, index].mean()
            result[f"joint/{name}/mean_envelope_occupancy"] = occupancy[:, index].mean()
            result[f"joint/{name}/saturation_fraction"] = saturated[:, index].mean()
        return result
