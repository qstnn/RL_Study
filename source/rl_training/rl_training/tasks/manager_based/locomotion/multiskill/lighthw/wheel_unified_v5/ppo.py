"""V5 受控 PPO；部署仍是原生 ActorCritic 的单 MLP。

复用 RSL-RL 的 rollout、GAE、动作分布和标准保存格式。仅替换前馈 PPO
更新以提供固定分组学习率、KL 停止条件和 Critic 预热，不修改安装包。
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from rsl_rl.algorithms.ppo import PPO


class V5PPO(PPO):
    """单 Adam 两组参数；额外训练状态必须与模型/Adam 一起恢复。

    worker 保存 training_state_dict()，runner.load 后调用
    load_training_state_dict()。全局 RNG 的保存/恢复仍由 worker 负责。
    """

    STATE_VERSION = 3

    def __init__(self, policy, *, critic_learning_rate=1.e-4, kl_stop_threshold=.01,
                 critic_warmup_updates=50, initial_std_by_action=None,
                 min_std_by_action=None, max_std_by_action=None,
                 reference_loss_coef=0., reference_batch_size=256, reference_max_group_mse=0.,
                 reference_action_scale=None, actor_input_mask=(205, 285), **kwargs):
        if kwargs.get("schedule", "fixed") != "fixed":
            raise ValueError("V5 requires fixed actor/critic learning rates")
        if policy.is_recurrent or kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg") or kwargs.get("multi_gpu_cfg"):
            raise ValueError("V5PPO supports single-device feed-forward PPO only")
        if not hasattr(policy, "log_std") or policy.log_std.shape != (16,):
            raise ValueError("V5 requires the standard 16-dimensional log_std parameter")
        if critic_learning_rate <= 0 or kl_stop_threshold <= 0 or critic_warmup_updates < 0:
            raise ValueError("Invalid critic learning rate, KL threshold or warmup length")
        if (not math.isfinite(reference_loss_coef) or not math.isfinite(reference_max_group_mse)
                or reference_loss_coef < 0 or reference_batch_size < 1 or reference_max_group_mse < 0):
            raise ValueError("Invalid reference loss configuration")
        super().__init__(policy, **kwargs)
        self.critic_learning_rate = float(critic_learning_rate)
        self.kl_stop_threshold = float(kl_stop_threshold)
        self.critic_warmup_updates = int(critic_warmup_updates)
        self.completed_updates = 0
        self.reference_loss_coef = float(reference_loss_coef)
        self.reference_batch_size = int(reference_batch_size)
        self.reference_max_group_mse = float(reference_max_group_mse)
        self.actor_parameters = list(policy.actor.parameters()) + [policy.log_std]
        self.critic_parameters = list(policy.critic.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": self.actor_parameters, "lr": self.learning_rate, "name": "actor"},
            {"params": self.critic_parameters, "lr": self.critic_learning_rate, "name": "critic"},
        ])
        self.std_min = self._action_vector(min_std_by_action, 1.e-4, "min_std_by_action")
        self.std_max = self._action_vector(max_std_by_action, 1., "max_std_by_action")
        if (self.std_min > self.std_max).any():
            raise ValueError("Exploration lower bound exceeds upper bound")
        self.reference_action_scale = self._action_vector(reference_action_scale, 1., "reference_action_scale")
        self.reference_observations = None
        self.reference_actions = None
        self.reference_group_ids = None
        self._reference_members = []
        self.actor_input_mask = tuple(actor_input_mask)
        start, stop = self.actor_input_mask
        if (policy.actor[0].weight.shape[1] != 300 or policy.critic[0].weight.shape[1] != 252
                or (start, stop) != (205, 285)):
            raise ValueError("V5 requires Actor300/Critic252 and the audited previous-action mask")
        mask = torch.ones_like(policy.actor[0].weight)
        mask[:, start:stop] = 0.
        self._input_mask_hook = policy.actor[0].weight.register_hook(lambda gradient: gradient * mask)
        if initial_std_by_action is not None:
            initial = self._action_vector(initial_std_by_action, 1., "initial_std_by_action")
            if ((initial < self.std_min) | (initial > self.std_max)).any():
                raise ValueError("Initial exploration std must lie within configured bounds")
            with torch.no_grad():
                policy.log_std.copy_(initial.log())
        self._project_constraints()

    def _action_vector(self, value, default, name):
        tensor = torch.as_tensor(default if value is None else value, dtype=self.policy.log_std.dtype,
                                 device=self.device).expand(16).clone()
        if not torch.isfinite(tensor).all() or (tensor <= 0).any():
            raise ValueError(f"{name} must contain 16 finite positive values or one scalar")
        return tensor

    @torch.no_grad()
    def _project_constraints(self):
        """投影探索范围并清零屏蔽输入权重/Adam 动量，保证单 Actor 导出不变。"""
        start, stop = self.actor_input_mask
        weight = self.policy.actor[0].weight
        weight[:, start:stop] = 0.
        for value in self.optimizer.state.get(weight, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                value[:, start:stop] = 0.
        self.policy.log_std.clamp_(self.std_min.log(), self.std_max.log())

    def set_reference_data(self, observations, actions, successful, group_ids=None):
        """仅缓存外部完整验收成功的样本，拒绝缺失的成功标签。

        successful 必须逐行提供；这里只检查形状、标签和有限性，不自行
        声称物理成功。缓存只用于辅助动作损失，不作为 PPO 的回报/优势。
        """
        observations = torch.as_tensor(observations, device=self.device).detach()
        actions = torch.as_tensor(actions, device=self.device).detach()
        successful = torch.as_tensor(successful, device=self.device)
        count = observations.shape[0]
        if observations.shape != (count, 300) or actions.shape != (count, 16):
            raise ValueError("Reference observations/actions must have shapes [N,300]/[N,16]")
        if successful.dtype != torch.bool or successful.shape != (count,):
            raise ValueError("Reference cache requires explicit per-sample boolean success labels")
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise ValueError("Reference cache contains non-finite values")
        if group_ids is None:
            group_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        group_ids = torch.as_tensor(group_ids, device=self.device)
        if group_ids.dtype not in (torch.int32, torch.int64) or group_ids.shape != (count,) or (group_ids < 0).any():
            raise ValueError("Reference group ids must be nonnegative integers of shape [N]")
        self.reference_group_ids = group_ids[successful].long().clone()
        self._reference_members = [(self.reference_group_ids == key).nonzero().flatten()
                                   for key in torch.unique(self.reference_group_ids, sorted=True)]
        self.reference_observations = observations[successful].to(self.policy.log_std.dtype).clone()
        self.reference_actions = actions[successful].to(self.policy.log_std.dtype).clone()
        if not len(self.reference_observations):
            self.reference_observations = self.reference_actions = self.reference_group_ids = None
            self._reference_members = []

    @torch.no_grad()
    def set_reference_policy(self, reference_actor, observations, successful):
        """冻结旧 Actor 只生成成功动作缓存，不注册为部署模型的子模块。"""
        observations = torch.as_tensor(observations, dtype=self.policy.log_std.dtype, device=self.device)
        self.set_reference_data(observations, reference_actor(observations), successful)

    def _reference_loss(self):
        if self.reference_loss_coef == 0. or self.reference_observations is None:
            return self.policy.log_std.new_zeros(())
        # 先均匀选工况/阶段组，再在组内选帧；长轨迹或易任务不再压低稀有工况权重。
        groups = torch.randint(len(self._reference_members), (self.reference_batch_size,), device=self.device)
        indices = torch.empty(self.reference_batch_size, dtype=torch.long, device=self.device)
        for group, members in enumerate(self._reference_members):
            selected = (groups == group).nonzero().flatten()
            if len(selected):
                indices[selected] = members[torch.randint(len(members), (len(selected),), device=self.device)]
        predicted = self.policy.act_inference(self.reference_observations[indices])
        return ((predicted - self.reference_actions[indices]) / self.reference_action_scale).square().mean()

    @torch.no_grad()
    def _reference_diagnostics(self):
        """全缓存按工况/阶段计算动作误差，返回最大组与均衡组均值。

        这是固定成功输入上的函数保持边界，不声称等价于闭环行为通过。
        归一化尺度是显式 action scale，不能以全缓存平均掩盖稀有工况。
        """
        if self.reference_observations is None:
            return {"reference_group_mse_max": 0., "reference_group_mse_mean": 0., "reference_group_count": 0.}
        errors = []
        for start in range(0, len(self.reference_observations), 8192):
            pred = self.policy.act_inference(self.reference_observations[start:start+8192])
            errors.append(((pred-self.reference_actions[start:start+8192])/self.reference_action_scale).square().mean(-1))
        error = torch.cat(errors)
        means = torch.stack([error[members].mean() for members in self._reference_members])
        return {"reference_group_mse_max": float(means.max()), "reference_group_mse_mean": float(means.mean()),
                "reference_group_count": float(len(means))}

    @staticmethod
    def _kl(old_mean, old_std, new_mean, new_std):
        """逐样本解析 KL(old||new)，不是单次动作 log-prob 差。"""
        return (torch.log(new_std / old_std) +
                (old_std.square() + (old_mean - new_mean).square()) /
                (2. * new_std.square()) - .5).sum(-1).clamp_min(0.)

    @torch.no_grad()
    def _rollout_diagnostics(self):
        observations = self.storage.observations.flatten(0, 1)
        critic_observations = self.storage.privileged_observations.flatten(0, 1)
        old_mean = self.storage.mu.flatten(0, 1)
        old_std = self.storage.sigma.flatten(0, 1)
        predictions, kl_sum = [], 0.
        # 分块避免完整 1024×64 rollout 的诊断额外保留大矩阵。
        for start in range(0, len(observations), 8192):
            end = start + 8192
            mean = self.policy.act_inference(observations[start:end])
            kl_sum += float(self._kl(old_mean[start:end], old_std[start:end], mean,
                                    self.policy.log_std.exp().expand_as(mean)).sum())
            predictions.append(self.policy.evaluate(critic_observations[start:end]).flatten())
        returns = self.storage.returns.flatten()
        variance = returns.var(unbiased=False)
        if variance > 1.e-12:
            explained = 1. - (returns - torch.cat(predictions)).var(unbiased=False) / variance
            explained, defined = float(explained), 1.
        else:
            explained, defined = 0., 0.
        return {"kl_rollout": kl_sum / len(observations), "value_explained_variance": explained,
                "value_explained_variance_defined": defined}

    @torch.no_grad()
    def _rollout_kl(self):
        """对完整本轮 rollout 核验 Actor 变化，不以单个 minibatch 代替。"""
        observations = self.storage.observations.flatten(0, 1)
        old_mean = self.storage.mu.flatten(0, 1)
        old_std = self.storage.sigma.flatten(0, 1)
        total = 0.
        for start in range(0, len(observations), 8192):
            end = start + 8192
            mean = self.policy.act_inference(observations[start:end])
            total += float(self._kl(old_mean[start:end], old_std[start:end], mean,
                                   self.policy.log_std.exp().expand_as(mean)).sum())
        return total / len(observations)

    @torch.no_grad()
    def _accept_actor_proposal(self, previous_parameters, previous_adam):
        """同时检查完整 rollout KL 和每组成功参考误差，按 1,1/2,...,1/256 回退 Actor 参数步长。

        接受小步时保留这次完整梯度的 Adam 动量，只缩放参数位移，不更改
        固定学习率。全部失败则恢复 Actor、log_std 及其 Adam 状态；Critic
        已完成的参数/Adam 更新保持。调用方停止本轮剩余 Actor 更新。
        """
        proposed = [parameter.detach().clone() for parameter in self.actor_parameters]
        largest_finite_kl = 0.
        nonfinite = 0
        for backtracks in range(9):
            fraction = 2. ** -backtracks
            if backtracks:
                for parameter, previous, candidate in zip(self.actor_parameters, previous_parameters, proposed):
                    torch.lerp(previous, candidate, fraction, out=parameter)
            kl = self._rollout_kl()
            if torch.isfinite(torch.tensor(kl)):
                largest_finite_kl = max(largest_finite_kl, kl)
                reference_ok = True
                if kl <= self.kl_stop_threshold and self.reference_max_group_mse > 0.:
                    reference = self._reference_diagnostics()["reference_group_mse_max"]
                    reference_ok = bool(torch.isfinite(torch.tensor(reference))) and reference <= self.reference_max_group_mse
                if kl <= self.kl_stop_threshold and reference_ok:
                    return dict(accepted=True, fraction=fraction, backtracks=backtracks,
                                proposal_kl_max=largest_finite_kl, nonfinite=nonfinite)
            else:
                nonfinite += 1
        for parameter, previous, state in zip(self.actor_parameters, previous_parameters, previous_adam):
            parameter.copy_(previous)
            if state is None:
                self.optimizer.state.pop(parameter, None)
            else:
                self.optimizer.state[parameter] = state
        return dict(accepted=False, fraction=0., backtracks=8,
                    proposal_kl_max=largest_finite_kl, nonfinite=nonfinite)

    def update(self):
        """Clipped PPO 加整轮 KL 回退；Critic 继续完成计划中的所有 minibatches。

        预热只优化 Critic。正式 Actor 每次 Adam proposal 都在完整 rollout
        上核验，接受不超过 KL 上限的步长；失败则恢复 Actor/其 Adam 状态，
        冻结本轮剩余 Actor 更新。两组分别裁剪梯度，保持独立学习率。
        """
        warmup = self.completed_updates < self.critic_warmup_updates
        if self.reference_loss_coef > 0. and self.reference_observations is None and not warmup:
            raise RuntimeError("Enabled reference loss requires a verified-success reference cache")
        totals = dict(value_function=0., surrogate=0., entropy=0., reference=0.,
                      actor_gradient_norm=0., critic_gradient_norm=0.)
        max_kl, stopped, steps = 0., False, 0
        actor_steps, proposals, rejections, backtracks, nonfinite = 0, 0, 0, 0, 0
        fractions = []
        proposal_kl_max = 0.
        actor_enabled = not warmup
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs, critic_obs, actions, target_values, advantages, returns, old_log_prob,
             old_mean, old_std, _, _, _) in generator:
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1.e-8)
            self.policy.update_distribution(obs)
            kl = float(self._kl(old_mean, old_std, self.policy.action_mean,
                                self.policy.action_std).mean().detach())
            if not torch.isfinite(torch.tensor(kl)):
                raise FloatingPointError("Non-finite PPO KL")
            max_kl = max(max_kl, kl)
            if actor_enabled and kl > self.kl_stop_threshold:
                actor_enabled = False
                stopped = True
            value = self.policy.evaluate(critic_obs)
            if self.use_clipped_value_loss and not warmup:
                clipped = target_values + (value - target_values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((value - returns).square(), (clipped - returns).square()).mean()
            else:
                value_loss = (value - returns).square().mean()
            ratio = (self.policy.get_actions_log_prob(actions) - old_log_prob.squeeze(-1)).exp()
            advantage = advantages.squeeze(-1)
            surrogate = torch.maximum(-advantage * ratio,
                                      -advantage * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)).mean()
            entropy = self.policy.entropy.mean()
            reference = self._reference_loss() if actor_enabled else value.new_zeros(())
            loss = self.value_loss_coef * value_loss
            if actor_enabled:
                loss = loss + surrogate - self.entropy_coef * entropy + self.reference_loss_coef * reference
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm, error_if_nonfinite=True)
            critic_norm = nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm, error_if_nonfinite=True)
            if actor_enabled:
                previous_parameters = [parameter.detach().clone() for parameter in self.actor_parameters]
                previous_adam = [deepcopy(self.optimizer.state.get(parameter)) for parameter in self.actor_parameters]
            self.optimizer.step()
            self._project_constraints()
            if actor_enabled:
                proposals += 1
                accepted = self._accept_actor_proposal(previous_parameters, previous_adam)
                backtracks += accepted["backtracks"]
                nonfinite += accepted["nonfinite"]
                proposal_kl_max = max(proposal_kl_max, accepted["proposal_kl_max"])
                if accepted["accepted"]:
                    actor_steps += 1
                    fractions.append(accepted["fraction"])
                else:
                    rejections += 1
                    actor_enabled = False
                    stopped = True
            steps += 1
            for name, metric in (("value_function", value_loss), ("surrogate", surrogate), ("entropy", entropy),
                                 ("reference", reference), ("actor_gradient_norm", actor_norm),
                                 ("critic_gradient_norm", critic_norm)):
                totals[name] += float(metric.detach())
        result = {name: value / max(1, steps) for name, value in totals.items()}
        result.update(self._rollout_diagnostics())
        if result["kl_rollout"] > self.kl_stop_threshold + 1.e-8:
            raise RuntimeError("Actor proposal protection failed the final full-rollout KL check")
        result.update(kl_minibatch_max=max_kl, kl_stopped=float(stopped), optimizer_steps=float(steps),
                      actor_optimizer_steps=float(actor_steps), actor_proposal_steps=float(proposals),
                      actor_rejected_steps=float(rejections), actor_backtracks_total=float(backtracks),
                      actor_accepted_fraction_mean=sum(fractions)/max(1,len(fractions)),
                      actor_accepted_fraction_min=min(fractions) if fractions else 0.,
                      actor_proposal_kl_max=proposal_kl_max, actor_nonfinite_proposal_checks=float(nonfinite),
                      critic_only=float(warmup),
                      actor_learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                      critic_learning_rate=float(self.optimizer.param_groups[1]["lr"]),
                      leg_std_mean=float(self.policy.log_std[:12].exp().mean().detach()),
                      wheel_std_mean=float(self.policy.log_std[12:].exp().mean().detach()))
        result.update(self._reference_diagnostics())
        self.storage.clear()
        self.completed_updates += 1
        return result

    def _configuration(self):
        """恢复不可静默改变优化器设置；初始 std 由模型中 log_std 恢复。"""
        names = ("learning_rate", "critic_learning_rate", "kl_stop_threshold", "critic_warmup_updates",
                 "reference_loss_coef", "reference_batch_size", "reference_max_group_mse", "num_learning_epochs", "num_mini_batches",
                 "clip_param", "gamma", "lam", "value_loss_coef", "entropy_coef", "max_grad_norm",
                 "use_clipped_value_loss", "normalize_advantage_per_mini_batch")
        values = {name: getattr(self, name) for name in names}
        values.update(std_min=self.std_min.detach().cpu().tolist(), std_max=self.std_max.detach().cpu().tolist(),
                      reference_action_scale=self.reference_action_scale.detach().cpu().tolist(),
                      actor_input_mask=list(self.actor_input_mask))
        return values

    def training_state_dict(self):
        """除标准 model_state_dict/optimizer_state_dict 外的可恢复训练状态。"""
        return {"version": self.STATE_VERSION, "configuration": self._configuration(),
                "completed_updates": self.completed_updates,
                "critic_warmup_updates": self.critic_warmup_updates,
                "reference_observations": (None if self.reference_observations is None else
                                           self.reference_observations.detach().cpu().clone()),
                "reference_actions": (None if self.reference_actions is None else
                                      self.reference_actions.detach().cpu().clone()),
                "reference_group_ids": (None if self.reference_group_ids is None else
                                        self.reference_group_ids.detach().cpu().clone())}

    def load_training_state_dict(self, state):
        state = deepcopy(state)
        if (state["version"] != self.STATE_VERSION or state["configuration"] != self._configuration()
                or state["critic_warmup_updates"] != self.critic_warmup_updates):
            raise ValueError("Incompatible V5 optimizer training state")
        if int(state["completed_updates"]) < 0:
            raise ValueError("Invalid completed PPO update count")
        self.completed_updates = int(state["completed_updates"])
        observations, actions = state["reference_observations"], state["reference_actions"]
        if ((observations is None) != (actions is None)
                or (observations is None) != (state["reference_group_ids"] is None)):
            raise ValueError("Incomplete reference cache")
        if observations is not None:
            self.set_reference_data(observations, actions, torch.ones(len(observations), dtype=torch.bool),
                                    group_ids=state["reference_group_ids"])
        else:
            self.reference_observations = self.reference_actions = self.reference_group_ids = None
            self._reference_members = []
        expected_lrs = (self.learning_rate, self.critic_learning_rate)
        if len(self.optimizer.param_groups) != 2 or tuple(g["name"] for g in self.optimizer.param_groups) != ("actor", "critic"):
            raise ValueError("Resume must restore both named V5 Adam parameter groups")
        if tuple(g["lr"] for g in self.optimizer.param_groups) != expected_lrs:
            raise ValueError("Resume learning rates differ from the fixed V5 configuration")
        # 严格恢复，不以投影悄悄修复损坏的 checkpoint。
        start, stop = self.actor_input_mask
        weight = self.policy.actor[0].weight
        tensors = [weight] + [v for v in self.optimizer.state.get(weight, {}).values()
                             if isinstance(v, torch.Tensor) and v.shape == weight.shape]
        if any(torch.count_nonzero(t[:, start:stop]) for t in tensors):
            raise ValueError("Resume contains nonzero masked Actor weights or Adam moments")
        std = self.policy.log_std.exp()
        if not torch.isfinite(std).all() or ((std < self.std_min * (1. - 1.e-6)) | (std > self.std_max * (1. + 1.e-6))).any():
            raise ValueError("Resume exploration std violates configured bounds")
