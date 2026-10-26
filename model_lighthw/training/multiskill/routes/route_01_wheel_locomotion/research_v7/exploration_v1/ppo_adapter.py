"""Exploration RSL adapter: trainable towing AND non-towing residuals, 302/252/16.

The installed RSL PPO clip/ratio/value loss and GAE are used unchanged. Timeout
reward bootstrap uses explicit pre-reset terminal observations, rather than the
installed RSL helper's pre-action value. Checkpoints resume numerical policy/
optimizer state only, not an Isaac environment or an unfinished rollout.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from migration_v1.contract import sha256
from .contract import ACTOR_DIM, CRITIC_DIM, ACTION_DIM, RESET_TIME_SCALE_S, PREPARATION_SCALE_S
from .checkpoint import source_signatures
from .policy import ExplorationActorCritic
from migration_v1.schedule import permitted_catalog, validate_pre_action_conditions

VERSION = 'r1_exploration_v1_controlled_rsl_adapter'


def _source_signature():
    from migration_v1 import schedule
    paths = [Path(__file__), Path(inspect.getfile(PPO)), Path(inspect.getfile(RolloutStorage)),
             Path(schedule.__file__), schedule.SPEC, schedule.FULL_PROTOCOL]
    result = dict(source_signatures())
    result.update({str(path.resolve()): sha256(path) for path in paths})
    return result


def _model_signature(policy):
    result = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        result.update(name.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            result.update(str(value.dtype).encode()); result.update(value.numpy().tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    return result.hexdigest()


class ControlledPPOAdapter:
    """Collect supplied transitions, propose standard PPO updates, audit full KL.

`conditions` are caller-supplied PRE-ACTION [slope,payload,tow,phase]. This class
can verify values/coverage, not prove they came from a real environment. Default
mode requires all 64 conditions before the first action. Smaller CPU protocol
checks must explicitly choose diagnostic mode and never authorize training.
"""
    def __init__(self, policy, *, num_envs, rollout_steps, mode='full_first_batch',
                 execution_kind='synthetic_cpu', num_learning_epochs=1, num_mini_batches=1,
                 actor_lr=1.e-5, non_towing_lr=1.e-6, critic_lr=1.e-4, std_lr=1.e-5, non_towing_std_lr=1.e-6,
                 gamma=.995, lam=.95, clip_param=.2, value_loss_coef=1., entropy_coef=0.,
                 max_grad_norm=1., max_kl=.01, group_kl_limit=None):
        if type(policy) is not ExplorationActorCritic:
            raise ValueError('Controlled PPO requires the exploration conditional Gaussian policy')
        if mode not in ('full_first_batch', 'diagnostic'):
            raise ValueError('Explicit full_first_batch or diagnostic mode required')
        if execution_kind not in ('synthetic_cpu', 'runtime_short_validation', 'formal_training'):
            raise ValueError('Unknown exploration execution kind')
        if execution_kind != 'synthetic_cpu' and mode != 'full_first_batch':
            raise ValueError('Actual exploration runtime requires the full 64-condition first batch')
        sizes = (num_envs, rollout_steps, num_learning_epochs, num_mini_batches)
        if any(type(x) is not int or x < 1 for x in sizes):
            raise ValueError('Rollout/epoch/minibatch sizes must be positive integers')
        if num_envs*rollout_steps < 2 or num_envs*rollout_steps % num_mini_batches:
            raise ValueError('RSL minibatches must divide every rollout sample without dropping a tail')
        if mode == 'full_first_batch' and num_envs < 64:
            raise ValueError('Full first-batch mode needs at least 64 slots')
        for name, value in dict(gamma=gamma, lam=lam, clip_param=clip_param,
                value_loss_coef=value_loss_coef, entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm, max_kl=max_kl).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid PPO scalar: '+name)
        if not 0 < gamma <= 1 or not 0 <= lam <= 1 or not 0 < clip_param < 1 or max_grad_norm <= 0:
            raise ValueError('Invalid standard PPO/GAE parameter range')
        if group_kl_limit is not None and (not isinstance(group_kl_limit, (int,float))
                or not math.isfinite(group_kl_limit) or group_kl_limit < 0):
            raise ValueError('Group KL limit must be finite and nonnegative')
        self.policy = policy; self.device = next(policy.parameters()).device
        if execution_kind == 'synthetic_cpu' and self.device.type != 'cpu':
            raise ValueError('Synthetic CPU mode cannot run a GPU policy')
        if execution_kind != 'synthetic_cpu' and self.device.type != 'cuda':
            raise ValueError('Actual exploration runtime requires a GPU policy')
        self.config = dict(num_envs=num_envs, rollout_steps=rollout_steps, mode=mode,
            execution_kind=execution_kind, num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches, actor_lr=actor_lr, non_towing_lr=non_towing_lr,
            critic_lr=critic_lr, std_lr=std_lr, non_towing_std_lr=non_towing_std_lr,
            gamma=gamma, lam=lam, clip_param=clip_param, value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef, max_grad_norm=max_grad_norm, max_kl=max_kl,
            group_kl_limit=group_kl_limit, schedule='fixed', normalize_advantage_per_mini_batch=False)
        self.algorithm = PPO(policy, num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches, clip_param=clip_param, gamma=gamma, lam=lam,
            value_loss_coef=value_loss_coef, entropy_coef=entropy_coef, learning_rate=actor_lr,
            max_grad_norm=max_grad_norm, use_clipped_value_loss=True, schedule='fixed', desired_kl=None,
            device=str(self.device), normalize_advantage_per_mini_batch=False)
        # RSL briefly constructs an empty Adam over all parameters. Replace it
        # BEFORE any act/update; it never owns optimizer moments or takes a step.
        self.algorithm.optimizer = policy.make_optimizer(actor_lr=actor_lr, non_towing_lr=non_towing_lr,
            critic_lr=critic_lr, std_lr=std_lr, non_towing_std_lr=non_towing_std_lr)
        self.algorithm.init_storage('rl', num_envs, rollout_steps, [ACTOR_DIM], [CRITIC_DIM], [ACTION_DIM])
        self.sources = _source_signature(); self.catalog = permitted_catalog()
        self.first_condition_review = None; self.fixed_conditions = None
        self.condition_history = []; self.clock_history = []; self.timeout_bootstrap_history = []
        self.pending = False; self.returns_ready = False; self.collection_signature = None
        self.last_values = None; self.last_update_report = None
        self.counters = dict(update_attempts=0, accepted_updates=0, rejected_updates=0,
            rsl_update_calls=0, observed_transition_frames=0)
        policy.assert_invariants(self.optimizer)

    @property
    def optimizer(self):
        return self.algorithm.optimizer

    @property
    def storage(self):
        return self.algorithm.storage

    def _matrix(self, value, width, name):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.config['num_envs'],width)
                or value.device != self.device or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite float32 [num_envs,{width}] on the policy device')

    def _mask(self, value, name):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.config['num_envs'],)
                or value.dtype != torch.bool or value.device != self.device):
            raise ValueError(name+' must be a boolean vector on the policy device')

    def _unchanged_collection(self):
        if self.collection_signature != _model_signature(self.policy):
            raise ValueError('Policy/Critic changed while collecting the current rollout')

    @torch.no_grad()
    def act(self, observations, critic_observations, pre_action_conditions, *, pre_action_clock_seconds):
        if self.pending or self.returns_ready or self.storage.step >= self.config['rollout_steps']:
            raise RuntimeError('Record the pending step or finish the current full rollout before act')
        self._matrix(observations,ACTOR_DIM,'Actor observations')
        self._matrix(critic_observations,CRITIC_DIM,'Critic observations')
        self._matrix(pre_action_conditions,4,'Pre-action conditions')
        self._matrix(pre_action_clock_seconds,2,'Pre-action elapsed/preparation seconds')
        if (pre_action_clock_seconds < 0).any():
            raise ValueError('Causal task clocks must be nonnegative')
        scales = pre_action_clock_seconds.new_tensor([RESET_TIME_SCALE_S, PREPARATION_SCALE_S])
        expected_clocks = (pre_action_clock_seconds/scales).clamp(0.,1.)
        if not torch.allclose(observations[:,300:302], expected_clocks, atol=1.e-6, rtol=0.):
            raise ValueError('Actor causal clock columns disagree with actual pre-action metadata')
        first = self.first_condition_review is None
        review = validate_pre_action_conditions(pre_action_conditions,
            mode=self.config['mode'] if first else 'diagnostic')
        if not torch.allclose(observations[:,298], pre_action_conditions[:,1]/15.,atol=1.e-6,rtol=0.) or not torch.allclose(
                observations[:,299],pre_action_conditions[:,2]/40.,atol=1.e-6,rtol=0.):
            raise ValueError('Actor nominal contexts disagree with pre-action condition metadata')
        if self.fixed_conditions is not None and not torch.equal(pre_action_conditions[:,:3],self.fixed_conditions):
            raise ValueError('Condition assignment changed across collection/reset; only phase may change')
        if self.storage.step == 0:
            self.policy.assert_invariants(self.optimizer)
            self.collection_signature = _model_signature(self.policy)
        # RSL keeps observation references until record_step, so freeze copies
        # before an in-place simulator observation/reset update can replace them.
        actions = self.algorithm.act(observations.detach().clone(),critic_observations.detach().clone())
        self.pending = True
        self.condition_history.append(pre_action_conditions.detach().clone())
        self.clock_history.append(pre_action_clock_seconds.detach().clone())
        if first:
            self.first_condition_review = review
            self.fixed_conditions = pre_action_conditions[:,:3].detach().clone()
        return actions.detach().clone()

    @torch.no_grad()
    def record_step(self, rewards, terminated, truncated, *, terminal_critic_observations=None):
        if not self.pending:
            raise RuntimeError('record_step requires exactly one preceding act')
        if (not isinstance(rewards,torch.Tensor) or rewards.shape != (self.config['num_envs'],)
                or rewards.dtype != torch.float32 or rewards.device != self.device or not torch.isfinite(rewards).all()):
            raise ValueError('Rewards must be a finite float32 vector on the policy device')
        self._mask(terminated,'terminated'); self._mask(truncated,'truncated')
        timeouts = truncated & ~terminated  # Physical failure takes precedence.
        bootstrap = torch.zeros_like(rewards)
        if timeouts.any():
            if terminal_critic_observations is None:
                raise ValueError('Timeout requires actual physical pre-reset, post-command-preview Critic observations')
            self._matrix(terminal_critic_observations,CRITIC_DIM,'Terminal Critic observations')
            bootstrap[timeouts] = self.policy.evaluate(terminal_critic_observations[timeouts]).squeeze(-1)
        adjusted = rewards+self.config['gamma']*bootstrap
        # Empty infos avoids RSL's second timeout correction using V(s_t).
        self.algorithm.process_env_step(adjusted,terminated|truncated,{})
        self.timeout_bootstrap_history.append(dict(mask=timeouts.detach().clone(),
            terminal_values=bootstrap.detach().clone(),raw_rewards=rewards.detach().clone()))
        self.pending = False
        self.counters['observed_transition_frames'] += self.config['num_envs']

    @torch.no_grad()
    def compute_returns(self, last_critic_observations):
        if self.pending or self.returns_ready or self.storage.step != self.config['rollout_steps']:
            raise RuntimeError('Returns require a complete recorded rollout and explicit last state')
        self._matrix(last_critic_observations,CRITIC_DIM,'Last-state Critic observations')
        self._unchanged_collection()
        self.last_values = self.policy.evaluate(last_critic_observations).detach().clone()
        self.algorithm.compute_returns(last_critic_observations.detach().clone())
        if not torch.isfinite(self.storage.returns).all() or not torch.isfinite(self.storage.advantages).all():
            raise ValueError('Nonfinite RSL GAE/returns')
        self.returns_ready = True
        return dict(last_values=self.last_values.clone(),timeout_count=sum(int(x['mask'].sum())
            for x in self.timeout_bootstrap_history), bootstrap='explicit_last_state_and_post_command_terminal_preview')

    def _kl_groups(self):
        conditions = torch.stack(self.condition_history).flatten(0,1)
        classes = dict(unloaded=(conditions[:,1]==0)&(conditions[:,2]==0),
            payload_only=conditions[:,1]>0,towing_only=conditions[:,2]>0)
        groups = dict(classes)
        for name,mask in classes.items():
            for phase in range(4):
                groups[f'{name}/phase{phase}'] = mask&(conditions[:,3]==phase)
        # Only actually observed case x phase groups; absence is not failure.
        for row in torch.unique(conditions,dim=0):
            label = 'case/'+','.join(f'{float(value):g}' for value in row)
            groups[label] = (conditions==row).all(-1)
        return groups

    def adam_parameter_steps(self):
        """Actual named Adam step counters; no inference from configured epochs."""
        names = {id(parameter): name for name, parameter in self.policy.named_parameters()}
        result = {}
        for group in self.optimizer.param_groups:
            for parameter in group['params']:
                step = self.optimizer.state.get(parameter, {}).get('step', 0)
                result[names[id(parameter)]] = int(step.item() if isinstance(step, torch.Tensor) else step)
        return result

    def update(self):
        if not self.returns_ready or self.pending:
            raise RuntimeError('PPO update requires compute_returns for a full rollout')
        self._unchanged_collection(); self.policy.assert_invariants(self.optimizer)
        if _source_signature() != self.sources:
            raise ValueError('PPO integration sources changed during this run')
        observations = self.storage.observations.flatten(0,1).detach().clone()
        groups = self._kl_groups()
        limits = (None if self.config['group_kl_limit'] is None else
            {name:self.config['group_kl_limit'] for name,mask in groups.items() if mask.any()})
        transaction = self.policy.begin_update(self.optimizer,observations,groups=groups)
        # Compare with the exact sampled rollout Gaussian, rather than a newly
        # batched recomputation of its baseline (row/batch arithmetic can differ).
        transaction.old_mean.copy_(self.storage.mu.flatten(0,1))
        transaction.old_std.copy_(self.storage.sigma.flatten(0,1))
        self.counters['update_attempts'] += 1
        adam_before = self.adam_parameter_steps()
        adam_proposed = None
        try:
            with transaction:
                self.counters['rsl_update_calls'] += 1
                try:
                    losses = self.algorithm.update()  # Standard RSL ratio/clip/GAE path.
                finally:
                    # Read before transaction __exit__ can roll back a failed partial update.
                    adam_proposed = self.adam_parameter_steps()
                if any(not math.isfinite(value) for value in losses.values()):
                    raise ValueError('Nonfinite RSL update losses')
                report = transaction.finish(max_kl=self.config['max_kl'],group_max_kl=limits)
            self.counters['accepted_updates' if report['accepted'] else 'rejected_updates'] += 1
            report.update(losses=losses,execution_kind=self.config['execution_kind'],
                configured_minibatch_steps=self.config['num_learning_epochs']*self.config['num_mini_batches'],
                actual_pre_action_review=deepcopy(self.first_condition_review),counters=deepcopy(self.counters),
                adam_parameter_steps_before=adam_before, adam_parameter_steps_proposed=adam_proposed,
                adam_parameter_steps_retained=self.adam_parameter_steps(),
                environment_state_verified=False,training_runtime_ready=False,final_behavior_pass=False)
            self.last_update_report = deepcopy(report)
            return report
        except Exception as error:
            self.counters['rejected_updates'] += 1
            self.last_update_report = dict(accepted=False,reason='update_exception',detail=str(error),
                counters=deepcopy(self.counters), adam_parameter_steps_before=adam_before,
                adam_parameter_steps_proposed=adam_proposed, adam_parameter_steps_retained=self.adam_parameter_steps(),
                training_runtime_ready=False,final_behavior_pass=False)
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.storage.clear(); self.algorithm.transition.clear()
            self.condition_history.clear(); self.clock_history.clear(); self.timeout_bootstrap_history.clear()
            self.pending = False; self.returns_ready = False; self.collection_signature = None

    def _checkpoint_boundary(self):
        if self.pending or self.storage.step or self.returns_ready:
            raise RuntimeError('Checkpoint only at an empty rollout boundary; environment state is not saved')

    def checkpoint(self):
        self._checkpoint_boundary()
        if _source_signature() != self.sources:
            raise ValueError('PPO integration source drift')
        return deepcopy(dict(schema=VERSION,config=self.config,sources=self.sources,
            catalog_source_sha256=self.catalog['source_sha256'],policy=self.policy.checkpoint(self.optimizer),
            counters=self.counters,first_condition_review=self.first_condition_review,
            fixed_conditions=self.fixed_conditions,last_update_report=self.last_update_report,
            environment_state_saved=False,training_runtime_ready=False,final_behavior_pass=False))

    def _validate_saved_optimizer(self, saved):
        groups = saved.get('policy',{}).get('optimizer_state_dict',{}).get('param_groups',[])
        current = self.optimizer.state_dict()['param_groups']
        if len(groups) != len(current):
            raise ValueError('Saved optimizer requires five fixed Adam groups')
        for group, expected in zip(groups, current):
            # Fixed schedule: every optimizer hyperparameter must match the actual
            # newly constructed Adam, including LR, betas, epsilon and weight decay.
            for name, value in expected.items():
                if name != 'params' and group.get(name) != value:
                    raise ValueError('Saved fixed Adam hyperparameter differs: '+name)
            if set(group) != set(expected):
                raise ValueError('Saved Adam hyperparameter keys differ')

    def restore_checkpoint(self, checkpoint, *, allow_short_to_formal=False):
        self._checkpoint_boundary()
        saved_config = deepcopy(checkpoint.get('config'))
        if allow_short_to_formal:
            if (self.config['execution_kind'] != 'formal_training' or not isinstance(saved_config,dict)
                    or saved_config.get('execution_kind') != 'runtime_short_validation'):
                raise ValueError('Only explicit short-validation to formal resume conversion is allowed')
            saved_config['execution_kind'] = 'formal_training'
        if saved_config != self.config:
            raise ValueError('PPO adapter restore contract mismatch: config')
        self._validate_saved_optimizer(checkpoint)
        for name,expected in [('schema',VERSION),('sources',self.sources),
                ('catalog_source_sha256',self.catalog['source_sha256']),('environment_state_saved',False),
                ('training_runtime_ready',False),('final_behavior_pass',False)]:
            if checkpoint.get(name) != expected:
                raise ValueError('PPO adapter restore contract mismatch: '+name)
        if _source_signature() != self.sources:
            raise ValueError('PPO integration source drift')
        counts=checkpoint.get('counters',{})
        if (set(counts)!=set(self.counters) or any(type(v) is not int or v<0 for v in counts.values())
                or counts['accepted_updates']+counts['rejected_updates']!=counts['update_attempts']
                or counts['rsl_update_calls']!=counts['update_attempts']):
            raise ValueError('Invalid saved PPO update counters')
        fixed=checkpoint.get('fixed_conditions')
        review=checkpoint.get('first_condition_review')
        if (fixed is None)!=(review is None):
            raise ValueError('Incomplete saved first-condition audit')
        if fixed is not None:
            if not isinstance(fixed,torch.Tensor) or fixed.shape!=(self.config['num_envs'],3):
                raise ValueError('Invalid saved condition assignment')
            rows=torch.cat((fixed,torch.zeros(len(fixed),1,device=fixed.device)),dim=1)
            validate_pre_action_conditions(rows,mode=self.config['mode'])
        self.policy.restore_checkpoint(checkpoint['policy'],self.optimizer)
        self.counters=deepcopy(counts); self.first_condition_review=deepcopy(review)
        self.fixed_conditions=None if fixed is None else fixed.detach().clone().to(self.device)
        self.last_update_report=deepcopy(checkpoint.get('last_update_report'))

    def save(self, path):
        """Create a new file only; caller records its path/hash in the run manifest."""
        snapshot=self.checkpoint(); path=Path(path)
        with path.open('xb') as stream:
            torch.save(snapshot,stream)
        return dict(path=str(path.resolve()),sha256=sha256(path),environment_state_saved=False,
            training_runtime_ready=False,final_behavior_pass=False)

    def load(self, path, *, allow_short_to_formal=False):
        # Stage on CPU: load_state_dict casts model tensors and Adam moments to
        # their owning device, while non-capturable Adam step scalars stay CPU.
        self.restore_checkpoint(torch.load(Path(path),map_location='cpu',weights_only=True),
            allow_short_to_formal=allow_short_to_formal)
