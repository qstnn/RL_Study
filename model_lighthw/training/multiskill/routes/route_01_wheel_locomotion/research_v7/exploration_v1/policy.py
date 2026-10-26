"""Unified deployable residual policy and standard RSL Gaussian/Critic interface.

Only numeric policy methods and the previously reviewed whole-update transaction
are reused. No migration behavior gate is imported or weakened. No optimizer
step, environment launch, successful-behavior label or training loop lives here.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from migration_v1.contract import validate_actor_mask
from migration_v1.policy import tensor_fingerprint, _validate_network
from migration_v1.ppo_policy import (TowingResidualActorCritic, PolicyUpdateTransaction,
    critic_contract_252, _finite_matrix, _rng_state, _restore_rng, _fingerprint)
from .contract import (ACTOR_DIM, BASE_OBS_DIM, CRITIC_DIM, ACTION_DIM, MASK_START,
    MASK_END, PAYLOAD_COLUMN, TOWED_COLUMN, VERSION, contract, source_signatures, sha256)


def _residual():
    model = nn.Sequential(nn.Linear(ACTOR_DIM,512), nn.ELU(), nn.Linear(512,256),
        nn.ELU(), nn.Linear(256,128), nn.ELU(), nn.Linear(128,ACTION_DIM))
    with torch.no_grad():
        model[-1].weight.zero_(); model[-1].bias.zero_()
    _mask(model)
    return model


def _mask(model, optimizer=None):
    first = model[0].weight
    if first.shape != (512,ACTOR_DIM):
        raise ValueError('Residual input layer must be 302 -> 512')
    with torch.no_grad():
        first[:,MASK_START:MASK_END].zero_()
        if optimizer is not None:
            for value in optimizer.state.get(first, {}).values():
                if isinstance(value,torch.Tensor) and value.shape == first.shape:
                    value[:,MASK_START:MASK_END].zero_()
    if not hasattr(model, '_exploration_mask_hook'):
        # Build on the actual gradient device; moving the model cannot stale it.
        def mask_gradient(gradient):
            gradient = gradient.clone(); gradient[:,MASK_START:MASK_END] = 0.
            return gradient
        model._exploration_mask_hook = first.register_hook(mask_gradient)


def _check_residual(model, optimizer=None):
    if model[0].weight.shape != (512,ACTOR_DIM) or model[-1].weight.shape != (ACTION_DIM,128):
        raise ValueError('Residual architecture differs')
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Residual parameters must be finite float32')
    if torch.count_nonzero(model[0].weight[:,MASK_START:MASK_END]):
        raise ValueError('Residual reintroduced masked previous actions')
    if optimizer is not None:
        for value in optimizer.state.get(model[0].weight, {}).values():
            if isinstance(value,torch.Tensor) and value.shape == model[0].weight.shape:
                if torch.count_nonzero(value[:,MASK_START:MASK_END]):
                    raise ValueError('Residual Adam state reintroduced masked previous actions')


class UnifiedResidualActor(nn.Module):
    def __init__(self, base_actor, *, towing_residual=None):
        super().__init__()
        _validate_network(base_actor)
        validate_actor_mask(base_actor)
        if base_actor[0].weight.shape != (512,BASE_OBS_DIM) or base_actor[-1].weight.shape != (ACTION_DIM,128):
            raise ValueError('Frozen base must retain the original 300/512/256/128/16 architecture')
        first = next(base_actor.parameters())
        self.base_actor = deepcopy(base_actor).eval().requires_grad_(False)
        self.frozen_base_sha256 = tensor_fingerprint(self.base_actor)
        self.towing_residual = _residual().to(first.device)
        self.non_towing_residual = _residual().to(first.device)
        self.towing_initialization = 'zero'
        if towing_residual is not None:
            _validate_network(towing_residual)
            validate_actor_mask(towing_residual)
            old = towing_residual.state_dict()
            new = self.towing_residual.state_dict()
            if set(old) != set(new):
                raise ValueError('Towing initializer architecture differs')
            with torch.no_grad():
                for name, value in new.items():
                    if name == '0.weight':
                        if old[name].shape != (512,BASE_OBS_DIM):
                            raise ValueError('Towing initializer must use the original 300 observations')
                        value.zero_(); value[:,:BASE_OBS_DIM].copy_(old[name])
                    elif old[name].shape == value.shape:
                        value.copy_(old[name])
                    else:
                        raise ValueError('Towing initializer tensor shape differs: '+name)
            self.towing_initialization = 'supplied_300_residual_clock_columns_zero'
        self.assert_invariants()

    def train(self, mode=True):
        super().train(mode); self.base_actor.eval()
        return self

    def forward(self, observations):
        _finite_matrix(observations,ACTOR_DIM,'Exploration Actor observation')
        payload, tow = observations[:,PAYLOAD_COLUMN],observations[:,TOWED_COLUMN]
        if ((payload < 0)|(payload > 1)|(tow < 0)|(tow > 1)|((payload > 0)&(tow > 0))).any():
            raise ValueError('Nominal payload/towing context is mutually exclusive and bounded')
        if ((observations[:,BASE_OBS_DIM:] < 0)|(observations[:,BASE_OBS_DIM:] > 1)).any():
            raise ValueError('Declared task clocks must be normalized into [0,1]')
        visible = observations.clone(); visible[:,MASK_START:MASK_END] = 0.
        base = self.base_actor(visible[:,:BASE_OBS_DIM])
        delta = torch.where((tow > 0)[:,None], self.towing_residual(visible),
                           self.non_towing_residual(visible))
        result=base+delta
        if not torch.isfinite(result).all(): raise ValueError('Exploration Actor output is nonfinite')
        return result

    def trainable_parameters(self):
        return (p for branch in (self.towing_residual,self.non_towing_residual) for p in branch.parameters())

    def project_constraints(self, optimizer=None):
        for branch in (self.towing_residual,self.non_towing_residual): _mask(branch,optimizer)
        self.assert_invariants(optimizer)

    def assert_invariants(self, optimizer=None):
        validate_actor_mask(self.base_actor)
        if (self.base_actor.training or any(p.requires_grad for p in self.base_actor.parameters())
                or tensor_fingerprint(self.base_actor) != self.frozen_base_sha256):
            raise ValueError('Frozen source base changed')
        for branch in (self.towing_residual,self.non_towing_residual): _check_residual(branch,optimizer)
        if optimizer is not None:
            protected = {id(p) for p in self.base_actor.parameters()}
            if any(id(p) in protected for g in optimizer.param_groups for p in g['params']):
                raise ValueError('Optimizer must exclude the frozen source base')

    def get_extra_state(self):
        return dict(schema=VERSION,contract=contract(),base_sha256=self.frozen_base_sha256,
            towing_initialization=self.towing_initialization)

    def set_extra_state(self,state):
        if state != self.get_extra_state(): raise ValueError('Exploration mean graph contract differs')


class ExplorationActorCritic(TowingResidualActorCritic):
    """Nonrecurrent standard Gaussian, 302-D mean and fresh 252-D Critic.

    Generic Normal/logprob/entropy/Critic/reset and transactional load methods
    are inherited; all dimension, protection and parameter-group policies here
    are explicit. The base_std buffer is an immutable initial reference only;
    both branch log-std parameters, including non-towing, are trainable.
    """
    def __init__(self,actor,*,base_std,towing_std=None,critic_dim=CRITIC_DIM,
                 critic_hidden_dims=(512,256,128),inference_mode='row'):
        nn.Module.__init__(self)
        if type(actor) is not UnifiedResidualActor or critic_dim != CRITIC_DIM:
            raise ValueError('Expected the new 302 Actor and explicit 252 Critic')
        if inference_mode not in ('row','batch') or not critic_hidden_dims or any(type(n) is not int or n<1 for n in critic_hidden_dims):
            raise ValueError('Explicit inference mode and positive Critic widths are required')
        self.actor = actor; self.inference_mode = inference_mode
        first = next(actor.parameters())
        def standard(value):
            result = torch.as_tensor(value,dtype=torch.float32,device=first.device).detach().clone()
            if result.shape != (ACTION_DIM,) or not torch.isfinite(result).all() or not (result > 0).all():
                raise ValueError('Each initial standard deviation must be a positive finite 16-vector')
            return result
        base = standard(base_std); tow = standard(base_std if towing_std is None else towing_std)
        self.register_buffer('base_std',base)
        self.frozen_base_std_sha256 = _fingerprint(base)
        self.towing_log_std = nn.Parameter(tow.log())
        self.non_towing_log_std = nn.Parameter(base.log().clone())
        layers=[]; sizes=[CRITIC_DIM,*critic_hidden_dims,1]
        for index,(left,right) in enumerate(zip(sizes,sizes[1:])):
            layers.append(nn.Linear(left,right))
            if index < len(sizes)-2: layers.append(nn.ELU())
        self.critic=nn.Sequential(*layers).to(first.device)
        self.critic_hidden_dims=tuple(critic_hidden_dims); self.distribution=None
        self.source_record=dict(kind='supplied_tensors_only',checkpoint_provenance_verified=False)
        self.implementation_sha256=sha256(Path(__file__))
        self.sources=source_signatures()
        self.assert_invariants()

    def act_inference(self,observations):
        _finite_matrix(observations,ACTOR_DIM,'Exploration Actor observation')
        return (self.actor(observations) if self.inference_mode=='batch' else
                torch.cat([self.actor(row[None]) for row in observations]))

    def distribution_parameters(self,observations):
        mean=self.act_inference(observations)
        std=torch.where((observations[:,TOWED_COLUMN]>0)[:,None],
            self.towing_log_std.exp().expand_as(mean),self.non_towing_log_std.exp().expand_as(mean))
        if not torch.isfinite(std).all() or not (std>0).all(): raise ValueError('Invalid exploration Gaussian')
        return mean,std

    def parameter_groups(self):
        return [dict(name='towing_residual',params=list(self.actor.towing_residual.parameters())),
            dict(name='non_towing_residual',params=list(self.actor.non_towing_residual.parameters())),
            dict(name='critic',params=list(self.critic.parameters())),
            dict(name='towing_std',params=[self.towing_log_std]),
            dict(name='non_towing_std',params=[self.non_towing_log_std])]

    def make_optimizer(self,*,actor_lr=1.e-5,non_towing_lr=1.e-6,critic_lr=1.e-4,
                       std_lr=1.e-5,non_towing_std_lr=1.e-6):
        rates=(actor_lr,non_towing_lr,critic_lr,std_lr,non_towing_std_lr)
        if any(not isinstance(x,(float,int)) or not 0<x<float('inf') for x in rates):
            raise ValueError('All explicit optimizer rates must be positive finite')
        groups=self.parameter_groups()
        for group,rate in zip(groups,rates): group['lr']=float(rate)
        optimizer=torch.optim.Adam(groups); self.assert_invariants(optimizer)
        return optimizer

    def assert_invariants(self,optimizer=None):
        self.actor.assert_invariants(optimizer)
        if self.base_std.requires_grad or _fingerprint(self.base_std)!=self.frozen_base_std_sha256:
            raise ValueError('Initial standard-deviation provenance changed')
        for p in (*self.critic.parameters(),self.towing_log_std,self.non_towing_log_std):
            if p.dtype!=torch.float32 or not torch.isfinite(p).all(): raise ValueError('Nonfinite trainable policy tensor')
        for value in (self.towing_log_std.exp(),self.non_towing_log_std.exp()):
            if not torch.isfinite(value).all() or not (value>0).all(): raise ValueError('Invalid Gaussian variance')
        if optimizer is not None:
            expected=self.parameter_groups()
            if len(optimizer.param_groups)!=len(expected): raise ValueError('Expected five explicit parameter groups')
            for current,target in zip(optimizer.param_groups,expected):
                if current.get('name')!=target['name'] or [id(p) for p in current['params']]!=[id(p) for p in target['params']]:
                    raise ValueError('Optimizer parameter ownership/order differs')
            allowed={p for group in expected for p in group['params']}
            if any(p not in allowed for p in optimizer.state): raise ValueError('Foreign optimizer parameter')
            for p,state in optimizer.state.items():
                for name,value in state.items():
                    if isinstance(value,torch.Tensor) and (not torch.isfinite(value).all() or (name!='step' and value.shape!=p.shape)):
                        raise ValueError('Nonfinite or malformed optimizer state')

    def get_extra_state(self):
        return dict(schema=VERSION,contract=contract(),critic_contract=critic_contract_252(),
            critic_hidden_dims=list(self.critic_hidden_dims),inference_mode=self.inference_mode,
            source=deepcopy(self.source_record),sources=deepcopy(self.sources),
            initial_non_towing_std_sha256=self.frozen_base_std_sha256,
            distribution='nominal_towing_conditional_diagonal_Normal_raw_actions',
            behavior_accepted=False,formal_ppo_authorized=False)

    def set_extra_state(self,state):
        if state!=self.get_extra_state(): raise ValueError('Exploration source/schema/distribution differs')

    def _validate_restore(self,saved):
        current=self.state_dict()
        if set(saved)!=set(current): raise ValueError('Exploration restore keys differ')
        self.set_extra_state(saved['_extra_state']); self.actor.set_extra_state(saved['actor._extra_state'])
        for name,value in current.items():
            candidate=saved[name]
            if isinstance(value,torch.Tensor):
                if not isinstance(candidate,torch.Tensor) or candidate.shape!=value.shape or candidate.dtype!=value.dtype or not torch.isfinite(candidate).all():
                    raise ValueError('Exploration restore tensor differs: '+name)
                if (name.startswith('actor.base_actor.') or name=='base_std') and not torch.equal(candidate.to(value.device),value):
                    raise ValueError('Restore replaced protected initialization')
        for name in ('actor.towing_residual.0.weight','actor.non_towing_residual.0.weight'):
            if torch.count_nonzero(saved[name][:,MASK_START:MASK_END]): raise ValueError('Restore unmasked action history')

    def checkpoint(self,optimizer=None):
        self.assert_invariants(optimizer)
        if source_signatures()!=self.sources: raise ValueError('Exploration policy implementation sources changed')
        return deepcopy(dict(schema=VERSION,model_state_dict=self.state_dict(),
            optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
            rng_state=_rng_state(self.base_std.device),behavior_accepted=False,formal_ppo_authorized=False))

    def restore_checkpoint(self,checkpoint,optimizer=None):
        if checkpoint.get('schema')!=VERSION or checkpoint.get('formal_ppo_authorized') is not False or checkpoint.get('behavior_accepted') is not False:
            raise ValueError('Unexpected exploration checkpoint schema')
        if (checkpoint.get('optimizer_state_dict') is None)!=(optimizer is None): raise ValueError('Optimizer presence differs')
        self.assert_invariants(optimizer); self._validate_restore(checkpoint['model_state_dict'])
        original=self.checkpoint(optimizer)
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.assert_invariants(optimizer); _restore_rng(checkpoint['rng_state'],self.base_std.device)
        except Exception:
            self.load_state_dict(original['model_state_dict'])
            if optimizer is not None: optimizer.load_state_dict(original['optimizer_state_dict'])
            _restore_rng(original['rng_state'],self.base_std.device)
            raise

    def begin_update(self,optimizer,observations,*,groups=None):
        return PolicyUpdateTransaction(self,optimizer,observations,groups=groups)
