"""CPU-testable Gaussian/Critic adapter; no environment, PPO loop or admission grant.

Implements the installed nonrecurrent RSL-RL policy protocol. RSL's default
Adam(policy.parameters()) is deliberately rejected: use make_optimizer().
An update transaction rolls back ALL proposed parameters/Adam/RNG on rejection;
separate Critic warmup belongs outside the rejected Actor transaction.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from .contract import CONTRACT, sha256
from .policy import FrozenBaseTowingResidualActor, TOWED_COLUMN

VERSION = 'r1_migration_v1_conditional_gaussian_fresh_critic'


def critic_contract_252():
    """Declared single-frame layout; actual environment agreement is NOT verified.

236 named reference columns followed by the V6 16D privileged task context.
No old Critic weights or runtime normalization are inherited.
"""
    terms = [
        ('base_lin_vel', 3, 'body-frame linear velocity, scale 2, input clip [-100,100]'),
        ('base_ang_vel', 3, 'body-frame angular velocity, scale .5, input clip [-100,100]'),
        ('projected_gravity', 3, 'body-frame gravity direction, scale 1'),
        ('velocity_commands', 3, 'current applied x/y/yaw command, scale 1'),
        ('joint_pos', 16, 'named relative-to-default joint angles; wheel positions zero'),
        ('joint_vel', 16, 'named relative joint velocity, scale .05'),
        ('actions', 16, 'previous raw action in deployed action order; Critic only, scale 1'),
        ('height_scan', 176, 'reference height_scanner ray order, native height_scan, clip [-1,1]'),
        ('surface_normal_b', 3, 'privileged terrain normal in robot body frame'),
        ('payload_context', 1, 'nominal payload kg / 15'),
        ('heading_error', 1, 'heading relative to reset, radians'),
        ('towed_context', 1, 'nominal trailer kg / 40'),
        ('hold_route_displacement', 1, 'signed route displacement from hold origin / .05 m'),
        ('hold_lateral_displacement', 1, 'signed lateral displacement from hold origin / .05 m'),
        ('hold_elapsed', 1, 'time since applied command reached zero / 7 s'),
        ('requested_vx', 1, 'requested command before slew / 3.1 m/s'),
        ('phase', 4, 'command task phase 0/1/2/3 one-hot; not a teacher-ready state'),
        ('phase_elapsed', 1, 'elapsed command phase time / 30 s'),
        ('remaining_distance', 1, 'nonnegative remaining distance / declared goal; zero when inactive'),
    ]
    offset = 0; fields = []
    for name, width, semantics in terms:
        fields.append(dict(name=name, columns=[offset, offset+width], semantics=semantics))
        offset += width
    assert offset == 252
    return dict(version='migration_v1_declared_critic252_v1', dimension=252,
        history_frames=1, fields=fields, observation_joint_order=deepcopy(CONTRACT['observation_joint_order']),
        action_order=deepcopy(CONTRACT['action_order']), empirical_normalization=False,
        runtime_environment_verified=False, initialization='fresh_no_old_critic_transfer')


def _fingerprint(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode()+value.numpy().tobytes()).hexdigest()


def _finite_matrix(value, width, label):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or not len(value)
            or value.shape[1] != width or value.dtype != torch.float32 or not torch.isfinite(value).all()):
        raise ValueError(f'{label} requires nonempty finite float32 [N,{width}]')


def _rng_state(device):
    state = dict(cpu=torch.random.get_rng_state().clone())
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device).clone()
    return state


def _restore_rng(state, device):
    torch.random.set_rng_state(state['cpu'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


class TowingResidualActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, actor, *, base_std, critic_dim=252, critic_hidden_dims=(512, 256, 128),
                 inference_mode='row'):
        super().__init__()
        if type(actor) is not FrozenBaseTowingResidualActor:
            raise ValueError('Expected the verified FrozenBaseTowingResidualActor type')
        actor.assert_invariants()
        if critic_dim != 252 or isinstance(critic_dim, bool):
            raise ValueError('Fresh Critic requires the explicitly declared 252-column contract')
        if inference_mode not in ('row', 'batch'):
            raise ValueError('Inference mode must be explicit row or batch')
        if not critic_hidden_dims or any(type(n) is not int or n <= 0 for n in critic_hidden_dims):
            raise ValueError('Critic hidden widths must be positive integers')
        self.actor = actor  # Own the supplied verified mean graph; do not remap actor[0].
        first = next(actor.parameters()); self.inference_mode = inference_mode
        standard = torch.as_tensor(base_std, dtype=torch.float32, device=first.device).detach().clone()
        if standard.shape != (16,) or not torch.isfinite(standard).all() or not (standard > 0).all():
            raise ValueError('base_std must contain 16 finite positive values')
        self.register_buffer('base_std', standard)
        self.frozen_base_std_sha256 = _fingerprint(standard)
        self.towing_log_std = nn.Parameter(standard.log().clone())
        layers = []; sizes = [252, *critic_hidden_dims, 1]
        for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(sizes)-2:
                layers.append(nn.ELU())
        self.critic = nn.Sequential(*layers).to(first.device)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.distribution = None
        self.source_record = dict(kind='supplied_tensors_only', checkpoint_provenance_verified=False)
        self.implementation_sha256 = sha256(Path(__file__))
        self.assert_invariants()

    @classmethod
    def from_student_checkpoint(cls, checkpoint, *, base_std, device='cpu', inference_mode='row', **kwargs):
        """Only this constructor claims the existing strict student loader was run."""
        from .student import load_student
        interface, manifest = load_student(checkpoint, device=device, inference_mode=inference_mode)
        result = cls(interface.actor, base_std=base_std, inference_mode=inference_mode, **kwargs)
        path = Path(checkpoint).resolve(strict=True)
        result.source_record = dict(kind='strict_student_loader', checkpoint_provenance_verified=True,
            student_checkpoint=str(path), student_checkpoint_sha256=sha256(path),
            student_manifest_sha256=sha256(path.with_name('manifest.json')),
            baseline=deepcopy(manifest['baseline']), sources=deepcopy(manifest['sources']))
        return result

    def train(self, mode=True):
        super().train(mode)
        self.actor.base_actor.eval()
        return self

    def forward(self, observations):
        return self.act_inference(observations)

    def reset(self, dones=None):
        return None

    def act_inference(self, observations):
        _finite_matrix(observations, 300, 'Actor observation')
        return (self.actor(observations) if self.inference_mode == 'batch' else
            torch.cat([self.actor(row[None]) for row in observations], dim=0))

    def distribution_parameters(self, observations):
        mean = self.act_inference(observations)
        towing_std = self.towing_log_std.exp()
        if not torch.isfinite(towing_std).all() or not (towing_std > 0).all():
            raise ValueError('Towing Gaussian standard deviation is nonfinite or nonpositive')
        std = torch.where((observations[:, TOWED_COLUMN] > 0)[:, None],
            towing_std.expand_as(mean), self.base_std.expand_as(mean))
        return mean, std

    def update_distribution(self, observations):
        mean, std = self.distribution_parameters(observations)
        self.distribution = Normal(mean, std, validate_args=True)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def _distribution(self):
        if self.distribution is None:
            raise RuntimeError('Call act/update_distribution before accessing its distribution')
        return self.distribution

    @property
    def action_mean(self):
        return self._distribution().mean

    @property
    def action_std(self):
        return self._distribution().stddev

    @property
    def entropy(self):
        return self._distribution().entropy().sum(-1)

    def get_actions_log_prob(self, actions):
        _finite_matrix(actions, 16, 'Raw Gaussian action')
        if actions.shape != self.action_mean.shape:
            raise ValueError('Action batch does not match current Gaussian')
        return self._distribution().log_prob(actions).sum(-1)

    def evaluate(self, critic_observations, **kwargs):
        _finite_matrix(critic_observations, 252, 'Critic observation')
        result = self.critic(critic_observations)
        if not torch.isfinite(result).all():
            raise ValueError('Critic returned nonfinite values')
        return result

    def parameter_groups(self):
        return [dict(name='towing_residual', params=list(self.actor.trainable_parameters())),
            dict(name='critic', params=list(self.critic.parameters())),
            dict(name='towing_std', params=[self.towing_log_std])]

    def make_optimizer(self, *, actor_lr=1.e-5, critic_lr=1.e-4, std_lr=1.e-5):
        rates = (actor_lr, critic_lr, std_lr)
        if any(not isinstance(x, (float, int)) or not 0 < x < float('inf') for x in rates):
            raise ValueError('Optimizer learning rates must be finite and positive')
        groups = self.parameter_groups()
        for group, rate in zip(groups, rates):
            group['lr'] = float(rate)
        optimizer = torch.optim.Adam(groups)
        self.assert_invariants(optimizer)
        return optimizer

    def assert_invariants(self, optimizer=None):
        self.actor.assert_invariants(optimizer)
        if self.base_std.requires_grad or _fingerprint(self.base_std) != self.frozen_base_std_sha256:
            raise ValueError('Protected non-towing standard deviation changed')
        for parameter in [*self.critic.parameters(), self.towing_log_std]:
            if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
                raise ValueError('Trainable Gaussian/Critic tensors must be finite float32')
        if not torch.isfinite(self.towing_log_std.exp()).all() or not (self.towing_log_std.exp() > 0).all():
            raise ValueError('Invalid towing standard deviation')
        if optimizer is not None:
            expected = self.parameter_groups()
            if len(optimizer.param_groups) != len(expected):
                raise ValueError('Optimizer must contain the three explicit trainable parameter groups')
            for group, target in zip(optimizer.param_groups, expected):
                if group.get('name') != target['name'] or [id(p) for p in group['params']] != [id(p) for p in target['params']]:
                    raise ValueError('Optimizer parameter ownership/order differs from protected policy contract')
            allowed = {p for group in expected for p in group['params']}
            if any(p not in allowed for p in optimizer.state):
                raise ValueError('Optimizer state contains a foreign/protected parameter')
            for parameter, state in optimizer.state.items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if not torch.isfinite(value).all():
                            raise ValueError('Nonfinite optimizer state')
                        if name != 'step' and value.shape != parameter.shape:
                            raise ValueError('Optimizer moment shape differs from its parameter')

    def project_constraints(self, optimizer=None):
        self.actor.project_constraints(optimizer)
        self.assert_invariants(optimizer)

    def get_extra_state(self):
        return dict(version=VERSION, policy_dim=300, action_dim=16,
            critic_contract=critic_contract_252(), critic_hidden_dims=list(self.critic_hidden_dims),
            inference_mode=self.inference_mode, source=deepcopy(self.source_record),
            implementation_sha256=self.implementation_sha256,
            base_std_sha256=self.frozen_base_std_sha256,
            base_std_origin='explicit caller-supplied vector; no claim of old checkpoint std equality',
            distribution='conditional_diagonal_Normal_raw_actions',
            gate='latest nominal towed context > 0', empirical_normalization=False,
            behavior_accepted=False, formal_ppo_authorized=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('PPO policy source/distribution/Critic contract mismatch')

    def _validate_restore(self, saved):
        current = self.state_dict()
        if set(saved) != set(current):
            raise ValueError('PPO restore keys differ')
        self.set_extra_state(saved['_extra_state'])
        self.actor.set_extra_state(saved['actor._extra_state'])
        for name, value in current.items():
            candidate = saved[name]
            if isinstance(value, torch.Tensor):
                if (not isinstance(candidate, torch.Tensor) or candidate.shape != value.shape
                        or candidate.dtype != value.dtype or not torch.isfinite(candidate).all()):
                    raise ValueError('PPO restore tensor shape/dtype/finiteness mismatch: '+name)
                if name.startswith('actor.base_actor.') or name == 'base_std':
                    if not torch.equal(candidate.to(value.device), value):
                        raise ValueError('Restore attempted to replace protected base/distribution')
        first_name = next(self.actor.residual_actor.named_parameters())[0]
        begin, end = CONTRACT['ignored_policy_columns']
        if torch.count_nonzero(saved['actor.residual_actor.'+first_name][:, begin:end]):
            raise ValueError('Restore reintroduced masked action history')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or assign:
            raise ValueError('PPO restore requires strict=True and assign=False')
        self._validate_restore(state_dict)
        old = deepcopy(self.state_dict())
        try:
            super().load_state_dict(state_dict, strict=True, assign=False)
            self.assert_invariants()
        except Exception:
            super().load_state_dict(old, strict=True, assign=False)
            raise
        self.distribution = None
        return True  # Installed RSL OnPolicyRunner expects a resume boolean.

    def checkpoint(self, optimizer=None):
        self.assert_invariants(optimizer)
        return deepcopy(dict(schema=VERSION, model_state_dict=self.state_dict(),
            optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
            rng_state=_rng_state(self.base_std.device), formal_ppo_authorized=False))

    def restore_checkpoint(self, checkpoint, optimizer=None):
        if checkpoint.get('schema') != VERSION or checkpoint.get('formal_ppo_authorized') is not False:
            raise ValueError('Unexpected PPO policy checkpoint schema')
        if (checkpoint.get('optimizer_state_dict') is None) != (optimizer is None):
            raise ValueError('Optimizer restore presence mismatch')
        self.assert_invariants(optimizer)
        self._validate_restore(checkpoint['model_state_dict'])
        original = self.checkpoint(optimizer)
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.assert_invariants(optimizer)
            _restore_rng(checkpoint['rng_state'], self.base_std.device)
        except Exception:
            self.load_state_dict(original['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(original['optimizer_state_dict'])
            _restore_rng(original['rng_state'], self.base_std.device)
            raise

    def begin_update(self, optimizer, observations, *, groups=None):
        return PolicyUpdateTransaction(self, optimizer, observations, groups=groups)


class PolicyUpdateTransaction:
    """Caller proposes updates; this object does no backward, Adam step or PPO.

All supplied observations are checked after the proposal, including per-group
mean KL. Missing groups are reported, never replaced with invented samples.
Use as a context manager so an exception/uncommitted exit also restores state.
"""
    def __init__(self, policy, optimizer, observations, *, groups=None):
        policy.assert_invariants(optimizer)
        self.policy = policy; self.optimizer = optimizer
        self.observations = observations.detach().clone()
        with torch.no_grad():
            mean, std = policy.distribution_parameters(self.observations)
        self.old_mean = mean.detach().clone(); self.old_std = std.detach().clone()
        self.groups = {} if groups is None else {name: mask.detach().clone() for name, mask in groups.items()}
        for name, mask in self.groups.items():
            if (not isinstance(name, str) or not name or mask.dtype != torch.bool
                    or mask.shape != (len(observations),) or mask.device != observations.device):
                raise ValueError('KL groups require named boolean masks of the observation batch')
        self.saved = policy.checkpoint(optimizer)
        self.closed = False

    def __enter__(self):
        return self

    def rollback(self):
        if self.closed:
            raise RuntimeError('Update transaction already closed')
        # Base corruption is an error, not a request to silently repair its tensors.
        self.policy.actor.base_actor.eval()
        self.policy._validate_restore(self.saved['model_state_dict'])
        self.policy.load_state_dict(self.saved['model_state_dict'])
        self.optimizer.load_state_dict(self.saved['optimizer_state_dict'])
        _restore_rng(self.saved['rng_state'], self.policy.base_std.device)
        self.policy.assert_invariants(self.optimizer)
        self.closed = True

    def finish(self, *, max_kl, group_max_kl=None):
        if self.closed:
            raise RuntimeError('Update transaction already closed')
        if not isinstance(max_kl, (int, float)) or not 0 <= max_kl < float('inf'):
            raise ValueError('KL threshold must be finite and nonnegative')
        limits = {} if group_max_kl is None else dict(group_max_kl)
        if set(limits)-set(self.groups) or any(not 0 <= x < float('inf') for x in limits.values()):
            raise ValueError('Group KL limits must be finite, nonnegative and refer to declared groups')
        try:
            self.policy.project_constraints(self.optimizer)
            with torch.no_grad():
                mean, std = self.policy.distribution_parameters(self.observations)
                kl = (torch.log(std/self.old_std)+(self.old_std.square()+(self.old_mean-mean).square())
                    /(2*std.square())-.5).sum(-1).clamp_min(0.)
            if not torch.isfinite(kl).all():
                raise ValueError('Nonfinite conditional Gaussian KL')
            means = {name: float(kl[mask].mean()) if mask.any() else None for name, mask in self.groups.items()}
            accepted = bool(float(kl.mean()) <= max_kl and all(means[name] is not None
                and means[name] <= limit for name, limit in limits.items()))
            report = dict(accepted=accepted, reason='accepted' if accepted else 'kl_limit_or_missing_group',
                samples=len(kl), mean_kl=float(kl.mean()), max_sample_kl=float(kl.max()), group_mean_kl=means,
                group_counts={name: int(mask.sum()) for name, mask in self.groups.items()},
                synthetic_or_runtime_context='caller_must_record', formal_ppo_authorized=False)
        except (ValueError, RuntimeError) as error:
            self.rollback()
            return dict(accepted=False, reason='invalid_proposal', detail=str(error), formal_ppo_authorized=False)
        if accepted:
            self.closed = True
            self.policy.distribution = None
        else:
            self.rollback()
        return report

    def __exit__(self, exc_type, exc, traceback):
        if not self.closed:
            self.rollback()
        return False
