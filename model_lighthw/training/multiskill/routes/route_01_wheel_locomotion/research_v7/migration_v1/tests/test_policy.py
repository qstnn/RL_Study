"""纯CPU合成不变量检查：不加载实际Actor，不执行optimizer.step/BC/PPO。"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1.contract import install_actor_mask
from migration_v1.policy import FrozenBaseTowingResidualActor, tensor_fingerprint


@pytest.fixture
def base():
    torch.manual_seed(31)
    network = nn.Sequential(nn.Linear(300, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16))
    install_actor_mask(network)
    return network


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(32)
    observation = torch.randn(6, 300, generator=generator)
    observation[:, 298:] = torch.tensor([[0., 0.], [1., 0.], [0., .5], [0., 1.], [.333333, 0.], [0., 0.]])
    return observation


def test_zero_residual_is_exact_base_for_all_tasks(base, inputs):
    before = tensor_fingerprint(base)
    actor = FrozenBaseTowingResidualActor(base)
    assert torch.equal(actor(inputs), base(inputs))
    assert tensor_fingerprint(base) == before
    assert actor.base_actor is not base
    assert actor.get_extra_state()['formal_ppo_authorized'] is False
    assert actor.get_extra_state()['behavior_accepted'] is False
    assert actor.get_extra_state()['checkpoint_provenance_verified'] is False


def test_nonzero_residual_preserves_unloaded_and_payload_exactly(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    # 人工改动只验证门控，不来自训练或真实教师拟合。
    with torch.no_grad():
        actor.residual_actor[-1].bias.copy_(torch.arange(16)*.1+.3)
        actor.residual_actor[-1].weight.fill_(.01)
    protected = inputs[:, 299] == 0
    assert torch.equal(actor(inputs)[protected], base(inputs)[protected])
    assert not torch.equal(actor(inputs)[~protected], base(inputs)[~protected])
    # gate只依赖已知名义拖载当前帧；命令为0或背载15kg不能激活拖载修正。
    inputs[:, 42] = 0.
    assert torch.equal(actor(inputs)[protected], base(inputs)[protected])
    assert not torch.equal(actor(inputs)[~protected], base(inputs)[~protected])


def test_action_history_ignored_in_both_branches_and_inputs_unchanged(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    with torch.no_grad():
        actor.residual_actor[-1].weight.fill_(.02)
    changed = inputs.clone()
    changed[:, 205:285] += 900.
    before = changed.clone()
    assert torch.equal(actor(inputs), actor(changed))
    assert torch.equal(changed, before)


def test_freeze_survives_train_mode_and_original_base_changes(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    reference = actor(inputs).detach().clone()
    actor.train()
    assert actor.residual_actor.training
    assert not actor.base_actor.training
    assert not any(p.requires_grad for p in actor.base_actor.parameters())
    with torch.no_grad():
        base[-1].bias.add_(2.)
    assert torch.equal(actor(inputs), reference)
    actor.assert_invariants()
    with torch.no_grad():
        actor.base_actor[-1].bias.add_(1.)
    with pytest.raises(ValueError, match='base tensors changed'):
        actor.assert_invariants()


def test_protected_samples_have_zero_residual_gradient(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    actor(inputs[inputs[:, 299] == 0]).sum().backward()
    assert all(p.grad is None for p in actor.base_actor.parameters())
    assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in actor.residual_actor.parameters())


def test_towing_gradient_mask_and_adam_projection_without_step(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    with torch.no_grad():
        actor.residual_actor[-1].weight.fill_(.02)
    actor(inputs).sum().backward()
    first = actor.residual_actor[0].weight
    assert torch.count_nonzero(first.grad[:, :205])
    assert not torch.count_nonzero(first.grad[:, 205:285])
    assert all(p.grad is None for p in actor.base_actor.parameters())
    optimizer = torch.optim.Adam(actor.trainable_parameters())
    optimizer.state[first]['exp_avg'] = torch.ones_like(first)
    optimizer.state[first]['exp_avg_sq'] = torch.ones_like(first)
    actor.project_constraints(optimizer)
    actor.assert_invariants(optimizer)
    assert not torch.count_nonzero(optimizer.state[first]['exp_avg'][:, 205:285])
    assert torch.count_nonzero(optimizer.state[first]['exp_avg'][:, :205])
    bad_optimizer = torch.optim.Adam(actor.parameters())
    with pytest.raises(ValueError, match='exclude the protected base'):
        actor.assert_invariants(bad_optimizer)


def test_restore_roundtrip_and_base_tampering_rejected_before_mutation(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    with torch.no_grad():
        actor.residual_actor[-1].bias.fill_(.7)
    restored = FrozenBaseTowingResidualActor(base)
    restored.load_state_dict(deepcopy(actor.state_dict()))
    assert torch.equal(restored(inputs), actor(inputs))
    before = tensor_fingerprint(restored.base_actor)
    tampered = deepcopy(actor.state_dict())
    tampered['base_actor.6.bias'][0] += 1.
    with pytest.raises(ValueError, match='replace protected base'):
        restored.load_state_dict(tampered)
    assert tensor_fingerprint(restored.base_actor) == before


@pytest.mark.parametrize('change', ['source', 'missing_metadata'])
def test_restore_rejects_other_source_contract_and_missing_metadata(base, change):
    actor = FrozenBaseTowingResidualActor(base)
    tampered = deepcopy(actor.state_dict())
    if change == 'source':
        tampered['_extra_state']['contract_source_sha256']['wheel_unified_v6/spec.py'] = '0'*64
    else:
        del tampered['_extra_state']
    with pytest.raises(ValueError, match='signature mismatch'):
        actor.load_state_dict(tampered)
    with pytest.raises(ValueError, match='strict=True'):
        actor.load_state_dict(actor.state_dict(), strict=False)


def test_unmasked_or_nonfinite_base_is_rejected_without_silent_change(base):
    with torch.no_grad():
        base[0].weight[0, 205] = .1
    with pytest.raises(ValueError, match='unmasked'):
        FrozenBaseTowingResidualActor(base)
    assert base[0].weight[0, 205] == .1
    with torch.no_grad():
        base[0].weight[0, 205] = 0.
        base[-1].bias[0] = float('nan')
    with pytest.raises(ValueError, match='finite float32'):
        FrozenBaseTowingResidualActor(base)


@pytest.mark.parametrize('kind', ['dimension', 'float64', 'nonfinite', 'negative_context', 'simultaneous_loads'])
def test_invalid_input_rejected(base, inputs, kind):
    actor = FrozenBaseTowingResidualActor(base)
    if kind == 'dimension':
        inputs = inputs[:, :299]
    elif kind == 'float64':
        inputs = inputs.double()
    elif kind == 'nonfinite':
        inputs[0, 12] = float('nan')
    elif kind == 'negative_context':
        inputs[0, 299] = -.1
    else:
        inputs[0, 298:] = 1.
    with pytest.raises(ValueError):
        actor(inputs)


def test_nan_residual_cannot_contaminate_protected_branch(base, inputs):
    actor = FrozenBaseTowingResidualActor(base)
    with torch.no_grad():
        actor.residual_actor[-1].bias.fill_(float('nan'))
    protected = inputs[inputs[:, 299] == 0]
    assert torch.equal(actor(protected), base(protected))
    with pytest.raises(ValueError, match='nonfinite'):
        actor(inputs)
    with pytest.raises(ValueError, match='finite float32'):
        actor.assert_invariants()
