"""Synthetic CPU contract checks only; no robot fitting, rollout or simulator."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.distributions import Normal

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from migration_v1.contract import install_actor_mask
from exploration_v1.contract import append_task_clock, contract, source_signatures
from exploration_v1.policy import UnifiedResidualActor, ExplorationActorCritic


@pytest.fixture(scope='module',autouse=True)
def one_thread():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def base_network():
    model=nn.Sequential(nn.Linear(300,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                        nn.Linear(256,128),nn.ELU(),nn.Linear(128,16))
    install_actor_mask(model)
    return model


@pytest.fixture
def model():
    torch.manual_seed(881)
    return ExplorationActorCritic(UnifiedResidualActor(base_network()),base_std=[.02]*12+[.04]*4,
        towing_std=[.04]*12+[.08]*4,critic_hidden_dims=(16,8))


@pytest.fixture
def observation():
    data=torch.randn(4,300,generator=torch.Generator().manual_seed(882))
    data[:,298:]=torch.tensor([[0.,0.],[1.,0.],[0.,.5],[0.,1.]])
    return append_task_clock(data,torch.tensor([0.,3.,7.48,50.]),torch.tensor([3.,3.,3.,7.48]))


def same_tree(left,right):
    if isinstance(left,torch.Tensor): assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: same_tree(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert type(left)==type(right) and len(left)==len(right)
        for a,b in zip(left,right): same_tree(a,b)
    else: assert left==right


def test_clock_contract_and_clipping(observation):
    assert observation.shape==(4,302)
    torch.testing.assert_close(observation[:,300],torch.tensor([0.,.075,.187,1.]))
    torch.testing.assert_close(observation[:,301],torch.tensor([.375,.375,.375,.935]))
    assert not contract()['actor_privileged_phase_or_progress']


@pytest.mark.parametrize('clock',[-1.,float('nan'),[0.,1.]])
def test_invalid_clock_rejected(observation,clock):
    with pytest.raises(ValueError): append_task_clock(observation[:,:300],clock,3.)


def test_zero_residual_preserves_original_base_same_row_mode(model,observation):
    visible=observation[:,:300].clone(); visible[:,205:285]=0.
    expected=torch.cat([model.actor.base_actor(row[None]) for row in visible])
    assert torch.equal(model.act_inference(observation),expected)
    assert model.actor.base_actor.training is False
    assert all(not p.requires_grad for p in model.actor.base_actor.parameters())


def test_strict300_residual_transfer_has_zero_new_clock_weights(observation):
    torch.manual_seed(883); base=base_network(); old=base_network()
    model=UnifiedResidualActor(base,towing_residual=old)
    assert torch.count_nonzero(model.towing_residual[0].weight[:,300:])==0
    visible=observation.clone(); visible[:,205:285]=0.
    expected=base(visible[:,:300])+torch.where((visible[:,299]>0)[:,None],old(visible[:,:300]),0.)
    # 302-vs300 GEMM is not claimed bitwise identical, only the transferred function.
    torch.testing.assert_close(model(observation),expected,rtol=2.e-6,atol=1.e-6)


def test_clock_columns_can_influence_new_residual(model,observation):
    with torch.no_grad():
        branch=model.actor.non_towing_residual
        branch[-1].weight.fill_(.01); branch[0].weight[:,300:].fill_(.5)
    changed=observation.clone(); changed[:2,300]=.9
    assert not torch.equal(model.act_inference(observation)[:2],model.act_inference(changed)[:2])


def test_history_mask_and_disjoint_nominal_branches(model,observation):
    with torch.no_grad():
        model.actor.towing_residual[-1].bias.fill_(.2)
        model.actor.non_towing_residual[-1].bias.fill_(-.1)
    changed=observation.clone(); changed[:,205:285]=999.
    assert torch.equal(model.act_inference(observation),model.act_inference(changed))
    model.act_inference(observation[:2]).sum().backward()
    assert model.actor.non_towing_residual[-1].bias.grad.abs().sum()>0
    assert model.actor.towing_residual[-1].bias.grad.abs().sum()==0
    assert all(p.grad is None for p in model.actor.base_actor.parameters())


def test_standard_gaussian_protocol(model,observation):
    action=model.act(observation)
    normal=Normal(model.action_mean,model.action_std)
    assert action.shape==(4,16) and model.evaluate(torch.zeros(4,252)).shape==(4,1)
    assert torch.equal(model.get_actions_log_prob(action),normal.log_prob(action).sum(-1))
    assert torch.equal(model.entropy,normal.entropy().sum(-1))
    assert torch.equal(model.action_std[:2],model.non_towing_log_std.exp().expand(2,16))
    assert torch.equal(model.action_std[2:],model.towing_log_std.exp().expand(2,16))
    assert model.reset(torch.tensor([True,False])) is None


def test_non_towing_now_learns_without_modifying_base(model,observation):
    """One synthetic regression step demonstrates the intended gradient freedom."""
    optimizer=model.make_optimizer(); base=deepcopy(model.actor.base_actor.state_dict())
    original=model.act_inference(observation).detach().clone()
    loss=(model.act_inference(observation[:2])-(original[:2]+.1)).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step(); model.project_constraints(optimizer)
    after=model.act_inference(observation)
    assert not torch.equal(after[:2],original[:2])
    assert torch.equal(after[2:],original[2:])
    same_tree(base,model.actor.base_actor.state_dict())
    assert [g['name'] for g in optimizer.param_groups]==['towing_residual','non_towing_residual','critic','towing_std','non_towing_std']
    assert optimizer.param_groups[1]['lr']<optimizer.param_groups[0]['lr']
    for branch in (model.actor.towing_residual,model.actor.non_towing_residual):
        assert not torch.count_nonzero(branch[0].weight[:,205:285])


def test_rejected_update_restores_both_branches_critic_adam_rng(model,observation):
    optimizer=model.make_optimizer(); original=model.checkpoint(optimizer)
    with model.begin_update(optimizer,observation,groups={'empty_payload':observation[:,299]==0}) as transaction:
        with torch.no_grad():
            model.actor.towing_residual[-1].bias.add_(2.)
            model.actor.non_towing_residual[-1].bias.add_(2.)
            model.critic[-1].bias.add_(3.)
            model.non_towing_log_std.add_(.2)
        torch.rand(10)
        result=transaction.finish(max_kl=.001,group_max_kl={'empty_payload':.0001})
    assert not result['accepted']
    same_tree(original,model.checkpoint(optimizer))


def test_restore_rejects_unmasked_or_replaced_base_without_mutation(model):
    original=model.checkpoint()
    for key,col in [('actor.non_towing_residual.0.weight',205),('actor.base_actor.0.weight',0)]:
        changed=deepcopy(original); changed['model_state_dict'][key][0,col]+=.1
        with pytest.raises(ValueError): model.restore_checkpoint(changed)
        same_tree(original,model.checkpoint())


def test_resume_new_policy_restores_both_trainable_stds_and_numerical_state(model,observation):
    optimizer=model.make_optimizer()
    saved=model.checkpoint(optimizer)
    with torch.no_grad(): model.towing_log_std.add_(.2); model.non_towing_log_std.add_(.4)
    model.restore_checkpoint(saved,optimizer)
    same_tree(saved,model.checkpoint(optimizer))
    assert not saved['behavior_accepted'] and not saved['formal_ppo_authorized']


def test_checkpoint_dependency_scope_includes_transaction_not_runtime(model):
    names=list(source_signatures())
    assert any(name.endswith('/migration_v1/ppo_policy.py') for name in names)
    assert any(name.endswith('/exploration_v1/checkpoint.py') for name in names)
    assert not any(name.endswith('/exploration_v1/runtime.py') for name in names)


@pytest.mark.parametrize('mutation', ['wrongdim','truth_column','badcontext'])
def test_schema_rejects_invalid_input(model,observation,mutation):
    z=observation.clone()
    if mutation=='wrongdim': z=z[:,:300]
    elif mutation=='truth_column': z[0,300]=2.
    else: z[0,298:300]=1.
    with pytest.raises(ValueError): model.act_inference(z)
