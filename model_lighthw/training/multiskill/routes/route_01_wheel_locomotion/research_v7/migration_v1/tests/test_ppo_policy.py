"""Synthetic CPU numerical/protocol checks; no simulator, rollout or real PPO training."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1.contract import install_actor_mask
from migration_v1.policy import FrozenBaseTowingResidualActor, tensor_fingerprint
from migration_v1.ppo_policy import TowingResidualActorCritic, critic_contract_252


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def model():
    torch.manual_seed(125)
    base = nn.Sequential(nn.Linear(300,512), nn.ELU(), nn.Linear(512,256), nn.ELU(),
        nn.Linear(256,128), nn.ELU(), nn.Linear(128,16))
    install_actor_mask(base)
    actor = FrozenBaseTowingResidualActor(base)
    return TowingResidualActorCritic(actor, base_std=[.04]*12+[.08]*4, critic_hidden_dims=(32,16))


@pytest.fixture
def observation():
    z = torch.randn(6,300, generator=torch.Generator().manual_seed(126))
    z[:,298:] = torch.tensor([[0.,0.],[1.,0.],[0.,.5],[0.,1.],[.333,0.],[0.,0.]])
    return z


def same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same_tree(left[key], right[key])
    elif isinstance(left, (tuple,list)):
        assert len(left) == len(right)
        for a,b in zip(left,right): same_tree(a,b)
    else:
        assert left == right


def toy_step(model, optimizer, observation):
    """Synthetic objective, deliberately not PPO/GAE or teacher fitting."""
    model.update_distribution(observation)
    target = model.action_mean.detach()+.01
    loss = -model.get_actions_log_prob(target).mean()+model.evaluate(torch.zeros(len(observation),252)).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step(); model.project_constraints(optimizer)


def test_conditional_normal_uses_one_mean_std_logprob_entropy(model, observation):
    with torch.no_grad():
        model.actor.residual_actor[-1].bias.fill_(.2)
        model.towing_log_std.add_(.3)
    actions = model.act(observation)
    assert actions.shape == (6,16)
    reference = Normal(model.action_mean, model.action_std)
    torch.testing.assert_close(model.get_actions_log_prob(actions), reference.log_prob(actions).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(model.entropy, reference.entropy().sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(model.act_inference(observation), model.action_mean, rtol=0, atol=0)
    protected = observation[:,299] == 0
    assert torch.equal(model.action_std[protected], model.base_std.expand(protected.sum(),16))
    visible = observation.clone(); visible[:,205:285] = 0.
    expected = torch.cat([model.actor.base_actor(row[None]) for row in visible])
    assert torch.equal(model.action_mean[protected], expected[protected])


def test_non_towing_logprob_and_entropy_cannot_update_frozen_distribution(model, observation):
    protected = observation[observation[:,299] == 0]
    before = tensor_fingerprint(model.actor.base_actor); std = model.base_std.clone()
    model.update_distribution(protected)
    loss = model.get_actions_log_prob(model.action_mean.detach()+.02).sum()+model.entropy.sum()
    loss.backward()
    assert all(p.grad is None for p in model.actor.base_actor.parameters())
    assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in model.actor.residual_actor.parameters())
    assert not torch.count_nonzero(model.towing_log_std.grad)
    optimizer = model.make_optimizer(); optimizer.step(); model.project_constraints(optimizer)
    assert tensor_fingerprint(model.actor.base_actor) == before
    assert torch.equal(model.base_std, std)
    assert model.base_std.grad is None


def test_synthetic_updates_preserve_non_towing_and_masked_adam(model, observation):
    optimizer = model.make_optimizer()
    with torch.no_grad(): model.actor.residual_actor[-1].weight.fill_(.01)
    before_mean, before_std = (x.detach().clone() for x in model.distribution_parameters(observation))
    protected = observation[:,299] == 0; fingerprint = tensor_fingerprint(model.actor.base_actor)
    for _ in range(3): toy_step(model, optimizer, observation)
    mean,std = model.distribution_parameters(observation)
    assert torch.equal(mean[protected],before_mean[protected])
    assert torch.equal(std[protected],before_std[protected])
    assert not torch.equal(mean[~protected],before_mean[~protected])
    assert tensor_fingerprint(model.actor.base_actor) == fingerprint
    first = next(model.actor.residual_actor.parameters())
    for name in ('exp_avg','exp_avg_sq'):
        assert torch.count_nonzero(optimizer.state[first][name][:,:205])
        assert not torch.count_nonzero(optimizer.state[first][name][:,205:285])
    changed = observation.clone(); changed[:,205:285] += 1000
    assert torch.equal(model.act_inference(observation),model.act_inference(changed))


def test_optimizer_rejects_whole_model_and_foreign_group(model):
    with pytest.raises(ValueError,match='exclude the protected base'):
        model.assert_invariants(torch.optim.Adam(model.parameters()))
    optimizer = model.make_optimizer()
    optimizer.param_groups[2]['params'].append(nn.Parameter(torch.zeros(1)))
    with pytest.raises(ValueError,match='ownership/order'):
        model.assert_invariants(optimizer)


def test_fresh_critic_252_semantics_and_input_separation(model, observation):
    contract = critic_contract_252()
    assert contract['fields'][0]['columns'] == [0,3]
    assert contract['fields'][7]['name'] == 'height_scan'
    assert contract['fields'][7]['columns'] == [60,236]
    assert contract['fields'][-1]['columns'] == [251,252]
    assert contract['initialization'] == 'fresh_no_old_critic_transfer'
    assert not contract['runtime_environment_verified']
    before = model.act_inference(observation).detach().clone()
    value = model.evaluate(torch.randn(6,252)); assert value.shape == (6,1)
    assert torch.equal(before, model.act_inference(observation))
    with pytest.raises(ValueError,match='252'):
        model.evaluate(torch.zeros(6,268))
    with pytest.raises(ValueError,match='252-column'):
        TowingResidualActorCritic(model.actor,base_std=model.base_std,critic_dim=268)


def test_accepted_transaction_retains_proposal_and_reports_groups(model, observation):
    optimizer = model.make_optimizer(); initial = model.towing_log_std.detach().clone()
    groups = {'unloaded':observation[:,298:].sum(-1)==0,'payload':observation[:,298]>0,
        'towing':observation[:,299]>0,'absent':torch.zeros(6,dtype=torch.bool)}
    with model.begin_update(optimizer,observation,groups=groups) as tx:
        toy_step(model,optimizer,observation)
        report = tx.finish(max_kl=1.,group_max_kl={'towing':1.})
    assert report['accepted']
    assert report['group_mean_kl']['unloaded'] == report['group_mean_kl']['payload'] == 0.
    assert report['group_mean_kl']['absent'] is None
    assert report['group_counts']['towing'] == 2
    assert not torch.equal(initial,model.towing_log_std)


def test_rejected_kl_restores_parameters_adam_and_rng_exactly(model, observation):
    optimizer = model.make_optimizer(); toy_step(model,optimizer,observation)
    saved = model.checkpoint(optimizer)
    groups = {'towing':observation[:,299]>0}
    with model.begin_update(optimizer,observation,groups=groups) as tx:
        torch.randn(19)
        toy_step(model,optimizer,observation)
        with torch.no_grad(): model.actor.residual_actor[-1].bias.add_(1.)
        report = tx.finish(max_kl=1000.,group_max_kl={'towing':0.})
    assert not report['accepted']
    assert report['reason'] == 'kl_limit_or_missing_group'
    same_tree(saved,model.checkpoint(optimizer))


@pytest.mark.parametrize('failure',['exception','unfinished','nonfinite'])
def test_failed_or_uncommitted_transaction_restores_everything(model, observation, failure):
    optimizer = model.make_optimizer(); toy_step(model,optimizer,observation)
    saved = model.checkpoint(optimizer)
    try:
        with model.begin_update(optimizer,observation) as tx:
            toy_step(model,optimizer,observation); torch.randn(5)
            if failure == 'exception': raise RuntimeError('test proposal error')
            if failure == 'nonfinite':
                with torch.no_grad(): model.towing_log_std.fill_(float('nan'))
                report = tx.finish(max_kl=.01)
                assert not report['accepted'] and report['reason'] == 'invalid_proposal'
    except RuntimeError as error:
        assert failure == 'exception' and str(error) == 'test proposal error'
    same_tree(saved,model.checkpoint(optimizer))


def test_checkpoint_roundtrip_and_invalid_adam_restore_are_transactional(model, observation, tmp_path):
    optimizer = model.make_optimizer(); toy_step(model,optimizer,observation)
    saved = model.checkpoint(optimizer); path = tmp_path/'synthetic.pt'
    torch.save(saved,path); loaded = torch.load(path,weights_only=True)
    toy_step(model,optimizer,observation); model.restore_checkpoint(loaded,optimizer)
    same_tree(saved,model.checkpoint(optimizer))
    corrupted = deepcopy(saved)
    first_state = next(iter(corrupted['optimizer_state_dict']['state'].values()))
    first_state['exp_avg'] = torch.zeros(3)
    with pytest.raises(ValueError): model.restore_checkpoint(corrupted,optimizer)
    same_tree(saved,model.checkpoint(optimizer))


def test_saved_checkpoint_restores_new_model_and_optimizer_instances(model, observation):
    optimizer=model.make_optimizer(); toy_step(model,optimizer,observation)
    saved=model.checkpoint(optimizer)
    fresh_actor=FrozenBaseTowingResidualActor(deepcopy(model.actor.base_actor))
    restored=TowingResidualActorCritic(fresh_actor,base_std=model.base_std,critic_hidden_dims=(32,16))
    restored_optimizer=restored.make_optimizer()
    restored.restore_checkpoint(saved,restored_optimizer)
    same_tree(saved,restored.checkpoint(restored_optimizer))
    left=model.distribution_parameters(observation);right=restored.distribution_parameters(observation)
    same_tree(left,right)
    # One further identical synthetic step must also preserve Adam continuation.
    toy_step(model,optimizer,observation); toy_step(restored,restored_optimizer,observation)
    same_tree(model.state_dict(),restored.state_dict())
    same_tree(optimizer.state_dict(),restored_optimizer.state_dict())


@pytest.mark.parametrize('target',['source','base','base_std','mask','critic_nan','actor_metadata'])
def test_invalid_model_restore_rejected_without_partial_mutation(model, observation, target):
    saved = deepcopy(model.state_dict()); bad = deepcopy(saved)
    if target == 'source': bad['_extra_state']['implementation_sha256'] = '0'*64
    elif target == 'actor_metadata': bad['actor._extra_state']['version'] = 'other'
    elif target == 'base': bad['actor.base_actor.6.bias'][0] += 1.
    elif target == 'base_std': bad['base_std'][0] *= 2.
    elif target == 'mask': bad['actor.residual_actor.0.weight'][0,205] = 1.
    else: bad['critic.0.weight'][0,0] = float('nan')
    with pytest.raises(ValueError): model.load_state_dict(bad)
    same_tree(saved,model.state_dict())


def test_installed_rsl_nonrecurrent_act_protocol_without_real_ppo_update(model, observation):
    from rsl_rl.algorithms import PPO
    algorithm = PPO(model,device='cpu',schedule='fixed',desired_kl=None)
    with pytest.raises(ValueError,match='exclude the protected base'):
        model.assert_invariants(algorithm.optimizer)
    algorithm.optimizer = model.make_optimizer()
    model.assert_invariants(algorithm.optimizer)
    algorithm.init_storage('rl',6,2,[300],[252],[16])
    actions = algorithm.act(observation,torch.zeros(6,252))
    assert actions.shape == (6,16)
    assert algorithm.transition.values.shape == (6,1)
    assert algorithm.transition.action_sigma.shape == (6,16)
    assert algorithm.transition.actions_log_prob.shape == (6,)
    # Deliberately no process_env_step/compute_returns/update or simulator.


def test_unverified_constructor_does_not_claim_training_or_provenance(model):
    extra = model.get_extra_state()
    assert extra['source']['checkpoint_provenance_verified'] is False
    assert extra['formal_ppo_authorized'] is False
    assert extra['behavior_accepted'] is False
    model.train()
    assert not model.actor.base_actor.training
    assert model.actor.residual_actor.training and model.critic.training
