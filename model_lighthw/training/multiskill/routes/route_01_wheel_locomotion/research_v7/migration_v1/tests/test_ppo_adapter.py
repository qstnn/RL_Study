"""Synthetic CPU RSL updates only: no actual environment, teacher or real PPO run."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_ppo_policy import model, cpu_threads, same_tree
from migration_v1.policy import FrozenBaseTowingResidualActor, tensor_fingerprint
from migration_v1.ppo_policy import TowingResidualActorCritic
from migration_v1.ppo_adapter import ControlledPPOAdapter
from migration_v1.schedule import make_schedule


@pytest.fixture(scope='module',autouse=True)
def synthetic_update_counter():
    from rsl_rl.algorithms import PPO
    original=PPO.update; calls=[]
    def tracked(algorithm):
        calls.append(dict(device=algorithm.device,completed=False))
        result=original(algorithm)
        calls[-1]['completed']=True
        return result
    PPO.update=tracked
    yield calls
    PPO.update=original
    print(f'SYNTHETIC_CPU_RSL_UPDATE_CALLS={len(calls)}; '
        f'COMPLETED={sum(x["completed"] for x in calls)}; REAL_ENVIRONMENT_PPO_UPDATES=0')


def adapter(model,**kwargs):
    options=dict(num_envs=3,rollout_steps=2,mode='diagnostic',max_kl=100.)
    options.update(kwargs)
    return ControlledPPOAdapter(model,**options)


def inputs(n=3,phase=0):
    cases=torch.tensor([[0.,0.,0.,phase],[0.,15.,0.,phase],[25.,0.,40.,phase]])
    assert n==3
    obs=torch.randn(n,300,generator=torch.Generator().manual_seed(151+phase))
    obs[:,298]=cases[:,1]/15.; obs[:,299]=cases[:,2]/40.
    critic=torch.randn(n,252,generator=torch.Generator().manual_seed(152+phase))
    return obs,critic,cases


def fill(algorithm):
    for k in range(algorithm.config['rollout_steps']):
        obs,critic,cases=inputs(phase=k%4)
        algorithm.act(obs,critic,cases)
        algorithm.record_step(torch.tensor([.1,.2,.3])+k,
            torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    last=inputs(phase=3)[1]
    algorithm.compute_returns(last)


def test_real_rsl_loss_path_on_synthetic_cpu_rollout_preserves_protected_distribution(model):
    a=adapter(model,num_learning_epochs=2,num_mini_batches=2)
    obs=inputs()[0]; protected=obs[:,299]==0
    before_mean,before_std=[x.detach().clone() for x in model.distribution_parameters(obs)]
    base=tensor_fingerprint(model.actor.base_actor)
    fill(a); report=a.update()
    assert report['accepted']
    assert report['counters']['rsl_update_calls']==1
    assert report['configured_minibatch_steps']==4
    assert report['execution_kind']=='synthetic_cpu'
    assert report['losses'].keys()=={'value_function','surrogate','entropy'}
    mean,std=model.distribution_parameters(obs)
    assert torch.equal(mean[protected],before_mean[protected])
    assert torch.equal(std[protected],before_std[protected])
    assert tensor_fingerprint(model.actor.base_actor)==base
    model.assert_invariants(a.optimizer)
    assert a.storage.step==0 and not a.returns_ready


def test_timeout_terminal_value_and_true_last_state_match_hand_computed_gae(model,monkeypatch):
    # A transparent synthetic value function isolates terminal/reset semantics.
    monkeypatch.setattr(model,'evaluate',lambda z,**kwargs:z[:,:1])
    a=adapter(model,gamma=.9,lam=.8)
    obs,critic,cases=inputs(); critic[:,0]=torch.tensor([1.,2.,3.])
    a.act(obs,critic,cases)
    terminated=torch.tensor([False,False,True]);truncated=torch.tensor([False,True,True])
    raw=torch.tensor([.1,.2,.3]);terminal=torch.zeros(3,252);terminal[:,0]=torch.tensor([99.,5.,99.])
    with pytest.raises(ValueError,match='terminal Critic'):
        a.record_step(raw,terminated,truncated)
    a.record_step(raw,terminated,truncated,terminal_critic_observations=terminal)
    torch.testing.assert_close(raw,torch.tensor([.1,.2,.3]))
    torch.testing.assert_close(a.storage.rewards[0,:,0],torch.tensor([.1,4.7,.3]))
    critic[:,0]=torch.tensor([4.,6.,7.]);a.act(obs,critic,cases)
    a.record_step(torch.tensor([1.,2.,3.]),torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    last=torch.zeros(3,252);last[:,0]=torch.tensor([8.,9.,10.])
    report=a.compute_returns(last)
    torch.testing.assert_close(a.storage.returns[:,:,0],torch.tensor([[6.724,4.7,.3],[8.2,10.1,12.]]),rtol=1.e-6,atol=1.e-6)
    assert report['timeout_count']==1
    torch.testing.assert_close(report['last_values'][:,0],last[:,0])
    assert torch.isfinite(a.storage.advantages).all()


def test_pre_action_inputs_are_copied_before_in_place_reset_and_sampling_is_detached(model):
    a=adapter(model);obs,critic,cases=inputs(); saved=obs.clone();saved_critic=critic.clone()
    actions=a.act(obs,critic,cases); raw=actions.clone()
    assert not actions.requires_grad
    obs.fill_(99.);critic.fill_(99.);cases[:,3]=3.;actions.zero_()
    a.record_step(torch.ones(3),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    assert torch.equal(a.storage.observations[0],saved)
    assert torch.equal(a.storage.privileged_observations[0],saved_critic)
    assert torch.equal(a.storage.actions[0],raw)
    assert torch.equal(a.condition_history[0][:,3],torch.zeros(3))


def test_kl_rejection_restores_entire_policy_adam_rng_after_actual_synthetic_rsl_update(model):
    a=adapter(model,max_kl=0.,actor_lr=.001,critic_lr=.001,std_lr=.001)
    fill(a); saved=model.checkpoint(a.optimizer)
    report=a.update()
    assert not report['accepted'] and report['reason']=='kl_limit_or_missing_group'
    same_tree(saved,model.checkpoint(a.optimizer))
    assert a.counters['rejected_updates']==a.counters['rsl_update_calls']==1
    assert all(p.grad is None for p in model.parameters())


def test_synthetic_update_exception_rolls_back_and_is_not_swallowed(model,monkeypatch):
    a=adapter(model);fill(a);saved=model.checkpoint(a.optimizer)
    original=a.optimizer.step
    def mutate_then_fail(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('synthetic post-step failure')
    monkeypatch.setattr(a.optimizer,'step',mutate_then_fail)
    with pytest.raises(RuntimeError,match='synthetic post-step failure'):
        a.update()
    same_tree(saved,model.checkpoint(a.optimizer))
    assert a.last_update_report['reason']=='update_exception'
    assert a.counters['rejected_updates']==1 and a.storage.step==0


def test_new_adapter_checkpoint_restores_rng_adam_and_exact_next_update(model,tmp_path):
    a=adapter(model);fill(a);assert a.update()['accepted']
    path=tmp_path/'synthetic_ppo_adapter.pt';record=a.save(path)
    assert not record['environment_state_saved']
    with pytest.raises(FileExistsError):a.save(path)
    fill(a);assert a.update()['accepted']
    expected=a.checkpoint()
    fresh_actor=FrozenBaseTowingResidualActor(deepcopy(model.actor.base_actor))
    fresh=TowingResidualActorCritic(fresh_actor,base_std=model.base_std,critic_hidden_dims=(32,16))
    restored=adapter(fresh);restored.load(path)
    fill(restored);assert restored.update()['accepted']
    same_tree(expected,restored.checkpoint())


def test_all64_first_batch_and_only_observed_phases_are_used_for_kl(model):
    a=ControlledPPOAdapter(model,num_envs=64,rollout_steps=1,max_kl=100.,group_kl_limit=100.)
    cases=torch.tensor(make_schedule(64)['conditions'],dtype=torch.float32)
    metadata=torch.cat((cases,torch.zeros(64,1)),dim=1)
    obs=torch.zeros(64,300);obs[:,298]=cases[:,1]/15.;obs[:,299]=cases[:,2]/40.
    critic=torch.zeros(64,252)
    missing=metadata.clone();missing[-1]=missing[0]
    altered=obs.clone();altered[-1]=altered[0]
    with pytest.raises(ValueError,match='missing permitted'):
        a.act(altered,critic,missing)
    a.act(obs,critic,metadata)
    a.record_step(torch.linspace(.1,1.,64),torch.zeros(64,dtype=torch.bool),torch.zeros(64,dtype=torch.bool))
    a.compute_returns(critic);report=a.update()
    assert report['accepted']
    assert report['actual_pre_action_review']['supplied_metadata_covers_all_64']
    assert not report['actual_pre_action_review']['execution_provenance_verified']
    assert report['group_counts']['towing_only/phase1']==0
    assert report['group_mean_kl']['towing_only/phase1'] is None
    assert sum(name.startswith('case/') for name in report['group_counts'])==64


def test_rejects_silent_rsl_tail_drop_and_unfinished_checkpoint(model):
    with pytest.raises(ValueError,match='without dropping a tail'):
        adapter(model,num_mini_batches=4)
    with pytest.raises(ValueError,match='at least 64'):
        ControlledPPOAdapter(model,num_envs=3,rollout_steps=2)
    a=adapter(model);a.act(*inputs())
    with pytest.raises(RuntimeError,match='empty rollout'):
        a.checkpoint()
    with pytest.raises(RuntimeError,match='pending step'):
        a.act(*inputs())


def test_condition_context_and_model_drift_are_explicitly_rejected(model):
    a=adapter(model);obs,critic,cases=inputs();bad=obs.clone();bad[2,299]=.5
    with pytest.raises(ValueError,match='nominal contexts'):
        a.act(bad,critic,cases)
    for _ in range(2):
        a.act(obs,critic,cases)
        a.record_step(torch.zeros(3),torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    with torch.no_grad():model.critic[-1].bias.add_(1.)
    with pytest.raises(ValueError,match='changed while collecting'):
        a.compute_returns(critic)


def test_bad_source_checkpoint_rejected_before_any_policy_mutation(model):
    a=adapter(model);saved=a.checkpoint();bad=deepcopy(saved)
    bad['sources']['ppo_policy.py']='0'*64
    with pytest.raises(ValueError,match='restore contract mismatch'):
        a.restore_checkpoint(bad)
    same_tree(saved,a.checkpoint())
