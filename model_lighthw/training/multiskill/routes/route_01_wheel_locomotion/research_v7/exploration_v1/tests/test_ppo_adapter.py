"""Standard RSL on synthetic CPU tensors only; actual robot PPO updates stay zero."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exploration_v1.contract import append_task_clock
from exploration_v1.policy import ExplorationActorCritic, UnifiedResidualActor
from exploration_v1.ppo_adapter import ControlledPPOAdapter
from migration_v1.contract import install_actor_mask
from migration_v1.policy import tensor_fingerprint
from migration_v1.schedule import make_schedule, SPEC, FULL_PROTOCOL


@pytest.fixture(scope='module', autouse=True)
def accounting():
    from rsl_rl.algorithms import PPO
    original = PPO.update; calls = []
    def track(algorithm):
        calls.append(False)
        result = original(algorithm)
        calls[-1] = True
        return result
    PPO.update = track
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield calls
    PPO.update = original; torch.set_num_threads(previous)
    print(f'SYNTHETIC_CPU_RSL_UPDATE_CALLS={len(calls)}; COMPLETED={sum(calls)}; ROBOT_PPO_UPDATES=0')


def network():
    result = nn.Sequential(nn.Linear(300,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                           nn.Linear(256,128),nn.ELU(),nn.Linear(128,16))
    install_actor_mask(result)
    return result


@pytest.fixture
def model():
    torch.manual_seed(191)
    return ExplorationActorCritic(UnifiedResidualActor(network()),
        base_std=[.02]*12+[.10]*4,towing_std=[.10]*12+[.30]*4,critic_hidden_dims=(16,8))


def adapter(model, **options):
    config = dict(num_envs=3, rollout_steps=2, mode='diagnostic', max_kl=100.)
    config.update(options)
    return ControlledPPOAdapter(model, **config)


def inputs(phase=0):
    cases = torch.tensor([[0.,0.,0.,phase],[0.,15.,0.,phase],[25.,0.,40.,phase]])
    raw = torch.randn(3,300,generator=torch.Generator().manual_seed(291+phase))
    raw[:,298] = cases[:,1]/15.; raw[:,299] = cases[:,2]/40.
    clocks = torch.tensor([[phase*.02,3.],[phase*.02,3.],[phase*.02,3.]])
    observation = append_task_clock(raw,clocks[:,0],clocks[:,1])
    critic = torch.randn(3,252,generator=torch.Generator().manual_seed(391+phase))
    return observation,critic,cases,clocks


def act(a, data):
    obs,critic,conditions,clocks = data
    return a.act(obs,critic,conditions,pre_action_clock_seconds=clocks)


def fill(a):
    for phase in range(a.config['rollout_steps']):
        act(a,inputs(phase%4))
        a.record_step(torch.tensor([.1,.2,.3])+phase,torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    a.compute_returns(inputs(3)[1])


def same(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b


def test_standard_rsl_update_learns_both_residuals_freezes_base_and_records_actual_adam(model):
    a = adapter(model,num_learning_epochs=2,num_mini_batches=2)
    base = tensor_fingerprint(model.actor.base_actor)
    tow = tensor_fingerprint(model.actor.towing_residual)
    non = tensor_fingerprint(model.actor.non_towing_residual)
    fill(a); report = a.update()
    assert report['accepted'] and report['execution_kind']=='synthetic_cpu'
    assert tensor_fingerprint(model.actor.base_actor)==base
    assert tensor_fingerprint(model.actor.towing_residual)!=tow
    assert tensor_fingerprint(model.actor.non_towing_residual)!=non
    assert set(report['adam_parameter_steps_before'].values())=={0}
    assert set(report['adam_parameter_steps_proposed'].values())=={4}
    assert report['adam_parameter_steps_retained']==report['adam_parameter_steps_proposed']
    assert all(not name.startswith('actor.base_actor') for name in report['adam_parameter_steps_retained'])
    assert str(SPEC.resolve()) in a.sources and str(FULL_PROTOCOL.resolve()) in a.sources
    assert not report['training_runtime_ready'] and not report['final_behavior_pass']
    model.assert_invariants(a.optimizer)


def test_explicit_timeout_value_not_reset_or_pre_action_value_and_true_terminal_precedence(model,monkeypatch):
    monkeypatch.setattr(model,'evaluate',lambda z,**kwargs:z[:,:1])
    a=adapter(model,gamma=.9,lam=.8)
    data=inputs();data[1][:,0]=torch.tensor([1.,2.,3.]);act(a,data)
    terminal=torch.zeros(3,252);terminal[:,0]=torch.tensor([99.,5.,99.])
    flags=torch.tensor([False,False,True]);timeouts=torch.tensor([False,True,True])
    with pytest.raises(ValueError,match='post-command-preview'):
        a.record_step(torch.tensor([.1,.2,.3]),flags,timeouts)
    a.record_step(torch.tensor([.1,.2,.3]),flags,timeouts,terminal_critic_observations=terminal)
    data[1][:,0]=torch.tensor([4.,6.,7.]);act(a,data)
    a.record_step(torch.tensor([1.,2.,3.]),torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    last=torch.zeros(3,252);last[:,0]=torch.tensor([8.,9.,10.]);a.compute_returns(last)
    torch.testing.assert_close(a.storage.returns[:,:,0],torch.tensor([[6.724,4.7,.3],[8.2,10.1,12.]]))


def test_actual_clocks_checked_and_collection_snapshots_detached(model):
    a=adapter(model);data=inputs();obs,critic,conditions,clocks=data
    bad=clocks.clone();bad[0,0]=2.
    with pytest.raises(ValueError,match='clock columns'):
        a.act(obs,critic,conditions,pre_action_clock_seconds=bad)
    expected=obs.clone();action=act(a,data);saved=action.clone()
    assert not action.requires_grad
    obs.zero_();critic.zero_();clocks.zero_();action.zero_()
    a.record_step(torch.ones(3),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    assert torch.equal(a.storage.observations[0],expected)
    assert torch.equal(a.storage.actions[0],saved)
    assert a.clock_history[0][:,1].eq(3.).all()


def test_kl_rejection_restores_whole_policy_optimizer_rng(model):
    a=adapter(model,max_kl=0.,actor_lr=.001,non_towing_lr=.001,critic_lr=.001,std_lr=.001,non_towing_std_lr=.001)
    fill(a);before=model.checkpoint(a.optimizer);report=a.update()
    assert not report['accepted'];same(before,model.checkpoint(a.optimizer))
    assert set(report['adam_parameter_steps_proposed'].values())=={1}
    assert set(report['adam_parameter_steps_retained'].values())=={0}


def test_partial_adam_exception_captures_proposal_then_rolls_back(model,monkeypatch):
    a=adapter(model);fill(a);before=model.checkpoint(a.optimizer)
    original=a.optimizer.step
    def fail(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('injected after actual Adam step')
    monkeypatch.setattr(a.optimizer,'step',fail)
    with pytest.raises(RuntimeError,match='actual Adam'):
        a.update()
    same(before,model.checkpoint(a.optimizer))
    report=a.last_update_report
    assert set(report['adam_parameter_steps_proposed'].values())=={1}
    assert set(report['adam_parameter_steps_retained'].values())=={0}
    assert not report['accepted'] and a.storage.step==0


def test_saved_adapter_restored_into_new_model_exact_next_cpu_update(model,tmp_path):
    a=adapter(model);fill(a);assert a.update()['accepted']
    path=tmp_path/'synthetic.pt';a.save(path)
    with pytest.raises(FileExistsError):a.save(path)
    fill(a);assert a.update()['accepted'];expected=a.checkpoint()
    fresh=ExplorationActorCritic(UnifiedResidualActor(deepcopy(model.actor.base_actor)),
        base_std=[.02]*12+[.10]*4,towing_std=[.10]*12+[.30]*4,critic_hidden_dims=(16,8))
    b=adapter(fresh);b.load(path);fill(b);assert b.update()['accepted']
    same(expected,b.checkpoint())


def test_all64_first_batch_then_only_observed_stage_groups(model):
    a=ControlledPPOAdapter(model,num_envs=64,rollout_steps=1,max_kl=100.)
    cases=torch.tensor(make_schedule(64)['conditions'],dtype=torch.float32)
    conditions=torch.cat((cases,torch.zeros(64,1)),dim=1)
    raw=torch.zeros(64,300);raw[:,298]=cases[:,1]/15.;raw[:,299]=cases[:,2]/40.
    clocks=torch.tensor([[0.,3.]]*64);obs=append_task_clock(raw,clocks[:,0],clocks[:,1]);critic=torch.zeros(64,252)
    a.act(obs,critic,conditions,pre_action_clock_seconds=clocks)
    a.record_step(torch.linspace(.1,1.,64),torch.zeros(64,dtype=torch.bool),torch.zeros(64,dtype=torch.bool))
    a.compute_returns(critic);report=a.update()
    assert report['accepted'] and report['actual_pre_action_review']['supplied_metadata_covers_all_64']
    assert report['group_counts']['towing_only/phase1']==0
    assert not report['environment_state_verified']


@pytest.mark.parametrize('kind',['runtime_short_validation','formal_training'])
def test_real_runtime_rejects_diagnostic_or_cpu_execution(model,kind):
    with pytest.raises(ValueError,match='full 64'):
        adapter(model,execution_kind=kind)
    with pytest.raises(ValueError,match='GPU policy'):
        ControlledPPOAdapter(model,num_envs=64,rollout_steps=1,execution_kind=kind)


@pytest.mark.parametrize('field,value',[('lr',float('nan')),('lr',.8),('betas',(.1,.3)),('weight_decay',.1)])
def test_restore_rejects_changed_fixed_adam_before_mutating_model(model,field,value):
    a=adapter(model);saved=a.checkpoint();before=model.checkpoint(a.optimizer)
    saved['policy']['optimizer_state_dict']['param_groups'][0][field]=value
    with pytest.raises(ValueError,match='hyperparameter'):
        a.restore_checkpoint(saved)
    same(before,model.checkpoint(a.optimizer))


def test_explicit_kind_conversion_preserves_every_other_config_field(model):
    # Pure checkpoint-validation unit test, deliberately no GPU/actual-runtime claim.
    a=adapter(model);saved=a.checkpoint()
    a.config['execution_kind']='formal_training'
    saved['config']['execution_kind']='runtime_short_validation'
    with pytest.raises(ValueError,match='config'): a.restore_checkpoint(saved)
    a.restore_checkpoint(saved,allow_short_to_formal=True)
    changed=deepcopy(saved);changed['config']['gamma']=.5
    with pytest.raises(ValueError,match='config'):
        a.restore_checkpoint(changed,allow_short_to_formal=True)
    changed=deepcopy(saved);changed['config']['execution_kind']='synthetic_cpu'
    with pytest.raises(ValueError,match='Only explicit'):
        a.restore_checkpoint(changed,allow_short_to_formal=True)


def test_checkpoint_load_stages_on_cpu_for_native_adam_device_ownership(model,tmp_path,monkeypatch):
    a=adapter(model);path=tmp_path/'roundtrip.pt';a.save(path)
    original=torch.load;locations=[]
    def checked(*args,**kwargs):
        locations.append(kwargs.get('map_location'))
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',checked)
    a.load(path)
    assert locations==['cpu']
