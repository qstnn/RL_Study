"""CPU gates/files only. No App, real environment, optimizer or PPO update."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exploration_v1 import runtime,train
from migration_v1.dataset import EvidenceError


def test_missing_terminal_or_formal_evidence_rejected_before_app_or_output(tmp_path):
    before=set(sys.modules)
    with pytest.raises(EvidenceError,match='Both actual GPU'):
        runtime.prepare(tmp_path/'short')
    with pytest.raises(EvidenceError,match='actual validated'):
        runtime.prepare(tmp_path/'formal',kind='formal_training',iterations=10000)
    assert not (tmp_path/'short').exists() and not (tmp_path/'formal').exists()
    assert not any(name.startswith('isaaclab.app') for name in set(sys.modules)-before)


@pytest.mark.parametrize('options',[{'num_envs':63},{'num_envs':True},{'inference_mode':'automatic'},
    {'iterations':200},{'save_interval':0},{'towing_initialization':'unknown'}])
def test_unsupported_configuration_never_creates_output(tmp_path,options):
    with pytest.raises(EvidenceError): runtime.prepare(tmp_path/'out',**options)
    assert not (tmp_path/'out').exists()


def test_false_actual_terminal_review_cannot_be_promoted(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'compare_runs',lambda *args: {'terminal_preview_gpu_pair_verified':False,'errors':['real mismatch']})
    with pytest.raises(EvidenceError,match='has not passed'):
        runtime.prepare(tmp_path/'out',terminal_timeout_run=tmp_path/'timeout',terminal_reference_run=tmp_path/'reference')
    assert not (tmp_path/'out').exists()


def test_prepare_clock_and_parallelism_configuration_is_explicit_without_app(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'verify_terminal_pair',lambda *args: {'terminal_preview_gpu_pair_verified':True})
    monkeypatch.setattr(runtime,'initialize_policy',lambda **kwargs:(SimpleNamespace(assert_invariants=lambda:None),{'synthetic':True}))
    monkeypatch.setattr(runtime,'source_signature',lambda num_envs: {'synthetic':str(num_envs)})
    result=runtime.prepare(tmp_path/'out',terminal_timeout_run='one',terminal_reference_run='two',num_envs=512,inference_mode='batch')
    assert result['configuration']['algorithm']['num_envs']==512
    assert result['configuration']['inference_mode']=='batch'
    assert result['configuration']['base_std']==[.02]*12+[.1]*4
    assert result['configuration']['towing_std']==[.1]*12+[.3]*4
    assert not result['training_runtime_ready'] and not (tmp_path/'out').exists()


def test_external_rsl_archive_path_is_unambiguous():
    path=Path('/home/user/miniconda3/envs/lab/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py')
    relative=runtime.archive_relative(path)
    assert not relative.is_absolute() and relative.parts[0]=='external'
    assert str(relative).endswith('site-packages/rsl_rl/algorithms/ppo.py')
    assert '..' not in relative.parts


@pytest.mark.parametrize('error_file,returncode',[(False,0),(True,0),(False,1)])
def test_parent_checks_artifact_instead_of_kit_exit_zero(tmp_path,error_file,returncode):
    if error_file: (tmp_path/'runtime_error.json').write_text('{}')
    with pytest.raises(EvidenceError):
        train.verify_process_outcome({'output':tmp_path,'kind':'runtime_short_validation'},returncode)


def test_parser_formal_iterations_are_explicit_additional_proposals():
    args=train.build_parser().parse_args(['train','--output','somewhere','--iterations','10000',
        '--validated-short-run','short','--num-envs','512','--inference-mode','batch'])
    assert args.iterations==10000 and args.num_envs==512 and args.inference_mode=='batch'


def test_streamed_statistics_distinguish_command_phase_from_success():
    result={};conditions=np.array([[0,0,0,0],[10,15,0,3],[25,0,40,2]],float)
    for _ in range(2): runtime.accumulate_rollout_diagnostics(result,conditions,np.array([1.,2.,3.]),
        np.ones((3,26)),np.array([0,1,0]),np.array([0,0,1]),np.array([[0,3],[3,3],[4,3]]))
    assert result['payload_only']['pre_action_command_phase_counts']==[0,0,0,2]
    assert result['payload_only']['terminated']==2
    assert result['towing_only']['truncated']==2 and result['towing_only']['reward_sum']==6.
    assert all('success' not in name for group in result.values() for name in group)


def test_resume_checkpoint_cannot_enter_short_mode(tmp_path):
    with pytest.raises(EvidenceError,match='only for explicit formal'):
        runtime.prepare(tmp_path/'out',resume_checkpoint='old.pt')
    assert not (tmp_path/'out').exists()


def test_formal_resume_schema_sources_config_and_counter_validation(tmp_path,monkeypatch):
    import torch
    from exploration_v1.ppo_adapter import VERSION
    config=runtime._configuration('bc_preparation_9f406')
    provenance={'synthetic':True};prior={'checkpoint_sha256':'prior'}
    (tmp_path/'request.json').write_text(json.dumps({'kind':'formal_training','configuration':config,
        'validated_short':prior,'initialization_provenance':provenance}))
    (tmp_path/'source_manifest.json').write_text('{}')
    monkeypatch.setattr(runtime,'source_signature',lambda n: {})
    monkeypatch.setattr(runtime,'_source_signature',lambda: {})
    monkeypatch.setattr(runtime,'verify_sources',lambda *args: None)
    monkeypatch.setattr(runtime,'_verify_numerical_checkpoint_cpu',lambda *args: {'synthetic':True})
    counts=dict(update_attempts=12,accepted_updates=11,rejected_updates=1,rsl_update_calls=12,observed_transition_frames=18432)
    saved=dict(schema=VERSION,config=dict(config['algorithm'],execution_kind='formal_training',schedule='fixed',
        normalize_advantage_per_mini_batch=False),sources={},environment_state_saved=False,
        training_runtime_ready=False,final_behavior_pass=False,counters=counts)
    path=tmp_path/'proposal.pt';torch.save(saved,path)
    result=runtime.verify_formal_resume(path,config,prior)
    assert result['counters']==counts and not result['environment_state_saved']
    for key,value in [('schema','old'),('sources',{'foreign':'x'}),('config',{}),('counters',dict(counts,accepted_updates=10))]:
        changed=dict(saved);changed[key]=value;torch.save(changed,path)
        with pytest.raises(EvidenceError): runtime.verify_formal_resume(path,config,prior)


def test_additional_counter_budget_includes_accepted_and_rejected():
    before=dict(update_attempts=2,accepted_updates=2,rejected_updates=0,rsl_update_calls=2,observed_transition_frames=3072)
    after=dict(update_attempts=6,accepted_updates=5,rejected_updates=1,rsl_update_calls=6,observed_transition_frames=9216)
    result=runtime.counter_delta(before,after)
    assert result['update_attempts']==4 and result['accepted_updates']==3 and result['rejected_updates']==1
    with pytest.raises(EvidenceError): runtime.counter_delta(after,before)


def test_prepare_cli_never_invokes_worker_or_subprocess(tmp_path,monkeypatch,capsys):
    prepared=dict(output=tmp_path/'no-output',kind='runtime_short_validation',configuration={},sources={})
    monkeypatch.setattr(train,'prepare_args',lambda args: prepared)
    monkeypatch.setattr(train.subprocess,'run',lambda *args,**kwargs: pytest.fail('prepare must not run a process'))
    monkeypatch.setattr(runtime,'run',lambda *args: pytest.fail('prepare must not create an App'))
    assert train.main(['prepare','--output',str(prepared['output'])])==0
    assert not prepared['output'].exists()
    assert json.loads(capsys.readouterr().out)['robot_ppo_updates']==0


def test_formal_cli_forwards_all_options_and_requires_parent_check(tmp_path,monkeypatch):
    prepared=dict(output=tmp_path/'out',kind='formal_training')
    monkeypatch.setattr(train,'prepare_args',lambda args: prepared)
    calls=[]
    def launch(command,**kwargs): calls.append(command);return SimpleNamespace(returncode=0)
    def checked(value,code):
        assert value is prepared and code==0;calls.append('checked');return {'completed':True}
    monkeypatch.setattr(train.subprocess,'run',launch);monkeypatch.setattr(train,'verify_process_outcome',checked)
    assert train.main(['train','--output',str(prepared['output']),'--iterations','10000',
        '--num-envs','512','--inference-mode','batch','--validated-short-run','validated',
        '--resume-checkpoint','last.pt','--save-interval','200'])==0
    command=calls[0]
    for key,value in [('--num-envs','512'),('--inference-mode','batch'),('--iterations','10000'),
        ('--validated-short-run','validated'),('--resume-checkpoint','last.pt'),('--save-interval','200')]:
        assert command[command.index(key)+1]==value
    assert calls[1]=='checked'


@pytest.fixture
def formal_completion_fixture(tmp_path,monkeypatch):
    import torch
    config=runtime._configuration('bc_preparation_9f406');initial=dict(update_attempts=2,accepted_updates=2,
        rejected_updates=0,rsl_update_calls=2,observed_transition_frames=3072)
    counts=dict(update_attempts=4,accepted_updates=3,rejected_updates=1,rsl_update_calls=4,observed_transition_frames=6144)
    initial_path=tmp_path/'initial.pt';torch.save({'counters':initial},initial_path)
    prior={'checkpoint':str(initial_path),'checkpoint_sha256':runtime.sha256(initial_path)}
    prepared={'output':tmp_path,'kind':'formal_training','configuration':config,'iterations':2,'sources':{},'validated_short':prior,'resume':None}
    request={'kind':'formal_training','configuration':config,'iterations':2,'initial_counters':initial,
        'validated_short':prior,'initialization_provenance':{}}
    final=tmp_path/'final.pt';final.write_bytes(b'not a torch checkpoint')
    review={'kind':'formal_training','formal_requested_proposals_completed':True,'final_behavior_pass':False,
        'executed_update_attempts':2,'executed_control_steps':48,'counters':counts,
        'additional_counters':runtime.counter_delta(initial,counts),'accepted_updates':1,'rejected_updates':1,
        'final_checkpoint':{'file':'final.pt','sha256':runtime.sha256(final)}}
    for name,data in [('request.json',request),('source_manifest.json',{}),('runtime_review.json',review)]:
        (tmp_path/name).write_text(json.dumps(data))
    rows=[]
    for index,accepted in enumerate([True,False]):
        current=dict(update_attempts=3+index,accepted_updates=3,rejected_updates=index,rsl_update_calls=3+index,
            observed_transition_frames=4608+index*1536)
        rows.append({'counters':current,'accepted':accepted,'execution_kind':'formal_training',
            'adam_parameter_steps_before':{'p':16+8*index},'adam_parameter_steps_proposed':{'p':24+8*index},
            'adam_parameter_steps_retained':{'p':24}})
    (tmp_path/'updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    monkeypatch.setattr(runtime,'source_signature',lambda n: {})
    monkeypatch.setattr(runtime,'verify_sources',lambda *args: None)
    return prepared,review,counts


def test_formal_exit_zero_with_matching_sha_noncheckpoint_is_rejected(formal_completion_fixture):
    import pickle
    prepared,_,_=formal_completion_fixture
    with pytest.raises((EvidenceError,pickle.UnpicklingError,RuntimeError)):
        train.verify_process_outcome(prepared,0)


@pytest.mark.parametrize('field,value',[('executed_update_attempts',1),('executed_control_steps',47),
    ('accepted_updates',2),('counters',{}),('additional_counters',{})])
def test_formal_outer_gate_recomputes_counter_and_budget_deltas(formal_completion_fixture,monkeypatch,field,value):
    prepared,review,counts=formal_completion_fixture
    monkeypatch.setattr(runtime,'verify_formal_resume',lambda *args: {'counters':counts})
    assert runtime.validate_formal_completion(prepared)['numerical_completion_verified']
    review[field]=value;(prepared['output']/'runtime_review.json').write_text(json.dumps(review))
    with pytest.raises(EvidenceError): train.verify_process_outcome(prepared,0)


def test_failed_first_short_candidate_changes_only_towing_learning_rate():
    config=runtime._configuration('bc_preparation_9f406')['algorithm']
    assert [config[name] for name in ('actor_lr','non_towing_lr','critic_lr','std_lr','non_towing_std_lr')]==[1.e-6,1.e-6,1.e-4,1.e-5,1.e-6]
    assert config['max_kl']==.01 and config['group_kl_limit']==.02
    assert config['num_envs']==64 and config['rollout_steps']==24


def test_cpu_numerical_precheck_never_initializes_cuda_and_preserves_rng(monkeypatch):
    import torch
    from exploration_v1.ppo_adapter import ControlledPPOAdapter
    config=runtime._configuration('bc_preparation_9f406')
    policy,provenance=runtime.initialize_policy(towing_initialization='bc_preparation_9f406',
        base_std=runtime.BASE_STD,towing_std=runtime.TOWING_STD,device='cpu',inference_mode='row')
    adapter=ControlledPPOAdapter(policy,execution_kind='synthetic_cpu',**config['algorithm'])
    saved=adapter.checkpoint()
    # Synthetic CUDA-state byte vector: this test proves CPU staging, not its
    # validity for a GPU generator. Actual GPU exact restore is separately run.
    saved['policy']['rng_state']['cuda']=torch.zeros(16,dtype=torch.uint8)
    for name in ('_lazy_init','get_rng_state','set_rng_state'):
        monkeypatch.setattr(torch.cuda,name,lambda *args,**kwargs: pytest.fail('CPU preflight called CUDA'))
    before=torch.get_rng_state().clone()
    result=runtime._verify_numerical_checkpoint_cpu(saved,config,provenance)
    assert result['model_and_adam_strict_cpu_loaded'] and torch.equal(before,torch.get_rng_state())
    for bad in (torch.zeros(16),torch.zeros(2,8,dtype=torch.uint8)):
        saved['policy']['rng_state']['cuda']=bad
        with pytest.raises(EvidenceError,match='RNG byte vectors'):
            runtime._verify_numerical_checkpoint_cpu(saved,config,provenance)


def test_parent_records_actual_signal_without_marking_completion(tmp_path):
    (tmp_path/'runtime_review.json').write_text('{}')
    result=train.record_child_outcome({'output':tmp_path},-11)
    assert result['returncode']==-11 and result['signal']==11
    assert result['runtime_review_present'] and result['artifact_role']=='subprocess_exit_record_before_independent_validation'
    assert json.loads((tmp_path/'child_process.json').read_text())==result
    with pytest.raises(EvidenceError,match='did not exit successfully'):
        runtime.validate_short_run(tmp_path)


def test_early_runtime_ready_without_parent_exit_evidence_is_not_a_formal_source(tmp_path):
    (tmp_path/'runtime_review.json').write_text(json.dumps({'training_runtime_ready':True}))
    with pytest.raises(EvidenceError,match='parent-process exit record'):
        runtime.validate_short_run(tmp_path)
