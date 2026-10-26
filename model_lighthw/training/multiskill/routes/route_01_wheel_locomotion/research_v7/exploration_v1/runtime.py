"""Bounded real exploration PPO and manual formal continuation.

Runtime readiness is source/physics/algorithm/restore evidence, never a claim
that the actor already completes R1. Old migration behavior gates are unchanged.
This module creates an Isaac App only inside explicitly invoked run().
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from migration_v1.contract import REPO, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.runtime_checks import check_action_mapping
from migration_v1.schedule import validate_pre_action_conditions
from migration_v1.student_worker import write_json
from migration_v1.terminal_preview_probe import compare_runs
from . import environment
from .checkpoint import initialize_policy
from .ppo_adapter import ControlledPPOAdapter, _source_signature

VERSION = 'r1_exploration_v1_runtime'
BASE_STD = [.02]*12 + [.10]*4
TOWING_STD = [.10]*12 + [.30]*4
SHORT_ACCEPTED_UPDATES = 2
SHORT_MAX_ATTEMPTS = 4
ALGORITHM = dict(num_envs=64,rollout_steps=24,mode='full_first_batch',
    num_learning_epochs=2,num_mini_batches=4,actor_lr=1.e-6,non_towing_lr=1.e-6,
    critic_lr=1.e-4,std_lr=1.e-5,non_towing_std_lr=1.e-6,gamma=.995,lam=.95,
    clip_param=.2,value_loss_coef=1.,entropy_coef=0.,max_grad_norm=1.,max_kl=.01,group_kl_limit=.02)


def source_signature(num_envs=64):
    result=environment.source_signature(num_envs)
    result.update(_source_signature())
    native=REPO.parent/'Isaaclab/source/isaaclab/isaaclab'
    files=[Path(__file__),Path(__file__).with_name('train.py'),
        Path(inspect.getfile(compare_runs)),native/'utils/math.py',native/'envs/mdp/observations.py',
        native/'managers/termination_manager.py',native/'app/app_launcher.py',
        REPO.parent/'isaac-sim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py']
    result.update({str(path.resolve()):sha256(path) for path in files})
    return result


def archive_relative(path):
    path=Path(path).resolve()
    if path.is_relative_to(REPO): return Path('repository')/path.relative_to(REPO)
    if path.is_relative_to(REPO.parent): return Path('workspace')/path.relative_to(REPO.parent)
    # RSL is installed outside the workspace; retain its unambiguous absolute ancestry.
    return Path('external')/path.relative_to(path.anchor)


def verify_sources(sources, archive=None):
    for name,digest in sources.items():
        if sha256(name)!=digest: raise EvidenceError('Exploration execution source changed: '+name)
        if archive is not None and sha256(Path(archive)/archive_relative(name))!=digest:
            raise EvidenceError('Exploration archived source changed: '+name)


def _same(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(_same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return a==b


def _array(value): return value.detach().cpu().numpy().copy()


def counter_delta(before,after):
    if set(before)!=set(after) or any(type(after[k]) is not int or after[k]<before[k] for k in before):
        raise EvidenceError('Continuation counters moved backwards or changed fields')
    result={name:after[name]-before[name] for name in before}
    if (result['update_attempts']!=result['accepted_updates']+result['rejected_updates']
            or result['rsl_update_calls']!=result['update_attempts']):
        raise EvidenceError('Continuation proposal/acceptance counters disagree')
    return result


def _configuration(initialization,num_envs=64,inference_mode='row'):
    algorithm=dict(ALGORITHM,num_envs=num_envs)
    return dict(version=VERSION,algorithm=algorithm,device='cuda:0',seed=43,
        episode_seconds=65.,towing_initialization=initialization,base_std=BASE_STD,towing_std=TOWING_STD,
        critic_hidden_dims=[512,256,128],inference_mode=inference_mode,teacher_calls=0,
        observation_noise=False,physical_randomization=False,
        std_origin='Explicit exploration design, not restored old checkpoint std: non-tow raw legs .02/wheels .10; towing raw legs .10/wheels .30',
        ordinary_reset=True,source_physics='mu3_GPU16_4_original_passive_cart',
        final_behavior_pass=False)


def verify_terminal_pair(timeout_run,reference_run):
    review=compare_runs(timeout_run,reference_run)
    if review.get('terminal_preview_gpu_pair_verified') is not True:
        raise EvidenceError('Actual terminal preview pair has not passed: '+repr(review.get('errors')))
    sources=json.loads((Path(timeout_run)/'terminal_preview_source_manifest.json').read_text())
    verify_sources(sources)
    return review


def _verify_numerical_checkpoint_cpu(saved,configuration,provenance):
    """Strict model/Adam validation before App; CUDA RNG bytes stay unapplied."""
    rng=saved.get('policy',{}).get('rng_state',{})
    if set(rng)!={'cpu','cuda'} or any(not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8
            or value.ndim!=1 or not value.numel() for value in rng.values()):
        raise EvidenceError('Actual GPU checkpoint lacks CPU/CUDA RNG byte vectors')
    with torch.random.fork_rng(devices=[]):
        policy,current=initialize_policy(towing_initialization=configuration['towing_initialization'],
            base_std=configuration['base_std'],towing_std=configuration['towing_std'],device='cpu',
            inference_mode=configuration['inference_mode'],critic_hidden_dims=tuple(configuration['critic_hidden_dims']))
        if current!=provenance: raise EvidenceError('Saved initialization provenance differs')
        verifier=ControlledPPOAdapter(policy,execution_kind='synthetic_cpu',**configuration['algorithm'])
        verifier._validate_saved_optimizer(saved)
        local=deepcopy(saved['policy'])
        local['rng_state']={'cpu':local['rng_state']['cpu']}
        policy.restore_checkpoint(local,verifier.optimizer)
        policy.assert_invariants(verifier.optimizer)
    return dict(model_and_adam_strict_cpu_loaded=True,cpu_rng_loaded_in_local_fork=True,
        cuda_rng_structure_checked=True,cuda_rng_device_restore_requires_runtime=True,ppo_updates=0)


def validate_short_run(directory,*,expected_configuration=None):
    """Recompute raw source, mapping, coverage and numerical restore evidence."""
    root=Path(directory).resolve(strict=True)
    if (root/'runtime_error.json').exists(): raise EvidenceError('A failed runtime directory cannot authorize formal continuation')
    process_path=root/'child_process.json'
    if not process_path.is_file(): raise EvidenceError('Short run lacks its parent-process exit record')
    process=json.loads(process_path.read_text())
    if type(process.get('returncode')) is not int or process['returncode']!=0:
        raise EvidenceError('Short simulation subprocess did not exit successfully')
    if process.get('runtime_error_present') is not False or process.get('runtime_review_present') is not True:
        raise EvidenceError('Parent observed missing completion or a runtime error')
    request=json.loads((root/'request.json').read_text())
    review=json.loads((root/'runtime_review.json').read_text())
    if (request.get('kind')!='runtime_short_validation' or review.get('kind')!='runtime_short_validation'
            or review.get('training_runtime_ready') is not True or review.get('final_behavior_pass') is not False
            or review.get('teacher_calls')!=0 or review.get('executed_update_attempts',0)>SHORT_MAX_ATTEMPTS
            or review.get('accepted_updates')!=SHORT_ACCEPTED_UPDATES
            or review.get('new_instance_restore_verified') is not True):
        raise EvidenceError('Missing bounded real PPO / new-instance restore evidence')
    config=request['configuration']
    if expected_configuration is not None and config!=expected_configuration:
        raise EvidenceError('Short-validation configuration differs from requested formal continuation')
    if config!=_configuration(config.get('towing_initialization'),config['algorithm']['num_envs'],config.get('inference_mode')):
        raise EvidenceError('Short run used an unreviewed configuration')
    sources=json.loads((root/'source_manifest.json').read_text())
    if sources!=source_signature(config['algorithm']['num_envs']): raise EvidenceError('Runtime sources differ from the actual short validation')
    verify_sources(sources,root/'source_snapshot')
    terminal=verify_terminal_pair(request['terminal_timeout_run'],request['terminal_reference_run'])
    if terminal!=request['terminal_review']: raise EvidenceError('Terminal diagnostic raw evidence changed')
    final=review['final_checkpoint'];path=root/final['file']
    if sha256(path)!=final['sha256']: raise EvidenceError('Short final checkpoint SHA differs')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    from .ppo_adapter import VERSION as ADAPTER_VERSION
    if (saved.get('schema')!=ADAPTER_VERSION or saved.get('sources')!=_source_signature()
            or saved.get('environment_state_saved') is not False or saved.get('training_runtime_ready') is not False
            or saved.get('final_behavior_pass') is not False):
        raise EvidenceError('Short final adapter schema/source flags differ')
    if (saved['counters']['accepted_updates']!=2 or saved['config']!=dict(config['algorithm'],
            execution_kind='runtime_short_validation',schedule='fixed',normalize_advantage_per_mini_batch=False)):
        raise EvidenceError('Short numerical checkpoint counters/configuration differ')
    if saved['counters']!=review['counters']: raise EvidenceError('Short review differs from saved actual counters')
    if any(value!=0 for value in request.get('initial_counters',{}).values()) or not request.get('initial_counters'):
        raise EvidenceError('Short validation must start with fresh optimizer/update counters')
    restore=json.loads((root/'new_instance_restore.json').read_text())
    if (restore.get('checkpoint_sha256')!=sha256(root/'restore_boundary.pt')
            or any(restore.get(key) is not True for key in
                ('new_instance','all_model_optimizer_rng_exact','actual_state_gaussian_and_critic_exact','environment_continues_without_reset'))
            or restore.get('environment_restored') is not False):
        raise EvidenceError('New-instance actual-state restore evidence is missing or changed')
    numerical=_verify_numerical_checkpoint_cpu(saved,config,request['initialization_provenance'])
    accepted=[row for row in review['updates'] if row.get('accepted') is True]
    if len(accepted)!=2: raise EvidenceError('Need exactly two accepted actual updates')
    for row in accepted:
        before,after=row['adam_parameter_steps_before'],row['adam_parameter_steps_retained']
        if (row.get('execution_kind')!='runtime_short_validation' or not before or before.keys()!=after.keys()
                or after!=row['adam_parameter_steps_proposed'] or any(after[k]<=before[k] for k in before)):
            raise EvidenceError('Accepted update lacks actual retained Adam step evidence')
    trace=root/review['trace']['file']
    if sha256(trace)!=review['trace']['sha256']: raise EvidenceError('Short rollout trace SHA differs')
    with np.load(trace,allow_pickle=False) as data:
        if data['observation'].shape[1:]!=(config['algorithm']['num_envs'],302) or data['critic'].shape[1:]!=(config['algorithm']['num_envs'],252):
            raise EvidenceError('Short run did not collect actual 302/252 batches')
        if any(not np.isfinite(data[key]).all() for key in data.files):
            raise EvidenceError('Short raw rollout contains nonfinite arrays')
        expected_frames=review['executed_update_attempts']*config['algorithm']['rollout_steps']
        if (len(data['observation'])!=expected_frames
                or saved['counters']['observed_transition_frames']!=config['algorithm']['num_envs']*expected_frames):
            raise EvidenceError('Actual frame/transition counters disagree with complete rollouts')
        validate_pre_action_conditions(data['conditions'][0],mode='full_first_batch')
        for index in range(len(data['observation'])):
            check_action_mapping(**{key:data[key][index] for key in
                ('actor_raw','manager_raw','leg_target','wheel_target','default_legs','leg_clip')},
                wrapper_clip=request['wrapper_clip'])
        if len(data['observation'])!=review['executed_control_steps']:
            raise EvidenceError('Actual rollout frame count differs')
        if not np.allclose(data['observation'][:,:,300:302],
                np.clip(data['clocks']/np.array([40.,8.]),0.,1.),atol=1.e-6,rtol=0.):
            raise EvidenceError('Recorded exploration task clock columns differ')
    return dict(directory=str(root),checkpoint=str(path),checkpoint_sha256=sha256(path),
        review_sha256=sha256(root/'runtime_review.json'),training_runtime_ready=True,
        final_behavior_pass=False,configuration=config,terminal_review=terminal,numerical_cpu_precheck=numerical)


def verify_formal_resume(path,configuration,validated_short):
    """Resume numerical state only, with the original full runtime source archive."""
    path=Path(path).resolve(strict=True);root=path.parent
    request=json.loads((root/'request.json').read_text())
    sources=json.loads((root/'source_manifest.json').read_text())
    if (request.get('kind')!='formal_training' or request.get('configuration')!=configuration
            or request.get('validated_short',{}).get('checkpoint_sha256')!=validated_short['checkpoint_sha256']):
        raise EvidenceError('Formal resume run configuration/initial short evidence differs')
    if sources!=source_signature(configuration['algorithm']['num_envs']):
        raise EvidenceError('Formal resume runtime source contract differs')
    verify_sources(sources,root/'source_snapshot')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(configuration['algorithm'],execution_kind='formal_training',
        schedule='fixed',normalize_advantage_per_mini_batch=False)
    from .ppo_adapter import VERSION as ADAPTER_VERSION
    if (saved.get('schema')!=ADAPTER_VERSION or saved.get('config')!=expected
            or saved.get('sources')!=_source_signature() or saved.get('environment_state_saved') is not False
            or saved.get('training_runtime_ready') is not False or saved.get('final_behavior_pass') is not False):
        raise EvidenceError('Formal resume numerical schema/source/configuration differs')
    counts=saved.get('counters',{})
    keys={'update_attempts','accepted_updates','rejected_updates','rsl_update_calls','observed_transition_frames'}
    if (set(counts)!=keys or any(type(v) is not int or v<0 for v in counts.values())
            or counts['update_attempts']!=counts['accepted_updates']+counts['rejected_updates']
            or counts['rsl_update_calls']!=counts['update_attempts']):
        raise EvidenceError('Formal resume update counters are invalid')
    numerical=_verify_numerical_checkpoint_cpu(saved,configuration,request['initialization_provenance'])
    return dict(checkpoint=str(path),checkpoint_sha256=sha256(path),counters=counts,
        original_run=str(root),numerical_cpu_precheck=numerical,environment_state_saved=False,
        continuation='model/Adam/RNG restored; a fresh ordinary environment reset, not a physical trajectory continuation')


def validate_formal_completion(prepared):
    """Outer-process completion validation includes the actual numerical artifact."""
    root=prepared['output'];request=json.loads((root/'request.json').read_text())
    review=json.loads((root/'runtime_review.json').read_text())
    if (request.get('kind')!='formal_training' or request.get('configuration')!=prepared['configuration']
            or request.get('iterations')!=prepared['iterations'] or review.get('kind')!='formal_training'
            or review.get('formal_requested_proposals_completed') is not True
            or review.get('final_behavior_pass') is not False):
        raise EvidenceError('Formal request/completion contract differs')
    sources=json.loads((root/'source_manifest.json').read_text())
    if sources!=prepared['sources']: raise EvidenceError('Formal source manifest differs from preflight')
    verify_sources(sources,root/'source_snapshot')
    final=review['final_checkpoint']
    if Path(final['file']).name!=final['file']: raise EvidenceError('Final checkpoint must be in the actual run directory')
    path=root/final['file']
    if sha256(path)!=final['sha256']: raise EvidenceError('Formal final checkpoint SHA differs')
    verified=verify_formal_resume(path,prepared['configuration'],prepared['validated_short'])
    initial=(prepared['resume']['counters'] if prepared.get('resume') is not None else
        torch.load(prepared['validated_short']['checkpoint'],map_location='cpu',weights_only=True)['counters'])
    if request.get('initial_counters')!=initial: raise EvidenceError('Formal initial counters differ from resumed numerical state')
    delta=counter_delta(initial,verified['counters']);algorithm=prepared['configuration']['algorithm']
    controls=prepared['iterations']*algorithm['rollout_steps']
    if (delta['update_attempts']!=prepared['iterations'] or review.get('executed_update_attempts')!=prepared['iterations']
            or review.get('executed_control_steps')!=controls
            or delta['observed_transition_frames']!=controls*algorithm['num_envs']
            or review.get('counters')!=verified['counters'] or review.get('additional_counters')!=delta
            or review.get('accepted_updates')!=delta['accepted_updates'] or review.get('rejected_updates')!=delta['rejected_updates']):
        raise EvidenceError('Formal numerical counters differ from requested additional proposals/control steps')
    previous=initial;count=0
    with (root/'updates.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);change=counter_delta(previous,row['counters'])
            if (change['update_attempts']!=1 or row.get('execution_kind')!='formal_training'
                    or change['observed_transition_frames']!=algorithm['num_envs']*algorithm['rollout_steps']
                    or change['accepted_updates']!=int(row['accepted']) or change['rejected_updates']!=int(not row['accepted'])):
                raise EvidenceError('Formal per-proposal actual counters disagree')
            before=row['adam_parameter_steps_before'];after=row['adam_parameter_steps_retained']
            if not before or before.keys()!=after.keys() or any(after[k]<before[k] for k in before):
                raise EvidenceError('Formal actual Adam step records are malformed')
            if (row['accepted'] and (after!=row['adam_parameter_steps_proposed'] or not any(after[k]>before[k] for k in before))) or (not row['accepted'] and after!=before):
                raise EvidenceError('Formal retained Adam state disagrees with proposal acceptance')
            previous=row['counters'];count+=1
    if count!=prepared['iterations'] or previous!=verified['counters']:
        raise EvidenceError('Formal update log is incomplete')
    return dict(review,numerical_completion_verified=True)


def prepare(output,*,kind='runtime_short_validation',terminal_timeout_run=None,
            terminal_reference_run=None,towing_initialization='bc_preparation_9f406',
            validated_short_run=None,iterations=None,save_interval=100,num_envs=64,inference_mode='row',resume_checkpoint=None):
    """CPU-only gate, no mkdir/App. Behavior success is deliberately not required."""
    if kind not in ('runtime_short_validation','formal_training'): raise EvidenceError('Unknown execution kind')
    if towing_initialization not in ('bc_preparation_9f406','zero'): raise EvidenceError('Unknown initializer')
    if type(num_envs) is not int or num_envs<64: raise EvidenceError('At least 64 actual first-batch slots are required')
    if inference_mode not in ('row','batch'): raise EvidenceError('Choose row or batch numerical inference explicitly')
    output=Path(output).resolve()
    if output.exists(): raise EvidenceError('Exploration output must be a fresh directory')
    if type(save_interval) is not int or save_interval<1: raise EvidenceError('Save interval must be positive')
    configuration=_configuration(towing_initialization,num_envs,inference_mode)
    prior=None;resume=None
    if resume_checkpoint is not None and kind!='formal_training':
        raise EvidenceError('Resume checkpoint is supported only for explicit formal continuation')
    if kind=='formal_training':
        if type(iterations) is not int or iterations<1: raise EvidenceError('Formal iterations count additional PPO proposals and must be positive')
        if validated_short_run is None: raise EvidenceError('Formal continuation requires an actual validated exploration short run')
        prior=validate_short_run(validated_short_run,expected_configuration=configuration)
        request=json.loads((Path(prior['directory'])/'request.json').read_text())
        terminal_timeout_run=request['terminal_timeout_run'];terminal_reference_run=request['terminal_reference_run']
        if resume_checkpoint is not None:
            resume=verify_formal_resume(resume_checkpoint,configuration,prior)
    elif iterations is not None or validated_short_run is not None:
        raise EvidenceError('Bounded short mode has a fixed 2 accepted / maximum 4 proposals budget')
    if terminal_timeout_run is None or terminal_reference_run is None:
        raise EvidenceError('Both actual GPU terminal diagnostic directories are required before an App')
    terminal_review=verify_terminal_pair(terminal_timeout_run,terminal_reference_run)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        policy,provenance=initialize_policy(towing_initialization=towing_initialization,
            base_std=BASE_STD,towing_std=TOWING_STD,device='cpu',inference_mode=inference_mode)
    policy.assert_invariants();del policy
    return dict(version=VERSION,output=output,kind=kind,configuration=configuration,
        initialization_provenance=provenance,sources=source_signature(num_envs),
        terminal_timeout_run=str(Path(terminal_timeout_run).resolve()),
        terminal_reference_run=str(Path(terminal_reference_run).resolve()),terminal_review=terminal_review,
        validated_short=prior,resume=resume,iterations=iterations,save_interval=save_interval,
        training_runtime_ready=False,final_behavior_pass=False)


def _new_adapter(prepared):
    config=prepared['configuration']
    policy,provenance=initialize_policy(towing_initialization=config['towing_initialization'],
        base_std=config['base_std'],towing_std=config['towing_std'],device='cuda:0',
        inference_mode=config['inference_mode'],critic_hidden_dims=tuple(config['critic_hidden_dims']))
    if provenance!=prepared['initialization_provenance']: raise EvidenceError('Strict initialization provenance changed')
    return ControlledPPOAdapter(policy,execution_kind=prepared['kind'],**config['algorithm'])


def _checkpoint(a,path):
    info=a.save(path)
    return dict(file=path.name,sha256=info['sha256'],counters=deepcopy(a.counters),environment_state_saved=False)


def accumulate_rollout_diagnostics(result,conditions,rewards,reward_terms,terminated,truncated,clocks):
    """Streaming training statistics only; command phase never means behavior pass."""
    arrays=[np.asarray(value) for value in (conditions,rewards,reward_terms,terminated,truncated,clocks)]
    if any(not np.isfinite(value).all() for value in arrays):
        raise EvidenceError('Nonfinite rollout diagnostic input')
    conditions,rewards,reward_terms,terminated,truncated,clocks=arrays
    masks={'unloaded':(conditions[:,1]==0)&(conditions[:,2]==0),
        'payload_only':conditions[:,1]>0,'towing_only':conditions[:,2]>0}
    for name,mask in masks.items():
        entry=result.setdefault(name,dict(sample_count=0,reward_sum=0.,reward_terms_sum=[0.]*reward_terms.shape[1],
            terminated=0,truncated=0,pre_action_command_phase_counts=[0]*4,
            reset_elapsed_min_s=None,reset_elapsed_max_s=None))
        if not mask.any(): continue
        entry['sample_count']+=int(mask.sum());entry['reward_sum']+=float(rewards[mask].sum())
        entry['reward_terms_sum']=(np.asarray(entry['reward_terms_sum'])+reward_terms[mask].sum(0)).tolist()
        entry['terminated']+=int(terminated[mask].sum());entry['truncated']+=int(truncated[mask].sum())
        entry['pre_action_command_phase_counts']=(np.asarray(entry['pre_action_command_phase_counts'])+
            np.bincount(conditions[mask,3].astype(int),minlength=4)).tolist()
        low=float(clocks[mask,0].min());high=float(clocks[mask,0].max())
        entry['reset_elapsed_min_s']=low if entry['reset_elapsed_min_s'] is None else min(low,entry['reset_elapsed_min_s'])
        entry['reset_elapsed_max_s']=high if entry['reset_elapsed_max_s'] is None else max(high,entry['reset_elapsed_max_s'])


def run(prepared):
    """Explicit real execution; caller must invoke it in a checked subprocess."""
    verify_sources(prepared['sources'])
    verify_terminal_pair(prepared['terminal_timeout_run'],prepared['terminal_reference_run'])
    if prepared['kind']=='formal_training':
        validate_short_run(prepared['validated_short']['directory'],expected_configuration=prepared['configuration'])
        if prepared.get('resume') is not None:
            current=verify_formal_resume(prepared['resume']['checkpoint'],prepared['configuration'],prepared['validated_short'])
            if current!=prepared['resume']: raise EvidenceError('Formal resume checkpoint changed after preflight')
    out=prepared['output'];out.mkdir(parents=True,exist_ok=False)
    app=env=None;frames=[];updates=[];restored=False;adapter=None;proposals_executed=0;last_update=None
    try:
        for name,digest in prepared['sources'].items():
            target=out/'source_snapshot'/archive_relative(name);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(name,target)
        verify_sources(prepared['sources'],out/'source_snapshot')
        write_json(out/'source_manifest.json',prepared['sources'])
        from isaaclab.app import AppLauncher
        # Keep the native/legacy fast-shutdown default. The parent validates
        # persisted errors, raw evidence and numerical artifacts even if Kit exits 0.
        app=AppLauncher(headless=True,device='cuda:0').app
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        cfg,agent,manifest=environment.configure(out,prepared['configuration']['algorithm']['num_envs'],43,episode_seconds=65.)
        dump_yaml(str(out/'environment_config.yaml'),cfg);dump_yaml(str(out/'agent_config.yaml'),agent)
        env=RslRlVecEnvWrapper(environment.make_environment(cfg),clip_actions=agent.clip_actions)
        base=env.unwrapped;env.seed(43)
        obs300,extras=env.reset();observation=environment.observe(base,obs300)
        critic=extras['observations']['critic']
        first=environment.verify_pre_action(env,observation,out)
        torch.manual_seed(43);torch.cuda.manual_seed_all(43)
        adapter=_new_adapter(prepared)
        if prepared['kind']=='formal_training':
            if prepared.get('resume') is None:
                adapter.load(prepared['validated_short']['checkpoint'],allow_short_to_formal=True)
            else:
                adapter.load(prepared['resume']['checkpoint'])
            restored=True
        initial_counts=deepcopy(adapter.counters)
        request={key:value for key,value in prepared.items() if key not in ('output','sources')}
        request.update(wrapper_clip=env.clip_actions,environment_configuration=manifest,
            actual_first_batch=first,initial_counters=initial_counts,teacher_calls=0,
            formal_conversion=('short_to_formal_execution_kind_only' if prepared.get('resume') is None else 'formal_numeric_resume_fresh_environment')
                if prepared['kind']=='formal_training' else None)
        write_json(out/'request.json',request)
        mapping={};latest={}
        native_process=base.action_manager.process_action
        def record_mapping(action):
            native_process(action)
            legs=base.action_manager.get_term('joint_pos');wheels=base.action_manager.get_term('joint_vel')
            mapping.clear();mapping.update(actor_raw=_array(latest['raw']),manager_raw=_array(action),
                leg_target=_array(legs.processed_actions),wheel_target=_array(wheels.processed_actions),
                default_legs=_array(base.scene['robot'].data.default_joint_pos[:,legs._joint_ids]),
                leg_clip=_array(legs._clip))
            check_action_mapping(**mapping,wrapper_clip=env.clip_actions)
        base.action_manager.process_action=record_mapping
        limit=SHORT_MAX_ATTEMPTS if prepared['kind']=='runtime_short_validation' else prepared['iterations']
        controls=0
        for proposal in range(limit):
            diagnostics={}
            for _ in range(prepared['configuration']['algorithm']['rollout_steps']):
                if not app.is_running(): raise EvidenceError('App closed before requested PPO proposals completed')
                conditions=environment.pre_action_conditions(base)
                term=base.command_manager.get_term('base_velocity')
                clocks=torch.stack((base.episode_length_buf.float(),term.declared_preparation_steps.float()),-1)*base.step_dt
                with torch.no_grad():
                    action=adapter.act(observation,critic,conditions,pre_action_clock_seconds=clocks)
                    # Keep pre-action snapshots before native observation tensors can be overwritten.
                    saved=(dict(observation=_array(observation),critic=_array(critic),conditions=_array(conditions),clocks=_array(clocks))
                        if prepared['kind']=='runtime_short_validation' else None)
                    latest['raw']=action.detach().clone()
                    obs300,reward,done,extras=env.step(action)
                    capture=base.r1_terminal_capture.latest
                    if capture is None: raise EvidenceError('Native transition lacks termination capture bookkeeping')
                    terminal=capture.for_timeout_bootstrap()
                    adapter.record_step(reward,capture.terminated,capture.truncated,terminal_critic_observations=terminal)
                    observation=environment.observe(base,obs300);critic=extras['observations']['critic']
                    terms=base.reward_manager._step_reward
                    if not torch.isfinite(terms).all(): raise EvidenceError('Nonfinite actual reward term')
                    accumulate_rollout_diagnostics(diagnostics,_array(conditions),_array(reward),_array(terms),
                        _array(capture.terminated),_array(capture.truncated),_array(clocks))
                    if prepared['kind']=='runtime_short_validation':
                        frames.append(dict(saved,**deepcopy(mapping),reward=_array(reward),reward_terms=_array(terms),
                            terminated=_array(capture.terminated),truncated=_array(capture.truncated),
                            terminal_critic=_array(capture.post_command_critic),next_critic=_array(critic),
                            previewed=_array(capture.previewed)))
                controls+=1
            adapter.compute_returns(critic)
            result=adapter.update();proposals_executed+=1
            result['rollout_diagnostics']=diagnostics
            result['reward_term_names']=list(base.reward_manager.active_terms)
            result['phase_counts_are_behavior_success']=False
            last_update=result
            if prepared['kind']=='runtime_short_validation': updates.append(result)
            with (out/'updates.jsonl').open('a') as stream: stream.write(json.dumps(result,allow_nan=False)+'\n')
            print(f"PPO proposal {proposal+1}/{limit}: accepted={result['accepted']}; total accepted={adapter.counters['accepted_updates']}; rejected={adapter.counters['rejected_updates']}",flush=True)
            verify_sources(prepared['sources'])
            if prepared['kind']=='runtime_short_validation' and result['accepted'] and not restored:
                path=out/'restore_boundary.pt';_checkpoint(adapter,path)
                snapshot=adapter.checkpoint()
                with torch.no_grad():
                    before_mean,before_std=adapter.policy.distribution_parameters(observation)
                    before_value=adapter.policy.evaluate(critic)
                fresh=_new_adapter(prepared);fresh.load(path)
                if not _same(snapshot,fresh.checkpoint()): raise EvidenceError('New-instance model/Adam/RNG restore is not exact')
                with torch.no_grad():
                    after_mean,after_std=fresh.policy.distribution_parameters(observation)
                    after_value=fresh.policy.evaluate(critic)
                if not all(torch.equal(x,y) for x,y in ((before_mean,after_mean),(before_std,after_std),(before_value,after_value))):
                    raise EvidenceError('New-instance restore changed actual-state Gaussian or Critic')
                write_json(out/'new_instance_restore.json',dict(new_instance=True,all_model_optimizer_rng_exact=True,
                    actual_state_gaussian_and_critic_exact=True,checkpoint_sha256=sha256(path),
                    environment_restored=False,environment_continues_without_reset=True))
                adapter=fresh;restored=True
            if (proposal+1)%prepared['save_interval']==0:
                _checkpoint(adapter,out/f'proposal_{adapter.counters["update_attempts"]:06d}.pt')
            if prepared['kind']=='runtime_short_validation' and adapter.counters['accepted_updates']>=SHORT_ACCEPTED_UPDATES:
                break
        verify_sources(prepared['sources'],out/'source_snapshot');adapter.policy.assert_invariants(adapter.optimizer)
        final=_checkpoint(adapter,out/'final.pt')
        trace=None
        if frames:
            path=out/'rollout_trace.npz';np.savez_compressed(path,**{name:np.stack([row[name] for row in frames]) for name in frames[0]})
            trace=dict(file=path.name,sha256=sha256(path))
        ready=prepared['kind']=='runtime_short_validation' and restored and adapter.counters['accepted_updates']==2
        delta=counter_delta(initial_counts,adapter.counters)
        if delta['update_attempts']!=proposals_executed: raise EvidenceError('Executed proposal count differs from numerical state')
        review=dict(version=VERSION,kind=prepared['kind'],training_runtime_ready=ready,
            final_behavior_pass=False,inherited_training_runtime_ready=prepared['validated_short'] is not None,
            teacher_calls=0,executed_control_steps=controls,
            executed_update_attempts=proposals_executed,accepted_updates=delta['accepted_updates'],rejected_updates=delta['rejected_updates'],additional_counters=delta,
            new_instance_restore_verified=restored,updates=updates,last_update=last_update,counters=deepcopy(adapter.counters),
            final_checkpoint=final,trace=trace,environment_state_saved=False,
            actual_first_batch_verified=first['runtime_pre_action_verified'],
            actual_reward_terms=len(base.reward_manager.active_terms),sources_unchanged=True,
            formal_requested_proposals_completed=prepared['kind']=='formal_training' and proposals_executed==prepared['iterations'])
        write_json(out/'runtime_review.json',review)
        if prepared['kind']=='runtime_short_validation' and not ready:
            raise EvidenceError('Bounded short run did not accept two updates within four proposals')
        return review
    except Exception as error:
        write_json(out/'runtime_error.json',dict(version=VERSION,error=str(error),traceback=traceback.format_exc(),
            training_runtime_ready=False,final_behavior_pass=False,counters=None if adapter is None else adapter.counters,
            completed_update_reports=updates,executed_update_attempts=proposals_executed,teacher_calls=0))
        raise
    finally:
        if env is not None: env.close()
        if app is not None: app.close()
