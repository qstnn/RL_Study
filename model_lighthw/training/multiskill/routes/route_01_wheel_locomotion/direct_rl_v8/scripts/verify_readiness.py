"""复算V8训练入口准入；不以教师成功或短验任务成功作为PPO必要条件。"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from launch import ROOT,REPO,SOURCE,SPEC,sha256,write_json,now
from quality import case_key
from source_guard import CATALOG_RELATIVE,catalog_r1_sha256


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return True


def main():
    base=ROOT/'validation';checks={};details={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:raise AssertionError(name)
    initialization=ROOT/'initialization/model_init.pt';saved=SPEC.checkpoint(initialization)
    original=SPEC.initialization_checkpoint(ROOT.parent/'unified_v5/initialization/selected/model_init.pt')
    check('initial_actor_exact',all(torch.equal(v,saved['model_state_dict'][k]) for k,v in original['model_state_dict'].items() if k.startswith('actor.')))
    baseline=json.loads((base/'final_baseline/evaluation.json').read_text())
    required=set(saved['infos']['initialization']['required_initialization_cases'])
    passing={case_key(r['case']) for r in baseline['cases'] if r['success']}
    check('final_baseline_checkpoint_sha',baseline['sha256']==sha256(initialization))
    check('all15_original_protected_cases_retained',len(required)==15 and required.issubset(passing) and len(baseline['cases'])==18)
    reference=saved['infos']['initialization']['reference_cache']
    check('reference_sha',sha256(reference['path'])==reference['sha256'])
    with np.load(reference['path']) as z:
        check('reference_all_finite_and_successful',bool(z['successful'].all()) and np.isfinite(z['observation']).all() and np.isfinite(z['action']).all())
        check('reference_case_stage_balanced_groups',len(np.unique(z['group_id']))==59 and set(z['case_key'].tolist())==required)
    totals=[];metrics=[];phase_rows=[]
    runtime_names=[]
    for label,updates in (('ppo_fixed',12),('ppo_resume',4)):
        directory=base/label
        run=json.loads((directory/'training_result.json').read_text());runtime=json.loads((directory/'runtime_check.json').read_text())
        check(label+'_completed',run['status']=='completed' and run['updates_this_run']==updates)
        check(label+'_all64_and_no_observation_noise',runtime['unique_condition_count']==64 and not runtime['policy_enable_corruption'] and runtime['num_envs']==512)
        check(label+'_physical_interface',runtime['dimensions']==[300,268,16] and runtime['ground_friction']==3. and runtime['software_effort_limits_Nm']==[96.]*12+[48.]*4)
        data=SPEC.checkpoint(run['checkpoint']);check(label+'_finite_model_adam',finite_tree(data['model_state_dict']) and finite_tree(data['optimizer_state_dict']))
        check(label+'_mask',not torch.count_nonzero(data['model_state_dict']['actor.0.weight'][:,205:285]))
        rows=list(csv.DictReader((directory/'training/optimizer_metrics.csv').open()))
        check(label+'_updates_logged',len(rows)==updates)
        check(label+'_finite_losses',all(np.isfinite(float(v)) for r in rows for v in r.values()))
        check(label+'_full_rollout_KL',all(float(r['kl_rollout'])<=.01+1.e-8 for r in rows))
        check(label+'_actor_updates',sum(float(r['actor_optimizer_steps']) for r in rows)>0)
        groups=list(csv.DictReader((directory/'training/condition_optimizer_metrics.csv').open()))
        check(label+'_actual_rollout_accounting',sum(int(r['samples']) for r in groups)==updates*512*128)
        totals.append(run);metrics+=rows
        if label=='ppo_fixed':phase_rows=groups
        manifest=json.loads((directory/'source_manifest.json').read_text())
        for relative,digest in manifest.items():
            if '/wheel_direct_v8/' in relative or ('/direct_rl_v8/scripts/' in relative and not relative.endswith('/verify_readiness.py')):
                path=REPO/relative
                if path.name=='launch.py':
                    current=path.read_text().replace('from source_guard import validate_sources\n','')
                    updated='                try:\n                    validate_sources(REPO,record,sha256)\n                except ValueError as error:\n                    parser.error(str(error))'
                    original='                for relative,digest in record.get("validated_source_sha256",{}).items():\n                    path=REPO/relative\n                    if not path.is_file() or sha256(path)!=digest:\n                        parser.error("Validated V8 training source changed: "+relative)'
                    check(label+'_startup_guard_only_patch',current.count(updated)==1 and current.replace(updated,original)==(directory/'source_snapshot'/relative).read_text())
                else:
                    check(label+'_source_'+path.name,sha256(path)==digest)
                runtime_names.append(relative)
    resumed=json.loads((base/'ppo_resume/initialization.json').read_text())
    check('resume_exact_model_adam_algorithm_rng',all(resumed.get(k) for k in ('model_restored_exact','optimizer_restored_exact','algorithm_restored_exact','rng_restored')) and resumed['start_update']==12)
    check('resume_new_scene_explicit',resumed['simulation_state_restored'] is False)
    total=sum(int(r['samples']) for r in phase_rows)
    fractions={str(slope):sum(int(r['samples']) for r in phase_rows if int(r['slope_deg'])==slope and int(r['towed_kg'])==40 and int(r['phase'])==1)/total for slope in (25,-20)}
    check('hard_moving_samples_above_2percent_each',all(v>.02 for v in fractions.values()))
    phase_counts={str(phase):sum(int(r['samples']) for r in phase_rows if int(r['phase'])==phase) for phase in range(4)}
    check('all_four_task_phases_sampled',all(phase_counts.values()))
    first=[r for r in phase_rows if int(r['iteration'])==0]
    check('first_rollout_all64',len({(r['slope_deg'],r['payload_kg'],r['towed_kg']) for r in first})==64)
    cpu=(base/'cpu_tests.txt').read_text();check('cpu_tests_passed','25 passed' in cpu and 'FAILED' not in cpu)
    original_sources=json.loads((base/'original_source_sha256.json').read_text())
    check('old_v6_v7_runtime_and_urdf_unchanged',all(sha256(REPO/p)==digest for p,digest in original_sources.items()))
    shared=[SOURCE/'wheel_env_cfg.py',SOURCE/'ppo_cfg.py',SOURCE/'catalog.py',SOURCE/'rewards.py',ROOT.parents[2]/'common.py']
    # 使用最终GPU快照核对公共依赖；不把并发R4代码变更混入R1准入。
    snapshot=json.loads((base/'final_baseline/source_manifest.json').read_text())
    for path in shared:
        relative=str(path.relative_to(REPO))
        if relative==CATALOG_RELATIVE:
            before=(base/'final_baseline/source_snapshot'/relative).read_text()
            catalog_scope=catalog_r1_sha256(before)
            check('shared_catalog_R1_and_all_executable_code_unchanged',catalog_r1_sha256(path.read_text())==catalog_scope)
            details['unrelated_catalog_change']='Other routes literal directory metadata excluded; R1 entry and all remaining AST match final GPU snapshot'
        else:
            check('shared_'+path.name,sha256(path)==snapshot[relative])
        runtime_names.append(relative)
    # 环境的实际父配置与动作/观测约定。
    velocity=REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_lighthw'
    for name in ('rough_env_cfg.py','linear_leg_spec.py','reference_observations.py'):
        path=velocity/name;relative=str(path.relative_to(REPO));check('shared_'+name,sha256(path)==snapshot[relative]);runtime_names.append(relative)
    runtime_names.append(str((ROOT/'scripts/source_guard.py').relative_to(REPO)))
    details['startup_guard_validation']='CPU-only patch; original training/environment/policy sources still match GPU snapshots'
    validated={p:sha256(REPO/p) for p in sorted(set(runtime_names))}
    actor_metrics=[r for r in metrics if not float(r['critic_only'])]
    details.update(checked_at=now(),checks=checks,baseline_pass_count=baseline['pass_count'],baseline_case_count=18,
        reference_groups=59,reference_samples=reference['samples'],observation_noise=False,
        physical_and_initial_randomization=False,command_sequence='fixed',ppo_action_exploration=True,
        gpu_short_updates=sum(r['updates_this_run'] for r in totals),gpu_short_actor_rounds=sum(r['actor_update_rounds_this_run'] for r in totals),
        actor_optimizer_steps=sum(float(r['actor_optimizer_steps']) for r in metrics),maximum_rollout_kl=max(float(r['kl_rollout']) for r in metrics),
        mean_accepted_parameter_fraction=float(np.mean([float(r['actor_accepted_fraction_mean']) for r in actor_metrics])),
        hard_moving_sample_fraction=fractions,phase_samples=phase_counts,formal_training_updates=0,
        behavior_accepted=False,teacher_required=False,seed43_screen_only=True,noise_robustness_validated=False)
    write_json(base/'verification.json',details)
    write_json(ROOT/'initialization/readiness.json',dict(training_ready=True,formal_training_command_available=True,
        checkpoint_sha256=sha256(initialization),reference_sha256=reference['sha256'],validated_source_sha256=validated,
        catalog_r1_sha256=catalog_scope,
        verification=str(base/'verification.json'),verification_sha256=sha256(base/'verification.json'),
        teacher_required=False,full_R1_behavior_accepted=False,validated_num_envs=512,
        reason='Fixed-condition interface, sampling, finite PPO updates, exact optimizer resume and15 protected baseline cases verified; full heavy towing remains learning objective'))
    print(json.dumps({k:v for k,v in details.items() if k!='checks'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
