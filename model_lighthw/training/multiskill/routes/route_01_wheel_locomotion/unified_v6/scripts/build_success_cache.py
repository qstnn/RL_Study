"""按完整闭环验收筛选参考缓存，失败试验保留在清单中，不伪造重拖标签。

同一工况的全部已提供试验必须通过，并至少有指定数量的带观测噪声 seed。
教师关节控制只能显式输出 teacher 缓存，不能冒充部署 Actor 成功证据。
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from launch import PROTOCOL, now, sha256, write_json
from quality import case_key


def build(directories, output, *, min_noise_seeds=3, allow_teacher=False, variant='baseline'):
    if min_noise_seeds < 1:
        raise ValueError('At least one noisy seed is required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    grouped = defaultdict(list)
    source_hashes, contracts, physics = set(), set(), set()
    sources = []
    for directory in directories:
        directory = Path(directory).resolve(strict=True)
        request = json.loads((directory / 'request.json').read_text())
        result = json.loads((directory / 'evaluation.json').read_text())
        runtime=json.loads((directory/'runtime_check.json').read_text())
        if bool(runtime['policy_enable_corruption'])!=bool(request['noise']):
            raise ValueError('Requested noise differs from actual observation manager')
        if request.get('parallel_noise_seeds'):
            noise=json.loads((directory/'noise_streams.json').read_text())
            if (noise['seeds']!=request['parallel_noise_seeds'] or not noise['calls']
                    or not noise['native_additive_bounds'] or not noise['applied_before_history']):
                raise ValueError('Parallel noise streams were not verified')
        checkpoint = Path(request['checkpoint'])
        if sha256(checkpoint) != request['checkpoint_sha256'] or result['sha256'] != request['checkpoint_sha256']:
            raise ValueError('Evaluation source checkpoint SHA mismatch')
        source_hashes.add(request['checkpoint_sha256'])
        contracts.add(json.dumps(request['execution_contract'], sort_keys=True))
        physics.add(json.dumps(request['physics_override'], sort_keys=True))
        source_id = len(sources)
        sources.append(dict(directory=str(directory), request_sha256=sha256(directory/'request.json'),
                            evaluation_sha256=sha256(directory/'evaluation.json'),runtime_sha256=sha256(directory/'runtime_check.json')))
        indices = defaultdict(int)
        for result_row in result['cases']:
            seed = result_row['seed']; index = indices[seed]; indices[seed] += 1
            case = result_row['case']
            if case.get('variant', 'baseline') != variant:
                continue
            teacher = case.get('variant', 'baseline') != 'baseline' and case['slope_deg'] == 25. and case['towed_kg'] == 40.
            if teacher and not allow_teacher:
                raise ValueError('Teacher-controlled trajectories require --allow-teacher and cannot certify the deployed Actor')
            prefix = directory / f'seed{seed}_case{index:02d}'
            csv_path = prefix.with_suffix('.csv')
            with csv_path.open(newline='') as stream:
                rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
            checked = PROTOCOL.summarize(rows, case, result_row['failure_reason'])
            if checked['success'] != result_row['success'] or checked['checks'] != result_row['checks']:
                raise ValueError('Saved acceptance does not reproduce from measurements: '+str(prefix))
            trace_path = Path(str(prefix)+'_policy_trace.npz')
            with np.load(trace_path, allow_pickle=False) as trace:
                x, y, phase = trace['observation'], trace['action'], trace['phase']
                if x.shape != (len(rows),300) or y.shape != (len(rows),16) or phase.shape != (len(rows),):
                    raise ValueError('Observation/action/measurement length or dimensions differ')
                if request.get('action_noise') and 'executed_action' not in trace.files:
                    raise ValueError('Action-noise evaluation requires both mean and executed actions')
                executed=trace['executed_action'] if 'executed_action' in trace.files else y
                if executed.shape!=y.shape or not np.isfinite(executed).all():
                    raise ValueError('Invalid executed action trace')
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError('Non-finite trace')
                if not teacher and not np.array_equal(y, trace['actor_action']):
                    raise ValueError('Actor-only trace contains modified actions')
                if not np.isin(phase, [0,1,2,3]).all():
                    raise ValueError('Unknown task phase')
                # 静止工况的原生phase一直为0；必须另外保护验收末5秒，不能被早期帧平均掉。
                if case['speed']==0.:
                    stages=np.asarray(['initial_settle' if r['t']<3. else 'static_mid' if r['t']<15. else 'static_hold' for r in rows])
                else:
                    stages=np.asarray([('wait','move','brake')[int(ph)] if ph<3 else
                        'stop_settle' if r['t']-PROTOCOL.DT < r['eval_stop_time_s']+2. else 'stop_hold'
                        for ph,r in zip(phase,rows)])
                row = dict(source_id=source_id, seed=seed, noise_seed=case.get('noise_seed',seed), index=index, success=bool(checked['success']),
                           failed_checks=checked['failed_checks'], noisy=bool(request['noise']), action_noise=bool(request.get('action_noise',False)), teacher=teacher,
                           csv_sha256=sha256(csv_path), trace_sha256=sha256(trace_path),
                           observation=x.copy(), action=y.copy(), executed_action=executed.copy(), phase=phase.astype(np.int64), stage=stages)
            grouped[case_key(case)].append(row)
    if len(source_hashes) != 1 or len(contracts) != 1 or len(physics) != 1:
        raise ValueError('Do not merge checkpoints, execution contracts or physics configurations')
    arrays = defaultdict(list); decisions = []; groups = []
    for key, trials in sorted(grouped.items()):
        noisy_seeds = sorted({r['noise_seed'] for r in trials if r['noisy']})
        accepted = all(r['success'] for r in trials) and len(noisy_seeds) >= min_noise_seeds
        details = [{k:v for k,v in r.items() if k not in ('observation','action','executed_action','phase','stage')} for r in trials]
        decisions.append(dict(case_key=key, accepted=accepted, noise_seeds=noisy_seeds,
                              passed=sum(r['success'] for r in trials), total=len(trials), trials=details))
        if not accepted:
            continue
        stages = sorted(set(np.concatenate([r['stage'] for r in trials]).tolist()))
        for stage in stages:
            group_id = len(groups); count = 0
            for row in trials:
                mask = row['stage'] == stage; n = int(mask.sum())
                if not n: continue
                arrays['observation'].append(row['observation'][mask]); arrays['action'].append(row['action'][mask]);arrays['executed_action'].append(row['executed_action'][mask])
                arrays['group_id'].append(np.full(n,group_id,dtype=np.int64))
                arrays['case_key'].append(np.full(n,key))
                arrays['phase'].append(row['phase'][mask]);arrays['stage'].append(np.full(n,stage))
                arrays['source_episode'].append(np.full(n,f"{row['source_id']}:{row['seed']}:{row['index']}"))
                count += n
            groups.append(dict(group_id=group_id,case_key=key,stage=stage,samples=count))
    manifest = dict(created_at=now(),min_noise_seeds=min_noise_seeds,variant=variant,
                    cache_kind='teacher_labels_for_offline_fit_only' if any(r['teacher'] for rr in grouped.values() for r in rr) else 'deployed_actor_success_reference',
                    source_checkpoint_sha256=next(iter(source_hashes)),execution_contract=json.loads(next(iter(contracts))),
                    physics_override=json.loads(next(iter(physics))),sources=sources,cases=decisions,groups=groups,
                    note='All supplied trials per condition must pass; observation-noise seeds are explicit, action-noise flags are recorded per trial; world terrain/mass/friction are fixed. Cached action is the mean target, executed_action preserves disturbance.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if not arrays:
        write_json(output.with_suffix('.json'),dict(manifest,status='no_accepted_cases',samples=0))
        raise ValueError('No condition met the reliability filter; no cache written')
    packed = {k:np.concatenate(v) for k,v in arrays.items()}
    packed['successful'] = np.ones(len(packed['observation']),dtype=np.bool_)
    packed['cache_kind'] = np.asarray(manifest['cache_kind'])
    np.savez_compressed(output,**packed)
    manifest.update(status='filtered',samples=len(packed['observation']),case_count=sum(r['accepted'] for r in decisions),
                    cache_sha256=sha256(output),cache_path=str(output))
    write_json(output.with_suffix('.json'),manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-noise-seeds',type=int,default=3)
    parser.add_argument('--variant',default='baseline')
    parser.add_argument('--allow-teacher',action='store_true')
    args=parser.parse_args()
    result=build(args.evaluations,args.output,min_noise_seeds=args.min_noise_seeds,
                 allow_teacher=args.allow_teacher,variant=args.variant)
    print(json.dumps({k:result[k] for k in ('cache_kind','samples','case_count','cache_sha256')},ensure_ascii=False))


if __name__=='__main__':main()
