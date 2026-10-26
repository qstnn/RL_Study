"""V8可审计初始化：旧Actor零改动迁移，按新物理/协议重新筛选保持缓存。"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from launch import ROOT,REPO,SPEC,signature,sha256,write_json,now,PROTOCOL
from quality import case_key

ORIGINAL=ROOT.parent/'unified_v5/initialization/selected/model_init.pt'

def candidate():
    path=ROOT/'initialization/candidate_fixed.pt'
    if path.exists():raise FileExistsError(path)
    original=SPEC.initialization_checkpoint(ORIGINAL)
    saved=deepcopy(original);weights=saved['model_state_dict']
    previous=weights['critic.0.weight'];extended=previous.new_zeros((512,268))
    extended[:,:previous.shape[1]]=previous;weights['critic.0.weight']=extended
    weights['log_std']=torch.tensor(SPEC.CONTRACT['initial_noise_std_by_action']).log()
    settings=SimpleNamespace(ground_friction=3.,rolling_coefficient=.02,leg_torque_limit=96.,wheel_torque_limit=48.,hitch_height=.07,unloaded_acceleration=1.,loaded_acceleration=.3,critic_warmup_updates=20,eval_distance=1.)
    saved['infos']=dict(r1_unified_contract=SPEC.CONTRACT,environment_signature=signature(settings),
        initialization=dict(source_checkpoint=str(ORIGINAL),source_sha256=sha256(ORIGINAL),actor_weights_unchanged=True,
                            teacher_required=False,preparation_only=True))
    saved['optimizer_state_dict']={};saved['iter']=-1
    torch.save(saved,path)
    SPEC.checkpoint(path)
    for key,value in original['model_state_dict'].items():
        if key.startswith('actor.') and not torch.equal(value,weights[key]):raise AssertionError(key)
    write_json(ROOT/'initialization/candidate_fixed.json',dict(created_at=now(),source=str(ORIGINAL),source_sha256=sha256(ORIGINAL),candidate_sha256=sha256(path),actor_unchanged=True,critic_input=268,new_columns_zero=True,formal_training_updates=0))
    print(path)

def build_cache(directory):
    directory=Path(directory).resolve(strict=True)
    request=json.loads((directory/'request.json').read_text());result=json.loads((directory/'evaluation.json').read_text())
    runtime=json.loads((directory/'runtime_check.json').read_text())
    source=ROOT/'initialization/candidate_fixed.pt'
    executed=Path(result['checkpoint']).resolve(strict=True)
    if result['sha256']!=sha256(executed):raise ValueError('Baseline source SHA mismatch')
    if executed.parent!=source.resolve().parent or not executed.name.startswith('candidate_'):raise ValueError('Unexpected candidate source')
    previous=torch.load(executed,map_location='cpu',weights_only=False)
    current=SPEC.checkpoint(source)
    for key,value in current['model_state_dict'].items():
        if key.startswith('actor.') and not torch.equal(value,previous['model_state_dict'][key]):raise ValueError('Reference Actor changed')
    manifest=json.loads((directory/'source_manifest.json').read_text())
    protocol_relative=str((ROOT/'scripts/protocol.py').relative_to(REPO))
    if manifest[protocol_relative]!=sha256(ROOT/'scripts/protocol.py'):raise ValueError('Reference acceptance protocol changed')
    if request['ground_friction']!=3. or runtime['dimensions']!=[300,268,16] or request['suite']!='boundary':raise ValueError('Invalid baseline physics/interface')
    if runtime['policy_enable_corruption'] or request['eval_seeds']!=[43]:raise ValueError('Cache build expects one explicitly clean frozen screen; no robustness claim')
    expected={case_key(c) for c in PROTOCOL.cases()}
    if len(result['cases'])!=18 or {case_key(r['case']) for r in result['cases']}!=expected:raise ValueError('Incomplete baseline coverage')
    arrays={k:[] for k in ('observation','action','successful','group_id','case_key','stage')};groups=[];decisions=[]
    for index,item in enumerate(result['cases']):
        prefix=directory/f"seed43_case{index:02d}";csv_path=prefix.with_suffix('.csv')
        with csv_path.open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        checked=PROTOCOL.summarize(rows,item['case'],item['failure_reason'])
        if checked['checks']!=item['checks'] or checked['success']!=item['success']:raise ValueError('Acceptance recomputation mismatch')
        decisions.append(dict(case_key=case_key(item['case']),success=checked['success'],failed_checks=checked['failed_checks'],csv_sha256=sha256(csv_path)))
        if not checked['success']:continue
        with np.load(Path(str(prefix)+'_policy_trace.npz')) as f:
            x,y,phase=f['observation'],f['action'],f['phase']
        if x.shape!=(len(rows),300) or y.shape!=(len(rows),16) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid reference trace')
        if item['case']['speed']==0:
            stages=np.array(['initial_settle' if r['t']<3. else 'static_mid' if r['t']<15. else 'static_hold' for r in rows])
        else:
            stages=np.array([('wait','move','brake')[int(ph)] if ph<3 else 'stop_settle' if r['t']-PROTOCOL.DT<r['eval_stop_time_s']+2. else 'stop_hold' for ph,r in zip(phase,rows)])
        for stage in sorted(set(stages)):
            mask=stages==stage;n=int(mask.sum());key=case_key(item['case']);g=len(groups)
            groups.append(dict(group_id=g,case_key=key,stage=stage,samples=n))
            for name,value in dict(observation=x[mask],action=y[mask],successful=np.ones(n,dtype=bool),group_id=np.full(n,g,dtype=np.int64),case_key=np.full(n,key),stage=np.full(n,stage)).items():arrays[name].append(value)
    passing={r['case_key'] for r in decisions if r['success']}
    old=ROOT.parent/'research_v7/results/2026-10-20_servo_support/frozen_boundary_mu3/evaluation.json'
    required={case_key(r['case']) for r in json.loads(old.read_text())['cases'] if r['success']}
    if not required.issubset(passing):raise ValueError('New baseline lost previously validated cases: '+repr(sorted(required-passing)))
    output=ROOT/'initialization/success_reference.npz'
    if output.exists():raise FileExistsError(output)
    packed={k:np.concatenate(v) for k,v in arrays.items()};packed['cache_kind']=np.asarray('deployed_actor_success_reference')
    np.savez_compressed(output,**packed)
    case_stages={key:sorted({r['stage'] for r in groups if r['case_key']==key}) for key in passing}
    reference=dict(path=str(output.resolve()),sha256=sha256(output),case_keys=sorted(passing),case_stages=case_stages,
                   group_count=len(groups),samples=len(packed['action']),provenance='V8 mu3 clean single-seed complete-episode screen; no noisy robustness certification')
    saved=SPEC.checkpoint(source);saved['infos']['initialization'].update(reference_cache=reference,required_initialization_cases=sorted(passing),preparation_only=False)
    final=ROOT/'initialization/model_init.pt'
    if final.exists():raise FileExistsError(final)
    torch.save(saved,final)
    write_json(ROOT/'initialization/reference_manifest.json',dict(created_at=now(),source_evaluation=str(directory),source_evaluation_sha256=sha256(directory/'evaluation.json'),source_candidate_sha256=sha256(executed),fixed_candidate_sha256=sha256(source),final_initialization_sha256=sha256(final),reference=reference,groups=groups,cases=decisions,pass_count=len(passing),robustness_validated=False))
    print(json.dumps(dict(pass_count=len(passing),groups=len(groups),samples=reference['samples'],initialization=str(final))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('candidate','cache'));p.add_argument('--evaluation',type=Path);a=p.parse_args()
    if a.mode=='candidate':candidate()
    else:build_cache(a.evaluation)
