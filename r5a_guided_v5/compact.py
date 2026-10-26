"""USB仅带回选中模型与足够审查的证据；完整训练资料仍留在训练电脑。"""
from pathlib import Path
import collections,math,shutil
from .io import read,write,sha,seal,pack_review

def episode_summary(path):
 """流式汇总全部训练回合；分窗口统计与有界样例避免整份JSONL膨胀。"""
 groups={};examples=[];best=[];tail=collections.deque(maxlen=5)
 if not path.exists():return dict(count=0,windows=[],examples=[])
 def add(stats,k,v):
  if not isinstance(v,(int,float)) or not math.isfinite(v):return
  s=stats.setdefault(k,dict(n=0,sum=0.,sum_sq=0.,min=float(v),max=float(v)))
  s['n']+=1;s['sum']+=v;s['sum_sq']+=v*v;s['min']=min(s['min'],v);s['max']=max(s['max'],v)
 import json
 count=0;incomplete_final_record=False
 with path.open() as f:
  for line in f:
   if not line.strip():continue
   try:row=json.loads(line)
   except json.JSONDecodeError:
    # 中断可能截断最后一次写入；保留完整回合统计，不能因此丢掉整个诊断ZIP。
    # 中间损坏仍报错，不把内部缺失记录当作正常完整数据。
    if f.read().strip():raise
    incomplete_final_record=True;break
   count+=1;window=(max(0,row.get('completed_updates',0))//200)*200
   g=groups.setdefault(window,dict(count=0,metrics={},failure_codes={}))
   g['count']+=1;code=str(row.get('failure_code'));g['failure_codes'][code]=g['failure_codes'].get(code,0)+1
   for k,v in row.items():
    if k not in ('env_id','episode','completed_updates','reward_components'):add(g['metrics'],k,v)
   for k,v in row.get('reward_components',{}).items():add(g['metrics'],'reward/'+k,v)
   if count<=5:examples.append(row)
   tail.append(row);best.append(row);best.sort(key=lambda r:(r.get('success',False),r.get('qualified_landing',False),not r.get('failed',False),r.get('angle',0)),reverse=True);best=best[:10]
 for g in groups.values():
  for s in g['metrics'].values():
   s['mean']=s['sum']/s['n'];s['std']=math.sqrt(max(0.,s['sum_sq']/s['n']-s['mean']**2));del s['sum'];del s['sum_sq']
 return dict(count=count,windows=[dict(update_window_start=k,**v) for k,v in sorted(groups.items())],examples=examples+best+list(tail),example_selection='first 5 + best 10 + last 5; not a random sample',raw_episodes_retained_on_training_machine=True,incomplete_final_record_ignored=incomplete_final_record)

def export_review(work,review,state,selection=None):
 """固定白名单复制，最多两模型/单套源码/选中与对照回放；不复制USD缓存、其他模型或全量回合日志。"""
 work=Path(work);review=Path(review);review.mkdir(parents=True,exist_ok=False)
 def copy(source,relative):
  source=Path(source)
  if not source.is_file():return
  out=review/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out)
 def tail(source,relative,lines=200):
  if source.is_file():
   with source.open(errors='replace') as f:text=''.join(collections.deque(f,maxlen=lines))
   out=review/relative;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text)
 for name in ('session.json','launch_arguments.json','candidate_comparison.json','final_summary.json'):
  copy(work/name,name)
 if (work/'preparation').is_dir():
  shutil.copytree(work/'preparation',review/'preparation',ignore=shutil.ignore_patterns('candidate_audit'))
 audit_summaries=next((p for p in (work/'candidate_audit',work/'preparation/candidate_audit') if p.is_dir()),None)
 if audit_summaries:shutil.copytree(audit_summaries,review/'candidate_audit')
 runs=[p for p in (work/'training',work/'final_evaluation',work/'replay') if p.is_dir()]
 source_run=next((p for p in runs if (p/'source_snapshot').is_dir()),None)
 if source_run:
  for directory in ('source_snapshot','dependency_source'):
   if (source_run/directory).is_dir():shutil.copytree(source_run/directory,review/directory)
  copy(source_run/'source_manifest.json','source_manifest.json')
 for run in runs:
  for name in ('request.json','run.json','result.json','worker_status.json','worker_error.json','runtime_check.json','joint_bounds.json','environment.json','contract.json','assets.json','model.json','source_check.json','bootstrap.json','resume.json','evaluation.json'):
   copy(run/name,run.name+'/'+name)
  if (run/'params').is_dir():shutil.copytree(run/'params',review/run.name/'params')
  copy(run/'input/robot.urdf',run.name+'/input/robot.urdf');tail(run/'console.log',run.name+'/console_tail.log')
 train=work/'training'
 for name in ('updates.jsonl','training_curve.jsonl'):copy(train/name,'training/'+name)
 copy(train/'input/demonstrations.npz','training/input/demonstrations.npz')
 write(review/'training/episodes_summary.json',episode_summary(train/'episodes.jsonl'))
 copy(work/'training_progress.png','training_progress.png')
 recovered=None;recovery_errors=[]
 if not selection:
  # 中断包也带一个最近已保存、完整可读的PPO模型；明确没有做行为评估。
  from .io import checkpoint
  paths=list((train/'training').glob('model_*.pt'))+[p for p in (train/'final.pt',train/'initial.pt') if p.is_file()]
  for p in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True):
   try:info=checkpoint(p)['infos']
   except Exception as error:recovery_errors.append(dict(file=p.name,error=str(error)));continue
   recovered=dict(source=str(p),completed_updates=info['completed_updates'],sha256=sha(p),evaluated=False)
   copy(p,'latest_saved.pt');write(review/'latest_saved.json',recovered);break
 retained_roles=[];missing_evidence=[];secondary=None
 def evidence(row,folder,checkpoint_name,role,video=False):
  model=Path(row['checkpoint'])
  copy(model,checkpoint_name)
  if not model.is_file():missing_evidence.append(role+'/model.pt')
  cases=row.get('cases',[])[:1]
  for case in cases:
   prefix=case['artifact_prefix'];run=Path(row['run'])
   if Path(prefix).name!=prefix:raise ValueError('Unsafe artifact prefix')
   suffixes=('.json','_trace.npz','_trace.json','_policy.npz')
   for suffix in suffixes+('.png',)+ (('.mp4',) if video else ()):
    source=run/(prefix+suffix)
    copy(source,folder+'/'+prefix+suffix)
    if suffix in suffixes and not source.is_file():missing_evidence.append(role+'/'+prefix+suffix)
  if not cases:missing_evidence.append(role+'/cases')
  retained_roles.append(dict(role=role,label=row['label'],checkpoint=checkpoint_name,
   checkpoint_sha256=sha(model) if model.is_file() else None,raw_case_count=len(cases),total_evaluated_cases=len(row.get('cases',[])),video_included=video))
 if selection:
  evidence(selection,'selected','selected.pt','selected',video=True)
  write(review/'selection.json',selection)
  comparison_path=work/'candidate_comparison.json'
  if state.get('mode')=='train' and comparison_path.is_file():
   comparison=read(comparison_path)
   if selection['label']=='session_start_baseline':
    from .cli import select_candidate
    pool=[r for r in comparison if r['label']!='session_start_baseline']
    if pool:
     secondary=dict(select_candidate(pool),run=selection['run'],evidence_role='best_new_candidate',
      selection_scope='best available trained candidate retained for audit; not the selected deployable model')
   else:
    baseline=next((r for r in comparison if r['label']=='session_start_baseline'),None)
    if baseline:secondary=dict(baseline,run=selection['run'],evidence_role='session_start_baseline',selection_scope='same-session baseline retained for contact comparison')
   if secondary:
    evidence(secondary,'secondary','secondary.pt',secondary['evidence_role'])
    write(review/'secondary_selection.json',secondary)
   else:missing_evidence.append('secondary_candidate')
 manifest=dict(export_kind='compact_guided_v5_review',status=state['status'],full_work_folder_on_training_machine=str(work),
  selection_scope='session-start baseline plus last up to 3 saved candidates, evaluated once after continuous training' if state.get('mode')=='train' else 'single explicitly evaluated checkpoint; no candidate comparison',
  candidate_audit_summary='candidate_audit/provenance.json' if audit_summaries else None,
  retained_evidence_roles=retained_roles,missing_raw_evidence=missing_evidence,raw_case_limit_per_model=1,video_limit_models=1,
  omitted=['other checkpoints','full episode JSONL','full console logs','repeated source snapshots','machine USD caches'],
  behavior_status=selection['behavior_status'] if selection else ('unevaluated_saved_checkpoint' if recovered else 'no_evaluated_model'),
  selected_checkpoint_sha256=sha(review/'selected.pt') if (review/'selected.pt').exists() else None,
  recovered_checkpoint=recovered,checkpoint_recovery_errors=recovery_errors)
 write(review/'review_manifest.json',manifest)
 (review/'README.md').write_text('# R5A Guided v5 精简USB审查包\n\n'+
  ('训练期间没有分段评估；结束后统一比较少量候选。' if state.get('mode')=='train' else '本次只评估指定模型，没有比较多个候选。')+
  ('本包含选中模型及其回放，以及训练对照模型的原始接触轨迹（若存在）；对照模型不是最终优选。' if selection else ('本包包含latest_saved.pt，尚未评估，不代表已成功。' if recovered else '尚无保存模型或回放。'))+'\n\n'+
  '行为状态：`'+manifest['behavior_status']+'`。best_available_not_passed表示尚未完成任务，不可将选中理解为成功。\n\n'+
  '全部训练回合已作分窗口统计，原始JSONL/其余模型留在训练电脑：`'+str(work)+'`。USB保留全程更新日志、曲线、最多20回合样例、单套源码/依赖、运行读回，最多两模型各一条1kHz物理/50Hz策略轨迹；仅选中模型保留录像。各模型角色见review_manifest.json和secondary_selection.json，缺失证据明确列出。\n')
 seal(review);return pack_review(review)
