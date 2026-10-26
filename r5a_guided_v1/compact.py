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
 """固定白名单复制，单模型/单套源码/选中回放；不复制USD缓存、其他模型或全量回合日志。"""
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
  shutil.copytree(work/'preparation',review/'preparation')
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
 if selection:
  copy(Path(selection['checkpoint']),'selected.pt')
  for case in selection['cases']:
   prefix=case['artifact_prefix'];run=Path(selection['run'])
   for suffix in ('.json','.png','.mp4','_trace.npz','_trace.json','_policy.npz'):copy(run/(prefix+suffix),'selected/'+prefix+suffix)
  write(review/'selection.json',selection)
 manifest=dict(export_kind='compact_guided_v1_review',status=state['status'],full_work_folder_on_training_machine=str(work),
  selection_scope='best of last up to 3 saved candidates, evaluated once after continuous training' if state.get('mode')=='train' else 'single explicitly evaluated checkpoint; no candidate comparison',
  omitted=['other checkpoints','full episode JSONL','full console logs','repeated source snapshots','machine USD caches'],
  behavior_status=selection['behavior_status'] if selection else 'no_evaluated_model',
  selected_checkpoint_sha256=sha(review/'selected.pt') if (review/'selected.pt').exists() else None)
 write(review/'review_manifest.json',manifest)
 (review/'README.md').write_text('# R5A Guided v1 精简USB审查包\n\n'+
  ('训练期间没有分段评估；结束后统一比较少量候选。' if state.get('mode')=='train' else '本次只评估指定模型，没有比较多个候选。')+
  ('本包含选中模型及其回放。' if selection else '本包尚无评估后的选中模型或回放。')+'\n\n'+
  '行为状态：`'+manifest['behavior_status']+'`。best_available_not_passed表示尚未完成任务，不可将选中理解为成功。\n\n'+
  '全部训练回合已作分窗口统计，原始JSONL/其余模型留在训练电脑：`'+str(work)+'`。USB保留全程更新日志、曲线、最多20回合样例、单套源码/依赖、运行读回、选中回放的1kHz原始物理数据/50Hz策略数据/录像。\n')
 seal(review);return pack_review(review)
