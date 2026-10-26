"""自主空翻离线单命令：训练→固定种子评估→录像→单文件夹回收。"""
import argparse,fcntl,json,math,os,platform,shutil,signal,subprocess,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_guided_v4.io import ROOT,read,write,stamp,now,sha,checkpoint,prepare_run,seal,pack_review,asset_manifest
from r5a_guided_v4.spec import CONTRACT as C

def child(session,mode,name,a,model=None,seeds=None,episodes=1,updates=0,video=False,candidates=None):
 run=session/name
 request=dict(mode=mode,seed=a.seed,device=a.device,num_envs=a.num_envs if mode=='train' else 1,
  headless=mode!='play' or a.headless,video=video,checkpoint=str(model) if model else None,
  eval_seeds=seeds or [43],eval_episodes=episodes,updates=updates,save_every=a.save_every,candidates=candidates)
 if mode=='train':
  request.update(initial_actor=str(a.initial_actor) if a.initial_actor else None,
   initial_actor_sha256=sha(a.initial_actor) if a.initial_actor else None,
   demonstrations=str(a.demonstrations),demonstrations_sha256=sha(a.demonstrations),preparation=str(a.preparation) if a.preparation else None)
 prepare_run(run,request)
 command=[sys.executable,'-u',str(ROOT/'r5a_guided_v4/worker.py'),'--run',str(run)]
 write(run/'run.json',dict(status='running',command=command,started_at=now(),request=request))
 proc=None;status='failed';code=1
 try:
  print('[AUTOFLIP RUN]',run,flush=True)
  with (run/'console.log').open('w') as stream:
   proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,start_new_session=True)
   for line in proc.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
   code=proc.wait()
  if code or (run/'worker_error.json').exists():raise RuntimeError(f'子进程失败 {code}；查看 {run}/console.log 和 worker_error.json')
  for filename in ('runtime_check.json','result.json'):
   if not (run/filename).is_file():raise RuntimeError(f'缺少 {filename}；先查看 {run}/console.log')
  result=read(run/'result.json');status='completed';return run,result
 except KeyboardInterrupt:status='interrupted';code=130;raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  write(run/'run.json',dict(status=status,returncode=code,finished_at=now(),command=command,request=request))

def summary(result):
 cases=result['cases'];n=len(cases)
 return dict(success_rate=sum(x['success'] for x in cases)/n,qualified_rate=sum(x['qualified_landing'] for x in cases)/n,
  valid_flight_rate=sum(x['flight_valid'] for x in cases)/n,launch_rate=sum(x['launched'] for x in cases)/n,
  mean_valid_angle_deg=sum(max(0,min(360,x['airborne_angle_deg'])) if x['flight_valid'] else 0 for x in cases)/n,
  physics_failure_rate=sum(x.get('physics_consistent') is False for x in cases)/n,
  momentum_unchecked_rate=sum(x.get('physics_consistent') is None for x in cases)/n,
  mean_joint_progress=sum(x['joint_progress'] if x['flight_valid'] and not x['failed'] else 0 for x in cases)/n,
  mean_raw_recovery_score=sum(x['recovery_score'] if x['flight_valid'] and not x['failed'] else 0 for x in cases)/n,
  mean_recovery_score=sum(x['recovery_score'] if x['qualified_landing'] and not x['failed'] else 0 for x in cases)/n,
  mean_stable_s=sum(x['stable_s'] for x in cases)/n,
  mean_end_tilt_deg=sum(x['end_tilt_deg'] for x in cases)/n,
  mean_impact_bodyweights=sum(x['peak_support_bodyweights'] for x in cases)/n,
  mean_flight_rise_m=sum(x['flight_rise_m'] for x in cases)/n,failed_rate=sum(x['failed'] for x in cases)/n)
def rank(x):
 # 合格任务优先；部分动作使用近完整圈进度，跳高不再补偿缺角。
 return (-x['physics_failure_rate'],x['success_rate'],x['qualified_rate'],x['mean_stable_s'],x['mean_recovery_score'],
  x['valid_flight_rate'],-x['failed_rate'],x.get('mean_joint_progress',0),x.get('mean_raw_recovery_score',0),-x['mean_impact_bodyweights'])

def landing_quality(x):
 """仅在通过原完整协议的候选内比较；冲击、末姿态与恢复共同排序。"""
 return x['mean_raw_recovery_score']-.2*math.log1p(x['mean_impact_bodyweights']/10.)-.15*(x['mean_end_tilt_deg']/15.)**2

def select_candidate(comparison):
 checked=[x for x in comparison if x['metrics']['physics_failure_rate']==0 and x['metrics'].get('momentum_unchecked_rate',0)==0]
 full=[x for x in checked if x['metrics']['success_rate']==1 and x['metrics']['qualified_rate']==1]
 if full:return max(full,key=lambda x:(landing_quality(x['metrics']),x['metrics']['mean_stable_s']))
 return max(checked or comparison,key=lambda x:rank(x['metrics']))

def positive(x):
 x=int(x)
 if x<1:raise argparse.ArgumentTypeError('Must be positive')
 return x

def behavior_status(metrics):
 """物理未检查和物理失败都只作诊断回收，不能称合格候选。"""
 if metrics['physics_failure_rate']>0:return 'physics_rejected'
 if metrics.get('momentum_unchecked_rate',0)>0:return 'physics_unchecked'
 return 'fixed_condition_passed' if metrics['success_rate']==1 else 'best_available_not_passed'

def saved_candidates(run,limit):
 # 更新数从检查点契约读取，不根据model_文件名推断；最终模型覆盖同计数的重复存档。
 found={}
 for path in sorted((run/'training').glob('model_*.pt'))+[run/'final.pt']:
  if not path.is_file():continue
  data=checkpoint(path);n=data['infos']['completed_updates'];found[n]=path
 return [dict(label=f'update_{n:06d}',checkpoint=str(found[n]),completed_updates=n) for n in sorted(found)[-limit:]]

def main(argv=None):
 p=argparse.ArgumentParser(description='连续训练到结束，仅末尾统一评估；USB导出精简优选证据')
 p.add_argument('mode',choices=['check','train','evaluate','play','verify','collect'])
 p.add_argument('--num-envs',type=positive,default=512);p.add_argument('--updates',type=positive,default=3000)
 p.add_argument('--save-every',type=positive,default=200);p.add_argument('--candidates',type=int,choices=[1,2,3],default=3)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0')
 p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);p.add_argument('--folder',type=Path)
 p.add_argument('--initial-actor',type=Path);p.add_argument('--demonstrations',type=Path);p.add_argument('--preparation',type=Path)
 p.add_argument('--eval-episodes',type=positive,default=1);p.add_argument('--play-episodes',type=positive,default=1)
 p.add_argument('--headless',action='store_true');p.add_argument('--no-video',action='store_true')
 a=p.parse_args(argv)
 from r5a_guided_v4.compact import export_review
 if a.mode in ('collect','verify'):
  if not a.folder:p.error('--folder required')
  folder=a.folder.resolve(strict=True)
  if a.mode=='verify':
   hashes=read(folder/'SHA256.json');bad=[]
   for name,h in hashes.items():
    path=(folder/name).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or sha(path)!=h:bad.append(name)
   if bad:raise ValueError('SHA mismatch: '+str(bad))
   print('[VERIFIED]',len(hashes),'files');return 0
  if (folder/'review_manifest.json').exists():seal(folder);print('[USB REVIEW ZIP]',pack_review(folder));return 0
  state=read(folder/'session.json')
  if state['status']=='running':raise ValueError('Session marked running; do not collect an active run')
  selection=read(folder/'selection.json') if (folder/'selection.json').exists() else None
  dest=ROOT/'usb_review/route_05a'/(folder.name+'_recollect_'+stamp())
  print('[USB REVIEW ZIP]',export_review(folder,dest,state,selection));return 0
 if a.mode=='check':
  from r5a_guided_v4.tests import run_tests
  from r5a_guided_v4.station import verify_baseline_manifest
  verify_baseline_manifest()
  assets=asset_manifest();run_tests()
  print('[AUTOFLIP CHECK] CPU/资产通过；不是GPU或空翻行为验收。',C['version'],'mass=',assets['mass_kg']);return 0
 if a.mode=='train' and a.num_envs<4:p.error('PPO requires at least 4 environments')
 if a.mode=='train':
  if not a.demonstrations:p.error('Use bash r5a_guided_v4.sh train to verify teacher and prepare demonstrations first')
  a.demonstrations=a.demonstrations.resolve(strict=True)
  if not a.checkpoint:
   if not a.initial_actor:p.error('Missing supervised initialization')
   a.initial_actor=a.initial_actor.resolve(strict=True)
   if not a.preparation:p.error('Missing autonomous student verification; use the station train command')
   a.preparation=a.preparation.resolve(strict=True)
 if a.checkpoint:a.checkpoint=a.checkpoint.resolve(strict=True);checkpoint(a.checkpoint)
 elif a.mode in ('evaluate','play'):
  marker=ROOT/'offline_sessions/route_05a/LATEST_GUIDED_V4_CHECKPOINT.txt'
  if not marker.exists():p.error('No Guided v4 model yet: provide a v3 --checkpoint')
  a.checkpoint=(ROOT/marker.read_text().strip()).resolve(strict=True);checkpoint(a.checkpoint)
 folder=(a.output or ROOT/'offline_sessions/route_05a'/('guided_v4_'+stamp())).resolve();folder.mkdir(parents=True,exist_ok=False)
 review=ROOT/'usb_review/route_05a'/folder.name
 if review.exists():review=review.with_name(review.name+'_'+stamp())
 state=dict(status='running',created_at=now(),kind='continuous_teacher_guided_v4',contract=C,requested_updates=a.updates if a.mode=='train' else 0,
  completed_updates=0,mode=a.mode,python=sys.version,platform=platform.platform(),review_folder=str(review),formal_behavior_acceptance='requires_review')
 write(folder/'session.json',state);write(folder/'launch_arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
 if a.preparation:
  shutil.copytree(a.preparation,folder/'preparation',ignore=shutil.ignore_patterns('asset_cache','source_snapshot','dependency_source','__pycache__','bc_initial_actor.pt'))
 lockdir=ROOT/'offline_sessions/route_05a';lockdir.mkdir(parents=True,exist_ok=True);code=0;selection=None
 with (lockdir/'guided_v4.lock').open('a') as lock:
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if a.mode=='train':
    print(f'[GUIDED CONTINUOUS] 连续训练 {a.updates} 次更新尝试；中途仅保存模型，结束后统一比较本轮起点基线和最后最多 {a.candidates} 个候选。',flush=True)
    run,result=child(folder,'train','training',a,a.checkpoint,updates=a.updates)
    state.update(completed_updates=a.updates,total_checkpoint_updates=result['completed_updates']);write(folder/'session.json',state)
    candidates=saved_candidates(run,a.candidates)
    initial=run/'initial.pt';initial_n=checkpoint(initial)['infos']['completed_updates']
    candidates.insert(0,dict(label='session_start_baseline',checkpoint=str(initial),completed_updates=initial_n))
    eval_run,evaluation=child(folder,'evaluate','final_evaluation',a,Path(candidates[-1]['checkpoint']),episodes=a.eval_episodes,video=not a.no_video,candidates=candidates)
    comparison=[]
    for candidate in candidates:
     cases=[x for x in evaluation['cases'] if x['candidate']==candidate['label']]
     metrics=summary(dict(cases=cases));comparison.append(dict(**candidate,metrics=metrics,cases=cases))
    write(folder/'candidate_comparison.json',comparison)
    best=select_candidate(comparison)
    metrics=best['metrics'];status=behavior_status(metrics)
    selection=dict(checkpoint=best['checkpoint'],label=best['label'],completed_updates=best['completed_updates'],metrics=metrics,cases=best['cases'],run=str(eval_run),behavior_status=status,selection_scope='session-start baseline plus last saved candidates; full success before impact/posture quality; not global-best proof',selected_is_baseline=best['label']=='session_start_baseline')
    write(folder/'final_summary.json',dict(behavior_status=status,selected_is_baseline=best['label']=='session_start_baseline',baseline_quality=landing_quality(comparison[0]['metrics']),selected_quality=landing_quality(metrics),**metrics));print('[GUIDED FINAL]',json.dumps(selection),flush=True)
    if status in ('physics_rejected','physics_unchecked'):code=1;state['error']='No selected candidate passed momentum screening; diagnostic export only'
   else:
    run,evaluation=child(folder,a.mode,'replay',a,a.checkpoint,seeds=[a.seed] if a.mode=='play' else [43],episodes=a.play_episodes if a.mode=='play' else a.eval_episodes,video=not a.no_video)
    metrics=summary(evaluation)
    selection=dict(checkpoint=str(a.checkpoint),label='selected',completed_updates=checkpoint(a.checkpoint)['infos']['completed_updates'],metrics=metrics,cases=evaluation['cases'],run=str(run),behavior_status=behavior_status(metrics))
   state['status']='completed' if not code else selection['behavior_status']
  except KeyboardInterrupt:code=130;state.update(status='interrupted',error='KeyboardInterrupt')
  except Exception as error:code=1;state.update(status='failed',error=str(error),traceback=traceback.format_exc());traceback.print_exc()
  finally:
   # 中断时父进程尚未收到result；以完整更新行恢复计数，不能把已训练误标为0。
   if a.mode=='train' and state['status']!='completed':
    log=folder/'training/updates.jsonl';completed=[]
    if log.is_file():
     for line in log.read_text().splitlines():
      try:completed.append(json.loads(line))
      except json.JSONDecodeError:break
    if completed:state.update(completed_updates=len(completed),total_checkpoint_updates=completed[-1]['completed_updates'])
   state['finished_at']=now();write(folder/'session.json',state)
   if selection:write(folder/'selection.json',selection)
   try:
    from r5a_guided_v4.report import plot_training
    plot_training(folder)
   except Exception as error:write(folder/'plot_error.json',dict(error=str(error)))
   print('[LOCAL FULL DATA]',folder,flush=True)
   try:
    archive=export_review(folder,review,state,selection)
    (lockdir/'LATEST_GUIDED_V4_REVIEW.txt').write_text(str(review)+'\n')
    if selection and selection['behavior_status'] not in ('physics_rejected','physics_unchecked'):
     selected=review/'selected.pt';(lockdir/'LATEST_GUIDED_V4_CHECKPOINT.txt').write_text(str(selected)+'\n')
    print('[USB REVIEW]',review,flush=True);print('[USB REVIEW ZIP]',archive,flush=True)
   except Exception as error:
    print('[EXPORT ERROR]',str(error),'完整资料仍在本机；用collect --folder完整目录重试',flush=True);code=code or 1
 return code
if __name__=='__main__':raise SystemExit(main())
