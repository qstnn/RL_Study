"""自主空翻离线单命令：训练→固定种子评估→录像→单文件夹回收。"""
import argparse,fcntl,json,os,platform,shutil,signal,subprocess,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_auto_v2.io import ROOT,read,write,stamp,now,sha,checkpoint,prepare_run,seal,pack_review,asset_manifest
from r5a_auto_v2.spec import CONTRACT as C

def child(session,mode,name,a,model=None,seeds=None,episodes=1,updates=0,video=False):
 run=session/name
 request=dict(mode=mode,seed=a.seed,device=a.device,num_envs=a.num_envs if mode=='train' else 1,
  headless=mode!='play' or a.headless,video=video,checkpoint=str(model) if model else None,
  eval_seeds=seeds or [43,44,45],eval_episodes=episodes,updates=updates)
 prepare_run(run,request)
 command=[sys.executable,'-u',str(ROOT/'r5a_auto_v2/worker.py'),'--run',str(run)]
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
  mean_recovery_score=sum(x['recovery_score'] if x['qualified_landing'] and not x['failed'] else 0 for x in cases)/n,
  mean_stable_s=sum(x['stable_s'] for x in cases)/n,
  mean_impact_bodyweights=sum(x['peak_support_bodyweights'] for x in cases)/n,
  mean_flight_rise_m=sum(x['flight_rise_m'] for x in cases)/n,failed_rate=sum(x['failed'] for x in cases)/n)
def rank(x):
 # 达到合格单圈后，先比较站稳/恢复，再比较毫米级高度；所有阶段先排除动力学异常。
 return (-x['physics_failure_rate'],x['success_rate'],x['qualified_rate'],x['mean_stable_s'],x['mean_recovery_score'],
  x['valid_flight_rate'],x['mean_valid_angle_deg'],x['launch_rate'],-x['failed_rate'],-x['mean_impact_bodyweights'],x['mean_flight_rise_m'])

def positive(x):
 x=int(x)
 if x<1:raise argparse.ArgumentTypeError('Must be positive')
 return x

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('mode',choices=['check','train','evaluate','play','verify','collect'])
 p.add_argument('--num-envs',type=positive,default=512);p.add_argument('--updates',type=positive,default=2000)
 p.add_argument('--interval',type=positive,default=200);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0')
 p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);p.add_argument('--folder',type=Path)
 p.add_argument('--eval-episodes',type=positive,default=4);p.add_argument('--play-episodes',type=positive,default=10)
 p.add_argument('--headless',action='store_true');p.add_argument('--no-video',action='store_true')
 a=p.parse_args(argv)
 if a.mode in ('collect','verify'):
  if not a.folder:p.error('--folder required')
  folder=a.folder.resolve(strict=True)
  if a.mode=='collect':
   state=read(folder/'session.json')
   if state['status']=='running':raise ValueError('Session marked running; do not seal an active run')
   seal(folder);print('[USB REVIEW ZIP]',pack_review(folder));return 0
  hashes=read(folder/'SHA256.json');bad=[]
  for name,h in hashes.items():
   path=(folder/name).resolve()
   if not path.is_relative_to(folder) or not path.is_file() or sha(path)!=h:bad.append(name)
  if bad:raise ValueError('SHA mismatch: '+str(bad))
  print('[VERIFIED]',len(hashes),'files');return 0
 if a.mode=='check':
  from r5a_auto_v2.tests import run_tests
  assets=asset_manifest();run_tests()
  print('[AUTOFLIP CHECK] CPU/资产通过；不是GPU或空翻行为验收。',C['version'],'mass=',assets['mass_kg']);return 0
 if a.mode=='train' and a.num_envs<4:p.error('PPO requires at least 4 environments')
 if a.checkpoint:a.checkpoint=a.checkpoint.resolve(strict=True);checkpoint(a.checkpoint)
 elif a.mode in ('evaluate','play'):
  marker=ROOT/'offline_sessions/route_05a/LATEST_AUTOFLIP_V2_CHECKPOINT.txt'
  if not marker.exists():p.error('No autonomous model yet: provide --checkpoint')
  a.checkpoint=(ROOT/marker.read_text().strip()).resolve(strict=True);checkpoint(a.checkpoint)
 folder=(a.output or ROOT/'usb_review/route_05a'/('autoflip_v2_'+stamp())).resolve();folder.mkdir(parents=True,exist_ok=False)
 state=dict(status='running',created_at=now(),kind='autonomous_backflip',contract=C,requested_updates=a.updates if a.mode=='train' else 0,
  completed_updates=0,mode=a.mode,python=sys.version,platform=platform.platform(),formal_behavior_acceptance='requires_review')
 write(folder/'session.json',state);write(folder/'launch_arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
 lockdir=ROOT/'offline_sessions/route_05a';lockdir.mkdir(parents=True,exist_ok=True);code=0
 with (lockdir/'autoflip_v2.lock').open('a') as lock:
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if a.mode=='train':
    print(f'[V2 BUDGET] 本次总预算 {a.updates} 更新，每段 {a.interval} 更新后自动复评并继续；从新策略开始' if not a.checkpoint else f'[V2 BUDGET] 从v2检查点追加 {a.updates} 更新，每段 {a.interval} 更新。',flush=True)
    current=a.checkpoint;best=None;best_stats=None;history=[];done=0
    while done<a.updates:
     count=min(a.interval,a.updates-done)
     run,result=child(folder,'train',f'train_{done+count:06d}',a,current,updates=count)
     current=run/'final.pt';done+=count;state.update(completed_updates=done,total_checkpoint_updates=result['completed_updates'],last_checkpoint=str(current.relative_to(folder)))
     write(folder/'session.json',state)
     _,evaluation=child(folder,'evaluate',f'eval_{done:06d}',a,current,episodes=2)
     metrics=summary(evaluation);history.append(dict(updates=done,**metrics));write(folder/'quality_history.json',history)
     print('[AUTOFLIP QUALITY]',json.dumps(history[-1]),flush=True)
     if metrics['physics_failure_rate']>0:raise RuntimeError('冻结轨迹未通过角动量筛查；停止续训，请带回审查ZIP')
     if best_stats is None or rank(metrics)>=rank(best_stats):best=current;best_stats=metrics
    selected=folder/'selected.pt';shutil.copy2(best,selected)
    shutil.copy2(current,folder/'last.pt')
    state.update(selected_checkpoint='selected.pt',selected_sha256=sha(selected),selection_metrics=best_stats)
    _,final=child(folder,'evaluate','final_evaluation',a,selected,episodes=a.eval_episodes)
    write(folder/'final_summary.json',summary(final))
    if not a.no_video:
     # 录像自动headless，交互play仍保留World自由相机。
     a.headless=True
     child(folder,'play','final_video',a,selected,seeds=[43,44,45],episodes=1,video=True)
    (lockdir/'LATEST_AUTOFLIP_V2_CHECKPOINT.txt').write_text(str(selected.relative_to(ROOT))+'\n' if selected.is_relative_to(ROOT) else str(selected)+'\n')
   else:
    child(folder,a.mode,'replay',a,a.checkpoint,seeds=[43,44,45] if a.mode=='evaluate' else [a.seed],
     episodes=a.eval_episodes if a.mode=='evaluate' else a.play_episodes,video=not a.no_video and a.mode=='play')
   state['status']='completed'
  except KeyboardInterrupt:code=130;state.update(status='interrupted',error='KeyboardInterrupt')
  except Exception as error:
   code=1;state.update(status='failed',error=str(error),traceback=traceback.format_exc());traceback.print_exc()
  finally:
   state['finished_at']=now();write(folder/'session.json',state)
   try:
    from r5a_auto_v2.report import plot_training
    plot_training(folder)
   except Exception as plot_error:write(folder/'plot_error.json',dict(error=str(plot_error)))
   seal(folder);(lockdir/'LATEST_AUTOFLIP_V2_REVIEW.txt').write_text(str(folder)+'\n')
   print('[USB REVIEW]',folder,flush=True)
   try:print('[USB REVIEW ZIP]',pack_review(folder),flush=True)
   except Exception as pack_error:
    print('[PACK ERROR]',str(pack_error),'请拷贝上述整个文件夹或用collect重试',flush=True);code=code or 1
 return code
if __name__=='__main__':raise SystemExit(main())
