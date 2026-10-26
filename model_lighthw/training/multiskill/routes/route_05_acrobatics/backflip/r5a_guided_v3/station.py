"""离线教师复核、学生状态纠偏、两次自主通过，再连续PPO及一次末尾评估。"""
import argparse,json,shutil,subprocess,sys,traceback,signal,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_guided_v3.io import read,write,sha,stamp,implementation,seal,pack_review

def teacher_passed(report):
 return (all(report.get(k) is True for k in ('success','protocol_success','qualified_landing','online_agrees','physics_consistent'))
  and report.get('failed') is False and report.get('duration_s',0)>=6. and report.get('stable_s',0)>=1.-1e-5)

def compact_preparation(work,out):
 """全部小型判定+一条教师/最终学生/最近纠偏轨迹；不拷所有BC模型和大轨迹。"""
 out.mkdir(parents=True,exist_ok=False)
 state=read(work/'preparation.json');keep=set(state.get('retained_evidence',[]))
 if state.get('last_correction'):keep.add(state['last_correction'])
 if not keep:
  # 教师尚未通过或子进程中断时，也保留最近已产生的原始轨迹，不能只剩摘要。
  traces=[p.parent for p in work.glob('*/*_trace.npz')]
  if traces:keep.add(max(traces,key=lambda p:p.stat().st_mtime).name)
 essential={state.get('actor',''),state.get('dataset','')}
 for p in work.rglob('*'):
  if not p.is_file() or p.is_relative_to(out):continue
  rel=p.relative_to(work)
  if any(x in rel.parts for x in ('source_snapshot','dependency_source','asset_cache','__pycache__','teacher_source')):continue
  selected=rel.parts[0] in keep
  if p.suffix=='.json' or str(rel) in essential or (selected and p.suffix in ('.npz','.png')):
   dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  elif p.suffix=='.log':
   from collections import deque
   dest=out/rel.with_name(p.stem+'_tail.log');dest.parent.mkdir(parents=True,exist_ok=True)
   with p.open(errors='replace') as f:dest.write_text(''.join(deque(f,maxlen=80)))
 write(out/'compact_scope.json',dict(raw_trace_folders=sorted(keep),actor=state.get('actor'),dataset=state.get('dataset'),
  full_data_retained_on_training_machine=str(work),not_exported='other intermediate models, all source copies, caches, and other raw traces'))

def verify_teacher_manifest():
 folder=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3/teacher';manifest=read(folder/'manifest.json')
 for rel,expected in manifest['files'].items():
  p=(folder/rel).resolve()
  if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha(p)!=expected:raise ValueError('Teacher asset SHA mismatch: '+rel)
 if not manifest['cpu_full_successes']:raise ValueError('No verified teacher evidence')
 for rel in manifest['cpu_full_successes']:
  if not teacher_passed(read(folder/rel)):raise ValueError('Teacher evidence is not a full successful replay')
 return manifest

def invoke_probe(work,name,teacher,device,seed,student=None,correction=None):
 out=work/name;command=[sys.executable,'-u',str(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3/teacher_verify.py'),'--output',str(out),'--teacher',str(teacher),'--device',device,'--seed',str(seed)]
 if student:command+=['--student',str(student)]
 if correction:
  for k,value in correction.items():command+=['--'+k.replace('_','-'),str(value)]
 log=work/(name+'_console.log');proc=None
 try:
  with log.open('w') as stream:
   proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
   for line in proc.stdout:
    stream.write(line);stream.flush()
    if '[TEACHER' in line or '[STUDENT' in line or 'Traceback' in line:print(line,end='',flush=True)
   code=proc.wait()
 finally:
  if proc is not None and proc.poll() is None:
   import os
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 if not (out/'result.json').exists() or (out/'error.txt').exists():raise RuntimeError(f'Probe failed ({code}); see {log}')
 if code and not device.startswith('cpu'):raise RuntimeError(f'Target GPU probe exited abnormally ({code}); see {log}')
 report=read(out/'result.json');report['process_returncode']=code
 write(work/(name+'_process.json'),dict(returncode=code,complete_report=True,cpu_shutdown_exception=bool(code)))
 return out,report

def merge_demonstrations(paths,destination):
 """仅使用已通过完整物理复核的教师/纠偏轨迹；重复固定初态不增加样本权重。"""
 import numpy as np
 arrays=[]
 for p in paths:
  with np.load(p,allow_pickle=False) as a:arrays.append({k:a[k] for k in ('observation','action','critic')})
 values={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
 _,ids=np.unique(np.concatenate([values['observation'],values['action']],axis=1),axis=0,return_index=True)
 ids=np.sort(ids);np.savez_compressed(destination,**{k:v[ids] for k,v in values.items()})
 return dict(frames=len(ids),before_deduplication=len(values['action']),sources=[dict(path=str(p),sha256=sha(p)) for p in paths])

def validate_ready(ready,device=None):
 """缓存与CLI训练均需完整自主证据；混合教师成功不能冒充学生成功。"""
 if not ready or (device is not None and ready.get('device')!=device) or ready.get('implementation_sha256')!=implementation():return False
 for k,h in (('actor','actor_sha256'),('demonstrations','demonstrations_sha256')):
  if not Path(ready.get(k,'')).is_file() or sha(Path(ready[k]))!=ready.get(h):return False
 proof=ready.get('autonomous_proof',[])
 return len(proof)==2 and all(x.get('controller')=='student_actor' and x.get('actor_sha256')==ready['actor_sha256'] and teacher_passed(x['report']) for x in proof)

def prepare(device,steps,rounds=6):
 manifest=verify_teacher_manifest();work=ROOT/'offline_sessions/route_05a'/('guided_v3_prepare_'+stamp());work.mkdir(parents=True,exist_ok=False)
 state=dict(status='running',device=device,bc_steps=steps,correction_rounds=rounds,ppo_updates=0,implementation_sha256=implementation(),
  teacher_attempts=[],student_attempts=[],corrections=[],retained_evidence=[])
 write(work/'preparation.json',state)
 try:
  selected=None;good_runs=[]
  for item in manifest['candidates']:
   candidate=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v3/teacher'/item;runs=[]
   for repeat in range(2):
    folder,report=invoke_probe(work,f'teacher_{candidate.stem}_{repeat}',candidate,device,43+repeat)
    passed=teacher_passed(report);state['teacher_attempts'].append(dict(candidate=item,repeat=repeat,passed=passed,result=str(folder.relative_to(work)/'result.json')))
    write(work/'preparation.json',state)
    if not passed:break
    runs.append(folder)
   if len(runs)==2:selected=candidate;good_runs=runs;break
  if selected is None:raise RuntimeError('教师未通过目标设备两次完整复核；已停止，PPO未启动。请回收诊断ZIP。')
  state['retained_evidence']=[good_runs[0].name]
  datasets=[p/'demonstrations.npz' for p in good_runs];initial=None;proof=[]
  from r5a_guided_v3.learning import pretrain
  for iteration in range(rounds+1):
   dataset=work/f'demonstrations_{iteration:02d}.npz';data=merge_demonstrations(datasets,dataset)
   actor=pretrain(dataset,work/f'bc_{iteration:02d}',device=device,steps=steps if iteration==0 else max(1500,steps//2),initial=initial)
   state.update(actor=str(actor.relative_to(work)),dataset=dataset.name,dataset_info=data,round=iteration);write(work/'preparation.json',state)
   proof=[]
   for repeat in range(2):
    folder,student=invoke_probe(work,f'student_{iteration:02d}_{repeat}',selected,device,43+repeat,student=actor)
    passed=teacher_passed(student);state['student_attempts'].append(dict(round=iteration,repeat=repeat,passed=passed,actor_sha256=sha(actor),report=student))
    state['retained_evidence']=[good_runs[0].name,folder.name]+([state['last_correction']] if state.get('last_correction') else [])
    write(work/'preparation.json',state)
    if not passed:break
    proof.append(dict(controller='student_actor',actor_sha256=sha(actor),report=student))
   if len(proof)==2:break
   if iteration==rounds:raise RuntimeError('纠偏预算已用完，学生仍未两次自主通过；不会进入长PPO。请回收诊断ZIP，不要原样重复长训。')
   # 小比例教师混合收集学生附近的状态；前缀接管针对真实漂移。只收全回合通过的数据。
   beta=max(.1,.5/(1+iteration*.5));plans=[dict(teacher_beta=beta),dict(student_until=min(55,25+5*iteration))]
   accepted=0
   for j,plan in enumerate(plans):
    folder,report=invoke_probe(work,f'correction_{iteration:02d}_{j}',selected,device,43,student=actor,correction=plan)
    eligible=teacher_passed(report) and read(folder/'correction.json')['eligible'];state['last_correction']=folder.name
    state['corrections'].append(dict(round=iteration,plan=plan,accepted=eligible,report=report,labels_sha256=sha(folder/'correction_labels.npz')))
    if eligible:datasets.append(folder/'correction_labels.npz');accepted+=1
    write(work/'preparation.json',state)
   if not accepted:
    # 更保守的混合仍在学生实际状态上标注；不能用失败轨迹或纯重复教师强行凑数据。
    folder,report=invoke_probe(work,f'correction_{iteration:02d}_fallback',selected,device,43,student=actor,correction=dict(teacher_beta=.85))
    eligible=teacher_passed(report) and read(folder/'correction.json')['eligible'];state['last_correction']=folder.name
    state['corrections'].append(dict(round=iteration,plan=dict(teacher_beta=.85),accepted=eligible,report=report,labels_sha256=sha(folder/'correction_labels.npz')))
    if eligible:datasets.append(folder/'correction_labels.npz')
    else:raise RuntimeError('当前学生附近未找到完整可恢复纠偏轨迹；PPO未启动，请回收诊断ZIP。')
   initial=actor
  state.update(status='prepared',selected_teacher=str(selected.relative_to(ROOT)),student_success=True,autonomous_proof=proof,
   initial_actor_sha256=sha(actor),demonstrations_sha256=sha(dataset),scope='two fixed-condition autonomous repeats; no robustness claim')
  write(work/'preparation.json',state);review=work/'compact';compact_preparation(work,review)
  ready=dict(actor=str(actor),demonstrations=str(dataset),preparation=str(review),device=device,actor_sha256=sha(actor),
   demonstrations_sha256=sha(dataset),implementation_sha256=implementation(),student_success=True,autonomous_proof=proof)
  assert validate_ready(ready,device);write(ROOT/'offline_sessions/route_05a/READY_GUIDED_V3.json',ready)
  print('[GUIDED PREPARED] 自主学生两次通过，准备连续PPO。',flush=True);return ready
 except BaseException as error:
  state.update(status='preparation_failed',error=str(error),traceback=traceback.format_exc());write(work/'preparation.json',state)
  review=ROOT/'usb_review/route_05a'/(work.name+'_diagnostic');compact_preparation(work,review);seal(review)
  print('[USB REVIEW ZIP]',pack_review(review),flush=True);raise

def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 if not args or args[0] in ('help','--help','-h'):
  print('bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v3.sh check\nbash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v3.sh train\nTrain: teacher checks -> BC/corrections -> two autonomous passes -> continuous 3000 PPO updates -> final evaluation -> USB ZIP');return 0
 mode=args[0]
 if mode not in ('prepare','train'):
  from r5a_guided_v3.cli import main as cli
  return cli(args)
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--device',default='cuda:0');p.add_argument('--bc-steps',type=int,default=5000);p.add_argument('--correction-rounds',type=int,default=6);p.add_argument('--checkpoint',type=Path)
 a,remaining=p.parse_known_args(args[1:])
 if a.bc_steps<1 or not 0<=a.correction_rounds<=12:raise ValueError('bc-steps positive; correction-rounds in [0,12]')
 from r5a_guided_v3.cli import main as cli
 if a.checkpoint:
  model=a.checkpoint.resolve(strict=True)
  paths=[model.parent/'input/demonstrations.npz',model.parent/'training/input/demonstrations.npz',model.parent.parent/'input/demonstrations.npz']
  demo=next((p for p in paths if p.is_file()),None)
  if demo is None:raise ValueError('Resume requires original training/input/demonstrations.npz beside checkpoint or USB bundle')
  return cli(['train','--device',a.device,*remaining,'--checkpoint',str(model),'--demonstrations',str(demo)])
 lockdir=ROOT/'offline_sessions/route_05a';lockdir.mkdir(parents=True,exist_ok=True)
 with (lockdir/'guided_v3_preparation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  marker=lockdir/'READY_GUIDED_V3.json';ready=read(marker) if marker.is_file() and mode!='prepare' else None
  if not validate_ready(ready,a.device):ready=prepare(a.device,a.bc_steps,a.correction_rounds)
 if mode=='prepare':return 0
 return cli(['train','--device',a.device,*remaining,'--initial-actor',ready['actor'],'--demonstrations',ready['demonstrations'],'--preparation',ready['preparation']])
if __name__=='__main__':raise SystemExit(main())
