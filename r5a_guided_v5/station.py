"""离线迁移已成功的v3 Actor、两次目标设备复核，再连续PPO及一次末尾基线对照。"""
import argparse,json,shutil,subprocess,sys,traceback,signal,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from r5a_guided_v5.io import read,write,sha,stamp,implementation,seal,pack_review

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
 if state.get('candidate_audit'):
  from r5a_guided_v5.candidate_prepare import copy_audit_summaries
  copy_audit_summaries(state['candidate_audit'],out/'candidate_audit')
 write(out/'compact_scope.json',dict(raw_trace_folders=sorted(keep),actor=state.get('actor'),dataset=state.get('dataset'),
  full_data_retained_on_training_machine=str(work),not_exported='other intermediate models, all source copies, caches, and other raw traces'))

def verify_teacher_manifest():
 folder=ROOT/'r5a_guided_v5/teacher';manifest=read(folder/'manifest.json')
 for rel,expected in manifest['files'].items():
  p=(folder/rel).resolve()
  if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha(p)!=expected:raise ValueError('Teacher asset SHA mismatch: '+rel)
 if not manifest['cpu_full_successes']:raise ValueError('No verified teacher evidence')
 for rel in manifest['cpu_full_successes']:
  if not teacher_passed(read(folder/rel)):raise ValueError('Teacher evidence is not a full successful replay')
 return manifest

def invoke_probe(work,name,teacher,device,seed,student=None,correction=None):
 out=work/name;command=[sys.executable,'-u',str(ROOT/'r5a_guided_v5/teacher_verify.py'),'--output',str(out),'--device',device,'--seed',str(seed)]
 if teacher:command+=['--teacher',str(teacher)]
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
 from r5a_guided_v5.candidate_prepare import validate_candidate_audit
 if not ready or not validate_candidate_audit(ready.get('candidate_audit')) or (device is not None and ready.get('device')!=device) or ready.get('implementation_sha256')!=implementation():return False
 for k,h in (('actor','actor_sha256'),('demonstrations','demonstrations_sha256')):
  if not Path(ready.get(k,'')).is_file() or sha(Path(ready[k]))!=ready.get(h):return False
 proof=ready.get('autonomous_proof',[])
 return len(proof)==2 and all(x.get('controller')=='student_actor' and x.get('actor_sha256')==ready['actor_sha256'] and teacher_passed(x['report']) for x in proof)

def verify_baseline_manifest():
 folder=ROOT/'r5a_guided_v5/baseline';m=read(folder/'manifest.json')
 for rel,h in m['files'].items():
  source=(folder/rel).resolve()
  if not source.is_relative_to(folder.resolve()) or not source.is_file() or sha(source)!=h:raise ValueError('Baseline asset SHA mismatch: '+rel)
 if m['source_checkpoint_sha256']!='4348eea44915334d37fbd132dbba34c61ed17b1096f454748416d2d5035a4a75':raise ValueError('Unexpected v3 baseline')
 if not teacher_passed(read(folder/'source_result.json')):raise ValueError('Baseline has no complete success evidence')
 return m

def prepare(device,steps=None,rounds=None,v4_session=None):
 """显式迁移已经成功的v3 Actor；无重新BC、无教师在线接管，目标设备两次复核。"""
 from r5a_guided_v5.candidate_prepare import ensure_candidate_audit
 candidate_audit=ensure_candidate_audit(v4_session)
 manifest=verify_baseline_manifest();base=ROOT/'r5a_guided_v5/baseline'
 work=ROOT/'offline_sessions/route_05a'/('guided_v5_prepare_'+stamp());work.mkdir(parents=True,exist_ok=False)
 actor=work/'actor.pt';dataset=work/'demonstrations.npz'
 shutil.copy2(base/'actor.pt',actor);shutil.copy2(base/'demonstrations.npz',dataset)
 state=dict(status='running',device=device,candidate_audit=candidate_audit,ppo_updates=0,bc_steps=0,implementation_sha256=implementation(),
  migration=manifest,actor=actor.name,dataset=dataset.name,retained_evidence=[],student_attempts=[])
 write(work/'preparation.json',state)
 try:
  proof=[]
  for repeat in range(2):
   folder,report=invoke_probe(work,f'baseline_{repeat}',None,device,43+repeat,student=actor)
   state['retained_evidence']=[folder.name]
   state['student_attempts'].append(dict(repeat=repeat,passed=teacher_passed(report),report=report));write(work/'preparation.json',state)
   if not teacher_passed(report):raise RuntimeError('v3成功基线未在当前设备复现；PPO未启动，请回收诊断ZIP。不要放宽判据。')
   proof.append(dict(controller='student_actor',actor_sha256=sha(actor),report=report))
  state.update(status='prepared',student_success=True,autonomous_proof=proof,initial_actor_sha256=sha(actor),
   demonstrations_sha256=sha(dataset),scope='exact v3 actor migration; two fixed-condition target-device autonomous repeats; no robustness claim')
  write(work/'preparation.json',state);review=work/'compact';compact_preparation(work,review)
  ready=dict(actor=str(actor),demonstrations=str(dataset),preparation=str(review),candidate_audit=candidate_audit,device=device,actor_sha256=sha(actor),
   demonstrations_sha256=sha(dataset),implementation_sha256=implementation(),student_success=True,autonomous_proof=proof)
  assert validate_ready(ready,device);write(ROOT/'offline_sessions/route_05a/READY_GUIDED_V5.json',ready)
  print('[GUIDED PREPARED] v3基线两次自主通过；新Critic/Adam，连续优化触地前缓冲和恢复。',flush=True);return ready
 except BaseException as error:
  state.update(status='preparation_failed',error=str(error),traceback=traceback.format_exc());write(work/'preparation.json',state)
  review=ROOT/'usb_review/route_05a'/(work.name+'_diagnostic');compact_preparation(work,review);seal(review)
  print('[USB REVIEW ZIP]',pack_review(review),flush=True);raise

def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 if not args or args[0] in ('help','--help','-h') or '--help' in args or '-h' in args:
  print('bash r5a_guided_v5.sh check\nbash r5a_guided_v5.sh audit --v4-session PATH\nbash r5a_guided_v5.sh train --v4-session PATH\nTrain: read-only v4 contact/reward audit -> successful v3 actor migration -> two autonomous passes -> continuous 3000 PPO updates -> final comparison -> two-role USB ZIP');return 0
 mode=args[0]
 if mode=='audit':
  p=argparse.ArgumentParser();p.add_argument('--v4-session',type=Path);a=p.parse_args(args[1:])
  from r5a_guided_v5.candidate_prepare import ensure_candidate_audit
  record=ensure_candidate_audit(a.v4_session,strict=False)
  print('[CANDIDATE AUDIT]', 'passed' if record['passed'] else 'incomplete_or_failed',flush=True)
  return 0 if record['passed'] else 1
 if mode not in ('prepare','train'):
  from r5a_guided_v5.cli import main as cli
  return cli(args)
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',type=Path);p.add_argument('--v4-session',type=Path)
 a,remaining=p.parse_known_args(args[1:])
 from r5a_guided_v5.cli import main as cli
 if a.checkpoint and mode=='prepare':raise ValueError('prepare不启动续训；--checkpoint只用于train')
 if a.checkpoint:
  model=a.checkpoint.resolve(strict=True)
  paths=[model.parent/'input/demonstrations.npz',model.parent/'training/input/demonstrations.npz',model.parent.parent/'input/demonstrations.npz']
  demo=next((p for p in paths if p.is_file()),None)
  if demo is None:raise ValueError('Resume requires original training/input/demonstrations.npz beside checkpoint or USB bundle')
  return cli(['train','--device',a.device,*remaining,*(['--v4-session',str(a.v4_session)] if a.v4_session else []),'--checkpoint',str(model),'--demonstrations',str(demo)])
 lockdir=ROOT/'offline_sessions/route_05a';lockdir.mkdir(parents=True,exist_ok=True)
 with (lockdir/'guided_v5_preparation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  marker=lockdir/'READY_GUIDED_V5.json';ready=read(marker) if marker.is_file() and mode!='prepare' else None
  if ready:
   from r5a_guided_v5.candidate_prepare import resolve_v4_session
   try:chosen_session=resolve_v4_session(a.v4_session)
   except (ValueError,OSError):ready=None
   else:
    if str(chosen_session)!=ready.get('candidate_audit',{}).get('session'):ready=None
  if not validate_ready(ready,a.device):ready=prepare(a.device,v4_session=a.v4_session)
 if mode=='prepare':return 0
 return cli(['train','--device',a.device,*remaining,'--v4-session',ready['candidate_audit']['session'],'--initial-actor',ready['actor'],'--demonstrations',ready['demonstrations'],'--preparation',ready['preparation']])
if __name__=='__main__':raise SystemExit(main())
