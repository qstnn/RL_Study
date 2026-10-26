"""Offline teacher verification -> supervised initialization -> continuous PPO."""
import argparse,json,shutil,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_guided_v2.io import read,write,sha,stamp,implementation,seal,pack_review

def teacher_passed(report):
 return (all(report.get(k) is True for k in ('success','protocol_success','qualified_landing','online_agrees','physics_consistent'))
  and report.get('failed') is False and report.get('duration_s',0)>=6. and report.get('stable_s',0)>=1.-1e-5)

def compact_preparation(work,out):
 out.mkdir(parents=True,exist_ok=False)
 for p in work.rglob('*'):
  if not p.is_file():continue
  if p.is_relative_to(out):continue
  rel=p.relative_to(work)
  if any(x in rel.parts for x in ('source_snapshot','dependency_source','asset_cache','__pycache__','teacher_source')):continue
  if p.suffix=='.log':
   from collections import deque
   dest=out/rel.with_name(p.stem+'_tail.log');dest.parent.mkdir(parents=True,exist_ok=True)
   with p.open(errors='replace') as f:dest.write_text(''.join(deque(f,maxlen=200)))
  elif p.suffix in ('.json','.npz','.pt','.png'):
   dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

def verify_teacher_manifest():
 folder=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2/teacher';manifest=read(folder/'manifest.json')
 for rel,expected in manifest['files'].items():
  p=(folder/rel).resolve()
  if not p.is_relative_to(folder.resolve()) or sha(p)!=expected:raise ValueError('Teacher asset SHA mismatch: '+rel)
 if not manifest['cpu_full_successes']:raise ValueError('No independently verified teacher')
 for rel in manifest['cpu_full_successes']:
  report=read(folder/rel)
  if not (report['success'] and report['physics_consistent'] and report['online_agrees'] and report['duration_s']>=6.):raise ValueError('Teacher evidence is not a full successful replay')
 return manifest

def invoke_probe(work,name,teacher,device,seed,student=None):
 out=work/name;command=[sys.executable,'-u',str(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2/teacher_verify.py'),'--output',str(out),'--teacher',str(teacher),'--device',device,'--seed',str(seed)]
 if student:command+=['--student',str(student)]
 log=work/(name+'_console.log')
 with log.open('w') as stream:
  proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
  for line in proc.stdout:
   stream.write(line);stream.flush()
   if '[TEACHER' in line or '[STUDENT' in line or 'Traceback' in line:print(line,end='',flush=True)
  code=proc.wait()
 if not (out/'result.json').exists() or (out/'error.txt').exists():raise RuntimeError(f'Probe failed ({code}); see {log}')
 if code and not device.startswith('cpu'):raise RuntimeError(f'Target GPU probe exited abnormally ({code}); see {log}')
 report=read(out/'result.json')
 # Some CPU-only Kit installations fail during shutdown after sealed evidence.
 # Retain the exit code explicitly rather than treating it as behavioral proof.
 report['process_returncode']=code;write(work/(name+'_process.json'),dict(returncode=code,complete_report=True))
 return out,report

def prepare(device,steps):
 manifest=verify_teacher_manifest()
 work=ROOT/'offline_sessions/route_05a'/('guided_v2_prepare_'+stamp());work.mkdir(parents=True,exist_ok=False)
 state=dict(status='running',device=device,bc_steps=steps,ppo_updates=0,implementation_sha256=implementation(),teacher_attempts=[])
 write(work/'preparation.json',state)
 try:
  import numpy as np
  selected=None;good_runs=[]
  for item in manifest['candidates']:
   candidate=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v2/teacher'/item
   print('[TEACHER CANDIDATE]',candidate.stem,'device='+device,flush=True)
   runs=[]
   for repeat in range(2):
    folder,report=invoke_probe(work,f'teacher_{candidate.stem}_{repeat}',candidate,device,43+repeat)
    passed=teacher_passed(report)
    state['teacher_attempts'].append(dict(candidate=item,repeat=repeat,passed=passed,result=str(folder.relative_to(work)/'result.json')))
    write(work/'preparation.json',state)
    if not passed:break
    runs.append(folder)
   if len(runs)==2:selected=candidate;good_runs=runs;break
  if selected is None:raise RuntimeError('所有教师候选均未通过目标设备两次完整复核；BC/PPO尚未开始。请取回上方诊断ZIP，查看其中failure_detail.json；无需反复重跑。')
  arrays=[np.load(p/'demonstrations.npz',allow_pickle=False) for p in good_runs]
  dataset=work/'demonstrations.npz'
  np.savez_compressed(dataset,**{k:np.concatenate([a[k] for a in arrays]) for k in ('observation','action','critic')})
  from r5a_guided_v2.learning import pretrain
  actor=pretrain(dataset,work/'bc',device=device,steps=steps)
  _,student=invoke_probe(work,'student_baseline',selected,device,43,student=actor)
  state.update(status='prepared',selected_teacher=str(selected.relative_to(ROOT)),teacher_repeat_scope='fixed conditions; not robustness trials',
   student_baseline=student,student_success=student['success'],initial_actor_sha256=sha(actor),demonstrations_sha256=sha(dataset))
  write(work/'preparation.json',state)
  review=work/'compact';compact_preparation(work,review)
  ready=dict(actor=str(actor),demonstrations=str(dataset),preparation=str(review),device=device,
   actor_sha256=sha(actor),demonstrations_sha256=sha(dataset),implementation_sha256=implementation(),student_success=student['success'])
  write(ROOT/'offline_sessions/route_05a/READY_GUIDED_V2.json',ready)
  print('[GUIDED PREPARED]',json.dumps({k:v for k,v in ready.items() if k!='implementation_sha256'}),flush=True)
  return ready
 except BaseException as error:
  state.update(status='preparation_failed',error=str(error),traceback=traceback.format_exc());write(work/'preparation.json',state)
  review=ROOT/'usb_review/route_05a'/(work.name+'_diagnostic');compact_preparation(work,review);seal(review)
  print('[USB REVIEW ZIP]',pack_review(review),flush=True)
  raise

def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 if not args or args[0] in ('help','--help','-h'):
  print('bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v2.sh check\nbash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v2.sh train\nOptional: prepare --device cpu --bc-steps 3000\nTrain verifies teacher, performs BC, records student baseline, then continuous PPO and one final evaluation.');return 0
 mode=args[0]
 if mode not in ('prepare','train'):
  from r5a_guided_v2.cli import main as cli
  return cli(args)
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--device',default='cuda:0');p.add_argument('--bc-steps',type=int,default=3000);p.add_argument('--checkpoint',type=Path)
 a,remaining=p.parse_known_args(args[1:])
 if a.bc_steps<1:raise ValueError('bc-steps must be positive')
 if mode=='prepare':prepare(a.device,a.bc_steps);return 0
 from r5a_guided_v2.cli import main as cli
 extra=[]
 if a.checkpoint:
  model=a.checkpoint.resolve(strict=True)
  paths=[model.parent/'input/demonstrations.npz',model.parent/'training/input/demonstrations.npz']
  demo=next((p for p in paths if p.is_file()),None)
  if demo is None:raise ValueError('Resume requires the original training/input/demonstrations.npz beside the checkpoint or USB bundle')
  extra=['--checkpoint',str(model),'--demonstrations',str(demo)]
 else:
  marker=ROOT/'offline_sessions/route_05a/READY_GUIDED_V2.json';ready=read(marker) if marker.exists() else None
  valid=ready is not None and ready['device']==a.device and ready['implementation_sha256']==implementation()
  if valid:
   valid=all(Path(ready[k]).is_file() and sha(ready[k])==ready[h] for k,h in (('actor','actor_sha256'),('demonstrations','demonstrations_sha256')))
  if not valid:ready=prepare(a.device,a.bc_steps)
  print('[GUIDED STUDENT BASELINE] success='+str(ready['student_success'])+'; PPO is a new learning run, not a claim of student mastery.',flush=True)
  extra=['--initial-actor',ready['actor'],'--demonstrations',ready['demonstrations'],'--preparation',ready['preparation']]
 return cli(['train','--device',a.device,*remaining,*extra])
if __name__=='__main__':raise SystemExit(main())
