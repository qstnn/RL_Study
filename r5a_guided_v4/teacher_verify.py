"""Frozen baseline/teacher verification; no PPO, root-state injection or contact bypass."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from r5a_guided_v4.io import prepare_run,write,sha,check_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--teacher',type=Path);p.add_argument('--seed',type=int,default=43);p.add_argument('--student',type=Path)
 p.add_argument('--teacher-beta',type=float);p.add_argument('--student-until',type=int);a=p.parse_args()
 correction=a.teacher_beta is not None or a.student_until is not None
 if correction and (not a.student or not a.teacher):p.error('Correction requires a student and a teacher')
 if a.teacher_beta is not None and not 0<=a.teacher_beta<=1:p.error('teacher-beta must be in [0,1]')
 if a.student_until is not None and not 20<=a.student_until<300:p.error('student-until must be in [20,300)')
 if a.teacher_beta is not None and a.student_until is not None:p.error('Choose mixing or prefix takeover, not both')
 run=a.output.resolve();prepare_run(run,dict(mode='teacher_verification' if a.teacher else 'teacher_baseline',device=a.device,teacher=str(a.teacher) if a.teacher else None,ppo_updates=0))
 import shutil
 sources={}
 for source in []:  # prepare_run already archives all guided sources and teacher assets

  dest=run/'teacher_source'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest);sources[source.name]=sha(source)
 write(run/'teacher_source_manifest.json',sources)
 app=env=None
 try:
  from isaaclab.app import AppLauncher
  app=AppLauncher(headless=True,device=a.device).app
  import torch,gymnasium as gym
  torch.set_num_threads(1)
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
  from r5a_guided_v4.env import AutoFlipCfg,AutoFlipEnv
  from r5a_guided_v4.worker import runtime
  from r5a_guided_v4.policy import SquashedActorCritic
  from r5a_guided_v4.record import Recorder
  class Wrapper(RslRlVecEnvWrapper):
   def __init__(self,*args,**kw):
    super().__init__(*args,**kw);self.num_privileged_obs=gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
  cfg=AutoFlipCfg();cfg.scene.num_envs=1;cfg.seed=a.seed;cfg.sim.device=a.device
  cfg.scene.robot.spawn.asset_path=str(run/'input/robot.urdf');cfg.scene.robot.spawn.usd_dir=str(run/'asset_cache');cfg.scene.robot.spawn.usd_file_name='robot.usd'
  env=Wrapper(AutoFlipEnv(cfg),clip_actions=1.);runtime(env,run);b=env.unwrapped
  metadata=json.loads((run/'runtime_check.json').read_text());metadata.update(no_reference_trajectory=bool(a.student) and not correction,controller='correction_teacher_student' if correction else ('autonomous_student' if a.student else 'teacher'),verification='full episode or native failure; correction is not autonomous acceptance');write(run/'runtime_check.json',metadata)
  actor=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu').to(a.device).eval()
  from r5a_guided_v4.learning import load_initial
  path=ROOT/'r5a_guided_v4/teacher/bc_initial_actor.pt';load_initial(actor,path)
  controller=lambda obs,env:actor.act_inference(obs)
  specification=None
  if a.teacher:
   specification=json.loads(a.teacher.read_text());parameters=specification['parameters']
   from r5a_guided_v4.teacher import control_handoff as module
   family=specification.get('controller_family','latent')
   if family!='handoff':raise ValueError('Only independently verified bounded teachers are eligible')
   Teacher=module.Teacher;DIM=module.DIM
   assert len(parameters)==DIM
   controller=Teacher(actor,[parameters])
  teacher_controller=controller
  if a.student:
   student=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu').to(a.device).eval()
   load_initial(student,a.student)
   controller=lambda obs,env:student.act_inference(obs)
  label='student' if a.student else 'teacher'
  student_sha=sha(a.student) if a.student else None
  teacher_sha=sha(a.teacher) if a.teacher else None
  write(run/'controller.json',dict(kind='correction_mixed' if correction else ('student_actor' if a.student else 'feedback_teacher'),source_sha256=sha(path),student_sha256=student_sha,teacher_sha256=teacher_sha,device=a.device,ppo_updates=0,specification=specification,teacher_beta=a.teacher_beta,student_until=a.student_until))
  with torch.inference_mode():
   obs,_=env.reset();rec=Recorder(b,run,label,False);steps=0;start=time.monotonic();labels=[]
   for _ in range(300):
    action=controller(obs,b)
    if correction:
     target=teacher_controller(obs,b);labels.append(target[0].clone())
     beta=a.teacher_beta if a.teacher_beta is not None else float(steps>=a.student_until)
     action=action*(1.-beta)+target*beta
    _,extra=env.get_observations();rec.policy_step(obs,action,extra['observations']['critic'])
    obs,_,done,_=env.step(action);steps+=1
    if steps%25==0:print('['+label.upper()+' BASELINE]',steps,'angle',float(b.state.angle[0])*180/3.14159265,'elapsed',time.monotonic()-start,flush=True)
    if done[0]:break
   report=rec.finish('terminated' if bool(b.reset_terminated[0]) else 'timeout',steps)
  check_sources(run)
  from r5a_guided_v4.failure_detail import explain
  detail=explain(run,label);write(run/'failure_detail.json',detail)
  if detail['failed']:print('['+label.upper()+' FAILURE]',json.dumps(detail),flush=True)
  if a.student and sha(a.student)!=student_sha:raise RuntimeError('Student changed during verification')
  if a.teacher and sha(a.teacher)!=teacher_sha:raise RuntimeError('Teacher changed during verification')
  assert all(sha(Path(__file__).parent/name)==h for name,h in sources.items()),'Teacher source changed during verification'
  if report['success'] and not a.student:
   import numpy as np
   data=np.load(run/'teacher_policy.npz');raw=np.load(run/'teacher_trace.npz')
   executed=raw['actions'][1::20]
   if executed.shape!=data['raw_action'].shape:raise ValueError('Demonstration action alignment mismatch')
   np.savez_compressed(run/'demonstrations.npz',observation=data['observation'],action=executed,critic=data['critic'])
  if correction:
   import numpy as np
   data=np.load(run/'student_policy.npz',allow_pickle=False)
   np.savez_compressed(run/'correction_labels.npz',observation=data['observation'],action=torch.stack(labels).cpu().numpy(),critic=data['critic'])
   write(run/'correction.json',dict(eligible=bool(report['success'] and report['physics_consistent'] and report['online_agrees']),
    autonomous=False,label_source='teacher on actual student/mixed states',teacher_beta=a.teacher_beta,student_until=a.student_until,
    student_sha256=student_sha,teacher_sha256=teacher_sha,labels_sha256=sha(run/'correction_labels.npz')))
  write(run/'result.json',report);print('['+label.upper()+' RESULT]',json.dumps(report),flush=True)
 except BaseException:
  (run/'error.txt').write_text(traceback.format_exc());raise
 finally:
  if env is not None:env.close()
  if app is not None:app.close()
if __name__=='__main__':main()
