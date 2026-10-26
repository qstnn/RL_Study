"""Frozen baseline/teacher verification; no PPO, root-state injection or contact bypass."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_auto_v6.io import prepare_run,write,sha,check_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--teacher',type=Path);p.add_argument('--seed',type=int,default=43);a=p.parse_args()
 run=a.output.resolve();prepare_run(run,dict(mode='teacher_verification' if a.teacher else 'teacher_baseline',device=a.device,teacher=str(a.teacher) if a.teacher else None,ppo_updates=0))
 import shutil
 sources={}
 for source in list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*_reference.json')):
  dest=run/'teacher_source'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest);sources[source.name]=sha(source)
 write(run/'teacher_source_manifest.json',sources)
 app=env=None
 try:
  from isaaclab.app import AppLauncher
  app=AppLauncher(headless=True,device=a.device).app
  import torch,gymnasium as gym
  torch.set_num_threads(1)
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
  from r5a_auto_v6.env import AutoFlipCfg,AutoFlipEnv
  from r5a_auto_v6.worker import runtime
  from r5a_auto_v6.policy import SquashedActorCritic
  from r5a_auto_v6.record import Recorder
  class Wrapper(RslRlVecEnvWrapper):
   def __init__(self,*args,**kw):
    super().__init__(*args,**kw);self.num_privileged_obs=gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
  cfg=AutoFlipCfg();cfg.scene.num_envs=1;cfg.seed=a.seed;cfg.sim.device=a.device
  cfg.scene.robot.spawn.asset_path=str(run/'input/robot.urdf');cfg.scene.robot.spawn.usd_dir=str(run/'asset_cache');cfg.scene.robot.spawn.usd_file_name='robot.usd'
  env=Wrapper(AutoFlipEnv(cfg),clip_actions=1.);runtime(env,run);b=env.unwrapped
  metadata=json.loads((run/'runtime_check.json').read_text());metadata.update(no_reference_trajectory=not bool(a.teacher),controller='searched teacher' if a.teacher else 'original frozen v5 actor',verification='unchanged production environment, full episode or native failure termination');write(run/'runtime_check.json',metadata)
  actor=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu').to(a.device).eval()
  path=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v6/bootstrap/actor_1201.pt';actor.load_state_dict(torch.load(path,map_location=a.device,weights_only=True),strict=False)
  controller=lambda obs,env:actor.act_inference(obs)
  specification=None
  if a.teacher:
   specification=json.loads(a.teacher.read_text());parameters=specification['parameters']
   from r5a_teacher_search.registry import controller as get_controller
   family=specification.get('controller_family','latent')
   if family in ('knots','force','kick','capture','open','split','split_bound','soft'):
    import importlib
    module=importlib.import_module('r5a_teacher_search.control_'+family)
   else:module=get_controller(family)
   Teacher=module.Teacher;DIM=module.DIM
   assert len(parameters)==DIM
   controller=Teacher(actor,[parameters])
  label='teacher' if a.teacher else 'baseline'
  write(run/'teacher.json',dict(kind='feedback_teacher' if a.teacher else 'unscaled_v5_actor',source_sha256=sha(path),device=a.device,ppo_updates=0,specification=specification))
  with torch.inference_mode():
   obs,_=env.reset();rec=Recorder(b,run,label,False);steps=0;start=time.monotonic()
   for _ in range(300):
    action=controller(obs,b);_,extra=env.get_observations();rec.policy_step(obs,action,extra['observations']['critic'])
    obs,_,done,_=env.step(action);steps+=1
    if steps%25==0:print('[TEACHER BASELINE]',steps,'angle',float(b.state.angle[0])*180/3.14159265,'elapsed',time.monotonic()-start,flush=True)
    if done[0]:break
   report=rec.finish('terminated' if bool(b.reset_terminated[0]) else 'timeout',steps)
  check_sources(run)
  assert all(sha(Path(__file__).parent/name)==h for name,h in sources.items()),'Teacher source changed during verification'
  write(run/'result.json',report);print('[TEACHER RESULT]',json.dumps(report),flush=True)
 except BaseException:
  (run/'error.txt').write_text(traceback.format_exc());raise
 finally:
  if env is not None:env.close()
  if app is not None:app.close()
if __name__=='__main__':main()
