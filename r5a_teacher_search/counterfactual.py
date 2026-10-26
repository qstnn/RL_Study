"""Frozen baseline/teacher verification; no PPO, root-state injection or contact bypass."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from r5a_auto_v6.io import prepare_run,write,sha,check_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--teacher',type=Path);p.add_argument('--control-landing-at',type=float);p.add_argument('--replay-state',type=Path,required=True);p.add_argument('--seed',type=int,default=43);a=p.parse_args()
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
  path=ROOT/'r5a_auto_v6/bootstrap/actor_1201.pt';actor.load_state_dict(torch.load(path,map_location=a.device,weights_only=True),strict=False)
  controller=lambda obs,env:actor.act_inference(obs)
  specification=None
  if a.teacher:
   specification=json.loads(a.teacher.read_text());parameters=specification['parameters']
   from r5a_teacher_search.registry import controller as get_controller
   family=specification.get('controller_family','latent')
   if family in ('knots','force','kick','capture','open','split','split_bound','soft','handoff'):
    import importlib
    module=importlib.import_module('r5a_teacher_search.control_'+family)
   else:module=get_controller(family)
   Teacher=module.Teacher;DIM=module.DIM
   assert len(parameters)==DIM
   controller=Teacher(actor,[parameters])
  if a.control_landing_at is not None:
   original_controller=controller
   from r5a_teacher_search.control_handoff import Overlay
   def controller(obs, env):
    early=(env.episode_physics_steps*env.physics_dt>=a.control_landing_at)&~env.state.landed
    state=Overlay(env.state,landed=env.state.landed|early,landing_t=torch.where(early,a.control_landing_at,env.state.landing_t))
    return original_controller(obs,Overlay(env,state=state))
   metadata.update(diagnostic_only=True,control_landing_at=a.control_landing_at,acceptance='counterfactual control-phase experiment; not a delivery teacher')
   write(run/'runtime_check.json',metadata)
  label='teacher' if a.teacher else 'baseline'
  write(run/'teacher.json',dict(kind='feedback_teacher' if a.teacher else 'unscaled_v5_actor',source_sha256=sha(path),device=a.device,ppo_updates=0,specification=specification))
  with torch.inference_mode():
   obs,_=env.reset();rec=Recorder(b,run,label,False);steps=0;start=time.monotonic()
   for step in range(75):
    if step==50:
     import numpy as np
     trace=np.load(a.replay_state,allow_pickle=False);k=1000
     root=torch.tensor(trace['root_state'][k:k+1],device=b.device)
     q=torch.tensor(trace['joint_position'][k:k+1],device=b.device)
     dq=torch.tensor(trace['joint_velocity'][k:k+1],device=b.device)
     b.robot.write_root_state_to_sim(root)
     b.robot.write_joint_state_to_sim(q,dq)
     b.robot.update(0.)
     b.state.prev_quat.copy_(root[:,3:7]);b.state.angle.copy_(torch.tensor(trace['online_angle'][k:k+1],device=b.device))
     b.current_metrics=b.metrics()
     write(run/'STATE_INJECTION_DIAGNOSTIC_ONLY.json',dict(time_s=1.,source=str(a.replay_state),source_sha256=sha(a.replay_state),purpose='paired CPU continuation from measured target-device airborne state',not_teacher_acceptance=True,not_training_data=True))
    action=controller(obs,b);_,extra=env.get_observations();rec.policy_step(obs,action,extra['observations']['critic'])
    obs,_,done,_=env.step(action);steps+=1
    if steps%25==0:print('[TEACHER BASELINE]',steps,'angle',float(b.state.angle[0])*180/3.14159265,'elapsed',time.monotonic()-start,flush=True)
    if done[0]:break
   import numpy as np
   b.physics_observer=None
   arrays={k:v[:rec.count].cpu().numpy() for k,v in rec.buffers.items()};arrays['t']=np.arange(rec.count)*.001
   np.savez_compressed(run/'diagnostic_trace.npz',**arrays)
   bad=np.flatnonzero(arrays['online_failed']);land=np.flatnonzero(arrays['online_landed'])
   i=int(bad[0]) if len(bad) else -1
   forces=arrays['contact_force'][i] if i>=0 else np.zeros((len(b.contacts.body_names),3))
   report=dict(diagnostic_only=True,teacher_acceptance=False,source_sha256=sha(a.replay_state),
    failed=bool(len(bad)),first_failure_s=float(arrays['t'][i]) if i>=0 else None,
    first_landing_s=float(arrays['t'][land[0]]) if len(land) else None,
    nonwheel_force_peak_N=float(arrays['body_force'][1001:].max()),
    first_failure_contacts=[dict(body=name,force_N=forces[j].tolist()) for j,name in enumerate(b.contacts.body_names) if np.linalg.norm(forces[j])>5],
    last_time_s=float(arrays['t'][-1]),angle_deg=float(np.degrees(arrays['online_angle'][-1])))
  check_sources(run)
  assert all(sha(Path(__file__).parent/name)==h for name,h in sources.items()),'Teacher source changed during verification'
  write(run/'result.json',report);print('[TEACHER RESULT]',json.dumps(report),flush=True)
 except BaseException:
  (run/'error.txt').write_text(traceback.format_exc());raise
 finally:
  if env is not None:env.close()
  if app is not None:app.close()
if __name__=='__main__':main()
