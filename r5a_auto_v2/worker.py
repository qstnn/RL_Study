"""受单目录离线入口管理的 Isaac/PPO 子进程。"""
import argparse,json,os,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_auto_v2.io import ROOT,read,write,now,implementation,checkpoint,check_sources,sha
from r5a_auto_v2.spec import CONTRACT as C,runner_config,LEG_JOINTS,WHEELS

def make_runner(folder):
 import torch,numpy as np
 from rsl_rl.runners import OnPolicyRunner
 import rsl_rl.runners.on_policy_runner as runner_module
 from r5a_auto_v2.policy import SquashedActorCritic
 from r5a_auto_v2.update_guard import guarded_update
 runner_module.SquashedActorCritic=SquashedActorCritic
 class Runner(OnPolicyRunner):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.completed_updates=0;self.accepted_updates=0;self.rejected_streak=0;self.git_status_repos=[]
   # 新任务的中性均值对应站姿；所有动作之后完全由网络输出。
   with torch.no_grad():
    self.alg.policy.actor[-1].weight.mul_(.01);self.alg.policy.actor[-1].bias.zero_()
    self.alg.policy.log_std[:12].fill_(np.log(.25));self.alg.policy.log_std[12:].fill_(np.log(.10))
   update=self.alg.update
   def checked_update():
    losses,row=guarded_update(self.alg,update,C['rollout_kl_limit'])
    self.completed_updates+=1
    self.accepted_updates+=int(row['accepted'])
    self.rejected_streak=0 if row['accepted'] else self.rejected_streak+1
    row.update(completed_updates=self.completed_updates,accepted_updates=self.accepted_updates)
    with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    if self.rejected_streak>=5:raise RuntimeError('连续5批更新被KL/有限值检查拒绝；停止并回收审查资料')
    if any(v is None for v in row['losses'].values()):raise RuntimeError('非有限PPO损失；已恢复模型与Adam，停止审查')
    return losses
   self.alg.update=checked_update
  def save(self,path,infos=None):
   info=dict(contract=C,implementation_sha256=implementation(),completed_updates=self.completed_updates,
    accepted_updates=self.accepted_updates,rejected_streak=self.rejected_streak,
    learning_rate=self.alg.learning_rate,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
    python_rng=random.getstate(),numpy_rng=np.random.get_state(),physics_state_saved=False)
   super().save(path,info)
  def log(self,locs,*a,**kw):
   super().log(locs,*a,**kw)
   rewards=locs.get('rewbuffer',[]);lengths=locs.get('lenbuffer',[])
   row=dict(completed_updates=self.completed_updates,mean_reward=float(np.mean(rewards)) if len(rewards) else None,
    mean_episode_length=float(np.mean(lengths)) if len(lengths) else None,
    collection_seconds=locs['collection_time'],learning_seconds=locs['learn_time'])
   with (folder/'training_curve.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 return Runner

def runtime(env,run):
 import torch
 import isaaclab.sim as sim
 from pxr import Usd
 b=env.unwrapped;d=b.robot.data;model=read(run/'model.json')
 assert [env.num_obs,env.num_privileged_obs,env.num_actions]==[420,104,16]
 assert [b.robot.joint_names[j] for j in b.joint_ids]==list(LEG_JOINTS+WHEELS)
 torch.testing.assert_close(b.mass,torch.full_like(b.mass,model['unloaded_robot_mass_kg']),atol=.002,rtol=1e-4)
 base_id=b.robot.find_bodies('Base_link')[0][0]
 torch.testing.assert_close(d.body_com_pos_b[0,base_id].cpu(),torch.tensor(model['base_com_m']),atol=2e-4,rtol=1e-3)
 torch.testing.assert_close(b.robot.root_physx_view.get_inertias()[0,base_id].cpu().reshape(3,3),torch.tensor(model['base_inertia_kg_m2']),atol=2e-4,rtol=.005)
 efforts=[24. if 'wheel' in n else 48. for n in b.robot.joint_names]
 speeds=[37.7 if 'wheel' in n else 12.57 for n in b.robot.joint_names]
 torch.testing.assert_close(d.joint_effort_limits[0],d.joint_effort_limits.new_tensor(efforts))
 torch.testing.assert_close(d.joint_vel_limits[0],d.joint_vel_limits.new_tensor(speeds))
 lo=d.joint_pos_limits[0,b.leg_ids,0];hi=d.joint_pos_limits[0,b.leg_ids,1]
 write(run/'joint_bounds.json',dict(actual_low=lo.cpu().tolist(),actual_high=hi.cpu().tolist(),target_low=C['action_lower'],target_high=C['action_upper']))
 if not ((lo<=lo.new_tensor(C['action_lower'])).all() and (hi>=hi.new_tensor(C['action_upper'])).all()):raise ValueError('Action limits exceed actual URDF limits; see joint_bounds.json')
 collision=[];angular_caps=[];gyroscopic=[]
 for p in Usd.PrimRange(sim.get_current_stage().GetPrimAtPath('/World/envs/env_0/Robot')):
  a=p.GetAttribute('physxArticulation:enabledSelfCollisions')
  if a and a.HasAuthoredValueOpinion():collision.append(bool(a.Get()))
  cap=p.GetAttribute('physxRigidBody:maxAngularVelocity')
  if cap and cap.HasAuthoredValueOpinion():angular_caps.append(float(cap.Get()))
  gyro=p.GetAttribute('physxRigidBody:enableGyroscopicForces')
  if gyro and gyro.HasAuthoredValueOpinion():gyroscopic.append(bool(gyro.Get()))
 assert collision and all(collision)
 assert len(angular_caps)==len(b.robot.body_names) and all(abs(x-C['rigid_max_angular_velocity_deg_s'])<.01 for x in angular_caps)
 assert len(gyroscopic)==len(b.robot.body_names) and all(gyroscopic)
 obs,extras=env.get_observations();again,_=env.get_observations();torch.testing.assert_close(obs,again,rtol=0,atol=0)
 assert torch.isfinite(obs).all() and torch.isfinite(extras['observations']['critic']).all()
 # PPO holds old observations until process_env_step; env.step must not mutate them.
 saved_observation=obs.clone();env.step(torch.zeros(env.num_envs,16,device=env.device))
 torch.testing.assert_close(obs,saved_observation,rtol=0,atol=0)
 env.reset()
 write(run/'runtime_check.json',dict(dimensions=[env.num_obs,env.num_privileged_obs,env.num_actions],
  rigid_angular_caps_deg_s=angular_caps,gyroscopic_forces=gyroscopic,
  mass_kg=float(b.mass[0]),effort_limits=efforts,velocity_limits=speeds,self_collisions=collision,
  observation_ownership_check=True,action_limits_within_URDF=True,physics_dt=b.physics_dt,policy_dt=b.step_dt,no_reference_trajectory=True,
  gravity=list(b.cfg.sim.gravity),task=C['version'],behavior_acceptance='not_evaluated'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);run=p.parse_args().run.resolve()
 app=env=runner=None;stage='inputs'
 def mark(value):
  nonlocal stage
  stage=value;write(run/'worker_status.json',dict(stage=stage,pid=os.getpid(),updated_at=now()))
 try:
  req=read(run/'request.json');check_sources(run);mark('app_start')
  from isaaclab.app import AppLauncher
  app=AppLauncher(headless=req['headless'],enable_cameras=req.get('video',False),device=req['device']).app
  import gymnasium as gym,numpy as np,torch
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
  from isaaclab.utils.io import dump_yaml
  import importlib.metadata as package_metadata,rsl_rl
  import shutil
  versions={}
  for name in ('torch','numpy','gymnasium','rsl-rl-lib','isaacsim','isaaclab'):
   try:versions[name]=package_metadata.version(name)
   except package_metadata.PackageNotFoundError:versions[name]='not registered'
  dependencies={}
  dep_root=Path(rsl_rl.__file__).resolve().parent
  for source in dep_root.rglob('*.py'):
   dest=run/'dependency_source/rsl_rl'/source.relative_to(dep_root);dest.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(source,dest);dependencies[str(source.relative_to(dep_root))]=sha(source)
  write(run/'environment.json',dict(python=sys.version,executable=sys.executable,versions=versions,
   cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),rsl_rl_path=str(dep_root),rsl_rl_sources=dependencies))
  from r5a_auto_v2.env import AutoFlipEnv,AutoFlipCfg
  from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a.appearance import sky_path
  class Wrapper(RslRlVecEnvWrapper):
   def __init__(self,*a,**k):
    super().__init__(*a,**k);self.num_privileged_obs=gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
  cfg=AutoFlipCfg();cfg.scene.num_envs=req['num_envs'];cfg.seed=req['seed'];cfg.sim.device=req['device']
  cfg.scene.robot.spawn.asset_path=str(run/'input/robot.urdf');cfg.scene.robot.spawn.usd_dir=str(run/'asset_cache');cfg.scene.robot.spawn.usd_file_name='auto.usd'
  if req.get('video') or not req['headless']:cfg.scene.sky_light.spawn.texture_file=sky_path()
  agent=runner_config();agent.update(seed=req['seed'],device=req['device'])
  dump_yaml(str(run/'params/env.yaml'),cfg);write(run/'params/agent.json',agent)
  mark('environment_creation');env=Wrapper(AutoFlipEnv(cfg,render_mode='rgb_array' if req.get('video') else None),clip_actions=1.)
  b=env.unwrapped
  def sink(rows):
   with (run/'episodes.jsonl').open('a') as f:
    for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
  mark('runtime_check');runtime(env,run)
  # Runtime probes are not training episodes; attach the sink only after reset.
  b.episode_count.zero_();b.last_reports=[];b.episode_sink=sink
  runner=make_runner(run)(env,agent,log_dir=str(run/'training'),device=req['device'])
  runner.logger_type='tensorboard'
  start=0
  if req.get('checkpoint'):
   saved=checkpoint(req['checkpoint']);info=runner.load(req['checkpoint'],load_optimizer=req['mode']=='train')
   start=info['completed_updates'];runner.completed_updates=start;runner.current_learning_iteration=start
   runner.accepted_updates=info['accepted_updates'];runner.rejected_streak=info['rejected_streak']
   runner.alg.learning_rate=info['learning_rate']
   if req['mode']=='train':
    torch.set_rng_state(info['torch_rng'].cpu());torch.cuda.set_rng_state_all(info['cuda_rng'])
    random.setstate(info['python_rng']);np.random.set_state(info['numpy_rng'])
   write(run/'resume.json',dict(checkpoint_sha256=sha(req['checkpoint']),completed_updates=start,
    optimizer_states=len(runner.alg.optimizer.state),physics_state_restored=False))
  mark(req['mode'])
  if req['mode']=='train':
   runner.save(str(run/'initial.pt'))
   runner.learn(num_learning_iterations=req['updates'],init_at_random_ep_len=False)
   final=run/'final.pt';runner.save(str(final));info=checkpoint(final)['infos']
   assert info['completed_updates']==start+req['updates']
   result=dict(status='completed',mode='train',checkpoint='final.pt',checkpoint_sha256=sha(final),
    start_completed_updates=start,completed_updates=info['completed_updates'],accepted_updates=info['accepted_updates'],ppo_updates=req['updates'],formal_behavior_acceptance='requires_review')
  else:
   from r5a_auto_v2.record import Recorder
   policy=runner.get_inference_policy(device=env.device);reports=[]
   for seed in req['eval_seeds']:
    env.seed(seed)
    for k in range(req['eval_episodes']):
     with torch.inference_mode():
      obs,_=env.reset();trace=Recorder(b,run,f'seed{seed}_case{k}',req.get('video',False));steps=0;reason='interrupted'
      try:
       for _ in range(b.max_episode_length):
        if not app.is_running():raise RuntimeError('Playback closed')
        begin=time.monotonic();act=policy(obs);_,feedback=env.get_observations();trace.policy_step(obs,act,feedback['observations']['critic'])
        obs,_,done,_=env.step(act);steps+=1
        if done[0]:reason='terminated' if bool(b.reset_terminated[0]) else 'timeout';break
        if not req['headless']:time.sleep(max(0,b.step_dt-(time.monotonic()-begin)))
      finally:report=trace.finish(reason,steps)
     report.update(seed=seed,episode=k);reports.append(report)
     print('[AUTOFLIP REVIEW]',json.dumps(report),flush=True)
   result=dict(status='completed',mode=req['mode'],cases=reports,ppo_updates=0,
    success_rate=sum(x['success'] for x in reports)/len(reports),formal_behavior_acceptance='requires_review')
   write(run/'evaluation.json',result)
  check_sources(run);write(run/'result.json',result);mark('completed')
 except BaseException as e:
  write(run/'worker_error.json',dict(stage=stage,error=str(e),traceback=traceback.format_exc(),failed_at=now()))
  traceback.print_exc();raise
 finally:
  if runner is not None and runner.writer is not None:runner.writer.close()
  if env is not None:env.close()
  if app is not None:app.close()
if __name__=='__main__':main()
