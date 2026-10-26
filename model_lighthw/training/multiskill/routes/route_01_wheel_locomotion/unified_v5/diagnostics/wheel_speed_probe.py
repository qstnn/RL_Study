"""冻结旧Actor的轮速诊断；所有覆盖仅本进程生效，PPO/梯度更新均为0。"""
import argparse,inspect,json,sys,textwrap,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT5=HERE.parent
ROOT=ROOT5.with_name('unified_v4')
sys.path.insert(0,str(ROOT/'scripts'))
from launch import SPEC,CART,module_from_file,write_json,now,sha256
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['cap','feedback','confirm','full'],default='cap');a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
models={'init':ROOT/'initialization/selected/model_init.pt','50':ROOT/'runs/2026-10-19_09-32-20.793780/training/model_49.pt'}
if a.mode=='cap':
 trials=[dict(model=m,slope_deg=s,clip=clip,damping=2.,feedforward_rad_s=0.) for m in models for s in (25.,0.,-25.) for clip in (37.,50.)]
elif a.mode=='feedback':
 trials=[dict(model=m,slope_deg=25.,clip=50.,damping=d,feedforward_rad_s=f) for m in models for d,f in ((2.,0.),(3.,0.),(4.,0.),(2.,1.),(2.,2.),(2.,3.))]
elif a.mode=='confirm':
 trials=[dict(model=m,slope_deg=s,clip=50.,damping=2.,feedforward_rad_s=1.,flat_feedforward_rad_s=f) for m in models for s,f in ((25.,0.),(0.,0.),(0.,.5),(-25.,0.))]
else:
 from launch import PROTOCOL
 trials=[dict(model=m,**case,clip=50.,damping=2.,feedforward_rad_s=1.,flat_feedforward_rad_s=.5) for m in models for case in PROTOCOL.cases()]
cases=[dict(slope_deg=t['slope_deg'],payload_kg=t.get('payload_kg',0.),towed_kg=t.get('towed_kg',0.),speed=t.get('speed',3.1),goal_distance_m=1.) for t in trials]
request=dict(created_at=now(),trials=trials,checkpoints={k:dict(path=str(v),sha256=sha256(v)) for k,v in models.items()},ppo_updates=0,observation_noise=False,action_sampling=False,wheel_solver_velocity_limit=60.,wheel_actuator_velocity_limit=60.,wheel_rigid_cap_deg_s=4000.,note='Same batch clip-only comparison uses raised numerical caps for both clip37 and clip50. Frozen policy means, no robustness claim. Feedforward if nonzero is a DIAGNOSTIC commanded-velocity gated output correction, not trained Actor.')
write_json(out/'request.json',request);CART.write_cart_urdf(out/'assets/cart.urdf')
(out/'source').mkdir();(out/'source'/'wheel_speed_probe.py').write_text(Path(__file__).read_text())
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app;env=None
try:
 import torch,numpy as np,gymnasium as gym
 from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from isaaclab.utils.io import dump_yaml
 worker=module_from_file('_r1_wheel_diagnostic_worker',ROOT/'scripts/worker.py')
 SPEC.register_task()
 from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw import wheel_rigid_cap_env_cfg as rigid
 namespace=dict(vars(rigid));source=inspect.getsource(rigid.spawn_lighthw_wheel_rigid_cap.__wrapped__) if hasattr(rigid.spawn_lighthw_wheel_rigid_cap,'__wrapped__') else inspect.getsource(rigid.spawn_lighthw_wheel_rigid_cap)
 # inspect follows __wrapped__ for the decorated spawner, retaining the clone decorator.
 source=textwrap.dedent(source).replace('3000.','4000.').replace('3000deg/s','4000deg/s')
 if '@sim_utils.clone' not in source:source='@sim_utils.clone\n'+source
 exec(compile(source,'<isolated_wheel_spawner>','exec'),namespace)
 cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
 cfg.scene.num_envs=len(cases);cfg.seed=agent.seed=43;cfg.episode_length_s=65.;cfg.fixed_speed_mps=3.1
 cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,0.,0.
 cfg.observations.policy.enable_corruption=False
 cfg.scene.robot.spawn.func=namespace['spawn_lighthw_wheel_rigid_cap']
 cfg.scene.robot.actuators['wheel'].velocity_limit=60.;cfg.scene.robot.actuators['wheel'].velocity_limit_sim=60.
 cfg.actions.joint_vel.clip={'.*_wheel_joint':(-50.,50.)}
 cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
 cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
 audit_source=textwrap.dedent(inspect.getsource(rigid.audit_runtime_caps)).replace('3000.','4000.')
 exec(compile(audit_source,'<isolated_wheel_caps_audit>','exec'),namespace);rigid.audit_runtime_caps=namespace['audit_runtime_caps']
 (out/'source'/'isolated_spawner.py').write_text(source+'\n'+audit_source)
 dump_yaml(str(out/'params/env.yaml'),cfg)
 env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
 base=env.unwrapped;robot=base.scene['robot'];term=base.action_manager.get_term('joint_vel');wheel_ids=term._joint_ids
 old_verify=worker.verify_runtime
 def verify(env,run):
  result=old_verify(env,run)
  clips=torch.tensor([t['clip'] for t in trials],device=env.device)[:,None]
  term._clip[:,:,0]=-clips;term._clip[:,:,1]=clips
  robot.actuators['wheel'].damping[:]=torch.tensor([t['damping'] for t in trials],device=env.device)[:,None]
  solver=robot.root_physx_view.get_dof_max_velocities()[:,wheel_ids]
  assert torch.allclose(solver,torch.full_like(solver,60.))
  assert torch.allclose(robot.actuators['wheel'].velocity_limit,torch.full_like(robot.actuators['wheel'].velocity_limit,60.))
  actual=dict(action_clip=term._clip.cpu().tolist(),wheel_damping=robot.actuators['wheel'].damping.cpu().tolist(),wheel_actuator_velocity_limit=robot.actuators['wheel'].velocity_limit.cpu().tolist(),wheel_solver_velocity_limit=solver.cpu().tolist(),wheel_scale=term._scale,checkpoint_policy_weights_unchanged=True)
  write_json(out/'runtime_overrides.json',actual)
  return result
 worker.verify_runtime=verify
 class Runner:
  def __init__(self):
   self.actors={};self.indices={};self.steps=0;self.extra=[]
   for name,path in models.items():
    saved=SPEC.checkpoint(path)
    actor=torch.nn.Sequential(torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16)).to(env.device).eval()
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in saved['model_state_dict'].items() if k.startswith('actor.')})
    self.actors[name]=actor;self.indices[name]=torch.tensor([i for i,t in enumerate(trials) if t['model']==name],device=env.device)
  def load(self,path,load_optimizer=False):assert not load_optimizer
  def get_inference_policy(self,device=None):
   def infer(obs):
    actions=torch.zeros((len(cases),16),device=obs.device)
    for name,actor in self.actors.items():
     ids=self.indices[name];actions[ids]=actor(obs[ids])
    raw=actions.clone();command=base.command_manager.get_command('base_velocity')[:,0]
    # Only command, not true speed/terrain, is used; clamp correction smoothly to zero at stop.
    gain=torch.tensor([t['feedforward_rad_s'] for t in trials],device=env.device)
    # 300 columns are term-major history; newest projected gravity is 27:30.
    # Only existing IMU and operator command enter this diagnostic correction.
    gravity=obs[:,27:30]
    uphill_gate=((-gravity[:,0]-.12)/.18).clamp(0.,1.)
    speed_gate=((command-2.)/1.1).clamp(0.,1.)
    unloaded=(obs[:,-2:].abs()<1.e-6).all(-1)
    gate=uphill_gate*speed_gate*unloaded
    flat_gate=(1.-gravity[:,0].abs()/.12).clamp(0.,1.)*speed_gate*unloaded
    flat_gain=torch.tensor([t.get('flat_feedforward_rad_s',0.) for t in trials],device=env.device)
    correction=gain*gate+flat_gain*flat_gate
    actions[:,12:]+=correction[:,None]/5.
    self.extra.append(dict(observation=obs.cpu().numpy(),raw_action=raw.cpu().numpy(),applied_action=actions.cpu().numpy(),wheel_vel=robot.data.joint_vel[:,wheel_ids].cpu().numpy(),command=command.cpu().numpy(),correction_gate=gate.cpu().numpy(),flat_correction_gate=flat_gate.cpu().numpy(),wheel_correction_rad_s=correction.cpu().numpy()))
    self.steps+=1
    if self.steps%250==0:print('WHEEL_PROBE_STEPS',self.steps,flush=True)
    return actions
   return infer
 runner=Runner();worker.protocol.cases=lambda distance:cases
 request=dict(mode='evaluate',eval_distance=1.,eval_seeds=[43],seed=43,checkpoint=str(models['init']),command_profile='slew',headless=True)
 with torch.inference_mode():worker.evaluate(env,runner,request,out,app)
 np.savez_compressed(out/'actor_trace.npz',**{k:np.stack([r[k] for r in runner.extra]) for k in runner.extra[0]})
 result=json.loads((out/'evaluation.json').read_text())
 for t,r in zip(trials,result['cases']):
  r['diagnostic_source_model']=t['model'];r['diagnostic_source_checkpoint']=str(models[t['model']]);r['diagnostic_source_sha256']=sha256(models[t['model']])
 result['diagnostic_mixed_checkpoint_note']='Per-case source checkpoint is authoritative; the original worker top-level checkpoint is only an interface placeholder.'
 write_json(out/'evaluation.json',result)
 summary=[dict(**t,success=r['success'],duration_s=r['observed']['duration_s'],termination=r['observed']['termination_causes'],metrics=r.get('metrics',{}),failed_checks=r['failed_checks']) for t,r in zip(trials,result['cases'])]
 write_json(out/'diagnostic_summary.json',dict(finished_at=now(),trials=summary,ppo_updates=0,weights_changed=False))
 for row in summary:print('WHEEL_PROBE_RESULT',json.dumps(row),flush=True)
except BaseException as exc:
 write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
 import os
 sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
 if env is not None:env.close()
 app.close()
