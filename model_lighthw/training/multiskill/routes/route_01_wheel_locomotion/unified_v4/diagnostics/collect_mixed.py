"""所有64工况同时采样教师标签；仅数据收集，非行为验收或PPO。"""
import argparse,sys,traceback,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from launch import SPEC,CART,REPO,module_from_file,snapshot,write_json,now
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=650);p.add_argument('--action-noise',type=float,default=0.);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
write_json(out/'request.json',dict(created_at=now(),steps=a.steps,num_envs=168,ppo_updates=0,purpose='all-condition teacher labels, no behavioral acceptance',action_noise_std=a.action_noise))
snapshot(out);CART.write_cart_urdf(out/'assets/cart.urdf')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app;env=None
try:
 import numpy as np,torch,gymnasium as gym
 from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from isaaclab.utils.io import dump_yaml
 from support_teacher import SupportTeacher
 worker=module_from_file('_v4_collect_worker',ROOT/'scripts/worker.py')
 SPEC.register_task();cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
 cfg.scene.num_envs=168;cfg.seed=agent.seed=47;cfg.observations.policy.enable_corruption=False
 cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
 cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
 dump_yaml(str(out/'params/env.yaml'),cfg)
 env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
 with torch.inference_mode():obs,_=env.reset()
 runtime=worker.verify_runtime(env,out)
 if runtime['unique_condition_count']!=64:raise RuntimeError('All 64 conditions must participate from first frame')
 teacher=SupportTeacher(env);xs=[];ys=[];dones=[]
 base=env.unwrapped
 context=torch.stack((base.r1_slope_rad*180/torch.pi,base.r1_payload_kg,base.r1_towed_kg),-1).cpu().numpy()
 with torch.inference_mode():
  for step in range(a.steps):
   action=teacher(obs)
   xs.append(obs.cpu().numpy().copy());ys.append(action.cpu().numpy().copy())
   executed=action+torch.randn_like(action)*a.action_noise
   obs,reward,done,extra=env.step(executed)
   if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():raise RuntimeError('Non-finite teacher collection')
   dones.append(done.cpu().numpy().copy())
   if step%100==0:print('COLLECT_PROGRESS',step,'terminations',int(np.stack(dones).sum()),flush=True)
 np.savez_compressed(out/'teacher_data.npz',observation=np.stack(xs),action=np.stack(ys),valid_lengths=np.full(env.num_envs,a.steps),done=np.stack(dones),conditions=context)
 write_json(out/'collection.json',dict(finished_at=now(),steps=a.steps,samples=a.steps*env.num_envs,ppo_updates=0,terminations_per_env=np.stack(dones).sum(0).tolist(),condition_metrics=base.drain_metrics(),motor_metrics=base.r1_torque_window.drain(),behavior_acceptance='not_evaluated'))
 print('COLLECTION_COMPLETE',flush=True)
except BaseException as exc:
 write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
 import os
 sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
 if env is not None:env.close()
 app.close()
