"""全协议支撑教师诊断；记录学生可见观测和特权教师标签，PPO0。"""
import argparse,sys,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launch import SPEC,CART,REPO,module_from_file,snapshot,write_json,now
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--friction',type=float,default=4.);p.add_argument('--short',type=float);p.add_argument('--tow-damping-grid',action='store_true');a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
write_json(out/'request.json',dict(created_at=now(),controller='privileged_support_teacher',friction=a.friction,short_s=a.short,ppo_updates=0,deployable=False));snapshot(out);CART.write_cart_urdf(out/'assets/cart.urdf')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app;env=None
try:
 import numpy as np,torch,gymnasium as gym
 from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from isaaclab.utils.io import dump_yaml
 from support_teacher import SupportTeacher
 worker=module_from_file('_v4_teacher_worker',ROOT/'scripts/worker.py')
 SPEC.register_task();cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
 cfg.scene.num_envs=16;cfg.seed=agent.seed=43;cfg.episode_length_s=65.;cfg.fixed_speed_mps=.3
 cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,15.,0.;cfg.observations.policy.enable_corruption=False
 cfg.ground_friction=a.friction
 cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
 cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
 dump_yaml(str(out/'params/env.yaml'),cfg)
 env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
 if a.tow_damping_grid:
  cases=[dict(slope_deg=25.,payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.) for _ in range(16)]
  worker.protocol.cases=lambda distance=1.:cases
  values=torch.tensor([3.,6.,9.,12.],device=env.device).repeat_interleave(4)
  env.unwrapped.scene['robot'].actuators['joint'].damping[:]=values[:,None]
  write_json(out/'damping_grid.json',dict(per_environment_leg_kd=values.cpu().tolist(),wheel_kd=2.,teacher='same controller and physical parameters except leg damping'))
 class Runner:
  def __init__(self):self.teacher=SupportTeacher(env);self.obs=[];self.actions=[]
  def load(self,path,load_optimizer=False):assert not load_optimizer
  def get_inference_policy(self,device=None):
   def infer(obs):
    action=self.teacher(obs)
    self.obs.append(obs.detach().cpu().numpy().copy());self.actions.append(action.detach().cpu().numpy().copy())
    if len(self.obs)%250==0:print('TEACHER_PROGRESS',len(self.obs),flush=True)
    return action
   return infer
 runner=Runner()
 if a.short:
  original=worker.protocol.max_steps;worker.protocol.max_steps=lambda c:min(original(c),round(a.short/.02))
 request=dict(mode='evaluate',eval_distance=1.,eval_seeds=[43],seed=43,checkpoint=str(ROOT/'diagnostics/support_teacher.py'),command_profile='slew',headless=True)
 with torch.inference_mode():worker.evaluate(env,runner,request,out,app)
 result=json.loads((out/'evaluation.json').read_text())
 np.savez_compressed(out/'teacher_data.npz',observation=np.stack(runner.obs),action=np.stack(runner.actions),valid_lengths=np.array([c['samples'] for c in result['cases']]))
 write_json(out/'teacher_contract.json',dict(deployable=False,privileged_teacher=True,actor_labels_only=True,physics_modified_by_teacher=False,complete=result['complete_count'],pass_count=result['pass_count'],finished_at=now()))
 print('TEACHER_RESULT',result['complete_count'],result['pass_count'],flush=True)
except BaseException as exc:
 write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
 import os
 sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
 if env is not None:env.close()
 app.close()
