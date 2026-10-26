"""多个标准V5 Actor在完整16工况中并排冻结验证；无附加控制器，无PPO。"""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from launch import SPEC,CART,PROTOCOL,module_from_file,write_json,sha256,now
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
models=[path.resolve(strict=True) for path in a.checkpoint]
if len(PROTOCOL.cases())!=16:raise ValueError('Candidate batch layout requires the declared 16 cases')
cases=PROTOCOL.cases()*len(models)
write_json(out/'request.json',dict(created_at=now(),models=[dict(path=str(path),sha256=sha256(path)) for path in models],cases=cases,ppo_updates=0,standard_actor_only=True))
CART.write_cart_urdf(out/'assets/cart.urdf')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app;env=None
try:
 import gymnasium as gym,numpy as np,torch
 from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from isaaclab.utils.io import dump_yaml
 SPEC.register_task();worker=module_from_file('_v5_frozen_worker',ROOT/'scripts/worker.py')
 cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
 cfg.scene.num_envs=len(cases);cfg.seed=agent.seed=43;cfg.episode_length_s=65.;cfg.fixed_speed_mps=3.1
 cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,0.,0.;cfg.observations.policy.enable_corruption=False
 cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
 cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
 dump_yaml(str(out/'params/env.yaml'),cfg)
 env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
 class Runner:
  def __init__(self):
   self.actors=[];self.observations=[];self.actions=[]
   for path in models:
    saved=SPEC.checkpoint(path)
    actor=torch.nn.Sequential(torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16)).to(env.device).eval()
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in saved['model_state_dict'].items() if k.startswith('actor.')});self.actors.append(actor)
  def load(self,path,load_optimizer=False):assert not load_optimizer
  def get_inference_policy(self,device=None):
   def infer(obs):
    action=torch.cat([actor(obs[16*i:16*(i+1)]) for i,actor in enumerate(self.actors)])
    self.observations.append(obs.cpu().numpy());self.actions.append(action.cpu().numpy())
    if len(self.actions)%250==0:print('FROZEN_CANDIDATE_STEPS',len(self.actions),flush=True)
    return action
   return infer
 runner=Runner();worker.protocol.cases=lambda distance:cases
 request=dict(mode='evaluate',eval_distance=1.,eval_seeds=[43],seed=43,checkpoint=str(models[0]),command_profile='slew',headless=True)
 with torch.inference_mode():worker.evaluate(env,runner,request,out,app)
 obs=np.stack(runner.observations);act=np.stack(runner.actions)
 result=json.loads((out/'evaluation.json').read_text());summaries=[]
 for m,path in enumerate(models):
  rows=result['cases'][m*16:(m+1)*16];summary=dict(checkpoint=str(path),sha256=sha256(path),pass_count=sum(r['success'] for r in rows),complete_count=sum(r['complete'] for r in rows),cases=rows)
  summaries.append(summary);xs=[];ys=[]
  for i,r in enumerate(rows):
   r['source_checkpoint']=str(path);r['source_sha256']=sha256(path)
   if r['success'] and r['complete']:
    n=r['samples'];xs.append(obs[:n,m*16+i]);ys.append(act[:n,m*16+i])
  if xs:
   x=np.concatenate(xs);y=np.concatenate(ys)
   np.savez_compressed(out/f'reference_success_model{m}.npz',observation=x,action=y,successful=np.ones(len(x),dtype=np.bool_))
  write_json(out/f'model{m}_evaluation.json',summary)
  print('FROZEN_CANDIDATE_RESULT',str(path),summary['pass_count'],summary['complete_count'],flush=True)
 result['per_model']=summaries;result['mixed_checkpoint_note']='Per-case source_checkpoint is authoritative; top-level checkpoint is only an interface placeholder.'
 write_json(out/'evaluation.json',result)
 for path,expected in zip(models,json.loads((out/'request.json').read_text())['models']):assert sha256(path)==expected['sha256']
except BaseException as exc:
 write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
 import os
 sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
 if env is not None:env.close()
 app.close()
