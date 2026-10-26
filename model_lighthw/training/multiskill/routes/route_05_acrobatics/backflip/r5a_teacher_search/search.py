"""Batched CEM search in unchanged Isaac PhysX; final verification is separate."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_auto_v6.io import prepare_run,write,sha,check_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
 p.add_argument('--population',type=int,default=32);p.add_argument('--generations',type=int,default=20)
 p.add_argument('--seconds',type=float,default=1.4);p.add_argument('--sigma',type=float,default=.35);p.add_argument('--seed',type=int,default=20261022)
 p.add_argument('--initial',type=Path);p.add_argument('--family',choices=('latent','actions'),default='latent');a=p.parse_args();run=a.output.resolve()
 prepare_run(run,dict(mode='teacher_cem',**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},ppo_updates=0))
 import shutil
 sources={}
 for name in ('search.py','control.py','control_actions.py'):
  source=Path(__file__).parent/name;dest=run/'teacher_source'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest);sources[name]=sha(source)
 write(run/'teacher_source_manifest.json',sources)
 app=env=None
 try:
  from isaaclab.app import AppLauncher
  app=AppLauncher(headless=True,device=a.device).app
  import torch,numpy as np,gymnasium as gym
  torch.set_num_threads(1)
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
  from r5a_auto_v6.env import AutoFlipCfg,AutoFlipEnv
  from r5a_auto_v6.worker import runtime
  from r5a_auto_v6.policy import SquashedActorCritic
  from r5a_teacher_search.control import Teacher,DIM,score
  if a.family=='actions':
   from r5a_teacher_search.control_actions import Teacher
  class Probe(AutoFlipEnv):
   def _get_dones(self):
    super()._get_dones()
    # Search keeps failures latched until batch end; strict full-horizon probe uses production env.
    return torch.zeros(self.num_envs,dtype=torch.bool,device=self.device),torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
   def _get_rewards(self):return torch.zeros(self.num_envs,device=self.device)
  class Wrapper(RslRlVecEnvWrapper):
   def __init__(self,*args,**kw):
    super().__init__(*args,**kw);self.num_privileged_obs=gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
  cfg=AutoFlipCfg();cfg.scene.num_envs=a.population;cfg.seed=43;cfg.sim.device=a.device
  cfg.scene.robot.spawn.asset_path=str(run/'input/robot.urdf');cfg.scene.robot.spawn.usd_dir=str(run/'asset_cache');cfg.scene.robot.spawn.usd_file_name='robot.usd'
  env=Wrapper(Probe(cfg),clip_actions=1.);runtime(env,run);b=env.unwrapped
  data=json.loads((run/'runtime_check.json').read_text());data.update(no_reference_trajectory=False,search_control='frozen actor plus optimized action residual',search_termination='failures remain latched; no early reset',acceptance='requires independent full 6s rollout');write(run/'runtime_check.json',data)
  actor=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu').to(a.device).eval().requires_grad_(False)
  actor.load_state_dict(torch.load(ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v6/bootstrap/actor_1201.pt',map_location=a.device,weights_only=True),strict=False)
  rng=np.random.default_rng(a.seed);mean=np.zeros(DIM);std=np.full(DIM,a.sigma);best=None;best_score=-1e9
  if a.initial:mean=np.asarray(json.loads(a.initial.read_text())['parameters']);best=mean.copy()
  elites=max(4,a.population//6);start=time.monotonic()
  with torch.inference_mode():
   for gen in range(a.generations):
    population=np.clip(mean+std*rng.normal(size=(a.population,DIM)),-3,3)
    population[0]=mean;population[1]=best if best is not None else 0.;population[2]=0.
    teacher=Teacher(actor,population);obs,_=env.reset()
    for step in range(round(a.seconds/.02)):
     action=teacher(obs,b);obs,_,_,_=env.step(action)
    scores=score(b.state,b.current_metrics).cpu().numpy();order=np.argsort(scores)[::-1]
    rows=[]
    for i in range(a.population):
     row=dict(generation=gen,candidate=i,score=float(scores[i]),parameters=population[i].tolist(),**b.state.report(i));rows.append(row)
    with (run/'candidates.jsonl').open('a') as f:
     for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    j=int(order[0]);r=rows[j]
    if scores[j]>best_score:
     best_score=float(scores[j]);best=population[j].copy();write(run/'best.json',dict(**r,controller_family=a.family,acceptance='search_candidate_not_validated',seconds=a.seconds))
    chosen=population[order[:elites]];mean=.3*mean+.7*chosen.mean(0)
    std=np.maximum(.10,.3*std+.7*chosen.std(0))
    write(run/'optimizer.json',dict(generation=gen,mean=mean.tolist(),std=std.tolist(),best_score=best_score,completed_candidates=(gen+1)*a.population))
    print('[CEM]',gen+1,'/',a.generations,'best',round(best_score,4),'angle',round(r['angle']*180/np.pi,2),'rise',round(r['flight_rise'],3),'flight',round(r['flight_time'],3),'H',round(r['release_H'],3),'failed',r['failed'],'landed',r['landed'],'seconds',round(time.monotonic()-start,1),flush=True)
  check_sources(run)
  assert all(sha(Path(__file__).parent/name)==h for name,h in sources.items()),'Search source changed'
  write(run/'result.json',dict(status='search_completed',generations=a.generations,candidates=a.population*a.generations,best_score=best_score,teacher_success_verified=False,ppo_updates=0))
 except BaseException:
  (run/'error.txt').write_text(traceback.format_exc());raise
 finally:
  if env is not None:env.close()
  if app is not None:app.close()
if __name__=='__main__':main()
