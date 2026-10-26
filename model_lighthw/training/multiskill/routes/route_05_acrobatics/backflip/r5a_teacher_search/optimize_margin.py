"""Batched CEM search in unchanged Isaac PhysX; final verification is separate."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5a_auto_v6.io import prepare_run,write,sha,check_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
 p.add_argument('--population',type=int,default=32);p.add_argument('--generations',type=int,default=20)
 p.add_argument('--seconds',type=float,default=1.4);p.add_argument('--sigma',type=float,default=.35);p.add_argument('--seed',type=int,default=20261022)
 p.add_argument('--initial',type=Path);p.add_argument('--family',choices=('split_bound',),default='split_bound');a=p.parse_args();run=a.output.resolve()
 prepare_run(run,dict(mode='teacher_cem',**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},ppo_updates=0))
 import shutil
 sources={}
 for name in ('optimize_margin.py','control.py','control_kick.py','control_split.py','control_split_bound.py','kick_reference.json','open_reference.json'):
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
  from r5a_teacher_search.control_split_bound import Teacher,DIM,score
  class Probe(AutoFlipEnv):
   def __init__(self,*args,**kw):
    super().__init__(*args,**kw);self.failed_before_launch=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    self.H_history=torch.zeros(self.num_envs,round(a.seconds/.001)+21,3,device=self.device)
   def momentum(self,m):
    h,inertia=super().momentum(m);self.measured_H=h;return h,inertia
   def _advance(self,tick):
    old=self._protocol_tick;super()._advance(tick)
    if tick==old:return
    if hasattr(self,'failed_before_launch'):self.failed_before_launch|=self.state.failed&~self.state.launched
    if hasattr(self,'H_history'):
     k=int(self.episode_physics_steps[0]);self.H_history[:,k]=self.measured_H
   def physics_screen(self):
    starts=torch.round(self.state.release_t/.001).long()+2
    ends=torch.where(self.state.landed,torch.round(self.state.landing_t/.001).long(),self.episode_physics_steps)-2
    starts=starts.clamp(0,self.H_history.shape[1]-1)
    ids=torch.arange(self.num_envs,device=self.device);h0=self.H_history[ids,starts]
    times=torch.arange(self.H_history.shape[1],device=self.device)
    mask=(times[None]>=starts[:,None])&(times[None]<ends[:,None])
    drift=torch.where(mask,(self.H_history-h0[:,None]).norm(dim=-1),0.).amax(-1)
    limit=torch.maximum(h0.norm(dim=-1)*.05,torch.full((self.num_envs,),.02,device=self.device))
    checked=self.state.launched&((ends-starts)>=21)
    return checked,checked&(drift<=limit),drift,limit
   def _reset_idx(self,ids):
    super()._reset_idx(ids)
    if hasattr(self,'failed_before_launch'):self.failed_before_launch[ids]=False
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
  rng=np.random.default_rng(a.seed);mean=np.zeros(DIM);std=np.full(DIM,a.sigma);std[:24]*=.35;best=None;best_score=-1e9
  if a.initial:mean=np.asarray(json.loads(a.initial.read_text())['parameters']);best=mean.copy()
  elites=max(4,a.population//6);start=time.monotonic();initial=mean.copy();cov=np.diag(std**2)
  with torch.inference_mode():
   for gen in range(a.generations):
    population=np.clip(rng.multivariate_normal(mean,cov,size=a.population),-3,3)
    population[-8:]=np.clip(mean+rng.normal(size=(8,DIM))*a.sigma*2,-3,3)
    population[:,26:30]=np.clip(population[:,26:30],-1,1)
    population[0]=mean;population[1]=best if best is not None else 0.;population[2]=initial;population[2,24]=-.8;population[3]=initial;population[3,24]=0.;population[4]=initial;population[4,24]=.8;
    teacher=Teacher(actor,population);obs,_=env.reset()
    for step in range(round(a.seconds/.02)):
     action=teacher(obs,b);obs,_,_,_=env.step(action)
    checked,physics_ok,drift,limit=b.physics_screen()
    progress=(b.state.angle/(2*np.pi)).clamp(0,1)
    upright=torch.exp(-(b.state.landing_tilt/.35).square())
    scores=(2.5*progress+.8*upright*progress+.3*(b.state.flight_time/.5).clamp(0,1)+.3*(b.state.flight_rise/.3).clamp(0,1)+.15*(b.state.release_H/7).clamp(0,1)
     -.3*(b.state.axis_error/.35).clamp(0,4)-.5*((b.state.angle-2*np.pi).clamp_min(0)/.5)
     -.75*b.state.failed.float()+3*(~b.state.failed&b.state.air_validated).float()*progress.square()
     +1.5*progress*torch.exp(-(b.current_metrics['tilt']/.3).square()-(b.current_metrics['com_vel'][:,:2].norm(dim=-1)/.3).square()-((b.current_metrics['com'][:,:2]-b.state.landing_xy).norm(dim=-1)/.15).square())
     +3*torch.exp(-((b.state.angle-2*np.pi)/.12).square())*(~b.state.failed).float()+.8*torch.exp(-(b.state.axis_error/.2).square())*progress
     +4*(b.state.qualified_landing&~b.state.failed).float()+5*b.state.stable_time.clamp(0,1)+20*b.state.success.float()
     -10*b.failed_before_launch.float()-5*(b.state.air_validated&~physics_ok).float()).cpu().numpy();order=np.argsort(scores)[::-1]
    rows=[]
    for i in range(a.population):
     row=dict(generation=gen,candidate=i,score=float(scores[i]),parameters=population[i].tolist(),failed_before_launch=bool(b.failed_before_launch[i]),momentum_checked=bool(checked[i]),physics_consistent=bool(physics_ok[i]),H_drift=float(drift[i]),H_limit=float(limit[i]),**b.state.report(i));rows.append(row)
    with (run/'candidates.jsonl').open('a') as f:
     for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    j=int(order[0]);r=rows[j]
    if scores[j]>best_score:
     best_score=float(scores[j]);best=population[j].copy();write(run/'best.json',dict(**r,controller_family=a.family,acceptance='search_candidate_not_validated',seconds=a.seconds))
    valid=[i for i in order if not rows[i]['failed'] and rows[i]['physics_consistent']]
    qualified=[i for i in valid if rows[i]['qualified_landing']]
    safe=qualified if qualified else valid
    chosen=population[(safe if safe else order.tolist())[:elites]]
    mean=.70*best+.30*chosen.mean(0)
    cov=.5*cov+.5*(np.cov(chosen,rowvar=False) if len(chosen)>1 else np.diag(std**2))+np.diag(np.r_[np.full(24,.00015),np.full(DIM-24,.0004)])
    std=np.sqrt(np.diag(cov))
    write(run/'optimizer.json',dict(generation=gen,mean=mean.tolist(),std=std.tolist(),best_score=best_score,completed_candidates=(gen+1)*a.population))
    print('[CEM]',gen+1,'/',a.generations,'best',round(best_score,4),'angle',round(r['angle']*180/np.pi,2),'rise',round(r['flight_rise'],3),'flight',round(r['flight_time'],3),'H',round(r['release_H'],3),'physics',r['physics_consistent'],'H_drift',round(r['H_drift'],4),'failed',r['failed'],'landed',r['landed'],'seconds',round(time.monotonic()-start,1),flush=True)
  check_sources(run)
  assert all(sha(Path(__file__).parent/name)==h for name,h in sources.items()),'Search source changed'
  write(run/'result.json',dict(status='search_completed',generations=a.generations,candidates=a.population*a.generations,best_score=best_score,teacher_success_verified=False,ppo_updates=0))
 except BaseException:
  (run/'error.txt').write_text(traceback.format_exc());raise
 finally:
  if env is not None:env.close()
  if app is not None:app.close()
if __name__=='__main__':main()
