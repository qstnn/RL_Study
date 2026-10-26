"""Phase-balanced supervised imitation, separate from PPO's on-policy buffer."""
import json,math
from pathlib import Path
import numpy as np
import torch
from .io import sha,write

class Demonstrations:
 def __init__(self,path,device):
  a=np.load(path,allow_pickle=False)
  obs=a['observation'];act=a['action']
  if obs.ndim!=2 or obs.shape[1]!=420 or act.shape!=(len(obs),16) or len(obs)<250:raise ValueError('Incomplete demonstration dimensions')
  if not np.isfinite(obs).all() or not np.isfinite(act).all() or np.max(np.abs(act))>math.tanh(3.)+1e-6:raise ValueError('Invalid demonstration actions')
  self.obs=torch.tensor(obs,device=device);self.action=torch.tensor(act,device=device)
  # The normalized float32 clock can reconstruct 0.4 s as 0.399999976.
  # Use policy ticks so the first executed command at tick 20 is not dropped.
  ticks=torch.round(self.obs[:,-12]*300).long();air=self.obs[:,-8]>.5;land=self.obs[:,-7]>.5
  masks=[(ticks>=20)&~air,air&~land,land&(ticks<100),land&(ticks>=100)]
  self.groups=[torch.where(mask)[0] for mask in masks]
  if any(len(x)==0 for x in self.groups):raise ValueError('Demonstration missing launch/flight/recovery/hold phase')
  self.source_sha256=sha(path)
 def sample(self,per_phase=64):
  ids=torch.cat([g[torch.randint(len(g),(per_phase,),device=g.device)] for g in self.groups])
  return self.obs[ids],self.action[ids]
 def loss(self,policy):
  obs,act=self.sample()
  # Latent loss avoids nearly vanishing tanh gradients for saturated motor targets.
  target=torch.atanh(act.clamp(-math.tanh(3.)+1e-6,math.tanh(3.)-1e-6))
  return (policy.latent_mean(obs)-target).square().mean()

def pretrain(dataset,output,device='cpu',steps=3000):
 from .policy import SquashedActorCritic
 torch.manual_seed(42);torch.set_num_threads(1)
 demo=Demonstrations(dataset,device)
 model=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu').to(device)
 initial=Path(__file__).parent/'teacher/bc_initial_actor.pt'
 model.load_state_dict(torch.load(initial,map_location=device,weights_only=True),strict=False)
 params=list(model.actor.parameters())+list(model.ground_actor.parameters());opt=torch.optim.Adam(params,lr=3e-4)
 records=[]
 for step in range(steps):
  opt.zero_grad(set_to_none=True);loss=demo.loss(model);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
  if (step+1)%200==0 or step==0 or step+1==steps:
   with torch.no_grad():
    prediction=model.act_inference(demo.obs)
    active=torch.round(demo.obs[:,-12]*300)>=20
    error=(prediction[active]-demo.action[active]).abs()
   row=dict(step=step+1,latent_loss=float(loss.detach()),active_action_MSE=float(error.square().mean()),active_action_max_error=float(error.max()),ignored_settle_frames=int((~active).sum()))
   records.append(row);print('[GUIDED BC]',json.dumps(row),flush=True)
 with torch.no_grad():model.log_std[:12].fill_(math.log(.03));model.log_std[12:].fill_(math.log(.02))
 state={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(('actor.','ground_actor.')) or k=='log_std'}
 output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.save(state,output/'actor.pt')
 report=dict(method='supervised teacher imitation, not PPO',steps=steps,seed=42,dataset_sha256=demo.source_sha256,
  initial_actor_sha256=sha(initial),actor_sha256=sha(output/'actor.pt'),phase_counts=[len(x) for x in demo.groups],curve=records,
  closed_loop_success='not_inferred_from_supervised_loss')
 write(output/'pretrain.json',report)
 return output/'actor.pt'

def auxiliary_update(alg,demo,completed_updates):
 from .spec import CONTRACT as C
 weight=C['demonstration_weight']*max(0.,1.-max(0,completed_updates-C['critic_warmup_updates'])/C['demonstration_decay_updates'])
 if weight<=0:return 0.,0.
 alg.optimizer.zero_grad(set_to_none=True);loss=demo.loss(alg.policy);(weight*loss).backward()
 torch.nn.utils.clip_grad_norm_(alg.policy.parameters(),1.);alg.optimizer.step()
 return float(loss.detach()),weight
