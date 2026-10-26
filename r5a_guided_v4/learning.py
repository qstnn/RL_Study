"""有来源的分相位模仿；纠偏样本与PPO on-policy buffer完全分开。"""
import collections,math
from pathlib import Path
import numpy as np
import torch
from .io import sha,write
from .spec import CONTRACT as C

ACTOR_PREFIXES=('actor.','ground_actor.','landing_actor.')

def actor_state(model):
 return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(ACTOR_PREFIXES) or k=='log_std'}

def load_initial(model,path):
 """只迁移显式模仿初值；旧两头Actor的空中权重复制为独立恢复头，不恢复Critic/Adam。"""
 state=torch.load(path,map_location=next(model.parameters()).device,weights_only=True)
 if 'model_state_dict' in state:raise ValueError('Use a supervised actor, not an older PPO checkpoint')
 state=dict(state)
 for k,v in list(state.items()):
  if k.startswith('actor.') and 'landing_actor.'+k[6:] not in state:state['landing_actor.'+k[6:]]=v.clone()
 expected=set(actor_state(model))
 if set(state)!=expected:raise ValueError('Unexpected supervised actor keys')
 for k,v in state.items():
  if not torch.isfinite(v).all() or v.shape!=model.state_dict()[k].shape:raise ValueError('Invalid actor tensor: '+k)
 torch.nn.Module.load_state_dict(model,state,strict=False)

class Demonstrations:
 def __init__(self,path,device):
  with np.load(path,allow_pickle=False) as a:obs=a['observation'];act=a['action']
  if obs.ndim!=2 or obs.shape[1]!=420 or act.shape!=(len(obs),16) or len(obs)<250:raise ValueError('Incomplete demonstration dimensions')
  if not np.isfinite(obs).all() or not np.isfinite(act).all() or np.max(np.abs(act))>math.tanh(3.)+1e-6:raise ValueError('Invalid demonstration actions')
  self.obs=torch.tensor(obs,device=device);self.action=torch.tensor(act,device=device)
  ticks=torch.round(self.obs[:,-12]*300).long();air=self.obs[:,-8]>.5;land=self.obs[:,-7]>.5
  from .landing import observation_prelanding
  late=observation_prelanding(self.obs)
  masks=[(ticks>=20)&~air,air&~land&~late,late,land&(ticks<100),land&(ticks>=100)]
  self.groups=[torch.where(m)[0] for m in masks]
  if any(len(g)==0 for g in self.groups):raise ValueError('Demonstration missing launch/flight/recovery/hold phase')
  self.source_sha256=sha(path)
 def sample(self,per_phase=64):
  ids=torch.cat([g[torch.randint(len(g),(per_phase,),device=g.device)] for g in self.groups])
  return self.obs[ids],self.action[ids]
 def loss(self,policy,full=False):
  if full:
   losses=[]
   for ids in self.groups:
    target=torch.atanh(self.action[ids].clamp(-math.tanh(3.),math.tanh(3.)))
    losses.append((policy.raw_mean(self.obs[ids])-target).square().mean())
   return sum(w*x for w,x in zip(C['demonstration_phase_weights'],losses))/sum(C['demonstration_phase_weights'])
  obs,act=self.sample();target=torch.atanh(act.clamp(-math.tanh(3.),math.tanh(3.)))
  err=(policy.raw_mean(obs)-target).square().mean(-1).reshape(len(self.groups),-1).mean(-1)
  weights=err.new_tensor(C['demonstration_phase_weights'])
  return (err*weights).sum()/weights.sum()

class Guidance:
 """只使用正常训练的已结束回合调整示范强度，无中途冻结评估；退化后自动恢复强约束。"""
 def __init__(self,rows=None):self.rows=collections.deque(rows or [],maxlen=C['demonstration_window_episodes'])
 def observe(self,rows):
  self.rows.extend((bool(r['success']),bool(r['qualified_landing'])) for r in rows)
 def summary(self):
  n=len(self.rows);success=sum(x[0] for x in self.rows)/max(n,1);qualified=sum(x[1] for x in self.rows)/max(n,1)
  mastered=n>=C['demonstration_min_episodes'] and success>=C['demonstration_success_rate'] and qualified>=C['demonstration_qualified_rate']
  return dict(episodes=n,success_rate=success,qualified_rate=qualified,mastered=mastered,
   weight=C['demonstration_min_weight'] if mastered else C['demonstration_weight'])


def pretrain(dataset,output,device='cpu',steps=3000,initial=None):
 from .policy import SquashedActorCritic
 torch.manual_seed(42);torch.set_num_threads(1)
 demo=Demonstrations(dataset,device)
 model=SquashedActorCritic(420,104,16,actor_hidden_dims=[256,256,128],critic_hidden_dims=[256,256,128],noise_std_type='log',activation='elu').to(device)
 initial=Path(initial or Path(__file__).parent/'teacher/bc_initial_actor.pt');load_initial(model,initial)
 params=[p for name,p in model.named_parameters() if name.startswith(ACTOR_PREFIXES)]
 opt=torch.optim.Adam(params,lr=3e-4);records=[]
 for step in range(steps):
  for group in opt.param_groups:group['lr']=3e-4 if step<steps*.65 else 3e-5
  opt.zero_grad(set_to_none=True);loss=demo.loss(model,full=step>=steps*.8);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
  if (step+1)%500==0 or step==0 or step+1==steps:
   with torch.no_grad():
    prediction=model.act_inference(demo.obs);active=torch.round(demo.obs[:,-12]*300)>=20;error=(prediction[active]-demo.action[active]).abs()
   row=dict(step=step+1,latent_loss=float(loss.detach()),active_action_MSE=float(error.square().mean()),active_action_max_error=float(error.max()))
   records.append(row);print('[GUIDED BC]',row,flush=True)
 with torch.no_grad():model.log_std[:12].fill_(math.log(.03));model.log_std[12:].fill_(math.log(.02))
 output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.save(actor_state(model),output/'actor.pt')
 write(output/'pretrain.json',dict(method='phase-balanced supervised imitation; no PPO',steps=steps,seed=42,
  dataset_sha256=demo.source_sha256,initial_actor_sha256=sha(initial),actor_sha256=sha(output/'actor.pt'),
  phase_counts=[len(g) for g in demo.groups],curve=records,closed_loop_success='requires_native_autonomous_replays'))
 return output/'actor.pt'

def auxiliary_update(alg,demo,completed_updates):
 """按训练行为保留示范；只更新Actor，PPO整批KL保护包括本次更新。"""
 guidance=getattr(alg,'guidance',None);weight=guidance.summary()['weight'] if guidance else C['demonstration_weight']
 alg.optimizer.zero_grad(set_to_none=True);loss=demo.loss(alg.policy);(weight*loss).backward()
 torch.nn.utils.clip_grad_norm_(alg.policy.parameters(),1.);alg.optimizer.step()
 return float(loss.detach()),weight
