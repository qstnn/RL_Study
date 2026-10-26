"""显式迁移已审查v5/1201：空中分支原样，地面特征保留并缩小输出层。"""
from pathlib import Path
import torch
from .io import read,sha,write
from .spec import CONTRACT as C
EXPECTED_SOURCE='845fb32576ca57fcdf60a6b5899ebf2e04bfde3682cf0b02a3d4bcd6b20456c9'

def reference_actions(state,obs,ground_scale=1.):
 """CPU/GPU同口径迁移复核，不改变执行策略。"""
 outputs=[]
 for branch in ('ground_actor','actor'):
  x=obs
  for k in (0,2,4,6):
   x=torch.nn.functional.linear(x,state[f'{branch}.{k}.weight'],state[f'{branch}.{k}.bias'])
   if k!=6:x=torch.nn.functional.elu(x)
  if branch=='ground_actor':x=x*ground_scale
  outputs.append(x)
 raw=torch.where((obs[:,-8]<.5)[:,None],outputs[0],outputs[1])
 return torch.tanh(3*torch.tanh(raw/3))

def initialize(runner,run):
 folder=Path(__file__).parent/'bootstrap';meta=read(folder/'provenance.json');path=folder/'actor_1201.pt'
 if meta['source_sha256']!=EXPECTED_SOURCE or sha(path)!=meta['actor_sha256']:raise ValueError('Bootstrap provenance mismatch')
 state=torch.load(path,map_location=runner.device,weights_only=True);model=runner.alg.policy;old=model.state_dict()
 expected={k for k in old if k.startswith(('actor.','ground_actor.')) or k=='log_std'}
 if set(state)!=expected:raise ValueError('Bootstrap actor keys mismatch')
 for k,v in state.items():
  if v.shape!=old[k].shape or not torch.isfinite(v).all():raise ValueError('Invalid bootstrap tensor: '+k)
 critic_before={k:v.clone() for k,v in old.items() if k.startswith('critic.')}
 model.load_state_dict(state,strict=False)
 scale=C['ground_actor_initial_output_scale']
 with torch.no_grad():
  model.ground_actor[-1].weight.mul_(scale);model.ground_actor[-1].bias.mul_(scale)
 for k,v in critic_before.items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
 if runner.alg.optimizer.state or runner.completed_updates!=0:raise ValueError('Initialization requires new optimizer and zero v6 updates')
 obs=torch.linspace(-1,1,420,device=runner.device).repeat(3,1);obs[:,-8]=torch.tensor([0.,1.,1.],device=runner.device)
 reference=reference_actions(state,obs,scale)
 with torch.no_grad():actual=model.act_inference(obs)
 torch.testing.assert_close(actual,reference,rtol=0,atol=1e-6)
 for k,v in state.items():
  if k.startswith('actor.'):torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
 write(Path(run)/'bootstrap.json',dict(**meta,critic_reset=True,optimizer_reset=True,v6_updates=0,
  migration_formula_max_error=float((actual-reference).abs().max()),ground_actor_reinitialized=False,
  ground_output_weight_scale=scale,ground_output_bias_scale=scale,air_actor_weights_unchanged=True,
  note='v5/1201 partial-skill baseline; ground output layer scaled for preparation exploration; no fixed motion reference'))
