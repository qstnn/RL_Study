"""显式 v2 Actor 迁移；禁止把旧奖励的 Critic、Adam 或更新计数带入 v3。"""
from pathlib import Path
import torch
from .io import read,sha,write
EXPECTED_SOURCE='40036a6837cb038e5d7eeaa84cebafd882bd1b1d894dc1dfb852694be296db9a'

def initialize(runner,run):
 folder=Path(__file__).parent/'bootstrap';meta=read(folder/'provenance.json');path=folder/'actor_1800.pt'
 if meta['source_sha256']!=EXPECTED_SOURCE or sha(path)!=meta['actor_sha256']:raise ValueError('Bootstrap provenance mismatch')
 state=torch.load(path,map_location=runner.device,weights_only=True)
 model=runner.alg.policy;old=model.state_dict()
 expected={k for k in old if k.startswith('actor.') or k=='log_std'}
 if set(state)!=expected:raise ValueError('Bootstrap actor keys mismatch')
 for k,v in state.items():
  if v.shape!=old[k].shape or not torch.isfinite(v).all():raise ValueError('Invalid bootstrap tensor: '+k)
 critic_before={k:v.clone() for k,v in old.items() if k.startswith('critic.')}
 model.load_state_dict(state,strict=False)
 for k,v in critic_before.items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
 if runner.alg.optimizer.state or runner.completed_updates!=0:raise ValueError('Bootstrap requires new optimizer and zero v3 updates')
 # 同一观测下，导入Actor与独立重算的有界v2输出必须完全等价。
 obs=torch.linspace(-1,1,420,device=runner.device).repeat(3,1)
 x=obs
 for k in (0,2,4,6):
  x=torch.nn.functional.linear(x,state[f'actor.{k}.weight'],state[f'actor.{k}.bias'])
  if k!=6:x=torch.nn.functional.elu(x)
 reference=torch.tanh(3*torch.tanh(x/3))
 with torch.no_grad():actual=model.act_inference(obs)
 torch.testing.assert_close(actual,reference,rtol=0,atol=1e-7)
 write(Path(run)/'bootstrap.json',dict(**meta,critic_reset=True,optimizer_reset=True,v3_updates=0,actor_equivalence_max_error=float((actual-reference).abs().max())))
