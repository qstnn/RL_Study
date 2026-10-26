"""完整复制正式v3/2000 Actor到两个分支，保留起跳经验；Critic/Adam重新开始。"""
from pathlib import Path
import torch
from .io import read,sha,write
from .spec import CONTRACT as C
EXPECTED_SOURCE='211d026f84b1acbe5416bf925096a3731b1897aa9b0a7af763911f36d705f59c'

def initialize(runner,run):
 folder=Path(__file__).parent/'bootstrap';meta=read(folder/'provenance.json');path=folder/'actor_2000.pt'
 if meta['source_sha256']!=EXPECTED_SOURCE or sha(path)!=meta['actor_sha256']:raise ValueError('Bootstrap provenance mismatch')
 state=torch.load(path,map_location=runner.device,weights_only=True);model=runner.alg.policy;old=model.state_dict()
 expected={k for k in old if k.startswith('actor.') or k=='log_std'}
 if set(state)!=expected:raise ValueError('Bootstrap actor keys mismatch')
 for k,v in state.items():
  if v.shape!=old[k].shape or not torch.isfinite(v).all():raise ValueError('Invalid bootstrap tensor: '+k)
 critic_before={k:v.clone() for k,v in old.items() if k.startswith('critic.')}
 model.load_state_dict(state,strict=False)
 ground={k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')};model.ground_actor.load_state_dict(ground)
 for k,v in critic_before.items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
 if runner.alg.optimizer.state or runner.completed_updates!=0:raise ValueError('Initialization requires new optimizer and zero v5 updates')
 obs=torch.linspace(-1,1,420,device=runner.device).repeat(3,1);obs[:,-8]=torch.tensor([0.,1.,1.],device=runner.device);x=obs
 for k in (0,2,4,6):
  x=torch.nn.functional.linear(x,state[f'actor.{k}.weight'],state[f'actor.{k}.bias'])
  if k!=6:x=torch.nn.functional.elu(x)
 reference=torch.tanh(3*torch.tanh(x/3))
 with torch.no_grad():actual=model.act_inference(obs)
 torch.testing.assert_close(actual,reference,rtol=0,atol=1e-7)
 write(Path(run)/'bootstrap.json',dict(**meta,critic_reset=True,optimizer_reset=True,v5_updates=0,
  all_phase_actor_equivalence_max_error=float((actual-reference).abs().max()),ground_actor_reinitialized=False,
  ground_output_weight_scale=1.,ground_output_bias_zero=False,
  note='Both phase actors match v3 on identical inputs; new rewards, exploration, critic and optimizer'))
