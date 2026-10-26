"""Load explicit supervised initialization; never reinterpret an older PPO checkpoint."""
from pathlib import Path
import torch
from .io import read,write,sha
def initialize(runner,run):
 req=read(Path(run)/'request.json');path=Path(req['initial_actor'])
 if sha(path)!=req['initial_actor_sha256']:raise ValueError('Initial actor changed')
 from .station import teacher_passed
 from .io import implementation
 prep=read(Path(req['preparation'])/'preparation.json');proof=prep.get('autonomous_proof',[])
 if (prep.get('status')!='prepared' or prep.get('implementation_sha256')!=implementation()
  or prep.get('device')!=str(runner.device) or prep.get('initial_actor_sha256')!=sha(path)
  or prep.get('demonstrations_sha256')!=req['demonstrations_sha256'] or len(proof)!=2
  or not all(x.get('controller')=='student_actor' and x.get('actor_sha256')==sha(path) and teacher_passed(x['report']) for x in proof)):
  raise ValueError('Student must pass two full autonomous replays on this device before PPO')
 state=torch.load(path,map_location=runner.device,weights_only=True)
 model=runner.alg.policy;old=model.state_dict()
 expected={k for k in old if k.startswith(('actor.','ground_actor.','landing_actor.')) or k=='log_std'}
 if set(state)!=expected:raise ValueError('Invalid supervised actor keys')
 for k,v in state.items():
  if v.shape!=old[k].shape or not torch.isfinite(v).all():raise ValueError('Invalid supervised actor tensor: '+k)
 critic={k:v.clone() for k,v in old.items() if k.startswith('critic.')}
 model.load_state_dict(state,strict=False)
 for k,v in critic.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
 if runner.alg.optimizer.state:raise ValueError('Initialization requires fresh PPO optimizer')
 runner.autonomous_initialization=True
 write(Path(run)/'bootstrap.json',dict(initial_actor_sha256=sha(path),demonstrations_sha256=req['demonstrations_sha256'],
  critic_reset=True,optimizer_reset=True,ppo_updates=0,autonomous_student_verified=True,proof=proof,method='explicit successful v3 actor migration after two autonomous passes; frozen ground; fresh critic and optimizer'))
