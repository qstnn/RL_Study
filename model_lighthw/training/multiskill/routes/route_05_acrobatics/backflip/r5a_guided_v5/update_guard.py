"""整批PPO更新后的KL保护：拒绝过大更新时恢复Actor、Critic和Adam，保留审计计数。"""
import copy,math
import torch
from .spec import CONTRACT as C

def guarded_update(alg,update,limit=.03):
 policy=alg.policy
 before={k:v.detach().clone() for k,v in policy.state_dict().items()}
 optimizer=copy.deepcopy(alg.optimizer.state_dict())
 old_lr=alg.learning_rate
 losses=update()
 with torch.no_grad():
  policy.log_std.clamp_(math.log(.01),math.log(.15))
  finite=all(bool(torch.isfinite(v).all()) for v in policy.state_dict().values())
  finite=finite and all(bool(torch.isfinite(v).all()) for state in alg.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
  loss_finite=all(math.isfinite(float(v)) for v in losses.values())
  storage=alg.storage
  kl=float('inf')
  if finite and loss_finite:
   policy.update_distribution(storage.observations.flatten(0,1))
   oldmu=storage.mu.flatten(0,1);olds=storage.sigma.flatten(0,1)
   mu=policy.action_mean;std=policy.action_std
   kl=float((torch.log(std/olds)+(olds.square()+(oldmu-mu).square())/(2*std.square())-.5).sum(-1).mean())
  accepted=finite and loss_finite and math.isfinite(kl) and kl<=limit
  if not accepted:
   policy.load_state_dict(before);alg.optimizer.load_state_dict(optimizer)
   alg.learning_rate=max(C['minimum_learning_rate'],old_lr*.5)
  else:alg.learning_rate=min(C['maximum_learning_rate'],old_lr*1.01)
  for group in alg.optimizer.param_groups:group['lr']=alg.learning_rate
  policy.update_distribution(storage.observations.flatten(0,1))
  std=policy.log_std.exp()
  changed=any(not torch.equal(v,policy.state_dict()[k]) for k,v in before.items() if k.startswith(('actor.','ground_actor.','landing_actor.')))
  row=dict(accepted=accepted,actor_changed=changed,rollout_KL=kl if math.isfinite(kl) else None,
   learning_rate=alg.learning_rate,noise_min=float(std.min()),noise_max=float(std.max()),
   effective_ground_std_min=float(std[:12].mul(C['ground_exploration_multiplier']).clamp(min=C['ground_exploration_leg_floor'],max=C['exploration_cap']).min()),
   near_action_boundary_fraction=float((storage.actions.abs()>.98).float().mean()),
   losses={k:float(v) if math.isfinite(float(v)) else None for k,v in losses.items()})
 return losses,row
