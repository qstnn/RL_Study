"""离线教师标签迁移：Route38初始化 -> 单个300维MLP；无环境真值输入。

交错样本验证只用于选择拟合检查点，不当作独立行为证据；必须另跑闭环评估。
"""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launch import SPEC,write_json,now,sha256,signature
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initialize',type=Path);p.add_argument('--steps',type=int,default=1800);p.add_argument('--mask-action-history',action='store_true',default=True);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(47);rng=np.random.default_rng(47)
from rsl_rl.modules import ActorCritic
net=ActorCritic(300,242,16,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',noise_std_type='log',init_noise_std=SPEC.CONTRACT["initial_noise_std"])
source=ROOT.parents[3]/'routes/route_38_ppo_stair_flights/runs/2026-10-10_02-16-06_263239_lighthw_stair_flights_from_selected_v1/training/model_9999.pt'
if args.initialize:
 saved=SPEC.initialization_checkpoint(args.initialize);net.load_state_dict(SPEC.transfer_policy(saved,net.state_dict()))
 source=args.initialize.resolve()
else:
 if sha256(source)!='dc570667b6c03f2d875eb8eae340a0feb8d431e0324ddfeab49bd7d7322c890d':raise RuntimeError('Route38 source mismatch')
 old=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
 state=net.state_dict();z=np.load(ROOT/'initialization/adapter_affine/velocity_adapter.npz')
 W=torch.from_numpy(z['weight']);b=torch.from_numpy(z['bias'])
 for name in state:
  if name.startswith('actor.') and name not in ('actor.0.weight','actor.0.bias'):state[name]=old[name]
 state['actor.0.weight'].zero_()
 state['actor.0.weight'][:,:290]=2.*old['actor.0.weight'][:,:3]@W
 state['actor.0.weight'][:,SPEC.latest_source_columns()]+=old['actor.0.weight'][:,3:]
 state['actor.0.bias']=old['actor.0.bias']+2.*old['actor.0.weight'][:,:3]@b
 net.load_state_dict(state)
if args.mask_action_history:
 mask=torch.ones_like(net.actor[0].weight);mask[:,205:285]=0.
 with torch.no_grad():net.actor[0].weight.mul_(mask)
 net.actor[0].weight.register_hook(lambda g:g*mask)
xs=[];ys=[];validation=[];groups=[];dataset_groups=[];ng=0
for path in args.data:
 start_group=ng
 z=np.load(path)
 observations,actions,lengths=z['observation'],z['action'],z['valid_lengths']
 for i,n in enumerate(lengths):
  x=observations[:n,i];y=actions[:n,i]
  good=np.isfinite(x).all(1)&np.isfinite(y).all(1)
  # Target wheel velocities above physical clip are equivalent, do not learn arbitrary overflow.
  y=y.copy();y[:,12:]=np.clip(y[:,12:],-7.4,7.4)
  indices=np.arange(n)[good]
  xs.append(x[good]);ys.append(y[good]);validation.extend((indices%5==0).tolist());groups.extend([ng]*len(indices));ng+=1
 dataset_groups.append(list(range(start_group,ng)))
x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys));val=torch.tensor(validation);group=np.array(groups)
indices={i:np.where((group==i)&~np.array(validation))[0] for i in range(ng)}
dataset_groups=[[i for i in g if len(indices[i])] for g in dataset_groups]
opt=torch.optim.Adam(net.actor.parameters(),lr=2.e-4)
weights=torch.tensor([1.28,2.24,2.24]*4+[1.]*4)
noise=torch.zeros(300);offset=0
for name,width in SPEC.POLICY_TERMS.items():
 magnitude={'base_ang_vel':.1,'projected_gravity':.05,'joint_pos':.01,'joint_vel':.075}.get(name,0.)
 noise[offset:offset+width*5]=magnitude;offset+=width*5
best=float('inf');beststate=None;rows=[]
with torch.no_grad():initial=float((((net.actor(x[val])-y[val])*weights)**2).mean())
for step in range(args.steps):
 selected=[rng.choice(dataset_groups[i]) for i in rng.integers(0,len(dataset_groups),size=512)]
 idx=np.array([rng.choice(indices[i]) for i in selected])
 batch=x[idx];target=y[idx]
 # Match declared proprioceptive noise; no noise or oracle added to task command fields.
 if step%2:batch=batch+(torch.rand_like(batch)*2-1)*noise
 prediction=net.actor(batch)
 loss=(((prediction-target)*weights)**2).mean()
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1.);opt.step()
 if (step+1)%100==0 or step+1==args.steps:
  with torch.no_grad():v=float((((net.actor(x[val])-y[val])*weights)**2).mean())
  rows.append(dict(step=step+1,train=float(loss),validation=v));print('DISTILL',rows[-1],flush=True)
  if v<best:best=v;beststate={k:v.detach().clone() for k,v in net.actor.state_dict().items()}
net.actor.load_state_dict(beststate)
with torch.no_grad():
 error=net.actor(x)-y
 stats=dict(raw_mae=error.abs().mean(0).tolist(),raw_p95=error.abs().quantile(.95,dim=0).tolist())
environment=signature(SimpleNamespace(ground_friction=4.,rolling_coefficient=.02,leg_torque_limit=96.,wheel_torque_limit=48.,hitch_height=.07,unloaded_acceleration=1.,loaded_acceleration=.3))
metadata=dict(r1_unified_contract=SPEC.CONTRACT,environment_signature=environment,initialization='Route38 weight transfer and offline teacher regression',source_checkpoint=str(source),source_sha256=sha256(source),completed_updates=0)
torch.save(dict(model_state_dict=net.state_dict(),optimizer_state_dict={},iter=-1,infos=metadata),out/'model_init.pt')
write_json(out/'fit.json',dict(created_at=now(),source=str(source),source_sha256=sha256(source),fitting_code_sha256=sha256(Path(__file__)),observation_noise_bounds=noise.tolist(),data=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in args.data],samples=len(x),offline_gradient_steps=args.steps,action_history_masked=args.mask_action_history,ppo_updates=0,initial_validation_loss=initial,best_validation_loss=best,curves=rows,stats=stats,actor_input='300 deployable proprioception and nominal task command only',validation='interleaved regression check only; separate GPU closed-loop required',checkpoint_sha256=sha256(out/'model_init.pt')))
print('SAVED',out/'model_init.pt',flush=True)
