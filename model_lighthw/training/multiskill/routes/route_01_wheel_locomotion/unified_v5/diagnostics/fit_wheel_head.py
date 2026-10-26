"""将成功冻结轨迹的轮速标签拟合回标准Actor；只改最终线性层4个轮输出，腿动作函数逐bit不变。

离线拟合误差不代表行为验收。正向校准只收完整成功case。可选失败负载域零增量边界
仅用于防止空载轮速校准外推，不作为成功标签，也不进入PPO成功参考缓存。
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scripts'))
from launch import SPEC,signature,module_from_file,write_json,sha256,now
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',default='50');p.add_argument('--output',type=Path,required=True)
p.add_argument('--loaded_boundary_weight',type=float,default=0.);p.add_argument('--uphill_label_scale',type=float,default=1.);p.add_argument('--ridge',type=float,nargs='+',default=[.01,.1,1.]);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(43)
request=json.loads((a.data/'request.json').read_text());evaluation=json.loads((a.data/'evaluation.json').read_text())
trace=np.load(a.data/'actor_trace.npz');source=Path(request['checkpoints'][a.model]['path'])
if sha256(source)!=request['checkpoints'][a.model]['sha256']:raise ValueError('Source checkpoint changed since labels were generated')
if len(request['trials'])!=len(evaluation['cases']) or trace['observation'].shape[1]!=len(request['trials']):raise ValueError('Trace/case layout mismatch')
saved=SPEC.initialization_checkpoint(source)
actor=torch.nn.Sequential(torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16)).eval()
actor.load_state_dict({k.removeprefix('actor.'):v for k,v in saved['model_state_dict'].items() if k.startswith('actor.')})
xs=[];ys=[];ids=[];weights=[];cases=[]
for i,(trial,result) in enumerate(zip(request['trials'],evaluation['cases'])):
    if trial['model']!=a.model:continue
    success=bool(result['success'] and result['complete'])
    loaded=trial.get('payload_kg',0.)>0. or trial.get('towed_kg',0.)>0.
    boundary=(not success) and loaded and a.loaded_boundary_weight>0.
    if not success and not boundary:continue
    n=result['samples']
    if not 0<n<=trace['observation'].shape[0]:raise ValueError('Invalid successful trajectory length')
    x=trace['observation'][:n,i].copy();y=trace['applied_action'][:n,i].copy()
    if boundary:y=trace['raw_action'][:n,i].copy()
    if trial['slope_deg']==25. and trial.get('payload_kg',0.)==0. and trial.get('towed_kg',0.)==0.:
        original=trace['raw_action'][:n,i,12:]
        y[:,12:]=original+a.uphill_label_scale*(y[:,12:]-original)
    xs.append(x);ys.append(y);ids.extend([i]*n)
    # 等case权重；静止/停车更重，避免微小偏差抵消原有稳定保持。
    w=np.where(abs(trace['command'][:n,i])<1.e-6,4.,1.).astype('float64')/n
    if boundary:w*=a.loaded_boundary_weight
    weights.append(w);cases.append(dict(index=i,case=result['case'],samples=n,source_success=success,boundary_only=boundary))
if sum(c['source_success'] for c in cases)<10:raise ValueError('Insufficient complete successful retention cases')
x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys));ids=np.array(ids);w=torch.from_numpy(np.concatenate(weights))
with torch.inference_mode():
    hidden=actor[:-1](x).double();raw=actor(x);delta=(y[:,12:]-raw[:,12:]).double()
z=torch.cat((hidden,torch.ones((len(hidden),1),dtype=torch.double)),dim=1)
normal=z.T@(w[:,None]*z);target=z.T@(w[:,None]*delta)
a.output.mkdir(parents=True,exist_ok=False)
base_source=dict(path=str(source),sha256=sha256(source));reports=[]
for ridge in a.ridge:
    correction=torch.linalg.solve(normal+ridge*torch.eye(z.shape[1],dtype=z.dtype),target)
    state={k:v.detach().clone() for k,v in saved['model_state_dict'].items()}
    old=state['critic.0.weight'];state['critic.0.weight']=torch.cat((old,torch.zeros((old.shape[0],252-old.shape[1]))),dim=1)
    state['actor.6.weight'][12:]+=correction[:-1].T.float();state['actor.6.bias'][12:]+=correction[-1].float()
    state['log_std']=torch.tensor([.005]*12+[.01]*4).log()
    residual=(z@correction-delta).float();by_case=[]
    for c in cases:
        mask=ids==c['index'];d=residual[mask]
        by_case.append(dict(**c,wheel_target_mae_rad_s=float(d.abs().mean()*5),wheel_target_abs_max_rad_s=float(d.abs().max()*5)))
    path=a.output/f'model_ridge_{ridge:g}.pt'
    args=argparse.Namespace(ground_friction=4.,rolling_coefficient=.02,leg_torque_limit=96.,wheel_torque_limit=48.,hitch_height=.07,unloaded_acceleration=1.,loaded_acceleration=.3,critic_warmup_updates=50)
    infos=dict(r1_unified_contract=SPEC.CONTRACT,environment_signature=signature(args),completed_updates=0,initialization=dict(actor_source=base_source,critic_source=base_source,offline_fit='wheel final-layer rows only',ridge=ridge,loaded_boundary_weight=a.loaded_boundary_weight,uphill_label_scale=a.uphill_label_scale,data=str(a.data),actor_ground_truth_inputs=False,ppo_updates=0))
    torch.save(dict(model_state_dict=state,optimizer_state_dict={},iter=-1,infos=infos),path)
    # 对任意输入腿网络及输出参数未变；同时验证实际输出逐bit相同。
    candidate=torch.nn.Sequential(*[torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16)]).eval()
    candidate.load_state_dict({k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')})
    with torch.inference_mode():torch.testing.assert_close(candidate(x)[:,:12],raw[:,:12],rtol=0,atol=0)
    reports.append(dict(path=str(path),sha256=sha256(path),ridge=ridge,by_case=by_case))
write_json(a.output/'offline_fit.json',dict(created_at=now(),source=base_source,data=str(a.data),fitting_cases=cases,successful_case_count=sum(c['source_success'] for c in cases),boundary_case_count=sum(c['boundary_only'] for c in cases),loaded_boundary_weight=a.loaded_boundary_weight,candidates=reports,uphill_label_scale=a.uphill_label_scale,label_note='Only source trajectories passed; scaled label candidates still require independent closed-loop validation',behavior_validated=False))
print(json.dumps(reports,ensure_ascii=False,indent=2))
