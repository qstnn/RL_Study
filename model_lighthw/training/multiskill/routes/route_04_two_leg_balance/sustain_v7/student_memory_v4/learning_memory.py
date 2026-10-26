"""配对单因素：同307维网络和6000步预算，仅暴露或屏蔽两维捕获记忆。"""
import math
import hashlib
import numpy as np
import torch
from common_memory import DATASET,INITIAL,INITIAL_SHA,read,write,sha,now,require,snapshot,sources,contract,initial_actor,load_actor
from capture_memory import augment
from model_memory import MemoryActor


def train(output,mode,device):
    require(DATASET);meta=read(DATASET/'dataset.json');assert sha(DATASET/'dataset.npz')==meta['dataset_sha256']
    torch.manual_seed(42);original=initial_actor();model=MemoryActor(original,mode).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);generator=torch.Generator(device=device).manual_seed(42)
    with np.load(DATASET/'dataset.npz') as z:
        raw=torch.tensor(z['observation'],device=device);memory=torch.tensor(z['capture_memory'],device=device)
        x=augment(raw,memory,mode);y=torch.tensor(z['action'],device=device)
        groups=[torch.tensor(np.flatnonzero(z['stage']==i),device=device) for i in range(4)]
    with torch.no_grad():initial_error=float((model(x)-original.to(device)(raw)).abs().max())
    assert initial_error<2e-6
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},output/'initial_actor_state.pt')
    @torch.no_grad()
    def metrics():
        e=(model(x)-y).square().mean(-1)
        return dict(mse=float(e.mean()),stage_mse=[float(e[g].mean()) for g in groups],action_max_error=float((model(x)-y).abs().max()))
    before=metrics();started=now();history=[];sample_hash=hashlib.sha256();steps=6000
    for step in range(steps):
        ids=torch.cat([g[torch.randint(len(g),(128,),device=device,generator=generator)] for g in groups])
        sample_hash.update(ids.cpu().numpy().tobytes())
        lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*step/steps)));optimizer.param_groups[0]['lr']=lr
        loss=(model(x[ids])-y[ids]).square().mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%1000==0:
            row=dict(step=step+1,lr=lr,**metrics());history.append(row);print('[MEMORY BC]',mode,row,flush=True)
    if mode=='zero':assert not model.trunk[0].weight[:,305:].any()
    value=dict(kind='sustain_memory_bc',contract=contract(),sources=sources(),memory_mode=mode,
        actor_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer_state_dict=optimizer.state_dict(),
        dataset=str(DATASET),dataset_sha256=meta['dataset_sha256'],initialization=str(INITIAL),initialization_sha256=INITIAL_SHA,
        steps=steps,seed=42,sample_sha256=sample_hash.hexdigest())
    torch.save(value,output/'actor.pt')
    write(output/'result.json',dict(started_at=started,finished_at=now(),memory_mode=mode,steps=steps,batch_size=512,device=device,
        parameter_count=sum(p.numel() for p in model.parameters()),initial_output_max_error=initial_error,
        sample_sha256=sample_hash.hexdigest(),dataset_frames=len(x),dataset_sha256=meta['dataset_sha256'],
        before=before,after=metrics(),history=history,actor_sha256=sha(output/'actor.pt'),ppo_updates=0,ppo_admission=False))
    load_actor(output/'actor.pt')
