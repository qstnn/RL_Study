"""配对相同容量、初始化、样本顺序、预算；只改变动作头路由。"""
import hashlib
import math
import numpy as np
import torch
from common_phase import HERE,previous,parent,read,write,sha,now,snapshot,require,sources,contract,load_actor,ACTOR,DATASET,DATASET_SHA
from model_phase import PhaseActor,observed_capture


def train(output,routing,device,steps=6000):
    assert steps==6000,'Matched finite budget is 6000 optimizer updates'
    previous.require(DATASET);assert sha(DATASET/'dataset.npz')==DATASET_SHA
    diagnostic=read(HERE/'validation/phase_information.json')
    assert diagnostic['dataset_sha256']==DATASET_SHA and diagnostic['dataset_pitch_gate_mismatch']==0
    torch.manual_seed(42)
    original,_=previous.load_actor(ACTOR)
    model=PhaseActor(original,routing).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    generator=torch.Generator(device=device).manual_seed(42)
    with np.load(DATASET/'dataset.npz') as z:
        x=torch.tensor(z['observation'],device=device);y=torch.tensor(z['action'],device=device)
        groups=[torch.tensor(np.flatnonzero(z['stage']==i),device=device) for i in range(4)]
    with torch.no_grad():
        initial_error=float((model(x)-original.to(device)(x)).abs().max())
    assert initial_error<2e-6
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},output/'initial_actor_state.pt')
    @torch.no_grad()
    def metrics():
        e=(model(x)-y).square().mean(-1);g=observed_capture(x)
        return dict(mse=float(e.mean()),stage_mse=[float(e[ids].mean()) for ids in groups],
                    gate_mse=[float(e[~g].mean()),float(e[g].mean())],action_max_error=float((model(x)-y).abs().max()))
    started=now();before=metrics();history=[];sample_hash=hashlib.sha256()
    for step in range(steps):
        ids=torch.cat([g[torch.randint(len(g),(128,),device=device,generator=generator)] for g in groups])
        sample_hash.update(ids.cpu().numpy().tobytes())
        lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.param_groups[0]['lr']=lr
        loss=(model(x[ids])-y[ids]).square().mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%1000==0:
            row=dict(step=step+1,lr=lr,**metrics());history.append(row);print('[PHASE BC]',routing,row,flush=True)
    if routing=='uniform':assert all(torch.equal(a,b) for a,b in zip(model.rise.parameters(),model.capture.parameters()))
    value=dict(kind='sustain_phase_bc',contract=contract(),sources=sources(),routing=routing,
        actor_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer_state_dict=optimizer.state_dict(),
        dataset=str(DATASET),dataset_sha256=DATASET_SHA,initialization=str(ACTOR),initialization_sha256=sha(ACTOR),
        steps=steps,seed=42,sample_sha256=sample_hash.hexdigest())
    torch.save(value,output/'actor.pt')
    write(output/'result.json',dict(started_at=started,finished_at=now(),routing=routing,steps=steps,batch_size=512,device=device,
        parameter_count=sum(p.numel() for p in model.parameters()),initial_output_max_error=initial_error,
        initial_weights_sha256=sha(output/'initial_actor_state.pt'),sample_sha256=sample_hash.hexdigest(),
        before=before,after=metrics(),history=history,actor_sha256=sha(output/'actor.pt'),
        dataset_frames=len(x),dataset_sha256=DATASET_SHA,ppo_updates=0,ppo_admission=False))
    require(output);load_actor(output/'actor.pt')
