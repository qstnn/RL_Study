"""只用实际成功执行的混合完成帧，拟合可解释捕获反馈头。"""
import numpy as np
import torch
from common_feedback import DATASET,PREFIX_SHA,URDF,BodyEstimator,prefix_actor,read,write,sha,now,snapshot,sources,contract,previous,load_actor
from model_feedback import FeedbackActor,feedback_basis


def fit(output,variant):
    torch.set_num_threads(4);previous.require(DATASET)
    with np.load(DATASET/'dataset.npz') as z:d={k:z[k] for k in z.files}
    selected=d['teacher_executed'] & (d['capture_memory'][:,1]>=1.)
    assert selected.sum()>100
    raw=d['observation'][selected];y=d['action'][selected].astype(np.float64);stage=d['stage'][selected]
    estimate=BodyEstimator(URDF).estimate(raw)
    estimates=np.stack((estimate['gap_m'],estimate['velocity_m_s']),-1)
    x=feedback_basis(torch.from_numpy(raw).double(),torch.from_numpy(estimates)).numpy()
    model=FeedbackActor(prefix_actor(),variant);coeff=np.zeros((16,6));records=[]
    # 长稳态不淹没反馈过渡；沿用原分组，两个出现的分组各半总权重。
    weights=np.zeros(len(stage))
    for s in np.unique(stage):weights[stage==s]=1./(stage==s).sum()
    weights/=weights.mean()
    for j in range(16):
        cols=np.flatnonzero(model.mask[j].numpy());valid=abs(y[:,j])<.999
        a=x[valid][:,cols];b=y[valid,j];w=weights[valid]
        scale=np.sqrt((a*a*w[:,None]).sum(0)/w.sum()).clip(1e-6);aw=a/scale*np.sqrt(w[:,None]);bw=b*np.sqrt(w)
        theta=np.linalg.lstsq(aw,bw,rcond=1e-8)[0]/scale
        coeff[j,cols]=theta
        records.append(dict(action=j,features=cols.tolist(),fit_frames=int(valid.sum()),condition=float(np.linalg.cond(aw))))
    with torch.no_grad():model.coefficients.copy_(torch.from_numpy(coeff).float())
    prediction=(x@coeff.T).clip(-1,1);error=prediction-y
    output.mkdir(parents=True,exist_ok=False);snapshot(output)
    value=dict(kind='structured_feedback_bc',contract=contract(),sources=sources(),variant=variant,prefix_sha256=PREFIX_SHA,
        dataset_sha256=sha(DATASET/'dataset.npz'),actor_state_dict=model.state_dict(),fit='weighted linear least squares; saturated targets excluded per output',selected_frames=int(selected.sum()))
    torch.save(value,output/'actor.pt')
    write(output/'result.json',dict(at=now(),variant=variant,frames=int(selected.sum()),selected_teacher_only=True,
        new_action_labels=0,fit_mse=float((error**2).mean()),fit_max_error=float(abs(error).max()),
        fit_rear_wheel_mae=float(abs(error[:,14:16]).mean()),stage_mse={str(s):float((error[stage==s]**2).mean()) for s in np.unique(stage)},
        coefficients=coeff.tolist(),fits=records,actor_sha256=sha(output/'actor.pt'),dataset_sha256=value['dataset_sha256'],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),ppo_updates=0,ppo_admission=False))
    load_actor(output/'actor.pt');print('[FEEDBACK FIT]',variant,'frames',int(selected.sum()),'MSE',float((error**2).mean()),'rear',coeff[14:].tolist(),flush=True)
