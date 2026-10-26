"""CPU: fit an affine proprioceptive velocity adapter; entire trajectories held out."""
from pathlib import Path
import json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
V3=ROOT.parent/'unified_v3'
torch.set_num_threads(2)
train_paths=[V3/'validation/calibration_route38_independent_01/route38/policy_inputs.npz',V3/'validation/calibration_wheelcap_d1_full_01/route53/policy_inputs.npz']
val_paths=[V3/'validation/calibration_wheelcap_d1_full_01/route38/policy_inputs.npz']
def data(paths):
    xs=[];ys=[];cases=[]
    for path in paths:
        z=np.load(path)
        for i,n in enumerate(z['valid_case_lengths']):
            xs.append(z['observation_history'][:n,i]);ys.append(z['source_observation'][:n,i,:3]/2);cases.extend([i]*n)
    return np.concatenate(xs).astype('float64'),np.concatenate(ys).astype('float64'),np.array(cases)
x,y,c=data(train_paths);vx,vy,vc=data(val_paths)
mean=x.mean(0);std=x.std(0).clip(.1);z=np.c_[(x-mean)/std,np.ones(len(x))];vz=np.c_[(vx-mean)/std,np.ones(len(vx))]
rows=[];best=None
for alpha in (.1,1.,10.,100.,1000.):
    penalty=np.eye(z.shape[1])*alpha;penalty[-1,-1]=0
    w=np.linalg.solve(z.T@z+penalty,z.T@y)
    pred=vz@w
    # Main selection metric is velocity RMSE on independent complete trajectories, all 16 cases.
    error=pred-vy;rmse=np.sqrt((error**2).mean(0))
    row=dict(alpha=alpha,rmse_mps=rmse.tolist(),mae_mps=np.abs(error).mean(0).tolist(),by_case={str(i):float(np.sqrt((error[vc==i]**2).mean())) for i in range(16)})
    rows.append(row)
    if best is None or rmse.mean()<best[0]:best=(rmse.mean(),w,alpha)
_,w,alpha=best
W=(w[:-1]/std[:,None]).T;b=w[-1]-mean@W.T
out=ROOT/'initialization/adapter_affine';out.mkdir(parents=True,exist_ok=False)
np.savez(out/'velocity_adapter.npz',weight=W.astype('float32'),bias=b.astype('float32'))
meta=dict(train_files=list(map(str,train_paths)),validation_files=list(map(str,val_paths)),train_samples=len(x),validation_samples=len(vx),selection=rows,selected_alpha=alpha,actor_truth_input=False,note='Offline accuracy only. Closed-loop GPU validation required.')
(out/'fit.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
