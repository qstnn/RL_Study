"""固定示范物理观测，仅回填学生历史动作；这是离线反事实诊断，不是新物理回放。"""
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_student import load_actor,read,write,sha


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path=HERE.parent/'runs/2026-10-22_sustain_30s_blend800/trace_seed43_blend800.npz'
    with np.load(path) as z:
        observations=z['policy_observation'][:500].copy()
        labels=z['actions'][::20][:500].copy()
    records=[]
    for name in ['2026-10-22_bc_01','2026-10-22_bc_02']:
        model,value=load_actor(HERE/'runs'/name/'actor.pt')
        x=torch.from_numpy(observations)
        direct=model(x).clamp(-1,1).numpy();direct[:10]=0
        history=torch.zeros(2,5,16)
        outputs=[]
        for k in range(len(x)):
            modified=x[k].reshape(2,5,61).clone()
            modified[:,:,34:50]=history
            action=model(modified.flatten(1)).clamp(-1,1)
            if k<10:action.zero_()
            outputs.append(action.numpy().copy())
            history[:,:-1]=history[:,1:].clone()
            history[:,-1]=action
        outputs=np.stack(outputs)
        rows=[]
        for slot in (0,1):
            error=np.max(abs(outputs[:,slot]-labels[:,slot]),axis=-1)
            ids=np.flatnonzero(error>.05)
            times=[125,160,200,250,400]
            rows.append(dict(slot=slot,first_max_component_error_over_0p05_s=float(ids[0]*.02) if len(ids) else None,
                checkpoints=[dict(t=k*.02,one_step_max_error=float(abs(direct[k,slot]-labels[k,slot]).max()),
                    self_history_max_error=float(error[k]),
                    rear_wheel_action_error=(outputs[k,slot,14:]-labels[k,slot,14:]).tolist()) for k in times]))
        records.append(dict(model=name,actor_sha256=sha(HERE/'runs'/name/'actor.pt'),rows=rows))
    write(HERE/'validation/action_history_counterfactual.json',dict(reference=str(path),reference_sha256=sha(path),
        mechanism='All non-action observation channels frozen to successful reference; only 5x16 past executed actions replaced recursively',
        limitation='Offline conditional diagnostic; no physical state evolution, not proof of full-system causality or success',records=records))
    print(records)


if __name__=='__main__':main()
