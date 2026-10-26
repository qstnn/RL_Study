"""从完整原控制序列计算因果记忆，再按原索引选取；不改变动作示范。"""
import numpy as np
import torch
from common_memory import DATASET,RAW_DATASET,RAW_SHA,previous,read,write,sha,now,snapshot,require
from capture_memory import CaptureMemory


def prepare():
    previous.previous.require(RAW_DATASET);assert sha(RAW_DATASET/'dataset.npz')==RAW_SHA
    meta=read(RAW_DATASET/'dataset.json')
    with np.load(RAW_DATASET/'dataset.npz') as z:data={k:z[k] for k in z.files}
    memory=np.zeros((len(data['stage']),2),np.float32);seen=np.zeros(len(memory),bool);replay=[]
    from verify_memory import reconstruct_memory
    for name in sorted({e['run']+'/'+e['trace'] for e in meta['episodes']}):
        from pathlib import Path
        path=Path(name)
        with np.load(path) as z:raw=z['policy_observation'].copy()
        tracker=CaptureMemory(raw.shape[1],'cpu')
        values=torch.stack([tracker.update(torch.from_numpy(frame)) for frame in raw]).numpy()
        independent=reconstruct_memory(raw)
        np.testing.assert_allclose(values,independent,atol=1e-7,rtol=0)
        for i,e in enumerate(meta['episodes']):
            if e['run']+'/'+e['trace']!=name:continue
            assert sha(path)==e['trace_sha256']
            chosen=data['episode']==i;ids=data['policy_index'][chosen];slot=e['slot']
            np.testing.assert_array_equal(data['observation'][chosen],raw[ids,slot])
            memory[chosen]=values[ids,slot];seen[chosen]=True
        replay.append(dict(trace=name,trace_sha256=sha(path),control_frames=int(np.prod(raw.shape[:2])),independent_memory_error=float(abs(values-independent).max())))
    assert seen.all()
    DATASET.mkdir(parents=True,exist_ok=False);snapshot(DATASET)
    data['capture_memory']=memory
    np.savez_compressed(DATASET/'dataset.npz',**data)
    write(DATASET/'dataset.json',dict(created_at=now(),frames=len(memory),raw_dataset=str(RAW_DATASET),raw_dataset_sha256=RAW_SHA,
        dataset_sha256=sha(DATASET/'dataset.npz'),original_arrays_unchanged=True,action_labels_added=0,
        memory_range=[memory.min(0).tolist(),memory.max(0).tolist()],stage_counts=meta['stage_counts'],episodes=meta['episodes'],trace_replay=replay))
    require(DATASET);print('[MEMORY DATA]',len(memory),'same observations/actions/stages; causal memory reconstructed before sampling',flush=True)
