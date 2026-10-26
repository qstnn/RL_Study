"""Explicit v1 -> v2 migration; frozen comparison and actor warm start are separate files."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil

from launch import ROUTE, implementation_hashes, now, sha256, spec, write_json, checkpoint


def migrate(source, destination, seed=42):
    import json
    import numpy as np
    import torch
    from numpy.core.multiarray import _reconstruct
    source=Path(source).resolve(strict=True); destination=Path(destination).resolve()
    legacy=json.loads((ROUTE/'compat/v1.json').read_text())
    with torch.serialization.safe_globals([_reconstruct,np.ndarray,np.dtype,type(np.dtype(np.uint32))]):
        saved=torch.load(source,map_location='cpu',weights_only=True)
    info=saved.get('infos',{})
    if info.get('r5a_contract')!=legacy['contract'] or info.get('implementation_sha256')!=legacy['implementation_sha256']:
        raise ValueError('Only the explicitly recognized R5A J1 v1 implementation can migrate; no R1/R6 or modified contracts')
    if not isinstance(info.get('completed_updates'),int) or info['completed_updates']<=0:
        raise ValueError('Invalid v1 training counter')
    weights=saved['model_state_dict']
    expected={'actor.0.weight':(512,285),'actor.2.weight':(256,512),'actor.4.weight':(128,256),
              'actor.6.weight':(16,128),'critic.0.weight':(512,86),'critic.2.weight':(256,512),
              'critic.4.weight':(128,256),'critic.6.weight':(1,128),'log_std':(16,)}
    for key,shape in expected.items():
        if key not in weights or tuple(weights[key].shape)!=shape:raise ValueError('Unexpected model shape: '+key)
    if not all(torch.isfinite(v).all() for v in weights.values()):raise ValueError('Nonfinite v1 checkpoint')
    destination.mkdir(parents=True,exist_ok=False)
    original_sha=sha256(source)
    shutil.copy2(source,destination/'source_v1.pt')
    if sha256(destination/'source_v1.pt')!=original_sha or sha256(source)!=original_sha:
        raise RuntimeError('Source checkpoint changed during migration')
    provenance=dict(created_at=now(),source_path=str(source),source_sha256=original_sha,
        source_completed_updates=info['completed_updates'],source_contract=legacy['contract']['version'],
        target_contract=spec.VERSION,migration_script_sha256=sha256(Path(__file__)),
        physics_state_restored=False,seed=seed)
    frozen=deepcopy(saved)
    frozen['infos'].update(r5a_contract=spec.CONTRACT,implementation_sha256=implementation_hashes(),
        initialization='frozen_migration',migration={**provenance,'kind':'frozen_all_weights_unchanged'})
    torch.save(frozen,destination/'evaluation.pt')
    warm=deepcopy(frozen)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index in (0,2,4,6):
            key=f'critic.{index}.weight';out_features,in_features=weights[key].shape
            layer=torch.nn.Linear(in_features,out_features)
            warm['model_state_dict'][key]=layer.weight.detach().clone()
            warm['model_state_dict'][f'critic.{index}.bias']=layer.bias.detach().clone()
    warm['model_state_dict']['log_std']=torch.full_like(weights['log_std'],math.log(spec.CONTRACT['refinement_noise_std']))
    warm['optimizer_state_dict']=None
    warm['iter']=0
    warm['infos']=dict(r5a_contract=spec.CONTRACT,implementation_sha256=implementation_hashes(),
        initialization='actor_warm_start',completed_updates=0,learning_rate=spec.CONTRACT['refinement_learning_rate'],
        physics_state_saved=False,migration={**provenance,'kind':'actor_preserved_critic_and_adam_reset',
        'v2_start_completed_updates':0,'exploration_std':spec.CONTRACT['refinement_noise_std']})
    torch.save(warm,destination/'warm_start.pt')
    # Verify exactly what was saved, not only in-memory intent.
    checked_frozen=checkpoint(destination/'evaluation.pt');checked_warm=checkpoint(destination/'warm_start.pt')
    assert all(torch.equal(weights[k],v) for k,v in checked_frozen['model_state_dict'].items())
    assert all(torch.equal(v,checked_warm['model_state_dict'][k]) for k,v in weights.items() if k.startswith('actor.'))
    assert checked_warm['optimizer_state_dict'] is None
    report={**provenance,'frozen_weights_max_error':0.,'warm_actor_weights_max_error':0.,
        'critic_reset':True,'optimizer_reset':True,'v2_start_completed_updates':0,
        'outputs':{p.name:sha256(p) for p in destination.glob('*.pt')}}
    write_json(destination/'migration.json',report)
    return destination/'evaluation.pt',destination/'warm_start.pt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    for path in migrate(args.checkpoint,args.output,args.seed):print(path)


if __name__=='__main__':main()
