"""已审查原 Actor -> v3，新增反馈列为零；Critic/Adam 显式重建。"""
import argparse
from copy import deepcopy
import math
import shutil
from pathlib import Path
import torch
from launch import spec,sha256,implementation_hashes,write_json,checkpoint,now

SOURCE_SHA='310c8e3e3df68d7fa503f3852924504610b6bd47b242fc8b9770a2a43b9cbcdb'


def actor(weights,x):
    for i in (0,2,4,6):
        x=torch.nn.functional.linear(x,weights[f'actor.{i}.weight'],weights[f'actor.{i}.bias'])
        if i!=6:x=torch.nn.functional.elu(x)
    return x


def migrate(source,destination,seed=42):
    source=Path(source).resolve(strict=True);destination=Path(destination)
    if sha256(source)!=SOURCE_SHA:raise ValueError('Highjump migration requires the reviewed original model999; source SHA differs')
    old=torch.load(source,map_location='cpu',weights_only=False)
    weights=old['model_state_dict']
    if weights['actor.0.weight'].shape!=(512,285):raise ValueError('Unexpected original Actor dimensions')
    if not all(torch.isfinite(v).all() for v in weights.values()):raise ValueError('Nonfinite source model')
    new=deepcopy(weights);first=torch.zeros(512,400,dtype=weights['actor.0.weight'].dtype)
    for frame in range(5):first[:,frame*80:frame*80+57]=weights['actor.0.weight'][:,frame*57:(frame+1)*57]
    new['actor.0.weight']=first
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index,inside,outside in ((0,spec.CONTRACT['critic_dim'],512),(2,512,256),(4,256,128),(6,128,1)):
            layer=torch.nn.Linear(inside,outside)
            new[f'critic.{index}.weight']=layer.weight.detach().clone()
            new[f'critic.{index}.bias']=layer.bias.detach().clone()
        x=torch.randn(2048,5,57);extra=torch.randn(2048,5,23)
        before=actor(weights,x.flatten(1));after=actor(new,torch.cat((x,extra),-1).flatten(1))
        error=float((before-after).abs().max())
        torch.testing.assert_close(before,after,atol=2e-6,rtol=1e-5)
    new['log_std']=torch.full_like(weights['log_std'],math.log(spec.CONTRACT['exploration_std']))
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source,destination/'source_v1.pt')
    assert sha256(destination/'source_v1.pt')==SOURCE_SHA
    provenance=dict(source_sha256=SOURCE_SHA,source_completed_updates=old['infos']['completed_updates'],
        created_at=now(),target_contract=spec.VERSION,actor_raw_output_max_error=error,
        critic_reset=True,adam_reset=True,added_actor_columns_zero=True,
        behavior_identical_claim=False,reference_and_rewards_changed=True)
    saved=dict(model_state_dict=new,optimizer_state_dict=None,iter=0,
        infos=dict(r5a_contract=spec.CONTRACT,implementation_sha256=implementation_hashes(),
            initialization='actor_warm_start',completed_updates=0,learning_rate=spec.CONTRACT['actor_learning_rate'],
            migration=provenance,physics_state_saved=False,critic_warmup_updates=spec.CONTRACT['critic_warmup_updates']))
    path=destination/'warm_start.pt';torch.save(saved,path);checkpoint(path)
    reference=deepcopy(saved)
    reference['model_state_dict']['actor.6.weight'].zero_();reference['model_state_dict']['actor.6.bias'].zero_()
    reference['infos']['migration']={**provenance,'initialization_variant':'reviewed_features_zero_residual_head','actor_raw_output_preserved':False}
    torch.save(reference,destination/'reference_start.pt');checkpoint(destination/'reference_start.pt')
    assert sha256(source)==SOURCE_SHA
    write_json(destination/'migration.json',{**provenance,'warm_start_sha256':sha256(path)})
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();print(migrate(a.checkpoint,a.output,a.seed))
