"""从实际文件核对来源、教师执行数据、有限BC和PPO准入前置条件。"""
from pathlib import Path
import argparse
import ast
import json
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_segments import parent,baseline,read,write,sha,require,load_actor,ACTOR,ACTOR_SHA,now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collection',type=Path,nargs='+',required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--autonomous',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    protected=0
    for name in ['source_manifest.json','comparison_source_manifest.json']:
        for rel,digest in read(parent.CONTROL/name).items():
            assert sha(parent.REPO/rel)==sha(parent.CONTROL/'source_snapshot'/rel)==digest
            protected+=1
    assert sha(parent.ACTOR)==parent.ACTOR_SHA and sha(ACTOR)==ACTOR_SHA
    require(a.dataset);meta=read(a.dataset/'dataset.json')
    assert sha(a.dataset/'dataset.npz')==meta['dataset_sha256']
    with np.load(a.dataset/'dataset.npz') as z:
        x=z['observation'].copy();y=z['action'].copy();stages=z['stage'].copy()
        for i,e in enumerate(meta['episodes']):
            file=Path(e['run'])/e['trace'];assert sha(file)==e['trace_sha256']
            with np.load(file) as src:
                slot=e['slot'];active=src['policy_active'][:,slot]
                mask=active if e['source_kind']==0 else active & src['policy_takeover'][:,slot]
                selected=z['episode']==i
                np.testing.assert_array_equal(z['policy_index'][selected],np.flatnonzero(mask))
                np.testing.assert_array_equal(x[selected],src['policy_observation'][mask,slot])
                np.testing.assert_array_equal(y[selected],src['actions'][::20,slot][mask])
        assert z['teacher_executed'][z['source_kind']>0].all()
    model,value=load_actor(a.training/'actor.pt');train=read(a.training/'result.json')
    with torch.no_grad():e=(model(torch.from_numpy(x))-torch.from_numpy(y)).square().mean(-1).numpy()
    mse=[float(e[stages==i].mean()) for i in range(4)]
    np.testing.assert_allclose(mse,train['after']['stage_mse'],atol=2e-10,rtol=1e-4)
    steps=sorted({int(s['step']) for s in value['optimizer_state_dict']['state'].values()});assert steps==[train['steps']]
    initial=torch.load(a.training/'initial_actor_state.pt',map_location='cpu',weights_only=True)
    source,_=load_actor(Path(value['initialization']))
    assert all(torch.equal(initial[k],v) for k,v in source.state_dict().items())
    records=[]
    for folder in a.collection+[a.autonomous]:
        require(folder);result=read(folder/'result.json');v=read(folder/'verification.json')
        assert read(folder/'session.json')['status']=='completed' and v['independent_physics_verified']
        for name,digest in result['trace_hashes'].items():assert sha(folder/name)==digest
        assert result['observation_passes']==sum(c['observation_pass'] for c in v['cases'])
        records.append(dict(run=str(folder.resolve()),cases=len(result['cases']),passes=result['observation_passes'],
            autonomous=result['autonomous'],physics_samples=v['physics_samples'],verification_sha256=sha(folder/'verification.json')))
    evaluation=read(a.autonomous/'verification.json')
    assert evaluation['autonomous'] and not evaluation['teacher_constructed']
    student_gate=evaluation['autonomous_10s_gate']
    for file in list(HERE.glob('*.py'))+list((HERE/'analysis').glob('*.py'))+list((HERE/'tests').glob('*.py')):
        ast.parse(file.read_text())
    report=dict(at=now(),protected_legacy_files=protected,parent_sources=len(parent.sources()),baseline_sources=len(baseline.sources()),
        old_actors_preserved=True,dataset_frames=len(x),dataset_source_counts=meta['source_counts'],
        student_execution_frames_excluded=meta['student_execution_excluded'],teacher_only_corrections_verified=True,
        actor_sha256=sha(a.training/'actor.pt'),cpu_stage_mse=mse,optimizer_steps=steps,initial_weights_exact=True,
        runs=records,physics_samples=sum(r['physics_samples'] for r in records),
        autonomous_10s_gate=student_gate,robot_ppo_update_save_resume_verified=False,robot_ppo_updates=0,ppo_admission=False,
        remaining=['real PPO update/save/resume'] if student_gate else ['six autonomous extra10s','real PPO update/save/resume'],
        analysis_sources={str(f.relative_to(HERE)):sha(f) for f in (HERE/'analysis').glob('*.py')},
        test_sources={str(f.relative_to(HERE)):sha(f) for f in (HERE/'tests').glob('*.py')})
    write(HERE/'validation/final_checks.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
