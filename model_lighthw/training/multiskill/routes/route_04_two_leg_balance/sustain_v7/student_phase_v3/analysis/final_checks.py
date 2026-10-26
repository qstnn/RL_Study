"""核对单因素、模型/Adam、教师全关和独立物理结论。"""
from pathlib import Path
import ast
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_phase import read,write,sha,now,require,load_actor,previous,parent,DATASET,DATASET_SHA,ACTOR


def main():
    torch.set_num_threads(4)
    models=[HERE/'runs/2026-10-23_uniform',HERE/'runs/2026-10-23_observed']
    records=[read(p/'result.json') for p in models]
    for key in ['parameter_count','dataset_sha256','dataset_frames','steps','batch_size','sample_sha256','device']:
        assert records[0][key]==records[1][key],key
    assert records[0]['routing']=='uniform' and records[1]['routing']=='observed'
    states=[torch.load(p/'initial_actor_state.pt',map_location='cpu',weights_only=True) for p in models]
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert sha(DATASET/'dataset.npz')==DATASET_SHA
    with np.load(DATASET/'dataset.npz') as z:
        x=torch.from_numpy(z['observation']);y=torch.from_numpy(z['action']);stage=z['stage']
    original,_=previous.load_actor(ACTOR)
    assert all(torch.equal(states[0]['trunk.'+k],v) for k,v in original.state_dict().items() if not k.startswith('6.'))
    for name in ['rise','capture']:
        for k in ['weight','bias']:assert torch.equal(states[0][name+'.'+k],original.state_dict()['6.'+k])
    mse=[]
    for folder,row in zip(models,records):
        model,value=load_actor(folder/'actor.pt')
        with torch.no_grad():error=(model(x)-y).square().mean(-1).numpy()
        actual=[float(error[stage==i].mean()) for i in range(4)]
        np.testing.assert_allclose(actual,row['after']['stage_mse'],rtol=1e-4,atol=2e-10)
        assert {int(v['step']) for v in value['optimizer_state_dict']['state'].values()}=={6000}
        mse.append(actual)
    runs=[]
    for name in ['2026-10-23_uniform_screen','2026-10-23_observed_screen']:
        folder=HERE/'runs'/name;require(folder)
        assert read(folder/'session.json')['status']=='completed'
        result=read(folder/'result.json');v=read(folder/'verification.json')
        assert result['autonomous'] and v['autonomous'] and not v['teacher_constructed']
        assert v['independent_physics_verified']
        for file,digest in result['trace_hashes'].items():assert sha(folder/file)==digest
        runs.append(dict(run=str(folder),cases=len(v['cases']),passes=sum(c['observation_pass'] for c in v['cases']),
            physics_samples=v['physics_samples'],autonomous_10s_gate=v['autonomous_10s_gate'],
            verification_sha256=sha(folder/'verification.json')))
    protected=0
    for name in ['source_manifest.json','comparison_source_manifest.json']:
        for rel,digest in read(parent.CONTROL/name).items():
            assert sha(parent.REPO/rel)==sha(parent.CONTROL/'source_snapshot'/rel)==digest;protected+=1
    for oldrun in ['2026-10-23_bc_03','2026-10-23_bc_03_autonomous']:
        previous.require(previous.HERE/'runs'/oldrun)
    for file in list(HERE.glob('*.py'))+list((HERE/'analysis').glob('*.py'))+list((HERE/'tests').glob('*.py')):ast.parse(file.read_text())
    write(HERE/'validation/final_checks.json',dict(at=now(),matched_capacity=True,matched_initial_weights=True,
        matched_minibatch_sequence=True,matched_data_budget_optimizer=True,parameter_count=records[0]['parameter_count'],
        sample_sha256=records[0]['sample_sha256'],dataset_sha256=DATASET_SHA,dataset_frames=7726,
        original_initialization_verified=True,cpu_stage_mse=mse,legacy_protected_sources=protected,
        old_student03_sources_preserved=True,runs=runs,physics_samples=sum(r['physics_samples'] for r in runs),
        robot_ppo_updates=0,robot_ppo_update_save_resume_verified=False,ppo_admission=False,
        analysis_sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'analysis').glob('*.py')}))
    print('Matched capacity, initialization, data, sample sequence, optimizer budget and independent autonomous verification passed.')


if __name__=='__main__':main()
