"""核对记忆单因素、无新动作标签、来源和真实行为证据。"""
from pathlib import Path
import sys
import ast
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_memory import previous,parent,DATASET,RAW_DATASET,RAW_SHA,read,write,sha,now,require,initial_actor,load_actor
from capture_memory import augment


def main():
    torch.set_num_threads(4);require(DATASET)
    with np.load(DATASET/'dataset.npz') as z:d={k:z[k] for k in z.files}
    with np.load(RAW_DATASET/'dataset.npz') as z:
        for key in z.files:np.testing.assert_array_equal(d[key],z[key])
    assert sha(RAW_DATASET/'dataset.npz')==RAW_SHA
    # 用上一轮独立真实状态教师重放的触发时刻检查，绝不用于新增动作标签。
    label_file=previous.HERE/'validation/phase_diagnostic_labels.npz'
    with np.load(label_file) as z:
        triggered=z['triggered'];u=z['blend_progress']
    expected=np.stack((triggered,np.where(triggered,u*u*(3.-2.*u),0.)),axis=-1)
    teacher_memory_error=float(abs(d['capture_memory']-expected).max())
    assert teacher_memory_error<3e-6
    folders=[HERE/'runs/2026-10-23_zero',HERE/'runs/2026-10-23_exposed']
    records=[read(p/'result.json') for p in folders]
    for key in ['parameter_count','dataset_frames','dataset_sha256','sample_sha256','steps','batch_size','device']:
        assert records[0][key]==records[1][key],key
    states=[torch.load(p/'initial_actor_state.pt',map_location='cpu',weights_only=True) for p in folders]
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    original=initial_actor()
    for k,v in original.state_dict().items():
        if k=='trunk.0.weight':
            assert torch.equal(states[0][k][:,:305],v) and not states[0][k][:,305:].any()
        else:assert torch.equal(states[0][k],v),k
    mse=[];new_column_norm=[]
    for folder,row in zip(folders,records):
        model,value=load_actor(folder/'actor.pt')
        x=augment(torch.from_numpy(d['observation']),torch.from_numpy(d['capture_memory']),model.mode)
        with torch.no_grad():error=(model(x)-torch.from_numpy(d['action'])).square().mean(-1).numpy()
        actual=[float(error[d['stage']==i].mean()) for i in range(4)]
        np.testing.assert_allclose(actual,row['after']['stage_mse'],rtol=1e-4,atol=2e-10)
        assert {int(v['step']) for v in value['optimizer_state_dict']['state'].values()}=={6000}
        mse.append(actual);new_column_norm.append(float(model.trunk[0].weight[:,305:].norm().detach()))
    assert new_column_norm[0]==0
    runs=[]
    for mode in ['zero','exposed']:
        folder=HERE/f'runs/2026-10-23_{mode}_screen';require(folder)
        result=read(folder/'result.json');v=read(folder/'verification_v4_1.json')
        assert v['physics_thresholds_unchanged'] and v['memory_atol']==3e-7
        assert sha(HERE/'analysis/verify_memory_v4_1.py')==v['reviewer_sha256']
        assert v['autonomous'] and not v['teacher_constructed'] and v['independent_memory_verified'] and v['independent_physics_verified']
        for f,digest in result['trace_hashes'].items():assert sha(folder/f)==digest
        runs.append(dict(run=str(folder),cases=len(v['cases']),passes=sum(c['observation_pass'] for c in v['cases']),
            physics_samples=v['physics_samples'],verification_sha256=sha(folder/'verification_v4_1.json'),autonomous_10s_gate=v['autonomous_10s_gate']))
    protected=0
    for name in ['source_manifest.json','comparison_source_manifest.json']:
        for rel,digest in read(parent.CONTROL/name).items():
            assert sha(parent.REPO/rel)==sha(parent.CONTROL/'source_snapshot'/rel)==digest;protected+=1
    for p in list(HERE.glob('*.py'))+list((HERE/'analysis').glob('*.py'))+list((HERE/'tests').glob('*.py')):ast.parse(p.read_text())
    write(HERE/'validation/final_checks.json',dict(at=now(),original_arrays_unchanged=True,new_action_labels=0,
        raw_dataset_sha256=RAW_SHA,teacher_progress_alignment_max_error=teacher_memory_error,teacher_diagnostic_sha256=sha(label_file),
        matched_capacity=True,matched_initialization=True,explicit_zero_column_migration_verified=True,
        matched_data_sampling_budget=True,sample_sha256=records[0]['sample_sha256'],parameter_count=records[0]['parameter_count'],
        cpu_stage_mse=mse,new_memory_weight_norm=new_column_norm,protected_legacy_sources=protected,
        runs=runs,physics_samples=sum(r['physics_samples'] for r in runs),robot_ppo_updates=0,
        robot_ppo_update_save_resume_verified=False,ppo_admission=False,
        analysis_sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'analysis').glob('*.py')}))
    print('Matched memory contrast, original data/actions, teacher progress alignment, models and independent physical evidence checked.')


if __name__=='__main__':main()
