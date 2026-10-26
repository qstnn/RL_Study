"""独立从50Hz/1kHz轨迹复算冻结审查；不运行仿真或PPO。"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
from review import V3,REPO,SOURCES,load_actor,SPEC,sha256,write_json
from launch import PROTOCOL

def audit(directory):
    torch.set_num_threads(1)
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['status']=='completed'
    manifest=json.loads((directory/'source_manifest.json').read_text())
    for relative,digest in manifest.items():
        assert sha256(directory/'source_snapshot'/relative)==digest,relative
        assert sha256(REPO/relative)==digest,relative
    for source in SOURCES.values():assert sha256(source['checkpoint'])==source['sha256']
    checks=[];table=[];initial_reference=None
    for entry in summary['candidates']:
        name=entry['candidate'];folder=directory/name
        result=json.loads((folder/'evaluation.json').read_text());assert result['sha256']==SOURCES[name]['sha256']
        inputs=np.load(folder/'policy_inputs.npz')
        lengths=inputs['valid_case_lengths'];obs=inputs['observation_history'];source_obs=inputs['source_observation'];raw=inputs['raw_action']
        assert obs.shape[:2]==source_obs.shape[:2]==raw.shape[:2]
        assert obs.shape[1:]==(16,290) and source_obs.shape[2]==SOURCES[name]['input_dim']
        assert all(np.isfinite(a).all() for a in (obs,source_obs,raw))
        if SOURCES[name]['input_dim']==60:assert np.array_equal(source_obs[:,:,3:],obs[:,:,SPEC.latest_source_columns()])
        else:assert np.array_equal(source_obs,obs)
        actor=load_actor(name);max_replay_error=0.;physics_samples=0
        initial=[c['initial_root_state'] for c in result['cases']]
        if initial_reference is None:initial_reference=initial
        initial_error=float(np.max(np.abs(np.asarray(initial)-np.asarray(initial_reference))))
        assert initial_error<1e-5
        for i,case in enumerate(result['cases']):
            prefix=f"seed43_case{i:02d}";n=int(lengths[i]);assert n==case['samples']
            with (folder/(prefix+'.csv')).open() as stream:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
            computed=PROTOCOL.summarize(rows,case['case'],case['failure_reason'])
            assert all(case[k]==v for k,v in computed.items()),(name,i,'protocol mismatch')
            with torch.inference_mode():expected=actor(torch.from_numpy(source_obs[:n,i]))
            error=float((expected-torch.from_numpy(raw[:n,i])).abs().max());max_replay_error=max(max_replay_error,error)
            assert error<2e-4,(name,i,error)
            trace=np.load(folder/(prefix+'_motor_trace.npz'))
            assert trace['applied'].shape==trace['requested'].shape==trace['velocity'].shape==(20*n,16)
            assert np.array_equal(trace['limits'],np.asarray([96.]*12+[48.]*4))
            assert list(trace['joint_names'])==SPEC.CONTRACT['action_order'] and float(trace['physics_dt'])==.001
            assert all(np.isfinite(trace[k]).all() for k in ('applied','requested','velocity'))
            assert np.all(np.abs(trace['applied'])<=trace['limits']+1e-4)
            physics_samples+=len(trace['applied'])
            metrics=case.get('metrics',{});observed=case['observed']
            table.append(dict(candidate=name,index=i,**case['case'],complete=case['complete'],success=case['success'],duration_s=observed['duration_s'],termination='|'.join(observed['termination_causes']),progress_m=observed['progress_m'],**metrics,failed_checks='|'.join(case['failed_checks']),leg_peak_Nm=float(np.abs(trace['applied'][:,:12]).max()),wheel_peak_Nm=float(np.abs(trace['applied'][:,12:]).max()),leg_clipped_fraction=float((np.abs(trace['requested'][:,:12])>96.+1e-4).mean()),wheel_clipped_fraction=float((np.abs(trace['requested'][:,12:])>48.+1e-4).mean())))
        checks.append(dict(candidate=name,cases_recomputed=16,source_action_replay_max_error=max_replay_error,initial_state_difference=initial_error,physics_sample_times=physics_samples,trajectory_finite=True,limits_verified=True))
    keys=list(dict.fromkeys(k for row in table for k in row))
    with (directory/'case_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(table)
    comparison=[]
    if (directory/'control_v3/evaluation.json').exists():
        control=json.loads((directory/'control_v3/evaluation.json').read_text())
        old=json.loads((V3/'runs/2026-10-16_16-29-49.804831/periodic_evaluations/update_0500/evaluation.json').read_text())
        comparison=[]
        for i,(a,b) in enumerate(zip(control['cases'],old['cases'])):
            shared=set(a.get('metrics',{}))&set(b.get('metrics',{}))
            comparison.append(dict(index=i,old_complete=b['complete'],new_complete=a['complete'],old_success=b['success'],new_success=a['success'],old_duration=b['observed']['duration_s'],new_duration=a['observed']['duration_s'],metric_deltas={k:a['metrics'][k]-b['metrics'][k] for k in sorted(shared)}))
    report=dict(status='passed',source_snapshot_files=len(manifest),source_checkpoints_unchanged=True,candidates=checks,historical_control_comparison=comparison,ppo_updates=0,note='Numerical audit is separate from behavioral pass; teacher velocity is privileged.')
    write_json(directory/'audit.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.directory.resolve()),ensure_ascii=False,indent=2))
