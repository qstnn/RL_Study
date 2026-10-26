"""从冻结原始轨迹复算校准结果；短测与完整门槛分别报告。"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
from probe import REPO,V3,SOURCES,SPEC,PROTOCOL,sha256,write_json
from review import load_actor

def run(root):
    torch.set_num_threads(1)
    summary=json.loads((root/'summary.json').read_text());plan=json.loads((root/'plan.json').read_text())
    assert summary['status']=='completed'
    source=json.loads((root/'source_manifest.json').read_text());drift=[]
    for relative,digest in source.items():
        assert sha256(root/'source_snapshot'/relative)==digest,relative
        if sha256(REPO/relative)!=digest:drift.append(relative)
    # 先前短测后允许诊断脚本补进度/观测，历史快照仍逐字节保留；生产代码不能漂移。
    assert all('/diagnostics/calibration/' in p for p in drift),drift
    for item in SOURCES.values():assert sha256(item['checkpoint'])==item['sha256']
    treatment=plan['treatment'];limits=np.asarray([treatment.get('leg_limit',96.)]*12+[treatment.get('wheel_limit',48.)]*4)
    rows=[];checks=[];initial=None
    for candidate in summary['candidates']:
        name=candidate['candidate'];folder=root/name;e=json.loads((folder/'evaluation.json').read_text())
        p=np.load(folder/'policy_inputs.npz');actor=load_actor(name);error=0.;total=0
        assert e['sha256']==SOURCES[name]['sha256']
        lengths=p['valid_case_lengths'];hist=p['observation_history'];obs=p['source_observation'];raw=p['raw_action']
        assert hist.shape[:2]==obs.shape[:2]==raw.shape[:2] and hist.shape[1:]==(16,290)
        assert all(np.isfinite(a).all() for a in (hist,obs,raw))
        assert np.array_equal(obs[:,:,3:],hist[:,:,SPEC.latest_source_columns()]) if obs.shape[-1]==60 else np.array_equal(obs,hist)
        state=np.asarray([c['initial_root_state'] for c in e['cases']])
        if initial is None:initial=state
        initial_error=float(np.abs(state-initial).max());assert initial_error<1e-5
        for i,c in enumerate(e['cases']):
            prefix=f'seed43_case{i:02d}';n=c['samples'];assert lengths[i]==n
            with (folder/(prefix+'.csv')).open() as stream:series=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
            computed=PROTOCOL.summarize(series,c['case'],c['failure_reason'])
            assert all(c[k]==v for k,v in computed.items()),(name,i,'protocol')
            with torch.inference_mode():pred=actor(torch.from_numpy(obs[:n,i]))
            error=max(error,float((pred-torch.from_numpy(raw[:n,i])).abs().max()));assert error<2e-4
            z=np.load(folder/(prefix+'_motor_trace.npz'))
            assert all(np.isfinite(z[k]).all() and z[k].shape==(20*n,16) for k in ('applied','requested','velocity'))
            assert np.array_equal(z['limits'],limits) and list(z['joint_names'])==SPEC.CONTRACT['action_order']
            assert np.all(abs(z['applied'])<=limits+1e-4);total+=20*n
            rows.append(dict(candidate=name,variant=plan['variant'],index=i,**c['case'],duration_s=c['observed']['duration_s'],complete=c['complete'],success=c['success'],termination='|'.join(c['observed']['termination_causes']),failure_reason=c['failure_reason'],progress_m=c['observed']['progress_m'],**c.get('metrics',{}),failed_checks='|'.join(c['failed_checks']),leg_peak_Nm=float(abs(z['applied'][:,:12]).max()),wheel_peak_Nm=float(abs(z['applied'][:,12:]).max()),leg_clipped_fraction=float((abs(z['requested'][:,:12])>limits[:12]+1e-4).mean()),wheel_clipped_fraction=float((abs(z['requested'][:,12:])>limits[12:]+1e-4).mean())))
        physical=json.loads((folder/'physical.json').read_text());assert len(physical['bodies'])==272
        for body in physical['bodies']:
            expected=treatment.get('wheel_cap',1000.) if body['name'].endswith('_wheel_link') else 1000.
            assert body['angular_cap_deg_s']==expected
        assert np.allclose(physical['wheel_damping'],treatment.get('wheel_damping',3.))
        assert np.allclose(np.asarray(physical['material'])[:,:,:2],treatment.get('friction',2.3))
        w=np.load(folder/'wheel_body_angular_velocity.npz');assert np.array_equal(w['valid_case_lengths'],lengths)
        assert w['angular_velocity_world_rad_s'].shape==(*hist.shape[:2],4,3) and np.isfinite(w['angular_velocity_world_rad_s']).all()
        runtime=json.loads((folder/'runtime_check.json').read_text())
        # 已完成旧版本探针的原报告保持；实际物理值单列，避免常数3冒充D1验证。
        runtime_damping=runtime['wheel_damping'];actual_damping=float(np.asarray(physical['wheel_damping']).flat[0])
        checks.append(dict(candidate=name,case_count=16,source_action_error=error,initial_state_error=initial_error,physics_sample_times=total,physical_caps_verified=True,zero_payload_equivalence=physical.get('zero_payload_equivalence'),actual_wheel_damping=actual_damping,legacy_runtime_wheel_damping=runtime_damping,legacy_runtime_report_mismatch=runtime_damping!=actual_damping))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'case_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result=dict(status='passed',full_protocol=plan['full_protocol'],source_snapshot_files=len(source),live_diagnostic_drift=drift,source_checkpoints_unchanged=True,candidates=checks,ppo_updates=0,note='Numerical and physical parameter audit is separate from behavioral acceptance.')
    write_json(root/'audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args();print(json.dumps(run(args.directory.resolve()),ensure_ascii=False,indent=2))
