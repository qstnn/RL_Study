"""可复算的物理重复性与教师行为门；进程正常退出不等于通过。"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from settings import GROUND_FRICTION

# 在运行前固定的研究容差，不根据某次试验结果自动放宽。
REPEAT_TOLERANCES=dict(body_position_m=.002,whole_com_m=.002,attitude_deg=.5,joint_position_rad=.01,
                       joint_velocity_rps=.5,normal_force_N=5.,friction_force_N=5.,applied_torque_Nm=1.)

def compare(reference,candidate):
    with np.load(reference) as a,np.load(candidate) as b:
        n=min(len(a['body_pos_w']),len(b['body_pos_w']))
        if n<2800:return dict(passed=False,reason='incomplete_2_8s_physical_control',samples=n)
        def gap(k):return float(np.max(np.abs(a[k][:n]-b[k][:n])))
        qa=Rotation.from_quat(np.roll(a['body_quat_w'][:n],-1,axis=1))
        qb=Rotation.from_quat(np.roll(b['body_quat_w'][:n],-1,axis=1))
        metrics=dict(body_position_m=gap('body_pos_w'),whole_com_m=gap('whole_com_w'),attitude_deg=float(np.rad2deg((qa.inv()*qb).magnitude()).max()),
            joint_position_rad=gap('joint_pos'),joint_velocity_rps=gap('joint_velocity'),normal_force_N=gap('contact_force_w'),friction_force_N=gap('friction_force_w'))
    ma=np.load(str(reference).replace('_physics_trace.npz','_motor_trace.npz'))
    mb=np.load(str(candidate).replace('_physics_trace.npz','_motor_trace.npz'))
    metrics['applied_torque_Nm']=float(np.max(abs(ma['applied'][:n]-mb['applied'][:n])))
    checks={k:np.isfinite(v) and v<=REPEAT_TOLERANCES[k] for k,v in metrics.items()}
    return dict(passed=all(checks.values()),samples=n,metrics=metrics,checks={k:bool(v) for k,v in checks.items()},
                reference=str(reference),candidate=str(candidate))


def physics_gate(runs):
    """每个坡度对照同槽重复、单槽/批量、交换槽位；不挑表现最好槽。"""
    refs={};comparisons=[];signatures=set()
    for directory in runs:
        directory=Path(directory);request=json.loads((directory/'request.json').read_text());seed=request['seed']
        signatures.add((request['checkpoint_sha256'],request.get('ground_friction'),request['position_iterations'],request['velocity_iterations'],request.get('actor_inference','batch')))
        if request.get('ground_friction')!=GROUND_FRICTION or request['noise'] or request.get('controller')!='frozen_baseline':
            return dict(passed=False,reason='physics_assumption_mismatch')
        for i,case in enumerate(request['cases']):
            path=directory/f'seed{seed}_case{i:02d}_physics_trace.npz';key=case['slope_deg']
            if key not in refs:refs[key]=path
            else:comparisons.append(compare(refs[key],path))
    return dict(passed=len(signatures)==1 and set(refs)=={25.,-20.} and len(comparisons)>=6 and all(x['passed'] for x in comparisons),tolerances=REPEAT_TOLERANCES,
                comparisons=comparisons,scope='2.8s frozen Actor with a fixed inference mode; action batching and perturbation robustness are separate.  two slopes, same-slot repeat, clone and slot swap. Not a guarantee for every future state.')


def teacher_gate(runs):
    results=[];signatures=set()
    for path in runs:
        path=Path(path);request=json.loads((path/'request.json').read_text());report=json.loads((path/'evaluation.json').read_text())
        manifest=json.loads((path/'research_source_manifest.json').read_text()) if (path/'research_source_manifest.json').exists() else {}
        signature=dict(checkpoint=request.get('checkpoint_sha256'),solver=[request.get('position_iterations'),request.get('velocity_iterations')],
            inference=request.get('actor_inference'),pose=request.get('pose'),options=request.get('teacher_options',{}),
            code={k:v for k,v in manifest.items() if '/scripts/' in k and k.endswith('.py')})
        signatures.add(json.dumps(signature,sort_keys=True))
        valid=report.get('teacher') and not report.get('short_diagnostic') and request['noise'] and request.get('ground_friction')==GROUND_FRICTION
        cases=report['cases'];keys={(r['case']['slope_deg'],r['case']['speed']) for r in cases}
        valid=valid and keys=={(s,v) for s in (25.,-20.) for v in (0.,.3)} and len(cases)==4
        valid=valid and all(c['success'] and c['complete'] for c in cases) and not report.get('teacher_failures')
        results.append(dict(path=str(path),seed=request['seed'],passed=bool(valid),pass_count=report['pass_count'],
            pose=request['pose']['id'],layout=[(c['slope_deg'],c['speed']) for c in request['cases']]))
    seeds={r['seed'] for r in results};poses={r['pose'] for r in results}
    coverage=len(signatures)==1 and len(seeds)>=3 and len(poses)==1 and all(sum(r['seed']==s for r in results)==2 for s in seeds)
    coverage=coverage and all(len({str(r['layout']) for r in results if r['seed']==s})==2 for s in seeds)
    return dict(passed=bool(coverage and all(r['passed'] for r in results)),coverage=coverage,runs=results,
        scope='One candidate/controller/source/physics; 3 observation-noise seeds x 2 slot layouts x 4 full cases. Fixed reset states; initial-state perturbation acceptance remains separate.')
