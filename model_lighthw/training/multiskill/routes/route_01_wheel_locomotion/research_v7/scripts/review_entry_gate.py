"""从原生轨迹重新审查迁移前的普通初态、CPU/GPU与上下坡证据；不启动训练。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from real_cart_audit import analyze


def readiness(rows):
    gates={}
    for slope,label in [(25.,'uphill'),(-20.,'downhill')]:
        for device in ['cpu','cuda']:
            for initial in ['loaded','ordinary']:
                name=f'{label}_{device}_{initial}'
                gates[name]=[r['run'] for r in rows if r['slope_deg']==slope and r['device'].startswith(device)
                    and r['initial_state']==initial and r['complete_teacher_pass'] and r['source_valid']]
    # 迁移采用目标GPU的普通初态完整教师；CPU保留为独立物理对照，不要求跨后端逐值相同。
    required=[f'{direction}_cuda_ordinary' for direction in ['uphill','downhill']]
    profiles={}
    for profile in sorted({r.get('physics_profile','unspecified') for r in rows}):
        available={f"{'uphill' if r['slope_deg']==25. else 'downhill'}_{'cuda' if r['device'].startswith('cuda') else 'cpu'}_{r['initial_state']}"
            for r in rows if r.get('physics_profile','unspecified')==profile and r['complete_teacher_pass'] and r['source_valid']}
        profiles[profile]=[name for name in required if name not in available]
    missing=min(profiles.values(),key=len) if profiles else list(required)
    return dict(gates=gates,required_gates=required,missing=missing,physics_profiles_missing=profiles,
        ready_for_teacher_collection=bool(profiles) and not missing,
        unified_actor_accepted=False,formal_ppo_authorized=False,
        note='Gate only permits the next teacher-data collection stage. No Actor training/export or PPO is launched.')


def review(runs,output):
    rows=[]
    for run in runs:
        run=Path(run).resolve(strict=True)
        request=json.loads((run/'request.json').read_text());report=analyze(run,write_report=False)
        bad=[]
        for filename in ['source_manifest.json','research_source_manifest.json']:
            manifest=json.loads((run/filename).read_text())
            for relative,sha in manifest.items():
                archive=run/'source_snapshot'/relative
                if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=sha:bad.append(relative)
        with np.load(run/'seed43_case00_policy_trace.npz',allow_pickle=False) as trace:
            valid_observation=trace['observation'].ndim==2 and trace['observation'].shape[1]==300 and np.isfinite(trace['observation']).all()
            valid_action=trace['executed_action'].shape==(len(trace['observation']),16) and np.isfinite(trace['executed_action']).all()
        runtime=json.loads((run/'runtime_check.json').read_text())
        matched=(request['physical_cart_mass_kg']==40 and request['ground_friction']==3 and request['rounded_cart_wheels']
            and request['execution_physics']['hitch_height_m']==.04 and request['execution_physics']['robot_hitch_offset_m']==.2
            and request['execution_physics']['rolling_coefficient']==.02 and request['execution_physics']['base_com_shift_m']==0)
        initial_path=run/'initial_state_runtime.json';initial=json.loads(initial_path.read_text()) if initial_path.exists() else {}
        gpu_confirmed=(not request['execution_physics']['device'].startswith('cuda') or
            (initial.get('cuda_available') and str(initial.get('robot_tensor_device','')).startswith('cuda')))
        ordinary_confirmed=True
        if request.get('initial_state_mode')=='ordinary':
            slope=np.deg2rad(request['slope_deg']);normal=np.array([-np.sin(slope),0,np.cos(slope)])
            default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])
            ordinary_confirmed=bool(initial and 'env_origin' in initial and initial.get('episode_step')==0 and initial.get('cart_mass_kg')==40
                and np.allclose(initial['joint_position_action_order'],default,atol=1.e-6)
                and np.allclose(initial['robot_root_state'][7:],0.,atol=1.e-6)
                and np.allclose(initial['cart_root_state'][7:],0.,atol=1.e-6)
                and abs((np.asarray(initial['robot_root_state'][:3])-initial['env_origin'])@normal-.416223)<1.e-5)
        profile=dict(request['execution_physics']);profile.pop('device',None)
        profile.update(robot_iterations=[request['position_iterations'],request['velocity_iterations']],
            cart_iterations=request['cart_solver_iterations'],rounded_cart_wheels=request['rounded_cart_wheels'],mass_kg=request['physical_cart_mass_kg'])
        profile_key=json.dumps(profile,sort_keys=True,separators=(',',':'))
        rows.append(dict(run=str(run),physics_profile=profile_key,slope_deg=request.get('slope_deg',25.),device=request['execution_physics']['device'],
            initial_state=request.get('initial_state_mode','loaded'),duration_s=report['duration_s'],
            cart_net_route_m=report['cart_net_route_m'],cart_max_progress_m=report['cart_max_progress_from_move_start_m'],
            max_tilt_deg=report['max_robot_tilt_deg'],complete_teacher_pass=report['move_brake_hold_pass'],
            hold=report['hold_2s_transition_5s'],failed_checks=[key for key,value in report['checks'].items() if not value],
            source_valid=bool(not bad and valid_observation and valid_action and matched and gpu_confirmed and ordinary_confirmed and not request['noise'] and not runtime['policy_enable_corruption']),
            collection_evidence_checks=dict(snapshot=not bad,observation=bool(valid_observation),action=bool(valid_action),
                physical_model=bool(matched),gpu_runtime=bool(gpu_confirmed),ordinary_reset=ordinary_confirmed),
            bad_snapshot_files=bad,request_sha256=hashlib.sha256((run/'request.json').read_bytes()).hexdigest(),
            physics_trace_sha256=hashlib.sha256((run/'seed43_case00_physics_trace.npz').read_bytes()).hexdigest(),
            policy_trace_sha256=hashlib.sha256((run/'seed43_case00_policy_trace.npz').read_bytes()).hexdigest()))
    result=dict(schema='r1_entry_gpu_braking_gate_v1',completed_runs=len(rows),runs=rows,**readiness(rows))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-ready',action='store_true');args=parser.parse_args()
    result=review(args.runs,args.output)
    print(json.dumps({key:result[key] for key in ['completed_runs','missing','ready_for_teacher_collection']},ensure_ascii=False))
    if args.require_ready and not result['ready_for_teacher_collection']:parser.exit(2,'Physical entry/GPU/braking prerequisites remain incomplete; migration is not ready.\n')
