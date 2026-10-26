"""离线核对教师请求与同一20ms执行区间；成功判据仍取原完整协议。"""
import argparse,json
from pathlib import Path
import numpy as np
from model import Model
from pose_search import basis


def stats(x):
    a=np.asarray(x,dtype=float)
    if not a.size:return None
    return dict(p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),max=float(np.max(a)))


def audit(run):
    report=json.loads((run/'evaluation.json').read_text())
    qp=json.loads((run/'qp_trace.json').read_text())
    model=Model();results=[]
    for i,case in enumerate(report['cases']):
        seed=case['seed'];prefix=f'seed{seed}_case{i:02d}'
        with np.load(run/(prefix+'_physics_trace.npz')) as raw:z={k:raw[k] for k in raw.files}
        with np.load(run/(prefix+'_motor_trace.npz')) as raw:mot={k:raw[k] for k in raw.files}
        stop=case.get('teacher_failure',{}).get('step',case['samples'])
        B=basis(case['case']['slope_deg']);normal=B[:,2]
        rows=[]
        for k in range(1,stop):
            info=qp[k].get(str(i),{})
            if not info.get('success') or 'tau' not in info:continue
            start=k*20;end=start+20;prior=start-1
            tau=np.asarray(info['tau']);applied=mot['applied'][start:end]
            f=np.asarray(info['contact_forces'])@normal
            measured=z['contact_force_w'][start:end]@normal
            qdd=np.asarray(info['acceleration'])[6:]
            actual_qdd=(z['joint_velocity'][end-1]-z['joint_velocity'][prior])/.02
            state=model.state(z['body_pos_w'][prior],z['body_quat_w'][prior],z['joint_pos'][prior])
            _,points,_=model.contacts(state,normal)
            geometric=z['native_contact_centroid_w'][prior]
            delta=points-geometric
            rows.append(dict(step=k,torque_peak_error=float(np.max(abs(applied-tau))),
                torque_mean_error=float(np.max(abs(applied.mean(0)-tau))),normal_mean_error=float(np.max(abs(f-measured.mean(0)))),
                mean_load_min=float(measured.mean(0).min()),predicted_load_min=float(f.min()),
                qdd_error=float(np.max(abs(actual_qdd-qdd))),
                contact_point_normal_error=float(np.max(abs(delta@normal))),
                contact_point_tangent_error=float(np.max(np.linalg.norm(delta@B[:,:2],axis=1)))))
        n=stop*20;load=z['contact_force_w'][:n]@normal
        gap=z['native_contact_min_separation_m'][:n];count=z['native_contact_count'][:n]
        closed=(count>0)&(gap<=1.e-4)&(gap>=-1.e-3)
        hold=slice(max(0,n-5000),n)
        result=dict(case=case['case'],success=case['success'],complete=case['complete'],teacher_failure=case.get('teacher_failure'),
            valid_teacher_frames=len(rows),fallback_excluded=stop<case['samples'],
            errors={key:stats([x[key] for x in rows]) for key in rows[0] if key!='step'} if rows else {},
            last5s_is_acceptance_hold=bool(case['complete'] and (not case['case']['speed'] or case.get('stop_time_s') is not None)),
            last5s_diagnostics=dict(instant_four_load_fraction=float(np.mean(np.all(load[hold]>=1,axis=1))),
                geometry_four_closed_fraction=float(np.mean(np.all(closed[hold],axis=1))),
                geometry_closed_zero_load_samples=int(np.sum(closed[hold]&(load[hold]<1))),
                mean_normal_N=load[hold].mean(0).tolist()),frames=rows)
        if 'native_contact_moment_com_w' in z:
            residuals=[];cop_residuals=[];impulse_errors=[]
            for row in rows:
                k=row['step'];j=k*20-1;end=j+20
                info=qp[k][str(i)];points=np.asarray(info['model_contact_points_w'])
                com=z['whole_com_w'][j];force=z['native_patch_force_w'][j]+z['friction_force_w'][j]
                actual=(z['native_contact_moment_com_w'][j]+z['native_friction_moment_com_w'][j]).sum(0)
                residuals.append(np.linalg.norm(np.cross(points-com,force).sum(0)-actual))
                cop_residuals.append(np.linalg.norm(np.cross(z['native_contact_cop_w'][j]-com,force).sum(0)-actual))
                mass=z['whole_mass_kg'][j]
                total=(z['native_patch_force_w'][j+1:end+1]+z['friction_force_w'][j+1:end+1]).sum(1)+z['applied_drawbar_force_w'][j+1:end+1]+np.array([0.,0.,-9.81*mass])
                impulse=total.sum(0)*.001
                dp=mass*(z['whole_com_velocity_w'][end]-z['whole_com_velocity_w'][j])
                impulse_errors.append(np.linalg.norm(impulse-dp))
            result['native_wrench_recorded']=True
            result['same_state_contact_moment_error_Nm']=dict(ideal_point=stats(residuals),native_normal_cop=stats(cop_residuals))
            result['whole_com_linear_impulse_error_Ns']=stats(impulse_errors)
            result['normal_force_reconstruction_max_N']=float(np.max(abs(z['native_patch_force_w'][:n]-z['native_ground_normal_w'][:n])))
            result['body_angular_velocity_max_rad_s']=np.linalg.norm(z['body_angular_velocity_w'][:n],axis=-1).max(0).tolist()
        results.append(result)
    return dict(source=str(run.resolve()),measurement='QP frame k compared with physical substeps 20k..20k+19; prior state 20k-1',
        teacher_passed=bool(results) and all(c['success'] and c['complete'] and not c['teacher_failure'] for c in results),
        ppo_ready=False,cases=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--require-pass',action='store_true');a=p.parse_args()
    result=audit(a.run);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    for c in result['cases']:
        e=c['errors'];print('EXECUTION_AUDIT',c['case'],'success',c['success'],'torque mean error',e.get('torque_mean_error'),'normal error',e.get('normal_mean_error'),'contact normal error',e.get('contact_point_normal_error'),flush=True)
    print('TEACHER_PASS',result['teacher_passed'],flush=True)
    return 2 if a.require_pass and not result['teacher_passed'] else 0

if __name__=='__main__':raise SystemExit(main())
