"""从完整 1 kHz 原始量复算起跳、整机角动量和首次触地前旋转。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from dynamics import centroidal,rotation


def air_segments(mask):
    """返回连续候选离地区间；几何净空转正可晚于法向力降为零。"""
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))


def analyze(run):
    run=Path(run);meta=json.loads((run/'physics.json').read_text());req=json.loads((run/'request.json').read_text())
    loaded=np.load(run/'trace.npz',allow_pickle=False);data={k:loaded[k] for k in loaded.files};loaded.close();t=data['t'];rows=[];curves={}
    wheel=[meta['contact_body_names'].index(x) for x in ('FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link')]
    other=[i for i in range(len(meta['contact_body_names'])) if i not in wheel]
    for k,profile in enumerate(req['profiles']):
        p=data['body_com_position'][:,k];v=data['body_com_velocity'][:,k]
        c,cv,h,locked,spin,orbital=centroidal(p,v,data['body_link_quaternion'][:,k],data['body_angular_velocity'][:,k],meta['masses'],meta['inertias_link_axes'])
        np.testing.assert_allclose(c,data['com'][:,k],atol=1e-5,rtol=0)
        np.testing.assert_allclose(cv,data['com_velocity'][:,k],atol=1e-5,rtol=0)
        q=data['root_state'][:,k,3:7];rot=rotation(q)
        axis=rot[0,:,1];forward=rot[0,:,0];forward[2]=0.;forward/=np.linalg.norm(forward)
        angle=np.unwrap(np.arctan2(rot[:,2,0],np.einsum('ti,i->t',rot[:,:,0],forward)))
        angle-=angle[0]
        hy=h@axis;iy=np.einsum('i,tij,j->t',axis,locked,axis)
        forces=np.linalg.norm(data['contact_force'][:,k],axis=-1)
        clear=(forces<2.).all(-1)&(data['wheel_clearance'][:,k]>0.).all(-1)
        candidate=(t>=.65)&clear&(cv[:,2]>.25)
        segments=air_segments(candidate)
        valid=[(a,b) for a,b in segments if b-a>=7]
        release=[valid[0][0]] if valid else []
        base=(t>=.2)&(t<=.49);pre=(t>=.5)&(t<=1.0)
        crouch_index=np.flatnonzero(pre)[np.argmin(c[pre,2])]
        row=dict(index=k,name=profile['name'],profile=profile,seed=req['seed'],released=bool(len(release)),
            stand_com_m=float(c[base,2].mean()),crouch_com_m=float(c[crouch_index,2]),
            crouch_depth_m=float(c[base,2].mean()-c[crouch_index,2]),
            terminal_j1_success=bool(data['legacy_success'][-1,k] and not data['legacy_failed'][-1,k]),
            terminal_stable_s=float(data['legacy_stable'][-1,k]),
            applied_torque_peak_Nm=float(np.abs(data['applied_torque'][1:,k]).max()),
            leg_speed_peak_rad_s=float(np.abs(data['joint_velocity'][1:,k, [i for i,n in enumerate(meta['joint_names']) if 'wheel' not in n]]).max()),
            rejected_short_air_gaps=sum(b-a<7 for a,b in segments),
            first_flight_turn_complete=False,full_backflip_success=False)
        if len(release):
            a=int(release[0]);landing=np.flatnonzero((np.arange(len(t))>a)&(forces>=2.).any(-1))
            b=int(landing[0]) if len(landing) else len(t)-1
            flight=slice(a,b);clean=(np.arange(len(t))>=a+3)&(np.arange(len(t))<b-3)&(forces<2.).all(-1)
            hrange=np.ptp(h[clean],axis=0) if clean.sum()>2 else np.full(3,np.nan)
            # 测得 H 不等于可执行刚体控制；这里仅提供固定收腿惯量的筛查量级。
            duration=float(t[b]-t[a]);hyp=float(-hy[a]);needed=.40910371*2*np.pi/duration if duration else 0.
            pitch_at_contact=float(angle[b]*180/np.pi)
            row.update(release_s=float(t[a]),release_vz_m_s=float(cv[a,2]),release_com_m=float(c[a,2]),
                flight_s=duration,flight_censored=not bool(len(landing)),flight_rise_m=float(c[flight,2].max()-c[a,2]),
                standing_rise_m=float(c[flight,2].max()-c[base,2].mean()),
                wheel_clearance_m=float(data['wheel_clearance'][flight,k].min(-1).max()),
                backflip_H_kg_m2_s=hyp,release_H_world=h[a].tolist(),
                release_spin_pitch=float(-(spin[a].sum(0)@axis)),release_orbital_pitch=float(-(orbital[a].sum(0)@axis)),
                release_locked_Iyy=float(iy[a]),min_flight_locked_Iyy=float(iy[flight].min()),
                fixed_tuck_required_H=needed,fixed_tuck_predicted_rotation_deg=float(hyp*duration/.40910371*180/np.pi),
                release_backflip_deg=float(angle[a]*180/np.pi),contact_backflip_deg=pitch_at_contact,
                first_flight_rotation_deg=float((angle[b]-angle[a])*180/np.pi),
                max_backflip_before_contact_deg=float(angle[a:b+1].max()*180/np.pi),
                max_lateral_axis_tilt_deg=float(np.arcsin(np.clip(np.abs(rot[a:b+1,2,1]),0,1)).max()*180/np.pi),
                angular_momentum_flight_range=hrange.tolist() if np.isfinite(hrange).all() else None,
                linear_ballistic_vz_residual=float(np.ptp(cv[clean,2]+9.81*t[clean])) if clean.any() else None,
                first_contact_s=float(t[b]),first_contact_body=bool((forces[b,other]>5.).any()),
                body_contact_before_release=bool((forces[(t>.5)&(t<t[a])][:,other]>5.).any()),
                first_flight_turn_complete=bool(len(landing) and 350<=(angle[b]-angle[a])*180/np.pi<=385),
                max_contact_force_N=float(np.linalg.norm(data['contact_force'][:,k],axis=-1).sum(-1).max()))
            # 完整翻转还必须四轮落地、无身体碰撞、末段至少 1 s 直立低速。
            post=t>=max(t[b]+.3,t[-1]-1.)
            upright=data['tilt'][:,k]<np.deg2rad(15)
            stable=upright&(np.linalg.norm(cv[:,:2],axis=-1)<.05)&(np.abs(cv[:,2])<.15)&(forces[:,wheel]>5.).all(-1)
            no_body=not (forces[(t>.5),:][:,other]>5.).any()
            row['full_backflip_success']=bool(row['first_flight_turn_complete'] and no_body and row['max_lateral_axis_tilt_deg']<20. and upright[b] and post.sum()>=1000 and stable[post].all())
        rows.append(row);curves[f'case{k}_com']=c;curves[f'case{k}_H']=h;curves[f'case{k}_Iyy']=iy;curves[f'case{k}_backflip_angle']=angle
    report=dict(analysis_version=3,min_clear_samples=7,contact_threshold_N=2.,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='analyzed',cases=rows,profiles=len(rows),ppo_updates=0,
        assumptions=['Fixed-tuck H estimate is a screening approximation, not full multibody or collision feasibility.',
                     'Backflip is negative rotation about initial body +Y. Air interval ends at ANY link contact, not only wheels.',
                     'All trials retain original 48/24 Nm and 12.57/37.7 rad/s configured motor limits.'],
        source_verified=json.loads((run/'source_check.json').read_text())['passed'])
    (run/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    keys=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    keys+=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(dict,list))}-set(keys))
    with (run/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(run/'dynamics.npz',t=t,**curves)
    best=sorted([r for r in rows if r['released']],key=lambda r:r['flight_rise_m'],reverse=True)
    print(json.dumps([ {k:r.get(k) for k in ('index','name','flight_rise_m','flight_s','backflip_H_kg_m2_s','max_backflip_before_contact_deg','terminal_j1_success','full_backflip_success')} for r in best[:8]],ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)
