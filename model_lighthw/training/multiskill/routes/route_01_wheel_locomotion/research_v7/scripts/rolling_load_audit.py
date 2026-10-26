"""允许部分车轮支撑的滚动拖载研究审查；独立于历史四轮/跟速门。

用户于2026-10-22取消同时四轮支撑要求。以下数值是新增研究判据，
在新增持续加载实验前固定；不是硬件标准，也不代表无滑移纯滚动。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from controlled_load import LoadSchedule, analyze_run


PROTOCOL = dict(version='2026-10-22_partial_support_rolling_v1',
    minimum_uphill_progress_m=1., maximum_weighted_slip_ratio=.25,
    spin_travel_ratio_bounds=[.75, 1.25], minimum_contact_fraction=.95,
    maximum_continuous_airborne_s=.05, maximum_tilt_deg=45.,
    four_wheel_contact_is_gate=False, command_tracking_is_gate=False,
    body_contact_sampling_hz=50, load_and_wheel_sampling_hz=1000)


def rolling_metrics(normal_load, contact, center_speed, spin_speed, slip_speed, dt=.001):
    """承载轮法向力加权：区分滚动、轮子空转、锁轮滑行和腾空。

    输入为时间×四轮。spin_speed=-[ω×(接触点−轮心)]·坡面切线；
    只有存在原生接触的正法向力参与。无承载时禁止除零变成通过。
    """
    arrays=[np.asarray(a) for a in (normal_load,contact,center_speed,spin_speed,slip_speed)]
    if any(a.shape!=arrays[0].shape for a in arrays) or arrays[0].ndim!=2:
        raise ValueError('Expected matching time x wheel arrays')
    if not all(np.isfinite(a).all() for a in arrays) or dt<=0:
        raise ValueError('Expected finite rolling observations and positive dt')
    normal_load,contact,center_speed,spin_speed,slip_speed=arrays
    weight=np.maximum(normal_load,0.)*(contact>0)
    supported=weight.sum(1)>1.
    longest=current=0
    for airborne in ~supported:
        current=current+1 if airborne else 0
        longest=max(longest,current)
    travel=float((weight*abs(center_speed)).sum())
    uphill=float((weight*center_speed).sum())
    valid=travel>1.e-8 and uphill>1.e-8 and weight.sum()>1.e-8
    slip_ratio=float((weight*slip_speed).sum()/travel) if valid else None
    spin_ratio=float((weight*spin_speed).sum()/uphill) if valid else None
    slip_mean=float((weight*slip_speed).sum()/weight.sum()) if valid else None
    return dict(weighted_slip_ratio=slip_ratio,spin_travel_ratio=spin_ratio,
        weighted_slip_speed_mps=slip_mean,any_loaded_wheel_fraction=float(supported.mean()),
        longest_airborne_s=longest*dt,
        rolling_dominates=bool(valid and slip_ratio<=PROTOCOL['maximum_weighted_slip_ratio']
            and PROTOCOL['spin_travel_ratio_bounds'][0]<=spin_ratio<=PROTOCOL['spin_travel_ratio_bounds'][1]
            and supported.mean()>=PROTOCOL['minimum_contact_fraction']
            and longest*dt<=PROTOCOL['maximum_continuous_airborne_s']))


def analyze_rolling_run(directory, schedule, write_report=True):
    directory=Path(directory)
    # Recompute force/motor/collision checks from traces; preserve historical JSON.
    legacy=analyze_run(directory,schedule,write_report=False)
    with np.load(directory/'seed43_case00_physics_trace.npz') as z:
        angle=np.deg2rad(25.)
        tangent=np.array([np.cos(angle),0.,np.sin(angle)])
        normal=np.array([-np.sin(angle),0.,np.cos(angle)])
        t=z['controlled_load_time_s']
        angular_contact=np.cross(z['wheel_angular_velocity_w'],z['native_contact_cop_w']-z['wheel_pos_w'])
        center=z['wheel_center_velocity_w']@tangent
        spin=-angular_contact@tangent
        slip_vector=z['wheel_center_velocity_w']+angular_contact
        slip=np.sqrt((slip_vector@tangent)**2+slip_vector[...,1]**2)
        normal_load=z['native_ground_normal_w']@normal
        contacts=z['native_contact_count']>0
        rows=[]
        for old,stage in zip(legacy['stages'],schedule.stages()):
            mask=(t>=stage['plateau_start_s']-1.e-6)&(t<stage['plateau_end_s']-1.e-6)
            row=dict(old)
            row['legacy_all_four_tracking_pass']=row.pop('sustained_rolling')
            row['legacy_checks']=row.pop('checks',{})
            row['rolling_load_1m_pass']=False
            if mask.any():
                row.update(rolling_metrics(normal_load[mask],contacts[mask],center[mask],spin[mask],slip[mask]))
                checks={k:old['checks'][k] for k in ('force_audit','complete_plateau','clearance','collision','termination','torque')}
                checks.update(uphill_1m=row['progress_m']>=PROTOCOL['minimum_uphill_progress_m'],
                              no_overturn=row['max_tilt_deg']<PROTOCOL['maximum_tilt_deg'],
                              rolling_dominates=row['rolling_dominates'])
                row.update(checks=checks,rolling_load_1m_pass=all(checks.values()))
            rows.append(row)
    report=dict(protocol=PROTOCOL,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_trailer_connected=False,ppo_updates=0,route_acceptance=False,stages=rows,
        highest_passed_load_N=max([r['level_N'] for r in rows if r['rolling_load_1m_pass']],default=None),
        limitations='Known backward hitch force only; no cart inertia or linkage, brake, hold, GPU or robustness acceptance. Non-wheel contact sampled at 50Hz; wheel/load data at 1kHz. Weighted slip tolerates small slip, not mathematical pure rolling.')
    if write_report:
        (directory/'rolling_load_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--require-load',type=float)
    args=parser.parse_args()
    request=json.loads((args.run/'request.json').read_text())
    report=analyze_rolling_run(args.run,LoadSchedule(**request['controlled_load']))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if args.require_load is not None and not any(r['level_N']==args.require_load and r['rolling_load_1m_pass'] for r in report['stages']):
        parser.exit(2,'Required load did not pass the rolling 1 m research protocol.\n')
