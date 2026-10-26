"""V4冻结验收：16工况，-25度不牵引；归零后2秒过渡+5秒稳停，负载无速度门槛。"""
import math
import numpy as np

DT=.02
STATIC_STEPS=1000
UNLOADED_STEPS=1500
MAX_DISTANCE_STEPS=3000


def cases(distance_m=1.):
    return [dict(slope_deg=s,payload_kg=p,towed_kg=t,speed=v,goal_distance_m=distance_m)
            for s in (0.,25.,-25.) for p,t in ((0.,0.),(15.,0.),(0.,40.))
            for v in (0.,3.1 if p==0 and t==0 else .3) if not (s==-25. and t>0)]


def is_distance_case(case):
    if case['payload_kg']>0 and case['towed_kg']>0:
        raise ValueError('Payload and towing are mutually exclusive')
    return (case['payload_kg']>0 or case['towed_kg']>0) and case['speed']>0


def max_steps(case):
    if case['speed']==0:return STATIC_STEPS
    return MAX_DISTANCE_STEPS if is_distance_case(case) else UNLOADED_STEPS


def finished(rows,case):
    if not rows:return False
    if case['speed']==0:return round(rows[-1]['t']/DT)>=STATIC_STEPS
    last=rows[-1]
    return (last.get('eval_phase',0)>=3 and last.get('eval_stop_time_s',-1)>=3.
            and round(last['t']/DT)>=round(last['eval_stop_time_s']/DT)+350)


def summarize(rows,case,reason):
    distance=is_distance_case(case);moving=case['speed']>0
    complete=finished(rows,case) and reason is None
    result=dict(case=case,samples=len(rows),complete=complete,failure_reason=reason,success=False,
                sample_hz=50,final_hardware_acceptance='not_evaluated',
                protocol='distance_slew_hold' if distance else 'timed_slew_hold' if moving else 'static_20s',
                moving_speed_acceptance='none' if case['payload_kg'] or case['towed_kg'] else 'unloaded_tracking')
    if not rows:
        result['failed_checks']=['no_samples'];return result
    if not all(math.isfinite(v) for row in rows for v in row.values()):
        result.update(failure_reason='non_finite_measurement',complete=False,failed_checks=['finite'])
        return result
    observed=dict(duration_s=rows[-1]['t'],peak_tilt_deg=max(r['tilt_deg'] for r in rows),
                  peak_drawbar_force_N_1kHz=max(r.get('drawbar_force_peak_N_1kHz',abs(r['drawbar_force_N'])) for r in rows),
                  body_contact_samples=sum(r['body_contact']>0 for r in rows),
                  termination_causes=[k.removeprefix('termination_') for k in rows[0]
                                      if k.startswith('termination_') and any(r[k]>0 for r in rows)],
                  progress_m=max(r.get('eval_progress_m',0.) for r in rows),
                  minimum_hitch_clearance_m=min(r.get('hitch_clearance_m',1.) for r in rows))
    result['observed']=observed
    if moving:
        stopped=[r['eval_stop_time_s'] for r in rows if r.get('eval_phase',0)>=3 and r.get('eval_stop_time_s',-1)>=3.]
        stop=stopped[0] if stopped else None
        stop_step=round(stop/DT) if stop is not None else None
        hold=[r for r in rows if stop_step is not None and stop_step+100<=round(r['t']/DT)<stop_step+350]
        if distance:
            cruise=[r for r in rows if r['t']>=3.5 and r.get('target_vx',0)>0]
        else:
            full=[r['t'] for r in rows if abs(r['command_vx']-case['speed'])<1e-4 and r.get('target_vx',0)>0]
            start=full[0]+1. if full else math.inf
            cruise=[r for r in rows if r['t']>=start and r.get('target_vx',0)>0 and abs(r['command_vx']-case['speed'])<1e-4]
        result['stop_time_s']=stop
    else:
        hold=[r for r in rows if 750<=round(r['t']/DT)<1000]
        cruise=hold
    result['window_samples']={'cruise':len(cruise),'hold':len(hold),'full_hold':len(hold)==250}
    checks=dict(complete=complete,full_hold_window=len(hold)==250,
                cruise_observed=bool(cruise) and (distance or not moving or len(cruise)>=100))
    if not cruise or not hold:
        checks['steady_windows_available']=False
        result.update(checks=checks,failed_checks=[k for k,v in checks.items() if not v]);return result
    avg=lambda rs,k:float(np.mean([r[k] for r in rs]))
    angle=math.radians(case['slope_deg'])
    # 路线切向+横向位移；斜坡上不能只用世界水平x距离代替沿坡漂移。
    drift=max(math.hypot((r['x']-hold[0]['x'])*math.cos(angle)+(r['z']-hold[0]['z'])*math.sin(angle),
                         r['y']-hold[0]['y']) for r in hold)
    metrics=dict(cruise_vx_mps=avg(cruise,'vx'),cruise_mae_mps=avg(cruise,'velocity_error'),
                 tilt_p95_deg=float(np.percentile([r['tilt_deg'] for r in rows],95)),
                 heading_abs_p95_deg=float(np.percentile([abs(r['heading_rad'])*180/math.pi for r in rows],95)),
                 height_error_p95_m=float(np.percentile([abs(r['height']-.416) for r in rows],95)),
                 hold_speed_mps=avg(hold,'speed'),hold_yaw_rate_rps=avg(hold,'yaw_rate_abs'),
                 hold_four_wheel_fraction=avg(hold,'four_wheel'),hold_drift_m=drift,
                 body_contact_samples=observed['body_contact_samples'],
                 peak_torque_Nm_50Hz=max(r['torque_abs_max'] for r in rows),
                 peak_torque_Nm_1kHz=max(r.get('torque_peak_Nm_1kHz',r['torque_abs_max']) for r in rows))
    unloaded=case['payload_kg']==0 and case['towed_kg']==0
    checks.update(unloaded_tracking=not unloaded or metrics['cruise_mae_mps']<=.15,
                  unloaded_actual_3mps=not unloaded or case['speed']<3. or metrics['cruise_vx_mps']>=3.,
                  loaded_distance=not distance or observed['progress_m']>=case.get('goal_distance_m',1.),
                  surface_posture=metrics['tilt_p95_deg']<=15.,heading=metrics['heading_abs_p95_deg']<=10.,
                  height=metrics['height_error_p95_m']<=.08,
                  hold_command_zero=all(abs(r['command_vx'])<1e-4 and abs(r['target_vx'])<1e-4 for r in hold),
                  hold_motion=metrics['hold_speed_mps']<=.05 and metrics['hold_yaw_rate_rps']<=.05,
                  hold_support=metrics['hold_four_wheel_fraction']>=.95,hold_drift=drift<=.05,
                  no_body_contact=metrics['body_contact_samples']==0,
                  drawbar_guard=observed['peak_drawbar_force_N_1kHz']<1499.,
                  drawbar_above_ground=case['towed_kg']==0 or observed['minimum_hitch_clearance_m']>=.005)
    result.update(metrics=metrics,checks=checks,success=all(checks.values()),failed_checks=[k for k,v in checks.items() if not v])
    return result
