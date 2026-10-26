"""V2验收：空载高速；载荷/拖车按行进距离与稳停验收，不设移动速度门槛。"""
import math
import numpy as np

DT=.02
DURATION=18.
STEPS=900
MAX_DISTANCE_STEPS=3000


def cases(distance_m=1.):
    return [dict(slope_deg=slope,payload_kg=load,towed_kg=tow,speed=speed,goal_distance_m=distance_m)
            for slope in (0.,25.,-25.) for load,tow in ((0.,0.),(15.,0.),(0.,40.))
            for speed in (0.,3.1 if load==0 and tow==0 else .3)]


def is_distance_case(case):
    if case['payload_kg']>0 and case['towed_kg']>0:
        raise ValueError('Payload and towing are mutually exclusive')
    return (case['payload_kg']>0 or case['towed_kg']>0) and case['speed']>0


def max_steps(case):
    return MAX_DISTANCE_STEPS if is_distance_case(case) else STEPS


def finished(rows,case):
    if not rows:
        return False
    if not is_distance_case(case):
        return len(rows)>=STEPS
    last=rows[-1]
    return (last.get('eval_phase',0)>=2 and last.get('eval_stop_time_s',-1)>=3
            and round(last['t']/DT)+1>=round(last['eval_stop_time_s']/DT)+350)


def summarize(rows,case,reason):
    distance=is_distance_case(case)
    complete=finished(rows,case) and reason is None
    result=dict(case=case,samples=len(rows),complete=complete,failure_reason=reason,success=False,
                sample_hz=50,final_hardware_acceptance='not_evaluated',
                protocol='distance_then_hold' if distance else 'fixed_18s_start_stop',
                moving_speed_acceptance='none' if case['payload_kg'] or case['towed_kg'] else 'unloaded_tracking')
    if not rows:
        return result
    if not all(math.isfinite(v) for row in rows for v in row.values()):
        result.update(failure_reason='non_finite_measurement',complete=False)
        return result
    observed=dict(duration_s=len(rows)*DT,peak_tilt_deg=max(r['tilt_deg'] for r in rows),
                  peak_drawbar_force_N_1kHz=max(r.get('drawbar_force_peak_N_1kHz',abs(r['drawbar_force_N'])) for r in rows),
                  body_contact_samples=sum(r['body_contact']>0 for r in rows),
                  termination_causes=[key.removeprefix('termination_') for key in rows[0]
                                      if key.startswith('termination_') and any(r[key]>0 for r in rows)],
                  progress_m=max(r.get('eval_progress_m',0.) for r in rows))
    result['observed']=observed
    if distance:
        stopped=[r['eval_stop_time_s'] for r in rows if r.get('eval_stop_time_s',-1)>=3.]
        stop=stopped[0] if stopped else None
        # 仿真stop来自float32；用控制步索引避免浮点舍入漏掉稳停窗口边界。
        stop_step=round(stop/DT) if stop is not None else None
        cruise=[r for r in rows if r['t']>=3.5 and (stop_step is None or round(r['t']/DT)+1<=stop_step)]
        hold=[r for r in rows if stop_step is not None and stop_step+100<=round(r['t']/DT)+1<stop_step+350]
        result['stop_time_s']=stop
    else:
        cruise=[r for r in rows if 5.<=r['t']<11.]
        hold=[r for r in rows if 13.<=r['t']<18.]
    result['window_samples']={'cruise':len(cruise),'hold':len(hold),'full_hold':len(hold)==250}
    if not cruise or not hold:
        return result
    avg=lambda rs,key:float(np.mean([r[key] for r in rs]))
    metrics=dict(cruise_vx_mps=avg(cruise,'vx'),cruise_mae_mps=avg(cruise,'velocity_error'),
                 tilt_p95_deg=float(np.percentile([r['tilt_deg'] for r in rows],95)),
                 heading_abs_p95_deg=float(np.percentile([abs(r['heading_rad'])*180/math.pi for r in rows],95)),
                 height_error_p95_m=float(np.percentile([abs(r['height']-.416) for r in rows],95)),
                 hold_speed_mps=avg(hold,'speed'),hold_yaw_rate_rps=avg(hold,'yaw_rate_abs'),
                 hold_four_wheel_fraction=avg(hold,'four_wheel'),
                 body_contact_samples=observed['body_contact_samples'],
                 peak_torque_Nm_50Hz=max(r['torque_abs_max'] for r in rows),
                 peak_torque_Nm_1kHz=max(r.get('torque_peak_Nm_1kHz',r['torque_abs_max']) for r in rows),
                 hold_drift_m=max(math.hypot(r['x']-hold[0]['x'],r['y']-hold[0]['y']) for r in hold))
    unloaded=case['payload_kg']==0 and case['towed_kg']==0
    checks=dict(complete=complete,full_hold_window=len(hold)==250,
                unloaded_tracking=not unloaded or metrics['cruise_mae_mps']<=.15,
                unloaded_actual_3mps=not unloaded or case['speed']<3. or metrics['cruise_vx_mps']>=3.,
                loaded_distance=not distance or observed['progress_m']>=case.get('goal_distance_m',1.),
                surface_posture=metrics['tilt_p95_deg']<=15.,heading=metrics['heading_abs_p95_deg']<=10.,
                height=metrics['height_error_p95_m']<=.08,
                hold_motion=metrics['hold_speed_mps']<=.05 and metrics['hold_yaw_rate_rps']<=.05,
                hold_support=metrics['hold_four_wheel_fraction']>=.95,
                no_body_contact=metrics['body_contact_samples']==0,
                drawbar_guard=observed['peak_drawbar_force_N_1kHz']<1499.,
                drawbar_above_ground=case['towed_kg']==0 or min(r.get('hitch_clearance_m',1.) for r in rows)>=.005)
    result.update(metrics=metrics,checks=checks,success=all(checks.values()))
    result['failed_checks']=[k for k,v in checks.items() if not v]
    return result
