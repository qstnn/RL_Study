"""隔离牵引台架的分级负载及原生轨迹审查；不代表 R1 完整路线验收。"""
from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LoadSchedule:
    levels_N: tuple = (0., 50., 100., 150., 173., 190., 220.)
    start_s: float = 5.
    ramp_s: float = 2.
    hold_s: float = 5.
    maximum_duration_s: float = 60.

    def __post_init__(self):
        if not self.levels_N or not all(np.isfinite(x) and 0 <= x <= 250 for x in self.levels_N):
            raise ValueError('Load levels must be finite and within [0,250] N')
        if any(b < a for a,b in zip(self.levels_N,self.levels_N[1:])):
            raise ValueError('Load staircase must be nondecreasing')
        if not all(np.isfinite(x) for x in (self.start_s,self.ramp_s,self.hold_s)) or self.start_s < 3 or self.ramp_s < .5 or self.hold_s < 2:
            raise ValueError('Invalid preparation, ramp or plateau duration')
        if self.maximum_duration_s not in (60.,180.) or self.duration_s > self.maximum_duration_s:
            raise ValueError('Research test exceeds its explicit 60s or 180s duration budget')

    @property
    def duration_s(self):
        return self.start_s+len(self.levels_N)*(self.ramp_s+self.hold_s)

    def value(self,time_s):
        """平滑增加反向负载；返回非负牛顿值，不按速度改变施力方向。"""
        t=np.asarray(time_s,dtype=float)
        out=np.zeros_like(t)
        previous=0.
        for i,level in enumerate(self.levels_N):
            begin=self.start_s+i*(self.ramp_s+self.hold_s)
            u=np.clip((t-begin)/self.ramp_s,0.,1.)
            out+=(level-previous)*u*u*(3.-2.*u)
            previous=level
        return out

    def stages(self):
        return [dict(level_N=float(level),ramp_start_s=self.start_s+i*(self.ramp_s+self.hold_s),
                     plateau_start_s=self.start_s+i*(self.ramp_s+self.hold_s)+self.ramp_s,
                     plateau_end_s=self.start_s+(i+1)*(self.ramp_s+self.hold_s))
                for i,level in enumerate(self.levels_N)]


def analyze_run(directory,schedule,write_report=True):
    """核查1kHz实际加载方向/力矩，再按预先给定窗口判断持续滚动。"""
    directory=Path(directory)
    z=np.load(directory/'seed43_case00_physics_trace.npz')
    motor=np.load(directory/'seed43_case00_motor_trace.npz')
    case=json.loads((directory/'seed43_case00.json').read_text())
    command_speed=float(case.get('case',{}).get('speed',.2))
    with (directory/'seed43_case00.csv').open() as stream:
        control_rows=list(csv.DictReader(stream))
    control_time=np.array([float(row['t']) for row in control_rows])
    body_contact=np.array([float(row['body_contact']) for row in control_rows])
    terminated=np.array([float(row['terminated']) for row in control_rows])
    angle=np.deg2rad(25.);tangent=np.array([np.cos(angle),0.,np.sin(angle)])
    normal=np.array([-np.sin(angle),0.,np.cos(angle)]);side=np.array([0.,1.,0.])
    t=z['controlled_load_time_s'];force=z['controlled_load_force_w_N']
    torque=z['controlled_load_torque_w_Nm'];arm=z['controlled_load_arm_w_m']
    expected=schedule.value(t)
    errors=dict(force_tangent_N=float(np.max(abs(force@tangent+expected))),
                force_normal_N=float(np.max(abs(force@normal))),force_side_N=float(np.max(abs(force@side))),
                moment_Nm=float(np.max(abs(torque-np.cross(arm,force)))),
                force_buffer_N=float(np.max(abs(force-z['controlled_load_buffer_force_w_N']))),
                torque_buffer_Nm=float(np.max(abs(torque-z['controlled_load_buffer_torque_w_Nm']))),
                disconnected_cart_force_N=float(np.max(z['controlled_load_cart_force_norm_N'])),
                disconnected_cart_torque_Nm=float(np.max(z['controlled_load_cart_torque_norm_Nm'])))
    finite=all(np.isfinite(z[k]).all() for k in z.files)
    force_valid=bool(finite and all(x < 1.e-3 for x in errors.values()) and np.min(z['controlled_load_cart_distance_m'])>50.)
    # CSV height and the last physics sample share the post-step root state.
    # Recover the ground plane, because the unloaded context masks hitch clearance as 1 m.
    end_indices=np.rint(control_time/.001).astype(int)-1
    plane_offsets=z['body_pos_w'][end_indices]@normal-np.array([float(row['height']) for row in control_rows])
    plane_valid=bool(np.isfinite(plane_offsets).all() and np.ptp(plane_offsets)<1.e-4)
    hitch_height=z['hitch_robot_w']@normal-np.median(plane_offsets)
    velocity=z['whole_com_velocity_w']@tangent
    position=z['whole_com_w']@tangent
    loads=z['native_ground_normal_w']@normal
    contact=z['native_contact_count']>0
    quat=z['body_quat_w'][:,[1,2,3,0]]
    R=Rotation.from_quat(quat).as_matrix()
    tilt=np.rad2deg(np.arccos(np.clip(R[:,:,2]@normal,-1,1)))
    heading=np.rad2deg(np.arctan2(R[:,:,0]@side,R[:,:,0]@tangent))
    lever=z['native_contact_cop_w']-z['wheel_pos_w']
    slip=z['wheel_center_velocity_w']+np.cross(z['wheel_angular_velocity_w'],lever)
    slip=np.sqrt((slip@tangent)**2+(slip@side)**2)
    actual_friction=(z['friction_force_w']@tangent).sum(1)
    rows=[]
    for stage in schedule.stages():
        mask=(t>=stage['plateau_start_s']-1.e-6)&(t<stage['plateau_end_s']-1.e-6)
        count=int(mask.sum());complete=count>=round(schedule.hold_s/.001)
        row=dict(stage,physical_samples=count,complete_plateau=complete,sustained_rolling=False)
        if count:
            control_mask=(control_time>stage['plateau_start_s']+1.e-6)&(control_time<=stage['plateau_end_s']+1.e-6)
            v=velocity[mask];N=loads[mask];valid_slip=slip[mask][contact[mask]]
            row.update(mean_speed_mps=float(v.mean()),speed_mae_mps=float(abs(v-command_speed).mean()),
                progress_m=float(position[mask][-1]-position[mask][0]),
                four_contact_fraction=float(contact[mask].all(1).mean()),mean_wheel_normal_N=N.mean(0).tolist(),
                max_tilt_deg=float(tilt[mask].max()),max_abs_heading_deg=float(abs(heading[mask]).max()),
                native_contact_slip_p95_mps=float(np.quantile(valid_slip,.95)) if len(valid_slip) else None,
                robot_hitch_min_height_m=float(hitch_height[mask].min()),
                body_contact_fraction_50Hz=float(body_contact[control_mask].mean()) if control_mask.any() else None,
                termination_in_plateau=bool(terminated[control_mask].any()),
                mean_ground_traction_N=float(actual_friction[mask].mean()),
                peak_motor_fraction=float((abs(motor['applied'][mask])/motor['limits']).max()))
            checks=dict(force_audit=force_valid,complete_plateau=complete,
                speed=row['mean_speed_mps']>=.1 and row['speed_mae_mps']<=.05,
                progress=row['progress_m']>=.1*schedule.hold_s-.002,
                support=row['four_contact_fraction']>=.95 and min(row['mean_wheel_normal_N'])>=1.,
                posture=row['max_tilt_deg']<=10.5 and row['max_abs_heading_deg']<=5.,
                clearance=plane_valid and row['robot_hitch_min_height_m']>=.005,
                collision=row['body_contact_fraction_50Hz']==0.,
                termination=not row['termination_in_plateau'],
                rolling=row['native_contact_slip_p95_mps'] is not None and row['native_contact_slip_p95_mps']<=.05,
                torque=row['peak_motor_fraction']<=1.+1.e-5)
            row.update(checks=checks,sustained_rolling=all(checks.values()))
        rows.append(row)
    import hashlib
    report=dict(test='controlled_backward_hitch_load',ppo_updates=0,physical_trailer_connected=False,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        route_acceptance=False,command_speed_mps=command_speed,load_audit_passed=force_valid,load_audit_errors=errors,stages=rows,
        ground_plane_audit_passed=plane_valid,
        termination_causes=[key for key in control_rows[-1] if key.startswith('termination_') and float(control_rows[-1][key])>0],
        duration_s=case['samples']*.02,teacher_failure=case.get('teacher_failure'),
        failure_reason=case.get('failure_reason'),max_applied_load_N=float((-force@tangent).max()),
        highest_passed_plateau_N=max([r['level_N'] for r in rows if r['sustained_rolling']],default=None),
        interpretation='A failed finite controller is not a proof of the physical towing limit. No cart inertia, spring, brake or hold acceptance is tested.')
    if write_report:
        (directory/'controlled_load_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--require-load',type=float,
                        help='Exit 2 unless this exact plateau passes every diagnostic check; not full-route acceptance')
    args=parser.parse_args()
    request=json.loads((args.run/'request.json').read_text())
    result=analyze_run(args.run,LoadSchedule(**request['controlled_load']))
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.require_load is not None:
        passed=any(row['level_N']==args.require_load and row['sustained_rolling'] for row in result['stages'])
        if not passed:parser.exit(2,f'Required {args.require_load:g} N plateau did not pass all diagnostic checks.\n')
