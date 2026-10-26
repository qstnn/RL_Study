"""Independent physical-cart audit; initialization/preload and route limits remain explicit."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rolling_load_audit import rolling_metrics,PROTOCOL


def cart_rolling_with_partial_contact(metrics):
    """按用户2026-10-22要求允许拖车间歇离地；保留接触时滚动证据。

    只取消拖车接触占比/连续离地硬门，仍拒绝无承载数据、锁轮滑行和空转。
    接触指标继续报告；不改机器人判据，也不覆盖旧版验收文件。
    """
    slip,spin=metrics['weighted_slip_ratio'],metrics['spin_travel_ratio']
    return bool(slip is not None and spin is not None and np.isfinite([slip,spin]).all()
        and slip<=PROTOCOL['maximum_weighted_slip_ratio']
        and PROTOCOL['spin_travel_ratio_bounds'][0]<=spin<=PROTOCOL['spin_travel_ratio_bounds'][1])


def drawbar_errors(z):
    """独立NumPy复算实际子步前状态对应的弹簧/阻尼和两端力矩。"""
    def a(key):return np.asarray(z['real_cart_'+key],dtype=np.float64)
    delta=a('robot_hitch_pre_w')-a('cart_hitch_pre_w')
    length=np.maximum(np.linalg.norm(delta,axis=1),1.e-6);direction=delta/length[:,None]
    rate=np.sum((a('robot_hitch_velocity_pre_w')-a('cart_hitch_velocity_pre_w'))*direction,axis=1)
    tension=2000.*(length-1.)+150.*rate
    force=np.clip(tension,-1500,1500)[:,None]*direction
    return dict(spring_force_max_N=float(abs(force+a('robot_force_w_N')).max()),
        robot_arm_torque_max_Nm=float(abs(np.cross(a('robot_arm_pre_w'),-force)-a('robot_torque_w_Nm')).max()),
        cart_arm_torque_max_Nm=float(abs(np.cross(a('cart_arm_pre_w'),force)-a('cart_torques_w_Nm')[:,0]).max()))


def movement_mask(time,start,stop):
    """只使用实际命令状态机起点；普通初态准备超过3秒也不计作拖行。"""
    time=np.asarray(time)
    mask=np.zeros(len(time),dtype=bool) if start is None else ((time>=start)&(time<(stop if stop is not None else time[-1])))
    # 单样本占位用于保留接触诊断形状；不产生距离，且另有起步硬门。
    if not mask.any():mask[-1]=True
    return mask


def analyze(path,write_report=True):
    path=Path(path);request=json.loads((path/'request.json').read_text())
    if not request.get('physical_trailer_connected'):raise ValueError('This audit requires a real connected cart')
    with np.load(path/'seed43_case00_physics_trace.npz') as raw:z={k:raw[k] for k in raw.files}
    with np.load(path/'seed43_case00_policy_trace.npz') as raw:a={k:raw[k] for k in raw.files}
    with np.load(path/'seed43_case00_motor_trace.npz') as raw:m={k:raw[k] for k in raw.files}
    rows=list(csv.DictReader((path/'seed43_case00.csv').open()));csv_data={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    case=json.loads((path/'seed43_case00.json').read_text());t=(np.arange(len(z['joint_pos']))+1)*.001
    slope=np.deg2rad(request.get('slope_deg',25.));tangent=np.array([np.cos(slope),0.,np.sin(slope)]);normal=np.array([-np.sin(slope),0.,np.cos(slope)])
    phase=csv_data['eval_phase'];moving=np.flatnonzero(phase>=1);stop=np.flatnonzero(phase==3)
    start=float(csv_data['t'][moving[0]]) if len(moving) else None
    stop_time=float(csv_data['t'][stop[0]]) if len(stop) else None
    end=stop_time or float(t[-1]);mask=movement_mask(t,start,stop_time)
    mass=z['real_cart_cart_masses_kg'];gravity=np.zeros_like(z['real_cart_cart_forces_w_N']);gravity[:,:,2]=-9.81*mass
    extra=z['real_cart_cart_forces_w_N']-gravity
    errors=dict(cart_mass_max_kg=float(abs(mass.sum(1)-request['physical_cart_mass_kg']).max()),
        action_reaction_max_N=float(abs(extra.sum(1)+z['real_cart_robot_force_w_N']).max()),
        nonbase_cart_force_max_N=float(abs(extra[:,1:]).max()),
        motor_limit_excess_Nm=float(np.maximum(abs(m['applied'])-m['limits'],0).max()),
        motor_clipping_error_Nm=float(abs(m['applied']-np.clip(m['requested'],-m['limits'],m['limits'])).max()))
    spring=drawbar_errors(z) if request.get('records_drawbar_pre_step') else None
    angular=np.cross(z['wheel_angular_velocity_w'],z['native_contact_cop_w']-z['wheel_pos_w'])
    slip=z['wheel_center_velocity_w']+angular
    robot_roll=rolling_metrics((z['native_ground_normal_w']@normal)[mask],(z['native_contact_count']>0)[mask],
        (z['wheel_center_velocity_w']@tangent)[mask],(-angular@tangent)[mask],np.linalg.norm(np.stack((slip@tangent,slip[:,:,1]),-1),axis=-1)[mask])
    R=Rotation.from_quat(z['real_cart_cart_quat_w'][:,[1,2,3,0]]).as_matrix();axis=R[:,:,1]
    radial=normal-axis*(axis@normal)[:,None];lever=-.14*radial/np.linalg.norm(radial,axis=1)[:,None]
    if request.get('rounded_cart_wheels'):lever=np.broadcast_to(-.14*normal,lever.shape)
    ca=np.cross(z['cart_wheel_angular_velocity_w'],lever[:,None,:]);cs=z['cart_wheel_center_velocity_w']+ca
    cart_roll=rolling_metrics((z['cart_ground_normal_w'][:,:4]@normal)[mask],(z['cart_ground_point_count'][:,:4]>0)[mask],
        (z['cart_wheel_center_velocity_w']@tangent)[mask],(-ca@tangent)[mask],np.linalg.norm(np.stack((cs@tangent,cs[:,:,1]),-1),axis=-1)[mask])
    if start is None:
        robot_roll['rolling_dominates']=False;cart_roll['rolling_dominates']=False
    cart_position=z['cart_pos_w']@tangent;robot_position=z['whole_com_w']@tangent
    ix=np.flatnonzero(mask);cart_progress=float(cart_position[ix[-1]]-cart_position[ix[0]]);robot_progress=float(robot_position[ix[-1]]-robot_position[ix[0]])
    body_R=Rotation.from_quat(z['body_quat_w'][:,[1,2,3,0]]).as_matrix();tilt=np.rad2deg(np.arccos(np.clip(body_R[:,:,2]@normal,-1,1)))
    hitch_min=float(np.minimum(z['hitch_robot_w']@normal,z['hitch_cart_w']@normal).min())
    cart_tilt=np.rad2deg(np.arccos(np.clip(R[:,:,2]@normal,-1,1)))
    complete=len(z['joint_pos'])>=round(request['trial_duration_s']/.001)
    health=dict(full_declared_window=complete,no_termination=not csv_data['terminated'].any(),
        no_robot_body_contact=not csv_data['body_contact'].any(),no_cart_body_contact=bool(np.linalg.norm(z['cart_ground_normal_w'][:,4],axis=1).max()<1.),
        no_overturn=bool(tilt.max()<45 and cart_tilt.max()<45),hitch_clearance=hitch_min>=.005,
        torque_limits=errors['motor_limit_excess_Nm']<1.e-4 and np.array_equal(m['limits'],np.r_[np.full(12,96),np.full(4,48)]),
        physical_mass_and_force=errors['cart_mass_max_kg']<1.e-4 and errors['action_reaction_max_N']<1.e-3 and errors['nonbase_cart_force_max_N']<1.e-3,
        no_drawbar_protection=bool(abs(z['drawbar_tension_interval_N']).max()<1499.))
    if spring is not None:health['independent_spring_and_moment']=max(spring.values())<.001
    checks=dict(health,commanded_movement_started=start is not None,cart_net_route_1m=cart_progress>=1,robot_net_route_1m=robot_progress>=1,
        robot_rolling=robot_roll['rolling_dominates'],cart_rolling=cart_rolling_with_partial_contact(cart_roll))
    hold=None
    if stop_time is not None and t[-1]>=stop_time+7.-.001:
        hm=(t>=stop_time+2.)&(t<stop_time+7.);hi=np.flatnonzero(hm)
        hold=dict(start_s=stop_time+2,end_s=stop_time+7,
            robot_max_drift_m=float(np.linalg.norm(z['whole_com_w'][hm]-z['whole_com_w'][hi[0]],axis=1).max()),
            cart_max_drift_m=float(np.linalg.norm(z['cart_pos_w'][hm]-z['cart_pos_w'][hi[0]],axis=1).max()),
            robot_speed_p95_mps=float(np.percentile(np.linalg.norm(z['whole_com_velocity_w'][hm],axis=1),95)),
            cart_speed_p95_mps=float(np.percentile(np.linalg.norm(z['cart_velocity_w'][hm],axis=1),95)))
        hold['pass']=bool(max(hold['robot_max_drift_m'],hold['cart_max_drift_m'])<=.05 and max(hold['robot_speed_p95_mps'],hold['cart_speed_p95_mps'])<=.05)
    report=dict(protocol='real_cart_loaded_pose_1m_partial_cart_contact_v2',
        cart_continuous_contact_is_gate=False,
        protocol_change='User explicitly allows intermittent/partial cart wheel support on 2026-10-22; rolling ratios unchanged; legacy v1 evidence preserved.',
        physical_cart_mass_kg=request['physical_cart_mass_kg'],duration_s=float(t[-1]),
        simulation_device=request['execution_physics']['device'],initial_state_mode=request.get('initial_state_mode','loaded'),
        slope_deg=float(np.rad2deg(slope)),distance_direction='uphill' if slope>0 else 'downhill',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cart_geometry='rounded_spherical_tread_variant' if request.get('rounded_cart_wheels') else 'original_cylindrical_wheels',
        declared_duration_s=request['trial_duration_s'],move_window_s=[start,end],command_zero_time_s=stop_time,
        cart_net_uphill_m=(cart_progress if slope>0 else None),robot_net_uphill_m=(robot_progress if slope>0 else None),cart_max_progress_from_move_start_m=float((cart_position[ix]-cart_position[ix[0]]).max()),
        cart_net_route_m=cart_progress,robot_net_route_m=robot_progress,
        preparation=dict(end_s=start if start is not None else float(t[-1]),
            cart_net_from_first_sample_m=float((cart_position[(t<start) if start is not None else np.ones(len(t),dtype=bool)]-cart_position[0])[-1]),
            cart_max_rollback_m=float(max(0.,-(cart_position[(t<start) if start is not None else np.ones(len(t),dtype=bool)]-cart_position[0]).min())),
            note='Preparation displacement is reported separately and never counts toward the commanded 1m.'),
        max_robot_tilt_deg=float(tilt.max()),max_cart_tilt_deg=float(cart_tilt.max()),minimum_hitch_height_m=hitch_min,
        peak_motor_fraction=float((abs(m['applied'])/m['limits']).max()),force_and_motor_errors=errors,
        independent_drawbar_errors=spring,
        robot_rolling=robot_roll,cart_rolling=cart_roll,checks=checks,real_cart_1m_pass=all(checks.values()),
        hold_2s_transition_5s=hold,move_brake_hold_pass=bool(all(checks.values()) and hold and hold['pass']),
        route_acceptance=False,ppo_updates=0,initial_condition=request['initial_condition'],
        initial_spring_energy_J=(request['initial_spring_energy_J'] if request['initial_spring_energy_J'] is not None else float(1000.*(np.linalg.norm(z['real_cart_robot_hitch_pre_w'][0]-z['real_cart_cart_hitch_pre_w'][0])-1.)**2)),cart_gravity_potential_change_J=cart_progress*request['physical_cart_mass_kg']*9.81*np.sin(slope),
        limitations=('One fixed initial condition on declared CPU/GPU; original robot, real passive cart. '+
            ('Native ordinary reset followed by controlled joint/wheel entry. ' if request.get('initial_state_mode')=='ordinary' else 'Loaded posture and preloaded spring are reset conditions, not ordinary R1 cold start. ')+
            'Robot rolling uses native contact CoP; cart rolling uses declared wheel geometry. Body contact sampled at50Hz. 64-condition route and deployment not accepted.'))
    if write_report:(path/'real_cart_analysis_partial_contact.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--require-pass',action='store_true')
    p.add_argument('--require-hold',action='store_true',help='Require the complete move, brake, 2s transition and 5s hold');args=p.parse_args()
    result=analyze(args.run);print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.require_pass and not result['real_cart_1m_pass']:p.exit(2,'Real cart did not pass the declared 1m protocol.\n')
    if args.require_hold and not result['move_brake_hold_pass']:p.exit(2,'Real cart did not pass the complete move/brake/hold protocol.\n')
