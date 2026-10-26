"""真实被动拖车的预承重姿态验证；仅初始化时指定姿态和弹簧预伸长。

之后采用原重力、惯量、弹簧和电机。不会施加已知负载替代拖车，不改根状态。
来源是独立恒载成功窗，非部署Actor；必须重新审查真实拖车净位移。
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import worker
from research_common import launch
from partial_support_control import BalanceOptions,PartialSupportControl,support_joint_torque,SCALE
from wheel_load_control import allocated_targets,heading_velocity_offset


def forwarded_ground_friction(arguments):
    """Read the exact value the downstream parser will use, including --name=value."""
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ground-friction',type=float,default=3.)
    return parser.parse_known_args(arguments)[0].ground_friction


def reference_from_run(run,begin,end):
    request=json.loads((run/'request.json').read_text())
    physical=request.get('physical_trailer_connected',False)
    if json.loads((run/'physics_recorder.json').read_text())['sample_hz']!=1000:raise ValueError('Reference currently requires archived 1kHz physics')
    if physical:
        report=json.loads((run/'real_cart_analysis_partial_contact.json').read_text())
        if not report.get('move_brake_hold_pass') or not report['move_window_s'][0]<=begin<end<=report['move_window_s'][1] or end-begin<2:
            raise ValueError('Physical reference requires complete move/brake/hold success and >=2s moving window')
        if request.get('slope_deg',25.)!=25.:raise ValueError('Base reference currently requires uphill source')
        stages=None
    else:
        report=json.loads((run/'rolling_load_analysis.json').read_text())
        stages=[s for s in report['stages'] if s.get('rolling_load_1m_pass') and s['plateau_start_s']<=begin and end<=s['plateau_end_s']]
        if len(stages)!=1 or end-begin<2:raise ValueError('Reference must be inside one complete successful plateau for at least 2s')
    with np.load(run/'seed43_case00_physics_trace.npz') as data,np.load(run/'seed43_case00_policy_trace.npz') as actions:
        p={k:data[k] for k in ('joint_pos','body_quat_w','body_pos_w','whole_com_w','wheel_pos_w','native_ground_normal_w','friction_force_w','native_contact_moment_com_w','native_friction_moment_com_w')}
        t=np.arange(len(p['joint_pos']))*.001+.001;mask=(t>=begin)&(t<end)
        R=Rotation.from_quat(p['body_quat_w'][mask][:,[1,2,3,0]]).mean()
        quat=R.as_quat()[[3,0,1,2]];matrix=R.as_matrix()
        force=p['native_ground_normal_w']+p['friction_force_w']
        moment=p['native_contact_moment_com_w']+p['native_friction_moment_com_w']-np.cross(p['wheel_pos_w']-p['whole_com_w'][:,None,:],force)
        q=p['joint_pos'][mask].mean(0)
        # Wheel rotation phase is periodic. Keep an actual final phase, not an average of turns.
        q[12:]=p['joint_pos'][np.flatnonzero(mask)[-1],12:]
        at=np.arange(len(actions['executed_action']))*.02;amask=(at>=begin)&(at<end)
        normal=np.array([-np.sin(np.deg2rad(25)),0.,np.cos(np.deg2rad(25))])
        angles=actions['balance_angles_rad'][amask].mean(0)
        result=dict(q=q.tolist(),quat=quat.tolist(),height_m=float((p['body_pos_w'][mask]@normal).mean()),
            leg_action=actions['executed_action'][amask,:12].mean(0).tolist(),angles_rad=angles.tolist(),
            force_body_N=(force[mask].mean(0)@matrix).tolist(),moment_body_Nm=(moment[mask].mean(0)@matrix).tolist(),
            source_load_N=(float(-(data['real_cart_robot_force_w_N'][mask]@np.array([normal[2],0.,-normal[0]])).mean()) if physical else stages[0]['level_N']),
            source_kind=('successful_real_cart' if physical else 'successful_known_force'),source_window_s=[begin,end],source_run=str(run),
            source_request_sha256=launch.sha256(run/'request.json'),source_trace_sha256=launch.sha256(run/'seed43_case00_physics_trace.npz'),
            source_actions_sha256=launch.sha256(run/'seed43_case00_policy_trace.npz'),source_request=request)
    return result


def cart_root_from_preload(robot_point,cart_local_world,tangent,normal,height,extension,origin):
    """在原坡面上按真实弹簧长度放置拖车；只用于重置，不在运行中调用。"""
    center=np.asarray(origin)+np.asarray(normal)*height
    center[1]=robot_point[1]-cart_local_world[1]
    gap=np.asarray(robot_point)-(center+cart_local_world)
    normal_gap=float(gap@normal);side_gap=float(gap[1]);length=1.+extension
    if normal_gap**2+side_gap**2>=length**2:raise ValueError('Hitch endpoints cannot attain requested preload length')
    distance=np.sqrt(length**2-normal_gap**2-side_gap**2)
    center+=tangent*(float(gap@tangent)-distance)
    return center


def retarget_preload(leg_action,actual_q,old_kp,new_kp):
    """在同一实际姿态保留原静态PD力矩，避免新刚度把旧目标偏差直接放大。"""
    default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5]);scale=np.tile([.08,.14,.14],4)
    action,q,old,new=[np.asarray(v,dtype=float) for v in (leg_action,actual_q,old_kp,new_kp)]
    if any(v.shape!=(12,) or not np.isfinite(v).all() for v in (action,q,old,new)) or np.any(old<=0) or np.any(new<=0):
        raise ValueError('Invalid preload servo state')
    target=default+action*scale
    return (q+(target-q)*old/new-default)/scale


def entry_fraction(time_s,duration_s):
    """普通初态连续进入的五次平滑插值，首尾速度/加速度为0；不改物理状态。"""
    if not np.isfinite([time_s,duration_s]).all() or duration_s<=0:raise ValueError('Invalid entry time')
    x=np.clip(time_s/duration_s,0.,1.)
    return float(x**3*(10.-15.*x+6.*x*x))


def entry_wheel_velocity(model,time_s,duration_s,reference_q,quat,tangent,normal):
    """调姿时按腿运动的轮心切向速度配轮速；不让轮速保持器阻碍支撑点后移。"""
    fraction=entry_fraction(time_s,duration_s);x=np.clip(time_s/duration_s,0.,1.)
    rate=30.*x*x*(1.-x)**2/duration_s
    default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5])
    q=np.asarray(reference_q).copy();v=np.zeros(22)
    v[6:18]=rate*(q[:12]-default);q[:12]=default+fraction*(q[:12]-default)
    state=model.state(np.zeros(3),quat,q);A,_,_=model.contacts(state,normal)
    speed=np.empty(4)
    for i in range(4):
        row=tangent@A[3*i:3*i+3]
        speed[i]=-float(row@v)/row[18+i]
    return speed


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--reference-window',type=float,nargs=2,required=True)
    parser.add_argument('--cart-mass',type=float,required=True)
    parser.add_argument('--trial-duration',type=float,default=30.)
    parser.add_argument('--move-speed',type=float,default=.2)
    parser.add_argument('--front-force-share',type=float,default=None)
    parser.add_argument('--wheel-heading-gain',type=float,default=0.)
    parser.add_argument('--cart-position-iterations',type=int,default=4)
    parser.add_argument('--cart-velocity-iterations',type=int,default=1)
    parser.add_argument('--rounded-cart-wheels',action='store_true',help='Explicit spherical-tread passive wheel geometry variant, original mass/radius')
    parser.add_argument('--diagnostic-compliant-drawbar',action='store_true',help='Isolated ordinary/live spring-damper variant; never original teacher/dataset evidence')
    parser.add_argument('--diagnostic-drawbar-stiffness',type=float,default=2000.,help='Explicit diagnostic stiffness N/m; baseline remains2000')
    parser.add_argument('--diagnostic-drawbar-damping',type=float,default=150.,help='Explicit diagnostic damping Ns/m; baseline remains150')
    parser.add_argument('--sagittal-servo-stiffness',type=float,default=None,help='Explicit thigh/calf gain; re-invert source preload so initial support torque is preserved')
    parser.add_argument('--sagittal-servo-damping',type=float,default=3.)
    parser.add_argument('--leg-force-gain',type=float,default=1.,help='Incremental measured force compensation around the preloaded reference; zero retains its existing load support')
    parser.add_argument('--reset-lateral-offset',type=float,default=0.,help='Explicit fixed world-y reset offset for contact/slot sensitivity checks; no randomization')
    parser.add_argument('--mirror-reference',action='store_true')
    parser.add_argument('--match-static-force-share',action='store_true')
    parser.add_argument('--static-preload',action='store_true')
    parser.add_argument('--leg-rate-limit',type=float,default=.6)
    parser.add_argument('--horizontal-attitude-gain',type=float,default=0.)
    parser.add_argument('--force-pair-average',type=float,default=0.)
    parser.add_argument('--initial-state',choices=('loaded','ordinary','continuation'),default='loaded')
    parser.add_argument('--initial-pose-progress',type=float,default=1.,help='Continuation only: 0=native geometry, 1=validated loaded geometry; never ordinary-entry evidence')
    parser.add_argument('--initial-route-velocity',type=float,default=0.,help='Isolated continuation at loaded geometry only: common robot/cart downhill velocity, not ordinary reset')
    parser.add_argument('--entry-ground-seeking',action='store_true')
    parser.add_argument('--entry-forward-m',type=float,default=0.)
    parser.add_argument('--entry-duration',type=float,default=1.)
    parser.add_argument('--entry-rear-first',type=float,default=0.,help='Optional first-stage duration: extend rear wheels 0.20m, then lower with fixed wheel centers')
    parser.add_argument('--move-start',type=float,default=3.)
    parser.add_argument('--fixed-command-clock',action='store_true',help='Use only the declared reset clock, never controller readiness, to start commands')
    parser.add_argument('--student-prefix-checkpoint',type=Path,help='Explicit student-prefix then executed-teacher recovery experiment; never autonomous evidence')
    parser.add_argument('--student-prefix-seconds',type=float,default=0.)
    parser.add_argument('--slope-deg',type=float,choices=(25.,-20.),default=25.)
    parser.add_argument('--wheel-load-source',choices=('nominal','drawbar'),default='nominal')
    parser.add_argument('--entry-mode',choices=('blend','geometry','cartesian'),default='blend')
    parser.add_argument('--entry-velocity-feedforward',action='store_true')
    parser.add_argument('--entry-wheel-kinematics',action='store_true')
    parser.add_argument('--entry-free-roll-capture',type=float,default=0.,help='Research entry only: follow measured wheel velocity while lowering, then blend braking/drive over this many seconds')
    parser.add_argument('--entry-load-share',action='store_true',help='Use measured front/rear normal load during entry; blend to declared traction share after lowering')
    parser.add_argument('--entry-torque-capture',action='store_true',help='Capture downhill motion with load-limited wheel torque and state-gated handoff')
    parser.add_argument('--entry-contact-traction',action='store_true',help='Torque capture only: allocate by each wheel actual previous-cycle normal force instead of axle pairs')
    parser.add_argument('--entry-capture-start',type=float,default=None,help='Torque capture only: overlap force establishment with lowering, seconds after physical reset')
    parser.add_argument('--record-momentum',action='store_true',help='Read-only native body mass/inertia and 1kHz momentum evidence')
    parser.add_argument('--record-native-dynamics',action='store_true',help='Read-only native M/C/g and COM Jacobian at every 20ms control boundary')
    parser.add_argument('--entry-capture-reference',choices=('wheel','body','ramp'),default='wheel',help='Torque capture velocity anchor; ramp decelerates a captured body-speed reference toward zero')
    parser.add_argument('--entry-capture-acceleration',type=float,default=.1,help='Ramp capture only: bounded reference deceleration in m/s2')
    parser.add_argument('--entry-speed-capture',type=float,default=0.,help='Optional bounded speed-command capture: maximum initial command-speed error m/s; zero disables')
    parser.add_argument('--coordinated-capture',action='store_true',help='Isolated native-dynamics joint leg/wheel capture at loaded geometry')
    parser.add_argument('--coord-servo-horizon',type=float,choices=(0.,.02),default=.02)
    parser.add_argument('--coord-contact-model',choices=('native','geometric'),default='native')
    parser.add_argument('--coord-leg-trust',type=float,default=0.,help='QP leg target maximum offset from nominal in rad; zero disables the extra bound')
    parser.add_argument('--coord-nominal-hips',action='store_true',help='Retain nominal hip position impedance in the joint capture QP')
    parser.add_argument('--coord-joint-impedance',action='store_true',help='Map original joint restoring effort through constrained native inertia')
    parser.add_argument('--coord-servo-discrete',action='store_true',help='Predict all 20 original 1ms servo steps under frozen contact dynamics')
    parser.add_argument('--coord-hold-only',action='store_true',help='Diagnostic capture/hold only; never hand off or authorize the movement phase')
    parser.add_argument('--ordinary-live-entry',action='store_true',help='Current-geometry support and measured leg/wheel kinematics during ordinary entry')
    parser.add_argument('--ordinary-continue-control',action='store_true',help='Keep the live ordinary controller after entry; readiness only releases the command clock')
    parser.add_argument('--ordinary-rolling-mean',action='store_true',help='Use previous 20ms measured leg velocity mean only for rolling kinematics; actual servo constraints stay instantaneous')
    parser.add_argument('--ordinary-native-bias',action='store_true',help='Use measured native Coriolis plus gravity in live entry; same independently checked mass matrix')
    parser.add_argument('--ordinary-motion-target-rate',type=float,default=0.,help='Optional leg target rate after preparation only; original position and instantaneous torque bounds take priority')
    parser.add_argument('--ordinary-hold-hip-preload',action='store_true',help='After preparation keep the reference hip preload; retain original position servos and torque bounds')
    parser.add_argument('--ordinary-wheel-torque',action='store_true',help='Realize allocated wheel torque at the control boundary using actual wheel speed plus torque/2')
    parser.add_argument('--ordinary-measured-traction-mu',type=float,default=0.,help='After preparation only, cap allocated wheel torque using measured wheel load and this control friction budget; physical mu stays unchanged')
    parser.add_argument('--ordinary-traction-budget-contact-only',action='store_true',help='Budget the wheel contact torque separately from inertial bias; retain original total servo limits')
    parser.add_argument('--ordinary-downhill-rolling-entry',action='store_true',help='During downhill preparation, roll while lowering, then brake after measured low-posture capture; preparation distance never counts toward 1m')
    parser.add_argument('--ordinary-downhill-settled-entry',action='store_true',help='Require low-posture angular and leg velocity convergence before the downhill preparation brake')
    parser.add_argument('--ordinary-motion-velocity-gain',type=float,default=1.5,help='Along-slope velocity feedback after preparation only; entry keeps its original gain')
    parser.add_argument('--ordinary-motion-acceleration-limit',type=float,default=.5,help='Along-slope feedback acceleration limit after preparation only in m/s2; entry keeps its original limit')
    parser.add_argument('--entry-support',choices=('off','fixed','adaptive','grounded','wbc','planned'),default='off',help='Isolated current-geometry force allocation during ordinary Cartesian entry')
    parser.add_argument('--entry-contact-gap',type=float,default=.006)
    parser.add_argument('--entry-recoil',action='store_true')
    parser.add_argument('--entry-wheel-torque',action='store_true')
    parser.add_argument('--entry-crouch-first',action='store_true')
    parser.add_argument('--reference-native-stance',action='store_true',help='Explicit native standing geometry candidate; requires new static preload computation')
    parser.add_argument('--entry-force-filter',type=float,default=0.)
    parser.add_argument('--entry-target-rate',type=float,default=0.)
    parser.add_argument('--spring-preload-scale',type=float,default=1.,help='Loaded-reset diagnostic only: scale initial physical spring strain, never gravity or running force')
    selected,remaining=parser.parse_known_args()
    def option(name,default):
        return float(remaining[remaining.index(name)+1]) if name in remaining else default
    ordinary_friction_mu=forwarded_ground_friction(remaining)
    from compliant_towing import diagnostic_parameters
    try:
        compliant_parameters=diagnostic_parameters(selected.diagnostic_compliant_drawbar,
            selected.diagnostic_drawbar_stiffness,selected.diagnostic_drawbar_damping,
            selected.initial_state,selected.ordinary_live_entry)
    except ValueError as error:
        parser.error(str(error))
    compliant_sources=None
    student_prefix=None
    if selected.student_prefix_checkpoint is not None:
        if (selected.initial_state!='ordinary' or not selected.ordinary_live_entry
            or not selected.ordinary_continue_control or not selected.ordinary_wheel_torque
            or not selected.fixed_command_clock):
            parser.error('Student-prefix recovery requires ordinary continuous wheel-torque teacher and an explicit fixed command clock')
        from student_prefix import StudentPrefix
        try:
            student_prefix=StudentPrefix(selected.student_prefix_checkpoint,selected.student_prefix_seconds,
                selected.slope_deg,selected.move_start)
        except (ValueError,RuntimeError,OSError) as error:
            parser.error(str(error))
    elif selected.student_prefix_seconds:
        parser.error('Student prefix duration requires an explicit source checkpoint')
    entering=selected.initial_state in ('ordinary','continuation')
    if selected.entry_rear_first:
        if (not selected.ordinary_live_entry or not selected.ordinary_continue_control
            or not selected.static_preload or selected.entry_crouch_first or selected.entry_forward_m
            or selected.entry_rear_first<.04 or selected.entry_duration-selected.entry_rear_first<.04):
            parser.error('Rear-first path needs continued ordinary entry, explicit static preload, zero forward shift and two positive grid stages')
    if selected.ordinary_continue_control and not selected.ordinary_live_entry:
        parser.error('Continued ordinary control requires --ordinary-live-entry')
    if selected.ordinary_rolling_mean and not selected.ordinary_live_entry:
        parser.error('Mean rolling kinematics requires --ordinary-live-entry')
    if selected.ordinary_native_bias and not selected.ordinary_live_entry:
        parser.error('Native ordinary bias requires --ordinary-live-entry')
    if not np.isfinite(selected.ordinary_motion_target_rate) or not 0<=selected.ordinary_motion_target_rate<=10.:
        parser.error('Ordinary motion target rate must be in [0,10]rad/s')
    if selected.ordinary_motion_target_rate and not selected.ordinary_continue_control:
        parser.error('Ordinary motion target rate requires continued ordinary control')
    if selected.ordinary_hold_hip_preload and not selected.ordinary_continue_control:
        parser.error('Ordinary hip preload requires continued ordinary control')
    if selected.ordinary_wheel_torque and not selected.ordinary_live_entry:
        parser.error('Ordinary wheel torque mode requires live ordinary entry')
    if not np.isfinite(selected.ordinary_measured_traction_mu) or not 0<=selected.ordinary_measured_traction_mu<=3.:
        parser.error('Measured traction budget must be in [0,3]')
    if selected.ordinary_measured_traction_mu and not (selected.ordinary_continue_control and selected.ordinary_wheel_torque):
        parser.error('Measured traction budget requires continued ordinary wheel torque control')
    if selected.ordinary_traction_budget_contact_only and not selected.ordinary_measured_traction_mu:
        parser.error('Contact-only traction budget requires a positive measured traction budget')
    if selected.ordinary_downhill_rolling_entry and not (selected.slope_deg<0 and selected.ordinary_continue_control and selected.ordinary_wheel_torque):
        parser.error('Downhill rolling entry requires a downhill slope and continued ordinary wheel torque control')
    if selected.ordinary_downhill_settled_entry and not selected.ordinary_downhill_rolling_entry:
        parser.error('Settled downhill entry requires downhill rolling preparation')
    if (not np.isfinite(selected.ordinary_motion_velocity_gain) or selected.ordinary_motion_velocity_gain<0
        or not np.isfinite(selected.ordinary_motion_acceleration_limit) or selected.ordinary_motion_acceleration_limit<0):
        parser.error('Ordinary motion velocity gain and acceleration limit must be finite and nonnegative')
    if (selected.ordinary_motion_velocity_gain!=1.5 or selected.ordinary_motion_acceleration_limit!=.5) and not selected.ordinary_continue_control:
        parser.error('Changed ordinary motion feedback requires continued ordinary control')
    if selected.ordinary_live_entry and (selected.initial_state!='ordinary' or selected.entry_mode!='cartesian'
        or selected.entry_support!='off' or selected.coordinated_capture or selected.entry_speed_capture
        or selected.entry_torque_capture or selected.entry_free_roll_capture or selected.entry_load_share
        or selected.sagittal_servo_stiffness not in (None,80.) or selected.sagittal_servo_damping!=3.):
        parser.error('Live ordinary entry requires ordinary Cartesian entry, original 80/3 servo, and no other controller override')
    if selected.coordinated_capture and (selected.initial_state!='continuation' or selected.initial_pose_progress!=1. or selected.entry_support!='off'
        or selected.entry_speed_capture or selected.entry_free_roll_capture or selected.entry_load_share or selected.entry_torque_capture
        or selected.sagittal_servo_stiffness not in (None,80.) or selected.sagittal_servo_damping!=3.):
        parser.error('Coordinated capture requires loaded-geometry continuation, original servos, no capture/support overrides')
    if not np.isfinite(selected.coord_leg_trust) or not 0<=selected.coord_leg_trust<=.2:
        parser.error('Coordinated leg trust must lie in [0,0.2]rad')
    if selected.coord_servo_discrete and selected.coord_servo_horizon!=.02:
        parser.error('Discrete servo requires a full 20ms horizon')
    if not selected.coordinated_capture and (selected.coord_servo_horizon!=.02 or selected.coord_contact_model!='native' or selected.coord_leg_trust or selected.coord_nominal_hips or selected.coord_joint_impedance or selected.coord_servo_discrete or selected.coord_hold_only):
        parser.error('Coord options require coordinated capture')
    if not np.isfinite(selected.entry_speed_capture) or not 0<=selected.entry_speed_capture<=.5:
        parser.error('Speed capture step must lie in [0,0.5]m/s')
    if selected.entry_speed_capture and (not entering or selected.entry_mode!='cartesian' or selected.entry_support!='off' or selected.entry_torque_capture or selected.entry_free_roll_capture or selected.entry_load_share):
        parser.error('Speed capture requires Cartesian entry with original allocation and no other capture override')
    if not np.isfinite(selected.initial_route_velocity) or not -2<=selected.initial_route_velocity<=0:
        parser.error('Initial route velocity must be within [-2,0]m/s')
    if selected.initial_route_velocity and (selected.initial_state!='continuation' or selected.initial_pose_progress!=1.):
        parser.error('Initial velocity diagnostic requires continuation at the loaded pose endpoint')
    if not np.isfinite(selected.initial_pose_progress) or not 0<=selected.initial_pose_progress<=1:
        parser.error('Initial pose progress must be in [0,1]')
    if selected.initial_state=='continuation' and (selected.entry_mode!='cartesian' or selected.entry_support!='off' or selected.entry_crouch_first or selected.entry_forward_m):
        parser.error('Continuation requires Cartesian entry, no support override, no crouch-first and no forward shift')
    if selected.initial_state!='continuation' and selected.initial_pose_progress!=1.:
        parser.error('Initial pose progress only applies to continuation reset')
    if not np.isfinite(selected.entry_free_roll_capture) or not 0<=selected.entry_free_roll_capture<=5.:
        parser.error('Free-roll capture duration must lie in [0,5]')
    if selected.entry_free_roll_capture and (not entering or selected.entry_mode!='cartesian' or selected.entry_support!='off'):
        parser.error('Free-roll capture requires Cartesian entry without a support override')
    if selected.entry_load_share and (not entering or selected.front_force_share is None or selected.entry_support!='off'):
        parser.error('Entry load share requires entry, a final front share and no support override')
    if selected.entry_torque_capture and (not selected.entry_free_roll_capture or not selected.entry_load_share):
        parser.error('Torque capture requires explicit free-roll capture duration and entry load share')
    if selected.entry_contact_traction and not selected.entry_torque_capture:
        parser.error('Individual contact traction requires torque capture')
    if selected.entry_capture_reference!='wheel' and not selected.entry_torque_capture:
        parser.error('Capture velocity reference requires torque capture')
    if not np.isfinite(selected.entry_capture_acceleration) or not .05<=selected.entry_capture_acceleration<=.5:
        parser.error('Capture acceleration must lie in [0.05,0.5]m/s2')
    if selected.entry_capture_acceleration!=.1 and selected.entry_capture_reference!='ramp':
        parser.error('Nondefault capture acceleration requires ramp reference')
    if selected.entry_capture_start is not None and (not selected.entry_torque_capture or not np.isfinite(selected.entry_capture_start) or not 0<=selected.entry_capture_start<=selected.entry_duration):
        parser.error('Capture start requires torque capture and lies within the entry duration')
    if not 5<=selected.cart_mass<=40 or not 20<=selected.trial_duration<=60 or not .1<=selected.move_speed<=.4:
        parser.error('Invalid physical cart mass/duration/speed')
    if not 4<=selected.cart_position_iterations<=32 or not 1<=selected.cart_velocity_iterations<=8:
        parser.error('Invalid cart-only solver iteration count')
    if selected.sagittal_servo_stiffness is not None and (not np.isfinite(selected.sagittal_servo_stiffness) or not 80<=selected.sagittal_servo_stiffness<=320):
        parser.error('Research sagittal stiffness must lie in [80,320]')
    if not np.isfinite(selected.sagittal_servo_damping) or not 3<=selected.sagittal_servo_damping<=12:parser.error('Sagittal damping must lie in [3,12]')
    if not np.isfinite(selected.leg_force_gain) or not 0<=selected.leg_force_gain<=1:parser.error('Leg force gain must lie in [0,1]')
    if not np.isfinite(selected.reset_lateral_offset) or abs(selected.reset_lateral_offset)>.5:parser.error('Lateral offset must be finite and within 0.5m')
    if not .2<=selected.entry_duration<=12 or not 3<=selected.move_start<=20 or selected.entry_duration>selected.move_start:parser.error('Invalid entry/movement timing')
    if selected.match_static_force_share and (not selected.static_preload or selected.front_force_share is None):
        parser.error('Static force matching needs --static-preload and --front-force-share')
    if (selected.entry_velocity_feedforward or selected.entry_ground_seeking) and (not entering or selected.entry_mode!='cartesian'):
        parser.error('Entry velocity/ground feedback needs ordinary initial state and Cartesian entry mode')
    if selected.entry_wheel_kinematics and not entering:parser.error('Entry wheel kinematics needs an entry initial state')
    if selected.entry_support!='off' and (selected.initial_state!='ordinary' or selected.entry_mode!='cartesian'):
        parser.error('Entry support requires ordinary Cartesian entry')
    if not .003<=selected.entry_contact_gap<=.03:parser.error('Entry contact gap must be within [0.003,0.03]m')
    if selected.entry_recoil and selected.entry_support!='wbc':parser.error('Entry recoil currently requires WBC')
    if not 0<=selected.entry_force_filter<=.5 or not 0<=selected.entry_target_rate<=5.:parser.error('Invalid entry filter/rate')
    if selected.entry_crouch_first and (selected.initial_state!='ordinary' or selected.entry_mode!='cartesian'):
        parser.error('Crouch-first entry requires ordinary Cartesian entry')
    if (selected.entry_wheel_torque or selected.entry_force_filter or selected.entry_target_rate) and selected.entry_support in ('off','wbc'):
        parser.error('Support torque/filter/rate options require a support allocator')
    if not 0<=selected.spring_preload_scale<=1. or (selected.spring_preload_scale!=1. and selected.initial_state=='ordinary'):
        parser.error('Spring preload diagnostic must be in [0,1] and use loaded reset')
    if '--output' not in remaining or any(x in remaining for x in ('--cases','--seconds','--pose-file','--balance-options','--wheel-boost-gain')):
        parser.error('Provide a fresh --output; this entry owns the case and controller')
    reference=reference_from_run(selected.reference_run.resolve(strict=True),*selected.reference_window)
    if selected.slope_deg!=25.:
        reference['source_quat_before_slope_rotation']=reference['quat']
        R=Rotation.from_euler('y',25.-selected.slope_deg,degrees=True)*Rotation.from_quat(np.asarray(reference['quat'])[[1,2,3,0]])
        reference['quat']=R.as_quat()[[3,0,1,2]].tolist()
    if selected.reference_native_stance:
        if not selected.static_preload or selected.mirror_reference:parser.error('Native stance reference requires static preload and no mirror')
        from entry_path import DEFAULT_Q
        reference.update(native_stance_candidate=True,source_q_before_native_override=reference['q'],
            q=DEFAULT_Q.tolist(),height_m=.416223,
            quat=Rotation.from_euler('y',-selected.slope_deg,degrees=True).as_quat()[[3,0,1,2]].tolist(),leg_action=[0.]*12)
    if selected.mirror_reference:
        if not selected.static_preload:parser.error('Mirroring needs explicit static preload recalculation')
        from static_preload import mirror_posture
        reference=mirror_posture(reference,selected.slope_deg)
    out=Path(remaining[remaining.index('--output')+1]).resolve();case_file=out.parent/(out.name+'_case.json')
    if out.exists() or case_file.exists():parser.error('Use a new output path')
    out.parent.mkdir(parents=True,exist_ok=True)
    entry_plan=None
    if entering and selected.entry_mode=='cartesian':
        if selected.entry_rear_first:
            from rear_entry_path import build_rear_path
            entry_plan=build_rear_path(reference,selected.slope_deg,selected.entry_rear_first,
                selected.entry_duration-selected.entry_rear_first)
            reference=entry_plan['endpoint_reference']
        else:
            from entry_path import build_path
            entry_plan=build_path(reference,selected.slope_deg,selected.entry_duration,forward_m=selected.entry_forward_m,crouch_first=selected.entry_crouch_first,
                start_progress=(selected.initial_pose_progress if selected.initial_state=='continuation' else 0.),
                preserve_wheel_phase=selected.initial_state=='continuation')
        launch.write_json(out.parent/(out.name+'_entry_path.json'),entry_plan)
    launch.write_json(case_file,[dict(slope_deg=selected.slope_deg,payload_kg=0.,towed_kg=selected.cart_mass,speed=selected.move_speed,goal_distance_m=1.2)])
    source=reference['source_request'];hip_kp=source.get('hip_servo_stiffness_Nm_rad',80.);sag_kp=source.get('sagittal_servo_stiffness_Nm_rad',80.)
    if selected.sagittal_servo_stiffness is not None:
        old=np.tile([hip_kp,sag_kp,sag_kp],4);sag_kp=selected.sagittal_servo_stiffness;new=np.tile([hip_kp,sag_kp,sag_kp],4)
        reference['servo_retarget']=dict(source_gain=old.tolist(),actual_gain=new.tolist(),source_leg_action=reference['leg_action'])
        reference['leg_action']=retarget_preload(reference['leg_action'],reference['q'][:12],old,new).tolist()
    if selected.static_preload:
        from static_preload import rebase
        reference=rebase(reference,selected.slope_deg,selected.cart_mass,option('--hitch-height',.07),option('--robot-hitch-offset',0.),
            np.tile([hip_kp,sag_kp,sag_kp],4),ordinary_friction_mu,front_share=(selected.front_force_share if selected.match_static_force_share else None))
    initial_reference=reference
    if selected.initial_state=='continuation':
        normal=np.array([-np.sin(np.deg2rad(selected.slope_deg)),0.,np.cos(np.deg2rad(selected.slope_deg))])
        initial_reference=dict(reference,q=entry_plan['q'][0],quat=entry_plan['body_quaternion_plan'][0],
            height_m=float(np.asarray(entry_plan['body_position_plan_m'][0])@normal))
    initial_joint_velocity=np.zeros(16)
    if selected.initial_route_velocity:
        from entry_initial_velocity import rolling_joint_velocity
        initial_joint_velocity=rolling_joint_velocity(initial_reference,selected.slope_deg,selected.initial_route_velocity)
    options=BalanceOptions(capture_start_s=3.,capture_end_s=4.,blend_s=1.,roll_gain=.8,rate_gain=.12,
        horizontal_attitude_gain=selected.horizontal_attitude_gain,leg_force_gain=selected.leg_force_gain,force_pair_average=selected.force_pair_average,contact_moment_gain=1.,hip_force_scale=0.,geometry_mode='iterative',load_geometry='desired',
        max_joint_delta_rad=.5,max_leg_rate_rad_s=selected.leg_rate_limit)
    balance=PartialSupportControl(options,servo_stiffness=np.tile([hip_kp,sag_kp,sag_kp],4))
    sample=tuple(np.asarray(reference[k]) for k in ('leg_action','q','angles_rad','force_body_N','moment_body_Nm'))
    balance.samples=[tuple(value.copy() for value in sample) for _ in range(50)]
    holder={};original_configure=worker.configure;original_module=launch.module_from_file;original_write=launch.write_json
    if selected.ordinary_live_entry:
        from ordinary_entry_controller import OrdinaryEntryController
        holder['ordinary']=OrdinaryEntryController(entry_plan,reference,selected.slope_deg,
            continue_control=selected.ordinary_continue_control,motion_target_rate=selected.ordinary_motion_target_rate,
            hold_hip_preload=selected.ordinary_hold_hip_preload,wheel_torque_control=selected.ordinary_wheel_torque,
            measured_traction_mu=selected.ordinary_measured_traction_mu,
            traction_budget_contact_only=selected.ordinary_traction_budget_contact_only,
            downhill_rolling_entry=selected.ordinary_downhill_rolling_entry,
            downhill_settled_entry=selected.ordinary_downhill_settled_entry,
            friction_mu=ordinary_friction_mu,
            motion_velocity_gain=selected.ordinary_motion_velocity_gain,
            motion_acceleration_limit=selected.ordinary_motion_acceleration_limit)
    if selected.coordinated_capture:
        from coordinated_capture import CoordinatedCapture
        holder['coord']=CoordinatedCapture(reference,selected.slope_deg,selected.coord_servo_horizon,selected.coord_contact_model,selected.coord_leg_trust,selected.coord_nominal_hips,selected.coord_joint_impedance,selected.coord_servo_discrete,selected.coord_hold_only)
    if selected.entry_speed_capture:
        from speed_capture import SpeedCapture
        holder['speed_capture']=SpeedCapture(0. if selected.initial_state=='continuation' else selected.entry_duration,selected.entry_speed_capture)
    if selected.entry_torque_capture:
        from entry_capture import TorqueCapture
        holder['torque_capture']=TorqueCapture(selected.entry_duration if selected.entry_capture_start is None else selected.entry_capture_start,
            selected.entry_free_roll_capture,selected.entry_capture_reference,selected.entry_capture_acceleration)
    if selected.entry_support!='off':
        from entry_support import EntrySupport
        holder['entry_support']=EntrySupport(entry_plan,reference,selected.slope_deg,
            np.tile([hip_kp,sag_kp,sag_kp],4),
            np.tile([source.get('hip_servo_damping_Nms',3.),selected.sagittal_servo_damping,selected.sagittal_servo_damping],4),
            adaptive=selected.entry_support=='adaptive',grounded=selected.entry_support=='grounded',wheel_torque=selected.entry_wheel_torque,planned=selected.entry_support=='planned',
            force_filter=selected.entry_force_filter,target_rate=selected.entry_target_rate)
        if selected.entry_support=='wbc':
            if sag_kp!=80. or hip_kp!=80. or selected.sagittal_servo_damping!=3.:
                parser.error('Entry WBC currently requires original 80/3 leg servo')
            from entry_wbc import EntryWBC
            holder['entry_support']=EntryWBC(entry_plan,reference,selected.slope_deg,selected.entry_contact_gap,selected.entry_recoil)

    def configure(*args,**kwargs):
        cfg,agent=original_configure(*args,**kwargs)
        # The production constructor catalog only contains 0/20/40kg. Its evaluation
        # API then sets the explicit case mass AND corresponding inertia before
        # the first measured reset; runtime verification checks that actual mass.
        cfg.fixed_towed_kg=40.;cfg.fixed_slope_deg=selected.slope_deg
        if selected.rounded_cart_wheels:
            from rounded_cart import replace_wheels
            original_write(out/'rounded_cart_geometry.json',replace_wheels(out/'assets/cart.urdf'))
        cfg.scene.trailer.spawn.articulation_props.solver_position_iteration_count=selected.cart_position_iterations
        cfg.scene.trailer.spawn.articulation_props.solver_velocity_iteration_count=selected.cart_velocity_iterations
        cfg.sim.physx.max_position_iteration_count=max(cfg.sim.physx.max_position_iteration_count,selected.cart_position_iterations)
        cfg.sim.physx.max_velocity_iteration_count=max(cfg.sim.physx.max_velocity_iteration_count,selected.cart_velocity_iterations)
        cfg.eval_distance_m=1.2;cfg.episode_length_s=selected.trial_duration+5
        cfg.scene.robot.actuators['joint'].stiffness={'.*_hip_joint':hip_kp,'.*_thigh_joint':sag_kp,'.*_calf_joint':sag_kp}
        cfg.scene.robot.actuators['joint'].damping={'.*_hip_joint':source.get('hip_servo_damping_Nms',3.),'.*_thigh_joint':selected.sagittal_servo_damping,'.*_calf_joint':selected.sagittal_servo_damping}
        import torch
        from isaaclab.utils.math import quat_apply
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import cart
        if selected.move_start!=3. or selected.entry_support!='off' or selected.entry_torque_capture or selected.entry_speed_capture or selected.coordinated_capture or selected.ordinary_live_entry:
            from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.commands import UnifiedVelocityCommand
            class EntryVelocityCommand(UnifiedVelocityCommand):
                def _update_command(self):
                    # 只延后状态机的运动起点；不改物理时钟、速度斜率或停车时钟。
                    if selected.fixed_command_clock:
                        self.cycle_start_step[:]=round((selected.move_start-3.)/self._env.step_dt)
                    elif ((selected.entry_support!='off' and not holder['entry_support'].finished)
                        or (selected.entry_torque_capture and not holder['torque_capture'].finished)
                        or (selected.entry_speed_capture and not holder['speed_capture'].finished)
                        or (selected.coordinated_capture and not holder['coord'].finished)
                        or (selected.ordinary_live_entry and not holder['ordinary'].finished)):
                        self.cycle_start_step[:]=self._env.episode_length_buf
                    elif (self._env.episode_length_buf*self._env.step_dt<selected.move_start).all():
                        self.cycle_start_step[:]=round((selected.move_start-3.)/self._env.step_dt)
                    super()._update_command()
            cfg.commands.base_velocity.class_type=EntryVelocityCommand
        original_reset=cfg.events.reset_base.func
        def reset_pose(env,env_ids):
            original_reset(env,env_ids)
            ids=torch.as_tensor(env_ids,device=env.device);robot=env.scene['robot'];trailer=env.scene['trailer']
            if selected.initial_state=='ordinary':
                if selected.reset_lateral_offset:
                    for asset in (robot,trailer):
                        pose=asset.data.root_state_w[ids,:7].clone();pose[:,1]+=selected.reset_lateral_offset
                        asset.write_root_pose_to_sim(pose,env_ids=ids)
                holder['env']=env
                return
            state=robot.data.default_root_state[ids].clone()
            state[:,:3]=env.scene.env_origins[ids]+env.r1_normal[ids]*initial_reference['height_m']
            state[:,1]+=selected.reset_lateral_offset
            state[:,3:7]=torch.as_tensor(initial_reference['quat'],device=env.device);state[:,7:]=0
            if selected.initial_route_velocity:state[:,7:10]=env.r1_tangent[ids]*selected.initial_route_velocity
            robot.write_root_pose_to_sim(state[:,:7],env_ids=ids);robot.write_root_velocity_to_sim(state[:,7:],env_ids=ids)
            cart_state=trailer.data.default_root_state[ids].clone();angle=env.r1_slope_rad[ids]
            cart_state[:,3:7]=0;cart_state[:,3]=torch.cos(angle/2);cart_state[:,5]=-torch.sin(angle/2);cart_state[:,7:]=0
            if selected.initial_route_velocity:cart_state[:,7:10]=env.r1_tangent[ids]*selected.initial_route_velocity
            rp=quat_apply(state[:,3:7],env.r1_robot_hitch[ids])+state[:,:3]
            cp_local=quat_apply(cart_state[:,3:7],env.r1_cart_hitch[ids])
            preload=selected.spring_preload_scale*selected.cart_mass*9.81*np.sin(np.deg2rad(selected.slope_deg))
            for j,index in enumerate(ids.tolist()):
                center=cart_root_from_preload(rp[j].cpu().numpy(),cp_local[j].cpu().numpy(),env.r1_tangent[index].cpu().numpy(),
                    env.r1_normal[index].cpu().numpy(),.30,preload/2000.,env.scene.env_origins[index].cpu().numpy())
                cart_state[j,:3]=torch.as_tensor(center,device=env.device)
            trailer.write_root_pose_to_sim(cart_state[:,:7],env_ids=ids);trailer.write_root_velocity_to_sim(cart_state[:,7:],env_ids=ids)
            cart_qd=torch.zeros_like(trailer.data.default_joint_vel[ids])
            if selected.initial_route_velocity:cart_qd[:]=selected.initial_route_velocity/.14
            trailer.write_joint_state_to_sim(trailer.data.default_joint_pos[ids],cart_qd,env_ids=ids)
            env.r1_eval_origin[ids]=state[:,:3];env.r1_eval_progress[ids]=0
            holder['env']=env
        cfg.events.reset_base.func=reset_pose
        def reset_joints(env,env_ids):
            robot=env.scene['robot'];ids=torch.as_tensor(env_ids,device=env.device)
            names=source['base_network_contract']['action_order']
            q=torch.as_tensor([initial_reference['q'][names.index(n)] for n in robot.joint_names],device=env.device).expand(len(ids),-1).clone()
            qd=torch.zeros_like(q)
            if selected.initial_route_velocity:
                qd[:]=torch.as_tensor([initial_joint_velocity[names.index(n)] for n in robot.joint_names],device=env.device)
            robot.write_joint_state_to_sim(q,qd,env_ids=ids)
        if selected.initial_state!='ordinary':
            cfg.events.reset_joints.func=reset_joints;cfg.events.reset_joints.params={}
        apply_original=cart.apply_cart_forces
        def apply_and_audit(env):
            if compliant_parameters is None:
                apply_original(env)
            else:
                from compliant_towing import apply_cart_forces
                apply_cart_forces(env,compliant_parameters)
            r,c=env.scene['robot'],env.scene['trailer']
            rh=quat_apply(r.data.root_quat_w,env.r1_robot_hitch)+r.data.root_pos_w
            ch=quat_apply(c.data.root_quat_w,env.r1_cart_hitch)+c.data.root_pos_w
            ra=rh-r.data.body_com_pos_w[:,env.r1_base_id]
            ca=ch-c.data.body_com_pos_w[:,env.r1_cart_base_id]
            env.r1_real_cart_audit=dict(robot_force_w_N=r._external_force_b[:,env.r1_base_id].clone(),
                robot_torque_w_Nm=r._external_torque_b[:,env.r1_base_id].clone(),cart_forces_w_N=c._external_force_b.clone(),
                cart_torques_w_Nm=c._external_torque_b.clone(),cart_masses_kg=env.r1_cart_masses.clone(),
                robot_hitch_pre_w=rh.clone(),cart_hitch_pre_w=ch.clone(),robot_arm_pre_w=ra.clone(),cart_arm_pre_w=ca.clone(),
                robot_hitch_velocity_pre_w=(r.data.root_com_lin_vel_w+torch.cross(r.data.root_ang_vel_w,ra,dim=-1)).clone(),
                cart_hitch_velocity_pre_w=(c.data.root_com_lin_vel_w+torch.cross(c.data.root_ang_vel_w,ca,dim=-1)).clone())
        cart.apply_cart_forces=apply_and_audit
        return cfg,agent

    def module_from_file(name,path):
        module=original_module(name,path)
        if name=='_r1_v7_physical_recorder':
            parent=module.PhysicsRecorder
            class Recorder(parent):
                def __init__(self,base,evaluation_worker):
                    super().__init__(base,evaluation_worker);holder['recorder']=self
                    from contact_state import record_contact_geometry
                    record_contact_geometry(self,base)
                    if selected.record_momentum:
                        from momentum_probe import record_momentum
                        record_momentum(self,base,out,original_write)
                    import torch
                    for key,shape in dict(robot_force_w_N=(3,),robot_torque_w_Nm=(3,),cart_forces_w_N=(5,3),cart_torques_w_Nm=(5,3),cart_masses_kg=(5,)).items():
                        self.buffers['real_cart_'+key]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device)
                        self.history['real_cart_'+key]=[]
                    for key in ('robot_hitch_pre_w','cart_hitch_pre_w','robot_arm_pre_w','cart_arm_pre_w','robot_hitch_velocity_pre_w','cart_hitch_velocity_pre_w'):
                        self.buffers['real_cart_'+key]=torch.zeros((base.num_envs,base.cfg.decimation,3),device=base.device)
                        self.history['real_cart_'+key]=[]
                    for key,shape in dict(cart_quat_w=(4,),cart_wheel_pos_w=(4,3)).items():
                        self.buffers['real_cart_'+key]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device)
                        self.history['real_cart_'+key]=[]
                    self.real_cart_wheels=base.scene['trailer'].find_bodies([f'cart_wheel_{i}' for i in range(4)],preserve_order=True)[0]
                    previous=base.r1_torque_recorder.sample
                    def sample_applied():
                        index=base.r1_torque_recorder.cursor;previous()
                        if base.r1_torque_recorder.cursor>index:
                            for key,value in base.r1_real_cart_audit.items():self.buffers['real_cart_'+key][:,index]=value
                    base.r1_torque_recorder.sample=sample_applied
                def sample(self):
                    index=self.cursor;super().sample();cart=self.base.scene['trailer']
                    self.buffers['real_cart_cart_quat_w'][:,index]=cart.data.root_quat_w
                    self.buffers['real_cart_cart_wheel_pos_w'][:,index]=cart.data.body_pos_w[:,self.real_cart_wheels]
            module.PhysicsRecorder=Recorder
        return module

    def transform(base,unused):
        import torch
        rec=holder['recorder'];buffers=rec.buffers;elapsed=float(base.episode_length_buf[0])*base.step_dt
        normal=base.r1_normal[0].cpu().numpy();tangent=base.r1_tangent[0].cpu().numpy()
        if elapsed==0:
            quat=np.asarray(initial_reference['quat']);q=np.asarray(initial_reference['q']);omega=np.zeros(3);velocity=selected.initial_route_velocity
            if selected.initial_state=='ordinary':
                robot=base.scene['robot'];ids=robot.find_joints(source['base_network_contract']['action_order'],preserve_order=True)[0]
                quat=robot.data.root_quat_w[0].cpu().numpy();q=robot.data.joint_pos[0,ids].cpu().numpy()
            original_write(out/'initial_state_runtime.json',dict(mode=selected.initial_state,
                initial_pose_progress=(selected.initial_pose_progress if selected.initial_state=='continuation' else None),
                initial_route_velocity_mps=selected.initial_route_velocity,
                env_origin=base.scene.env_origins[0].detach().cpu().tolist(),
                robot_root_state=base.scene['robot'].data.root_state_w[0].detach().cpu().tolist(),
                cart_root_state=base.scene['trailer'].data.root_state_w[0].detach().cpu().tolist(),
                joint_position_action_order=q.tolist(),cart_mass_kg=float(base.r1_cart_masses.sum()),
                episode_step=int(base.episode_length_buf[0]),simulation_device=str(base.device),
                robot_tensor_device=str(base.scene['robot'].data.root_state_w.device),
                cuda_available=bool(torch.cuda.is_available()),
                cuda_device_name=(torch.cuda.get_device_name(base.device) if str(base.device).startswith('cuda') else None),
                note='Captured immediately before first measured control step; no physical state change here.'))
            R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            force=np.asarray(reference['force_body_N'])@R.T;moment=np.asarray(reference['moment_body_Nm'])@R.T
            joints_v=initial_joint_velocity.copy();normals=force@normal
        else:
            q=buffers['joint_pos'][0,-1].cpu().numpy();quat=buffers['body_quat_w'][0,-1].cpu().numpy()
            omega=buffers['body_omega_b'][0,-1].cpu().numpy();joints_v=buffers['joint_velocity'][0,-1].cpu().numpy()
            f=buffers['native_ground_normal_w'][0]+buffers['friction_force_w'][0]
            m=buffers['native_contact_moment_com_w'][0]+buffers['native_friction_moment_com_w'][0]
            m-=torch.cross(buffers['wheel_pos_w'][0]-buffers['whole_com_w'][0,:,None,:],f,dim=-1)
            force=f.mean(0).cpu().numpy();moment=m.mean(0).cpu().numpy()
            velocity=float((buffers['whole_com_velocity_w'][0,-1]*base.r1_tangent[0]).sum())
            normals=(buffers['native_ground_normal_w'][0].cpu().numpy()@normal).mean(0)
        leg,extra=balance.step(elapsed+5.,np.asarray(reference['leg_action']),q,quat,normal,omega,force,moment,joints_v)
        if entering:
            if selected.entry_ground_seeking:extra['entry_contact_drop_m']=np.zeros(4)
            fraction=entry_fraction(elapsed,selected.entry_duration)
            if selected.entry_mode=='blend':leg=leg*fraction
            elif fraction<1. or (selected.entry_ground_seeking and elapsed<selected.move_start):
                default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5])
                desired=balance.reference_q_full.copy();desired[:12]=default+fraction*(desired[:12]-default)
                if entry_plan is not None:desired=np.asarray(entry_plan['q'][min(round(elapsed/.02),len(entry_plan['q'])-1)])
                if selected.entry_ground_seeking:
                    from partial_support_control import solve_foot_pose
                    from research_common import LEGS
                    local=balance.model.state(np.zeros(3),[1.,0.,0.,0.],desired)
                    centers=np.array([local['trans'][name+'_wheel_link'][:3,3] for name in LEGS])
                    R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();normal_body=R.T@normal
                    height=float((base.scene['robot'].data.root_pos_w[0]-base.scene.env_origins[0])@base.r1_normal[0])
                    gap=np.maximum(height+centers@normal_body-.0875,0.)
                    # 前轮目标向坡面补偿，后腿继续调姿；仅关节控制，不施加接触或根状态。
                    target=np.zeros(4);target[:2]=np.minimum(.5*gap[:2],.08)
                    previous_drop=holder.get('entry_contact_drop',np.zeros(4))
                    drop=previous_drop+.5*(target-previous_drop);holder['entry_contact_drop']=drop
                    desired=solve_foot_pose(balance.model,desired,centers,-drop[:,None]*normal_body)
                    extra['entry_contact_drop_m']=drop
                state=balance.model.state(np.zeros(3),quat,desired)
                ref_state=balance.model.state(np.zeros(3),quat,balance.reference_q_full)
                ff=(support_joint_torque(balance.model,state,balance.filtered_force,balance.filtered_moment)
                    -support_joint_torque(balance.model,ref_state,balance.filtered_force,balance.filtered_moment))/balance.servo_stiffness
                ff[::3]=0.
                leg+=(desired[:12]-balance.reference_q+selected.leg_force_gain*ff)/SCALE
            if selected.entry_velocity_feedforward:
                if entry_plan is None:raise ValueError('Entry velocity feedforward requires Cartesian path')
                qd=np.asarray(entry_plan['joint_velocity_rad_s'][min(round(elapsed/.02),len(entry_plan['q'])-1)])
                leg+=np.tile([source.get('hip_servo_damping_Nms',3.),selected.sagittal_servo_damping,selected.sagittal_servo_damping],4)*qd[:12]/balance.servo_stiffness/SCALE
            extra['entry_fraction']=fraction
        filtered=holder.get('normals',normals);filtered=filtered+(.02/.12)*(normals-filtered);holder['normals']=filtered
        cmd=float(base.command_manager.get_command('base_velocity')[0,0]);robot_mass=float(rec.masses.sum())
        if selected.entry_speed_capture and not holder['speed_capture'].finished:
            from partial_support_control import attitude
            cmd=holder['speed_capture'].command(elapsed,velocity)
            height=float((base.scene['robot'].data.root_pos_w[0]-base.scene.env_origins[0])@base.r1_normal[0])
            angles,_=attitude(quat,normal,omega);goal,_=attitude(reference['quat'],normal,np.zeros(3))
            extra.update(holder['speed_capture'].observe(cmd,velocity,height-reference['height_m'],angles-goal,omega,
                float(abs(q[:12]-np.asarray(reference['q'])[:12]).max()),joints_v[12:],normals))
        elif selected.entry_speed_capture:
            extra.update(entry_speed_reference_mps=cmd,entry_speed_ready_s=holder['speed_capture'].ready_s,entry_speed_finished=1.)
        rolling=-float(base.r1_rolling_torque.sum())/.14
        slope=np.deg2rad(selected.slope_deg)
        load=selected.cart_mass*9.81*np.sin(slope)+rolling
        if selected.wheel_load_source=='drawbar':
            if elapsed>0:load=-float((buffers['real_cart_robot_force_w_N'][0]@base.r1_tangent[0]).mean())
            elif compliant_parameters is not None:
                from compliant_towing import estimate_robot_force
                load=-float((estimate_robot_force(base,compliant_parameters)[0]@base.r1_tangent[0]))
                base.r1_compliant_initial_load_calls=getattr(base,'r1_compliant_initial_load_calls',0)+1
            else:
                from isaaclab.utils.math import quat_apply
                r,c=base.scene['robot'],base.scene['trailer']
                delta=(quat_apply(r.data.root_quat_w,base.r1_robot_hitch)+r.data.root_pos_w
                    -quat_apply(c.data.root_quat_w,base.r1_cart_hitch)-c.data.root_pos_w)[0]
                length=float(delta.norm());load=2000.*(length-1.)*float(delta@base.r1_tangent[0])/max(length,1.e-6)
        # Full true cart gravity is present from t=0; wheel targets supply all propulsion.
        front_share=selected.front_force_share
        if selected.entry_load_share:
            measured=np.maximum(filtered,0.)
            initial_share=float(np.clip(measured[:2].sum()/max(measured.sum(),1.e-9),.01,.99))
            capture=entry_fraction(max(0.,elapsed-selected.entry_duration),max(1.,selected.entry_free_roll_capture))
            if selected.entry_torque_capture:capture=holder['torque_capture'].handoff
            front_share=(1.-capture)*initial_share+capture*selected.front_force_share
            extra['entry_front_force_share']=front_share
        try:
            allocation=allocated_targets(cmd,velocity,load,filtered,robot_mass,slope,base.cfg.ground_friction,
                uniform=True,front_share=front_share,signed_load=True)
            wheel=allocation['target_rad_s'];wheel_torque=allocation['torque_feedforward_Nm']
        except ValueError:
            wheel=holder.get('wheel',np.zeros(4)).copy();wheel_torque=np.zeros(4)
        if selected.entry_contact_traction:
            try:
                contact=allocated_targets(cmd,velocity,load,normals,robot_mass,slope,base.cfg.ground_friction,signed_load=True)
                contact_torque=contact['torque_feedforward_Nm']
            except ValueError:contact_torque=np.zeros(4)
            capture=holder['torque_capture'].handoff
            wheel_torque=(1.-capture)*contact_torque+capture*wheel_torque
            extra['entry_capture_allocated_normal_N']=np.maximum(normals,0.)
        if selected.wheel_heading_gain and elapsed>0:
            R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();side=np.cross(normal,tangent)
            yaw=np.arctan2(R[:,0]@side,R[:,0]@tangent)
            y=(buffers['wheel_pos_w'][0,-1]-buffers['whole_com_w'][0,-1]).cpu().numpy()@side
            wheel+=heading_velocity_offset(yaw,float((R@omega)@normal),y,selected.wheel_heading_gain)
        previous=holder.get('wheel',wheel);wheel=previous+np.clip(wheel-previous,-.1,.1);holder['wheel']=wheel.copy()
        entry_wheel=np.zeros(4)
        if entering and selected.entry_wheel_kinematics:
            entry_wheel=entry_wheel_velocity(balance.model,elapsed,selected.entry_duration,balance.reference_q_full,quat,tangent,normal)
            if entry_plan is not None:entry_wheel=np.asarray(entry_plan['wheel_rad_s'][min(round(elapsed/.02),len(entry_plan['q'])-1)])
            wheel=np.clip(wheel+entry_wheel,-50.,50.)
        if selected.entry_torque_capture:
            from partial_support_control import attitude
            robot=base.scene['robot'];height=float((robot.data.root_pos_w[0]-base.scene.env_origins[0])@base.r1_normal[0])
            angles,_=attitude(quat,normal,omega);goal_angles,_=attitude(reference['quat'],normal,np.zeros(3))
            wheel,capture_trace=holder['torque_capture'].step(elapsed,joints_v[12:],wheel_torque,wheel,velocity,height,
                reference['height_m'],angles,goal_angles,omega,float(abs(q[:12]-np.asarray(reference['q'])[:12]).max()),normals)
            extra.update(capture_trace)
        elif selected.entry_free_roll_capture:
            capture=entry_fraction(max(0.,elapsed-selected.entry_duration),selected.entry_free_roll_capture)
            # 仅改原轮速目标：调姿期跟随实测轮速，允许整车自然顺坡滚动。
            # 重力、弹簧、被动车及原2Nm/(rad/s)速度伺服全程保留；零初始请求不等于周期内零力矩。
            wheel=(1.-capture)*joints_v[12:]+capture*wheel
            wheel=np.clip(wheel,-50.,50.)
            extra['entry_capture_fraction']=capture
        action=np.r_[leg,wheel/5.]
        if selected.coordinated_capture:
            action,coord_trace=holder['coord'].step(base,rec,elapsed,action)
            extra.update(coord_trace)
        if selected.ordinary_live_entry:
            from isaaclab.utils.math import quat_apply
            from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.cart_asset import drawbar_force
            robot,cart=base.scene['robot'],base.scene['trailer']
            rh=quat_apply(robot.data.root_quat_w,base.r1_robot_hitch)+robot.data.root_pos_w
            ch=quat_apply(cart.data.root_quat_w,base.r1_cart_hitch)+cart.data.root_pos_w
            rv=robot.data.root_com_lin_vel_w+torch.cross(robot.data.root_ang_vel_w,rh-robot.data.root_com_pos_w,dim=-1)
            cv=cart.data.root_com_lin_vel_w+torch.cross(cart.data.root_ang_vel_w,ch-cart.data.root_com_pos_w,dim=-1)
            if compliant_parameters is None:
                force_now=-drawbar_force(rh,ch,rv,cv,base.r1_towed_kg>0)[0][0].detach().cpu().numpy()
            else:
                from compliant_towing import estimate_robot_force
                force_now=estimate_robot_force(base,compliant_parameters)[0].detach().cpu().numpy()
                base.r1_compliant_live_estimator_calls=getattr(base,'r1_compliant_live_estimator_calls',0)+1
            native_bias=None
            if selected.ordinary_native_bias:
                from momentum_probe import native_dynamics
                native_bias=native_dynamics(base)['dynamics_bias']
            action,entry_trace=holder['ordinary'].step(elapsed,q,joints_v,quat,
                robot.data.root_ang_vel_w[0].detach().cpu().numpy(),
                (robot.data.root_pos_w[0]-base.scene.env_origins[0]).detach().cpu().numpy(),
                robot.data.root_com_lin_vel_w[0].detach().cpu().numpy(),force_now,
                base.r1_robot_hitch[0].detach().cpu().numpy(),np.zeros(4) if elapsed==0 else normals,action,command_velocity=cmd,
                rolling_joint_velocity=(buffers['joint_velocity'][0].mean(0).cpu().numpy()
                    if selected.ordinary_rolling_mean and elapsed>0 else None),native_bias22=native_bias)
            extra.update(entry_trace)
        if selected.entry_support!='off':
            controller=holder['entry_support'];robot=base.scene['robot']
            height=float((robot.data.root_pos_w[0]-base.scene.env_origins[0])@base.r1_normal[0])
            external=(buffers['real_cart_robot_force_w_N'][0].mean(0).cpu().numpy() if elapsed>0
                else -load*tangent)
            if selected.entry_support=='wbc':action,support_trace=controller.step(base,rec,elapsed,action)
            else:action,support_trace=controller.step(elapsed,q,joints_v,quat,omega,height,robot.data.root_com_lin_vel_w[0].cpu().numpy(),external,
                base.r1_robot_hitch[0].cpu().numpy(),normals,action)
            extra.update(support_trace)
        extra.update(teacher_time_s=elapsed,teacher_load_feedforward_N=load,teacher_wheel_target_rad_s=wheel,entry_wheel_offset_rad_s=entry_wheel)
        if selected.record_native_dynamics:
            from momentum_probe import native_dynamics
            extra.update(native_dynamics(base))
        result=torch.as_tensor(action[None],device=base.device,dtype=unused.dtype)
        trace={k:torch.as_tensor(np.asarray(v)[None],device=base.device) for k,v in extra.items()}
        if student_prefix is not None:
            result,prefix_trace=student_prefix.select(elapsed,holder['student_observation'],result)
            trace.update(prefix_trace)
        return result,trace

    def write_json(path,content):
        nonlocal compliant_sources
        if Path(path).name=='request.json':
            content=dict(content,physical_trailer_connected=True,formal_behavior_evaluation=False,route_acceptance=False,
                controller='loaded_reference_analytic_controller',actor_inference=student_prefix is not None,actor_weights_loaded_for_provenance=True,
                fixed_command_clock=selected.fixed_command_clock,
                student_prefix=None if student_prefix is None else student_prefix.metadata,
                records_drawbar_pre_step=True,
                records_native_momentum=selected.record_momentum,
                records_native_dynamics=selected.record_native_dynamics,
                entry_capture_reference=selected.entry_capture_reference,
                entry_capture_acceleration=selected.entry_capture_acceleration,
                entry_speed_capture=selected.entry_speed_capture,
                coordinated_capture=selected.coordinated_capture,
                coord_servo_horizon_s=selected.coord_servo_horizon,coord_contact_model=selected.coord_contact_model,
                coord_leg_trust_rad=selected.coord_leg_trust,
                coord_nominal_hips=selected.coord_nominal_hips,
                coord_joint_impedance=selected.coord_joint_impedance,
                coord_servo_discrete=selected.coord_servo_discrete,
                coord_hold_only=selected.coord_hold_only,
                ordinary_live_entry=selected.ordinary_live_entry,
                ordinary_continue_control=selected.ordinary_continue_control,
                ordinary_rolling_mean=selected.ordinary_rolling_mean,
                ordinary_native_bias=selected.ordinary_native_bias,
                ordinary_motion_target_rate_rad_s=selected.ordinary_motion_target_rate,
                ordinary_hold_hip_preload=selected.ordinary_hold_hip_preload,
                ordinary_wheel_torque_control=selected.ordinary_wheel_torque,
                ordinary_measured_traction_mu=selected.ordinary_measured_traction_mu,
                ordinary_traction_budget_contact_only=selected.ordinary_traction_budget_contact_only,
                ordinary_downhill_rolling_entry=selected.ordinary_downhill_rolling_entry,
                ordinary_downhill_settled_entry=selected.ordinary_downhill_settled_entry,
                ordinary_controller_friction_mu=ordinary_friction_mu,
                ordinary_motion_velocity_gain=selected.ordinary_motion_velocity_gain,
                ordinary_motion_acceleration_limit_mps2=selected.ordinary_motion_acceleration_limit,
                reset_lateral_offset_m=selected.reset_lateral_offset,
                initial_state_mode=selected.initial_state,entry_duration_s=selected.entry_duration,move_start_s=selected.move_start,slope_deg=selected.slope_deg,
                entry_rear_first_duration_s=selected.entry_rear_first,
                wheel_load_source=selected.wheel_load_source,entry_mode=selected.entry_mode,entry_wheel_kinematics=selected.entry_wheel_kinematics,
                entry_path=entry_plan,entry_velocity_feedforward=selected.entry_velocity_feedforward,entry_ground_seeking=selected.entry_ground_seeking,
                entry_free_roll_capture_s=selected.entry_free_roll_capture,
                entry_load_share=selected.entry_load_share,
                entry_torque_capture=selected.entry_torque_capture,
                entry_contact_traction=selected.entry_contact_traction,
                entry_capture_start_s=selected.entry_capture_start,
                entry_support=selected.entry_support,
                entry_contact_gap_m=selected.entry_contact_gap,entry_recoil=selected.entry_recoil,
                entry_wheel_torque=selected.entry_wheel_torque,
                entry_force_filter_s=selected.entry_force_filter,entry_target_rate_rad_s=selected.entry_target_rate,
                spring_preload_scale=selected.spring_preload_scale,
                initial_pose_progress=(selected.initial_pose_progress if selected.initial_state=='continuation' else None),
                initial_route_velocity_mps=selected.initial_route_velocity,
                initial_condition=(f'zero velocity at prior validated loaded posture; initial gravity-matched spring strain scaled by {selected.spring_preload_scale:g}; no state writes after reset' if selected.initial_state=='loaded'
                    else (f'intermediate reset at pose progress {selected.initial_pose_progress:g}; physical spring strain scale {selected.spring_preload_scale:g}; common route velocity {selected.initial_route_velocity:g}m/s, full gravity, no state writes after reset; NOT ordinary cold start' if selected.initial_state=='continuation'
                    else 'native ordinary reset: default joints, slope-aligned root at0.416223m, cart center1.73m behind; full mass/gravity from t0; no state writes after reset')),
                physical_cart_mass_kg=selected.cart_mass,reference=reference,balance_options=asdict(options),
                constructor_mass_context_kg=40.,measured_mass_source='evaluation case applied with native set_payload_properties before reset; runtime checked',
                front_force_share=selected.front_force_share,wheel_heading_gain=selected.wheel_heading_gain,
                cart_solver_iterations=[selected.cart_position_iterations,selected.cart_velocity_iterations],
                rounded_cart_wheels=selected.rounded_cart_wheels,
                sagittal_servo_stiffness_Nm_rad=sag_kp,hip_servo_stiffness_Nm_rad=hip_kp,
                sagittal_servo_damping_Nms=selected.sagittal_servo_damping,hip_servo_damping_Nms=source.get("hip_servo_damping_Nms",3.),
                trial_duration_s=selected.trial_duration,initial_spring_energy_J=(.5*2000*(selected.spring_preload_scale*selected.cart_mass*9.81*np.sin(np.deg2rad(selected.slope_deg))/2000)**2 if selected.initial_state!='ordinary' else None),
                physics_note='Original passive-cart per-link gravity, rolling resistance, inertia and equal/opposite spring forces. No known-load surrogate or gravity ramp.')
            if compliant_parameters is not None:
                from compliant_towing import freeze_sources,VARIANT
                compliant_sources=freeze_sources(Path(path).parent)
                declaration=compliant_parameters.metadata()
                content.update(physics_variant=VARIANT,diagnostic_drawbar=declaration,
                    diagnostic_drawbar_sources=compliant_sources,original_teacher_dataset_eligible=False)
                content['execution_physics']=dict(content['execution_physics'],drawbar=declaration)
                content['physics_note']+=' Explicit diagnostic drawbar stiffness/damping; original2000/150 teacher dataset is ineligible.'
            if student_prefix is not None:
                student_prefix.snapshot(Path(path).parent)
        if Path(path).name=='evaluation.json' and student_prefix is not None:
            original_write(Path(path).parent/'student_prefix_execution.json',student_prefix.execution_record())
        if Path(path).name=='evaluation.json' and compliant_parameters is not None:
            from compliant_towing import source_hashes
            if source_hashes()!=compliant_sources:
                raise ValueError('Compliant drawbar execution sources changed during the run')
            env=holder['env']
            original_write(Path(path).parent/'drawbar_runtime.json',dict(
                parameters=getattr(env,'r1_compliant_drawbar_parameters',None),sources=compliant_sources,
                physics_substeps=getattr(env,'r1_compliant_drawbar_physics_calls',0),
                live_estimator_frames=getattr(env,'r1_compliant_live_estimator_calls',0),
                initial_load_estimator_calls=getattr(env,'r1_compliant_initial_load_calls',0)))
        original_write(path,content)
    worker.configure=configure;launch.module_from_file=module_from_file;launch.write_json=write_json
    def source_observation(actor,obs,mode):
        if student_prefix is not None:
            holder['student_observation']=obs.detach().clone()
        return obs.new_zeros((obs.shape[0],16))
    worker.infer_actor=source_observation
    worker.action_transform=transform
    worker.protocol_v7.max_steps=lambda case:round(selected.trial_duration/.02)
    worker.protocol_v7.finished=lambda rows,case:False
    worker.protocol_v7.summarize=lambda rows,case,reason:dict(case=case,samples=len(rows),complete=False,success=False,failure_reason=reason,
        protocol='loaded_reference_real_cart',route_acceptance=False,failed_checks=['requires_independent_physical_cart_review'])
    sys.argv=[sys.argv[0],*remaining,'--cases',str(case_file),'--cart-contact-trace']
    worker.main()


if __name__=='__main__':main()
