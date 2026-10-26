"""后轮先后展、再降姿的隔离几何候选；不写入任何仿真状态。

阶段一保持普通根姿态及前轮世界轮心，后轮沿坡后移；阶段二固定四轮
世界轮心，只降低根 link 法向高度并旋转至来源姿态。返回的末端是新的
几何，来源姿态的受力/动作标定不可据此视为已验证。所有导数均对应本
路径，而非来源姿态的关节插值；不证明动力学、接触或教师行为可行。
"""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from entry_path import DEFAULT_Q
from model import Model
from partial_support_control import solve_foot_pose
from research_common import LEGS


DEFAULT_HEIGHT_M = .416223


def _blend(time, duration):
    """五次段及其一、二阶时间导数；两端明确静止，避免跨段差分尖峰。"""
    if time <= 0.:
        return 0., 0., 0.
    if time >= duration:
        return 1., 0., 0.
    x = time / duration
    return (x**3*(10.-15.*x+6.*x*x),
            30.*x*x*(1.-x)**2/duration,
            60.*x*(1.-x)*(1.-2.*x)/duration**2)


def _wheel_centers(state):
    return np.array([state['trans'][leg+'_wheel_link'][:3, 3] for leg in LEGS])


def _center_jacobians(state):
    return np.array([state['jac'](leg+'_wheel_link', center)[:3]
                     for leg, center in zip(LEGS, _wheel_centers(state))])


def _kinematic_rates(model, state, q, normal, tangent, velocity, omega,
                     acceleration, alpha, center_velocity, center_acceleration):
    """原 URDF Jacobian 解腿速/加速度及切向滚动，不含接触力假设。

Jdot 与 Adot 用沿实际广义速度的中心差分；根和轮心的轨迹导数解析
给定。静止端点直接返回零，不能由跨阶段数值差分产生假加速度。
"""
    rotation = state['trans']['Base_link'][:3, :3]
    position = state['trans']['Base_link'][:3, 3]
    offset = rotation @ model.base_offset
    base_v = np.r_[velocity+np.cross(omega, offset), omega]
    base_a = np.r_[acceleration+np.cross(alpha, offset)
                   +np.cross(omega, np.cross(omega, offset)), alpha]
    if not (np.any(base_v) or np.any(base_a) or np.any(center_velocity)
            or np.any(center_acceleration)):
        return np.zeros(16), np.zeros(16), np.zeros(4), np.zeros(4), 0., 0.

    jac = _center_jacobians(state)
    v = np.r_[base_v, np.zeros(16)]
    for k in range(4):
        columns = slice(6+3*k, 9+3*k)
        v[columns] = np.linalg.solve(jac[k, :, columns],
            center_velocity[k]-jac[k, :, :6]@base_v)

    A, _, _ = model.contacts(state, normal)
    rows = np.array([tangent@A[3*k:3*k+3] for k in range(4)])
    for k in range(4):
        if abs(rows[k, 18+k]) < .02:
            raise ValueError('Rear entry wheel rolling direction nearly singular')
        v[18+k] = -(rows[k]@v)/rows[k, 18+k]

    # Model 的平移广义坐标在根 COM，旋转增量及角速度均为世界坐标。
    epsilon = 1.e-5
    displaced = []
    for sign in (-1., 1.):
        delta = sign*epsilon
        R = Rotation.from_rotvec(omega*delta).as_matrix()@rotation
        p = position+offset+base_v[:3]*delta-R@model.base_offset
        quat = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
        displaced.append(model.state(p, quat, q+v[6:]*delta))
    jac_dot = (_center_jacobians(displaced[1])-_center_jacobians(displaced[0]))/(2*epsilon)
    a = np.r_[base_a, np.zeros(16)]
    for k in range(4):
        columns = slice(6+3*k, 9+3*k)
        a[columns] = np.linalg.solve(jac[k, :, columns],
            center_acceleration[k]-jac[k, :, :6]@base_a-jac_dot[k]@v)
    A_dot = (model.contacts(displaced[1], normal)[0]
             -model.contacts(displaced[0], normal)[0])/(2*epsilon)
    for k in range(4):
        a[18+k] = -(rows[k]@a+tangent@A_dot[3*k:3*k+3]@v)/rows[k, 18+k]

    rolling_residual = float(abs(rows@v).max())
    center_residual = float(abs(np.einsum('kij,j->ki', jac, v)-center_velocity).max())
    # 与 entry_path 同契约：q 的轮相位为零占位，腿导数保留16列；
    # 真正的滚动轮速/加速度单独输出，不把占位轮相位当驱动位置命令。
    qd, qdd = v[6:].copy(), a[6:].copy()
    qd[12:] = qdd[12:] = 0.
    return qd, qdd, v[18:].copy(), a[18:].copy(), rolling_residual, center_residual


def build_rear_path(reference, slope_deg, rear_duration, lower_duration,
                    rear_distance=.20, dt=.02, final_centers='fixed'):
    """返回 entry_path 兼容字典和显式 endpoint_reference；仅 CPU 几何。

body_velocity/acceleration_plan 指根 link 世界线速度/加速度，角速度
也用世界坐标。只沿法向改变根位置；普通轮相位从零开始，真实轮速在
    wheel_rad_s。两阶段分别停稳，所有端点的一、二阶导数都为零。

    final_centers='fixed' 保留原固定世界轮心候选；'reference' 在下沉段
    将四轮轮心平滑恢复至来源姿态。后者全段使用来源轮相位常数，从而
    末端16D q精确恢复、没有末帧相位跳变；这项初始轮相位变化显式记录。
"""
    scalars = [slope_deg, rear_duration, lower_duration, rear_distance, dt]
    if not np.isfinite(scalars).all() or dt <= 0. or rear_distance < 0.:
        raise ValueError('Expected finite path parameters, positive dt and nonnegative rear distance')
    if final_centers not in ('fixed', 'reference'):
        raise ValueError('Final centers must be fixed or reference')
    for duration in (rear_duration, lower_duration):
        if duration < 2*dt or abs(round(duration/dt)*dt-duration) > 1.e-8:
            raise ValueError('Each rear entry stage needs >=2 steps on the action grid')
    source_q = np.asarray(reference['q'], dtype=float)
    quat = np.asarray(reference['quat'], dtype=float)
    height = float(reference['height_m'])
    if (source_q.shape != (16,) or quat.shape != (4,) or not np.isfinite(source_q).all()
            or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1.e-12
            or not np.isfinite(height) or not 0. < height <= DEFAULT_HEIGHT_M):
        raise ValueError('Expected finite16D source q, nonzero quaternion and lower positive height')

    model = Model()
    slope = np.deg2rad(slope_deg)
    normal = np.array([-np.sin(slope), 0., np.cos(slope)])
    tangent = np.array([np.cos(slope), 0., np.sin(slope)])
    R0 = Rotation.from_euler('y', -slope)
    R1 = Rotation.from_quat(quat[[1, 2, 3, 0]])
    rotvec = (R1*R0.inv()).as_rotvec()
    p0 = normal*DEFAULT_HEIGHT_M
    delta_p = normal*(height-DEFAULT_HEIGHT_M)
    q = DEFAULT_Q.copy()
    if final_centers == 'reference':
        q[12:] = source_q[12:]
    centers0 = _wheel_centers(model.state(p0, R0.as_quat()[[3, 0, 1, 2]], q))
    rear_shift = np.zeros((4, 3)); rear_shift[2:] = -rear_distance*tangent
    recovery_shift = np.zeros((4, 3))
    if final_centers == 'reference':
        source_state = model.state(normal*height, quat, source_q)
        recovery_shift = _wheel_centers(source_state)-centers0-rear_shift
    first_steps, second_steps = round(rear_duration/dt), round(lower_duration/dt)
    times = np.arange(first_steps+second_steps+1)*dt
    output = {key: [] for key in (
        'q', 'joint_velocity_rad_s', 'joint_acceleration_rad_s2', 'wheel_rad_s',
        'wheel_acceleration_rad_s2', 'body_position_plan_m', 'body_quaternion_plan',
        'body_velocity_plan_mps', 'body_acceleration_plan_mps2',
        'body_angular_velocity_plan_rad_s', 'body_angular_acceleration_plan_rad_s2',
        'wheel_center_plan_m', 'wheel_center_velocity_plan_mps', 'wheel_center_acceleration_plan_mps2')}
    ik_errors, rolling_errors, center_velocity_errors = [], [], []
    for i, time in enumerate(times):
        if i <= first_steps:
            fraction, rate, accel = _blend(time if i < first_steps else rear_duration, rear_duration)
            p, R = p0, R0
            velocity = acceleration = omega = alpha = np.zeros(3)
            centers = centers0+fraction*rear_shift
            center_velocity, center_acceleration = rate*rear_shift, accel*rear_shift
        else:
            local_time = time-rear_duration if i < len(times)-1 else lower_duration
            fraction, rate, accel = _blend(local_time, lower_duration)
            p, R = p0+fraction*delta_p, Rotation.from_rotvec(fraction*rotvec)*R0
            velocity, acceleration = rate*delta_p, accel*delta_p
            omega, alpha = rate*rotvec, accel*rotvec
            centers = centers0+rear_shift+fraction*recovery_shift
            center_velocity, center_acceleration = rate*recovery_shift, accel*recovery_shift
        target = (centers-p)@R.as_matrix()
        q = solve_foot_pose(model, q, target, np.zeros((4, 3)))
        quat_now = R.as_quat()[[3, 0, 1, 2]]
        if final_centers == 'reference' and i == len(times)-1:
            if np.max(abs(q[:12]-source_q[:12])) > 1.e-6:
                raise ValueError('Reference endpoint reached a different IK branch')
            if np.any(source_q[:12] < model.limits[:12, 0]) or np.any(source_q[:12] > model.limits[:12, 1]):
                raise ValueError('Reference endpoint exceeds original joint limits')
            # Only roundoff is removed: no last-frame branch or wheel-phase jump.
            q, quat_now = source_q.copy(), quat.copy()
        state = model.state(p, quat_now, q)
        error = float(abs(_wheel_centers(state)-centers).max())
        if error > 1.e-6:
            raise ValueError(f'Rear entry path unreachable within original joint limits at {time:.6f}s: {error:.6g}m')
        qd, qdd, wheel, wheel_a, rolling_error, velocity_error = _kinematic_rates(
            model, state, q, normal, tangent, velocity, omega, acceleration, alpha,
            center_velocity, center_acceleration)
        values = (q, qd, qdd, wheel, wheel_a, p, quat_now, velocity, acceleration,
                  omega, alpha, centers, center_velocity, center_acceleration)
        for key, value in zip(output, values):
            output[key].append(np.asarray(value).tolist())
        ik_errors.append(error); rolling_errors.append(rolling_error); center_velocity_errors.append(velocity_error)

    endpoint = deepcopy(reference)
    endpoint.update(q=list(output['q'][-1]), quat=list(output['body_quaternion_plan'][-1]), height_m=height)
    endpoint['posture_variant'] = dict(
        name='rear_extension_then_normal_lowering', rear_distance_m=float(rear_distance),
        source_q=source_q.tolist(), source_quat=quat.tolist(), source_height_m=height,
        source_posture_variant=deepcopy(reference.get('posture_variant')),
        physically_validated=False, requires_static_rebase=True,
        note='Inherited force/action/source fields describe the source posture, not a validated calibration of this new geometry.')
    if final_centers == 'reference':
        endpoint['posture_variant'].update(name='rear_extension_then_reference_recovery',
            endpoint_matches_source_geometry=True, requires_static_rebase=False,
            note='The source endpoint geometry and calibration are retained; this entry transient has not been physically validated.')
    normal_acceleration = np.array(output['body_acceleration_plan_mps2'])@normal
    gravity_normal = 9.81*np.cos(slope)
    # 五次插值加速度极值在 x=(3±sqrt(3))/6；同时报告采样值，防止
    # 50Hz漏掉峰值。这里只比较根link计划加速度与自由落体，不替代全身
    # COM、牵引力和腿惯性审查，也不由该比较宣布动力学可达。
    peak_normal_acceleration = (DEFAULT_HEIGHT_M-height)*(10.*np.sqrt(3.)/3.)/lower_duration**2
    freefall_exceeded = normal_acceleration < -gravity_normal
    output.update(time_s=times.tolist(), endpoint_reference=endpoint,
        maximum_ik_error_m=max(ik_errors), maximum_tangent_rolling_residual_mps=max(rolling_errors),
        maximum_center_velocity_residual_mps=max(center_velocity_errors),
        maximum_planned_leg_speed_rad_s=float(abs(np.array(output['joint_velocity_rad_s'])[:, :12]).max()),
        maximum_planned_leg_acceleration_rad_s2=float(abs(np.array(output['joint_acceleration_rad_s2'])[:, :12]).max()),
        planned_root_forward_m=0., crouch_first=False, start_progress=0., preserve_wheel_phase=final_centers=='reference',
        final_centers=final_centers,
        rear_duration_s=float(rear_duration), lower_duration_s=float(lower_duration),
        rear_distance_m=float(rear_distance), stage_boundary_index=first_steps,
        planned_normal_acceleration_diagnostic=dict(
            sampled_min_mps2=float(normal_acceleration.min()),
            sampled_max_mps2=float(normal_acceleration.max()),
            analytic_min_mps2=-float(peak_normal_acceleration),
            analytic_max_mps2=float(peak_normal_acceleration),
            gravity_normal_mps2=float(gravity_normal),
            faster_downward_than_freefall_frame_fraction=float(freefall_exceeded.mean()),
            lower_stage_faster_downward_than_freefall_frame_fraction=float(freefall_exceeded[first_steps:].mean()),
            note='Root-link kinematic diagnostic only; gravity comparison does not establish whole-body dynamic feasibility.'),
        note='Unvalidated kinematic candidate only. Root poses/velocities are never commanded to simulation. '
             'Wheel phases in q are placeholders; rolling rates are separate. Source force/action calibration requires rebasing.')
    if final_centers == 'reference':
        output['note'] = ('Unvalidated kinematic candidate only. Root poses/velocities are never commanded to simulation. '
            'The source wheel phases are constant from reset; rolling rates are separate. '
            'The endpoint restores source q/quat/height exactly; entry dynamics remain unvalidated.')
    return output
