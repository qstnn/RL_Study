"""普通初态腿轮协调原型；只输出原16轴动作，不修改物理状态或外力。

腿参考直接来自进入路径，支撑偏置由当前构型/当前牵引载荷重算。
轮速基准由实测腿速和机身角速度求滚动运动学，再叠加原轮伺服力矩补偿。
静力/逆动力学分配可解不代表真实接触成立，需实际完整教师验收。
"""
import json
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial.transform import Rotation
from model import Model
from pose_search import basis
from qp import friction_rows, servo_bounds
from entry_path import DEFAULT_Q
from partial_support_control import SCALE
from coordinated_capture import CaptureHandoff


def root_link_acceleration_to_com(acceleration, alpha, omega_world, offset_world):
    """将根link原点加速度转换为根刚体COM加速度，匹配M的浮基坐标。"""
    return (np.asarray(acceleration)+np.cross(alpha, offset_world)
        + np.cross(omega_world, np.cross(omega_world, offset_world)))


def load_weighted_rolling_error(wheel_velocity, rolling_velocity, measured_normal):
    """按正的实测轮载平均绝对滚动误差，仅用于内部准备判据，不改行为验收。

总正轮载为零时返回0，调用方仍需独立满足至少有实际承载轮；零值不表示已接地。
"""
    weight = np.maximum(np.asarray(measured_normal, dtype=float), 0.)
    total = float(weight.sum())
    error = abs(np.asarray(wheel_velocity)-np.asarray(rolling_velocity))*.0875
    return float(weight@error/total) if total > 0. else 0.


def limit_motion_leg_target(desired, previous, q, qd, position_limits, rate):
    """速率窗为软约束，原位置/瞬时96Nm区间为硬约束；冲突必须显式记录。

若位置与力矩自身无可行交集，停止试验而不是返回越矩动作。
"""
    desired, previous, q, qd = [np.asarray(x, dtype=float) for x in (desired, previous, q, qd)]
    tau_lo, tau_hi = servo_bounds(q, qd, position_limits)
    lower = q[:12]+(tau_lo[:12]+3.*qd[:12])/80.
    upper = q[:12]+(tau_hi[:12]+3.*qd[:12])/80.
    if np.any(lower > upper+1.e-12):
        raise RuntimeError('Ordinary motion leg target has no original position/96Nm intersection')
    radius = .02*rate
    conflict = np.maximum(lower, previous-radius) > np.minimum(upper, previous+radius)
    rate_limited = np.clip(desired, previous-radius, previous+radius)
    final = np.clip(rate_limited, lower, upper)
    actual_tau = 80.*(final-q[:12])-3.*qd[:12]
    if np.max(abs(actual_tau)) > 96.+1.e-7:
        raise RuntimeError('Ordinary motion limiter violated original96Nm request bound')
    return final, dict(rate_adjustment=rate_limited-desired, physical_adjustment=final-rate_limited,
        conflict=conflict.astype(float), rate_window_excess=np.maximum(abs(final-previous)-radius, 0.),
        actual_torque=actual_tau)


def rolling_reference(model, state, normal, tangent, root_com_velocity, omega_world,
                      joint_velocity, command_velocity):
    """指定沿坡根COM速度，其余实际速度保留；返回四轮纯运动学基准及残差。

叠加轮力矩/2之后的最终目标不再声称严格无滑动，此处仅核查基准。
"""
    A, _, _ = model.contacts(state, normal)
    linear = np.asarray(root_com_velocity, dtype=float).copy()
    linear += tangent * (command_velocity - linear @ tangent)
    v = np.r_[linear, omega_world, joint_velocity].astype(float)
    v[18:] = 0.
    for k in range(4):
        row = tangent @ A[3*k:3*k+3]
        if abs(row[18+k]) < .02:
            raise ValueError('Wheel rolling direction nearly singular')
        v[18+k] = -(row @ v) / row[18+k]
    return v[18:].copy(), (A @ v).reshape(4, 3) @ tangent


class DownhillRollingEntry:
    """教师准备参考：1滚动、2制动、3零速稳定、4已放行；不写环境命令或物理状态。

低姿态连续.06s才从滚动转制动。理由码0无转移、1低姿态成立、2参考归零、3原ready放行。
速度参考为本步起点值；加速度前馈积分至下一步，饱和后不再继续加速。
"""
    def __init__(self):
        self.phase = 1; self.velocity_reference = 0.; self.low_pose_ready_s = 0.; self.phase_time_s = 0.

    def step(self, low_pose_ready, prepared=False):
        reason = 0
        if prepared and self.phase != 4:
            self.phase = 4; self.phase_time_s = 0.; reason = 3
        elif self.phase == 1:
            self.low_pose_ready_s = self.low_pose_ready_s+.02 if low_pose_ready else 0.
            if self.low_pose_ready_s >= .06-1.e-12:
                self.phase = 2; self.phase_time_s = 0.; reason = 1
        if self.phase == 2 and self.velocity_reference <= 1.e-12:
            self.velocity_reference = 0.; self.phase = 3; self.phase_time_s = 0.; reason = 2
        velocity = self.velocity_reference
        acceleration = (min(2., max(.8-velocity, 0.)/.02) if self.phase == 1
            else -min(1., velocity/.02) if self.phase == 2 else 0.)
        self.velocity_reference = float(np.clip(velocity+acceleration*.02, 0., .8))
        report = dict(phase=self.phase, phase_time_s=self.phase_time_s, transition_reason=reason,
            velocity_reference_mps=velocity, next_velocity_reference_mps=self.velocity_reference,
            acceleration_feedforward_mps2=acceleration, low_pose_ready_s=self.low_pose_ready_s,
            velocity_cap_active=float(self.phase == 1 and velocity >= .8-1.e-12),
            settling_allowed=float(self.phase >= 3), active=float(self.phase < 4))
        self.phase_time_s += .02
        return report


class OrdinaryEntryController:
    def __init__(self, plan, reference, slope, kp=80., kd=3., continue_control=False, motion_target_rate=0.,
                 hold_hip_preload=False, wheel_torque_control=False, measured_traction_mu=0.,
                 motion_velocity_gain=1.5, motion_acceleration_limit=.5, traction_budget_contact_only=False,
                 downhill_rolling_entry=False, friction_mu=3., downhill_settled_entry=False):
        if kp != 80. or kd != 3.:
            raise ValueError('Ordinary entry requires original 80/3 leg servos')
        self.model = Model(); self.B = basis(slope)
        if not np.isfinite(friction_mu) or friction_mu <= 0.:
            raise ValueError('friction_mu must be finite and positive')
        self.friction_mu = float(friction_mu)
        self.plan = plan; self.reference = reference
        self.qpath = np.asarray(plan['q']); self.vpath = np.asarray(plan['joint_velocity_rad_s'])
        self.ppath = np.asarray(plan['body_position_plan_m'])
        def derivative(key, shape, fallback):
            if key not in plan:
                return fallback()
            value = np.asarray(plan[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Explicit plan derivative {key} must be finite with shape{shape}')
            return value.copy()
        self.apath = derivative('joint_acceleration_rad_s2', self.qpath.shape,
            lambda: np.gradient(self.vpath, .02, axis=0))
        self.pv = derivative('body_velocity_plan_mps', self.ppath.shape,
            lambda: np.gradient(self.ppath, .02, axis=0))
        self.pa = derivative('body_acceleration_plan_mps2', self.ppath.shape,
            lambda: np.gradient(self.pv, .02, axis=0))
        self.explicit_derivatives = np.array([float(key in plan) for key in
            ('joint_acceleration_rad_s2', 'body_velocity_plan_mps', 'body_acceleration_plan_mps2')])
        self.rpath = np.asarray(plan['body_quaternion_plan'])
        self.duration = float(plan['time_s'][-1])
        self.handoff = CaptureHandoff(); self.finished = False
        self.previous_force = None; self.lateral = None
        self.continue_control = bool(continue_control); self.ready_time = 0.
        if not np.isfinite(motion_target_rate) or motion_target_rate < 0. or (motion_target_rate and not self.continue_control):
            raise ValueError('Motion target rate must be nonnegative and requires continuous control')
        self.motion_target_rate = float(motion_target_rate); self.previous_output_leg_target = None
        self.hold_hip_preload = bool(hold_hip_preload); self.wheel_torque_control = bool(wheel_torque_control)
        if (not np.isfinite(measured_traction_mu) or measured_traction_mu < 0.
                or (measured_traction_mu and not (self.continue_control and self.wheel_torque_control))):
            raise ValueError('Measured traction mu must be nonnegative and requires continuous wheel torque control')
        self.measured_traction_mu = float(measured_traction_mu)
        self.traction_budget_contact_only = bool(traction_budget_contact_only)
        self.downhill_rolling_entry = bool(downhill_rolling_entry)
        self.downhill_settled_entry = bool(downhill_settled_entry)
        if self.downhill_settled_entry and not self.downhill_rolling_entry:
            raise ValueError('Downhill settled entry requires downhill rolling entry')
        if self.downhill_rolling_entry and (not np.isfinite(slope) or slope >= 0.
                or not self.continue_control or not self.wheel_torque_control):
            raise ValueError('Downhill rolling entry requires negative slope and continuous wheel torque control')
        self.downhill_entry = DownhillRollingEntry() if self.downhill_rolling_entry else None
        for name,value in (('motion_velocity_gain',motion_velocity_gain),
                ('motion_acceleration_limit',motion_acceleration_limit)):
            if not np.isfinite(value) or value < 0.:
                raise ValueError(f'{name} must be finite and nonnegative')
        self.motion_velocity_gain = float(motion_velocity_gain)
        self.motion_acceleration_limit = float(motion_acceleration_limit)
        self.hip_ids = np.array([0, 3, 6, 9]); self.source_hip_preload = np.zeros(4)
        if self.hold_hip_preload:
            if not self.continue_control:
                raise ValueError('Holding hip preload requires continuous control')
            source_action = np.asarray(reference.get('leg_action', []), dtype=float)
            if source_action.shape != (12,) or not np.isfinite(source_action).all():
                raise ValueError('Holding hip preload requires finite12D source leg action')
            source_target = DEFAULT_Q[:12]+source_action*SCALE
            self.source_hip_preload = 80.*(source_target-np.asarray(reference['q'])[:12])[self.hip_ids]

    def advance_preparation(self, ready, time):
        """连续控制模式只放行命令时钟；之后仍输出自身动作，不混合旧教师。"""
        if self.continue_control:
            self.ready_time = self.ready_time+.02 if ready else 0.
            self.finished = self.finished or self.ready_time >= .4
            return 0.
        blend = self.handoff.advance(ready, time)
        self.ready_time = self.handoff.ready_s; self.finished = self.handoff.finished
        return blend

    def reference_kinematics(self, time):
        """路径结束后保持端点，绝不重复末项数值差分速度/加速度。"""
        idx = min(round(time/.02), len(self.qpath)-1)
        stopped = time >= self.duration
        return (idx, self.qpath[idx], np.zeros(16) if stopped else self.vpath[idx],
            np.zeros(16) if stopped else self.apath[idx],
            np.zeros(3) if stopped else self.pv[idx], np.zeros(3) if stopped else self.pa[idx])

    def step(self, time, q, qd, quat, omega_world, root_pos, root_com_velocity,
             external_world, hitch_local, measured_normal, nominal_action,
             command_velocity=0., rolling_joint_velocity=None, native_bias22=None):
        """root_pos为相对env_origin的实际根位置；其余向量采用world/action顺序。"""
        q, qd, quat, omega_world, root_pos, root_com_velocity, external_world, hitch_local, measured_normal = [
            np.asarray(x, dtype=float) for x in (q, qd, quat, omega_world, root_pos,
                root_com_velocity, external_world, hitch_local, measured_normal)]
        if not all(np.isfinite(x).all() for x in (q, qd, quat, omega_world, root_pos,
                root_com_velocity, external_world, hitch_local, measured_normal)):
            raise ValueError('Nonfinite ordinary entry state')
        prepared_at_start = self.finished
        n, t = self.B[:, 2], self.B[:, 0]
        if self.lateral is None:
            self.lateral = float(root_pos@self.B[:, 1])
        idx, desired, desired_v, desired_a, planned_root_v, planned_root_a = self.reference_kinematics(time)
        state = self.model.state(root_pos, quat, q)
        R = state['trans']['Base_link'][:3, :3]
        com_offset_world = R@self.model.base_offset
        root_link_velocity = root_com_velocity-np.cross(omega_world, com_offset_world)
        goal_R = Rotation.from_quat(self.rpath[idx][[1, 2, 3, 0]]).as_matrix()
        error = Rotation.from_matrix(goal_R @ R.T).as_rotvec()
        downhill = dict(phase=0, phase_time_s=0., transition_reason=0, velocity_reference_mps=0.,
            next_velocity_reference_mps=0., acceleration_feedforward_mps2=0., low_pose_ready_s=0.,
            velocity_cap_active=0., settling_allowed=1., active=0.)
        low_pose_checks = np.zeros(4); low_pose_errors = np.zeros(3)
        settled_entry_errors = np.array([abs(omega_world).max(), abs(qd[:12]-desired_v[:12]).max()])
        settled_entry_thresholds = np.array([.3, .5])
        settled_entry_checks = (settled_entry_errors < settled_entry_thresholds).astype(float)
        entry_brake_eligible = False
        route_velocity_reference = command_velocity
        if self.downhill_entry is not None:
            endpoint_R = Rotation.from_quat(self.rpath[-1][[1, 2, 3, 0]]).as_matrix()
            endpoint_error = Rotation.from_matrix(endpoint_R @ R.T).as_rotvec()
            low_pose_errors = np.array([abs(root_pos@n-self.reference['height_m']),
                abs(q[:12]-np.asarray(self.reference['q'])[:12]).max(), abs(endpoint_error).max()])
            low_pose_checks = np.r_[low_pose_errors < [.02, .18, .12], (measured_normal > 5.).any()].astype(float)
            entry_brake_eligible = bool(low_pose_checks.all() and
                (not self.downhill_settled_entry or settled_entry_checks.all()))
            downhill = self.downhill_entry.step(entry_brake_eligible, prepared_at_start)
            if downhill['active']:
                route_velocity_reference = downhill['velocity_reference_mps']
        # 可选前周期平均只用于轮滚动参考；实际PD、原力矩/位置界仍用瞬时qd。
        rolling_qd = qd if rolling_joint_velocity is None else np.asarray(rolling_joint_velocity, dtype=float)
        if rolling_qd.shape != (16,) or not np.isfinite(rolling_qd).all():
            raise ValueError('Rolling joint velocity must be finite16D in action order')
        rolling, residual = rolling_reference(self.model, state, n, t,
            root_com_velocity, omega_world, rolling_qd, route_velocity_reference)
        A, points, _ = self.model.contacts(state, n)
        M, gravity = self.model.dynamics(state)
        dynamics_bias = gravity if native_bias22 is None else np.asarray(native_bias22, dtype=float)
        if dynamics_bias.shape != (22,) or not np.isfinite(dynamics_bias).all():
            raise ValueError('Native dynamics bias must be finite22D in world/action order')
        hitch = root_pos + R @ hitch_local
        external = state['jac']('Base_link', hitch)[:3].T @ external_world
        # 参考加速度只用于当前构型的前馈；实际PD仍提供位置恢复，物理增益不变。
        acc = n*np.clip(planned_root_a@n + 20*((self.ppath[idx]-root_pos)@n)
            + 8*((planned_root_v-root_link_velocity)@n), -4., 4.)
        # 新速度反馈仅在本步开始已完成进入后启用；准备阶段保留原1.5/.5。
        velocity_gain = self.motion_velocity_gain if prepared_at_start else 1.5
        acceleration_limit = self.motion_acceleration_limit if prepared_at_start else .5
        route_acceleration_feedforward = 0.
        acceleration_lower, acceleration_upper = -acceleration_limit, acceleration_limit
        if downhill['active']:
            velocity_gain = 6.; route_acceleration_feedforward = downhill['acceleration_feedforward_mps2']
            acceleration_limit = 3.  # 旧标量字段记录最大绝对界；精确非对称界另存下方bounds。
            acceleration_lower, acceleration_upper = -2., 3.
        route_acceleration_feedback = velocity_gain*(route_velocity_reference-root_link_velocity@t)
        route_acceleration = np.clip(route_acceleration_feedforward+route_acceleration_feedback,
            acceleration_lower, acceleration_upper)
        acc += t*route_acceleration
        acc += self.B[:, 1]*np.clip(4*(self.lateral-root_pos@self.B[:, 1])-3*(root_link_velocity@self.B[:, 1]), -2., 2.)
        alpha = np.clip(20*error-5*omega_world, -20., 20.)
        com_acc = root_link_acceleration_to_com(acc, alpha, omega_world, com_offset_world)
        desired_acc = np.r_[com_acc, alpha, np.clip(desired_a[:12], -100., 100.), np.zeros(4)]
        bias = M@desired_acc + dynamics_bias-external
        wheel_pd = np.zeros(4) if self.wheel_torque_control else 2*(rolling-qd[12:])
        pd = np.r_[80*(desired[:12]-q[:12])+3*(desired_v[:12]-qd[:12]), wheel_pd]
        torque_offset = bias[6:]+pd
        F, fb = friction_rows(self.B, self.friction_mu, 0.)
        gaps = points@n
        # 几何邻近或上一周期有支撑即允许分配；允许全部/部分轮离地，无四轮硬门。
        mask = (gaps < .006) | (measured_normal > 1.)
        N = np.kron(np.eye(4), n[None, :])
        tl, tu = servo_bounds(q, qd, self.model.limits)
        # 仅已完成进入后的连续纯轮力矩模式使用上一周期实测轮载；不改变准备动作或物理μ。
        # 默认预算限制总瞬时轮请求；可选仅限制接触项，将总τ区间中心移至动力学bias。
        # 当前圆接触几何下接触项为-Jwheel.T F；native CoP法向偏心矩未来需单独分离。
        # 必须在分配前相交，不能事后裁剪轮动作；原±48Nm及速度目标界始终保留。
        budget_active = bool(self.measured_traction_mu and self.continue_control
            and self.wheel_torque_control and prepared_at_start)
        allowed_tau = np.full(4, 48.)
        budget_center = np.zeros(4)
        if budget_active:
            if measured_normal.shape != (4,):
                raise ValueError('Measured traction budget requires four measured normal loads')
            allowed_tau = self.measured_traction_mu*np.maximum(measured_normal, 0.)*.0875
            if self.traction_budget_contact_only:
                budget_center = bias[18:].copy()
            original_wheel_lower, original_wheel_upper = tl[12:].copy(), tu[12:].copy()
            tl[12:] = np.maximum(tl[12:], budget_center-allowed_tau)
            tu[12:] = np.minimum(tu[12:], budget_center+allowed_tau)
            if np.any(tl[12:] > tu[12:]+1.e-12):
                # 只扩充失败证据；原停止条件和控制界不变。控制器没有协议phase输入，
                # 因此只标识进入/命令状态，不能将零命令擅自称为停车保持阶段。
                diagnostic = dict(time_s=float(time),
                    controller_phase=('prepared_nonzero_command' if command_velocity != 0.
                        else 'prepared_zero_command'),
                    prepared_at_start=bool(prepared_at_start), command_velocity_mps=float(command_velocity),
                    wheel_order=['FR', 'FL', 'RR', 'RL'], actual_wheel_qd_rad_s=qd[12:].tolist(),
                    measured_normal_N=measured_normal.tolist(), measured_traction_mu=self.measured_traction_mu,
                    allowed_wheel_tau_Nm=allowed_tau.tolist(),
                    traction_budget_contact_only=self.traction_budget_contact_only,
                    budget_center_bias_Nm=budget_center.tolist(),
                    original_feasible_wheel_lower_Nm=original_wheel_lower.tolist(),
                    original_feasible_wheel_upper_Nm=original_wheel_upper.tolist(),
                    intersection_wheel_lower_Nm=tl[12:].tolist(), intersection_wheel_upper_Nm=tu[12:].tolist())
                raise RuntimeError('Measured traction budget has no original wheel speed/48Nm intersection: '
                    +json.dumps(diagnostic, sort_keys=True))
        G = np.vstack((F, N, -A.T[6:], A.T[6:]))
        bound = np.r_[fb, np.where(mask, 350., 0.), tu-torque_offset, torque_offset-tl]
        scale = np.maximum(np.linalg.norm(G, axis=1), 1.)
        Gn, bn = G/scale[:, None], bound/scale
        E, rhs = A.T[:6], bias[:6]
        seed = linprog(np.zeros(12), A_ub=Gn, b_ub=bn, bounds=[(None, None)]*12, method='highs')
        valid = bool(seed.success)
        success = False
        if valid:
            weights = np.array([1., 1., 1., 10., 10., 10.])
            H = E.T@(weights[:, None]*E)+np.eye(12)*.002
            linear = -E.T@(weights*rhs)
            x0 = seed.x
            if self.previous_force is not None and np.max(G@self.previous_force-bound) < 1.e-6:
                x0 = self.previous_force
            result = minimize(lambda f: .5*f@H@f+linear@f, x0,
                jac=lambda f: H@f+linear, method='SLSQP',
                constraints=[dict(type='ineq', fun=lambda f: bn-Gn@f, jac=lambda f: -Gn)],
                options=dict(maxiter=100, ftol=1.e-8))
            force = result.x if np.isfinite(result.x).all() else seed.x
            valid = bool(np.max(G@force-bound) < 1.e-4)
            success = bool(result.success and valid)
        if not valid:
            # 连续控制不能悄然回退旧教师；仅请求当前q/轮速，失败必须如实保留。
            # 这是无求解时的确定性回退，不是对真实下一周期稳定性的保证。
            force = np.zeros(12)
            if self.continue_control:
                target = np.clip(q[:12], self.model.limits[:12, 0], self.model.limits[:12, 1])
                action = np.r_[(target-DEFAULT_Q[:12])/SCALE, np.clip(qd[12:], -50., 50.)/5.]
            else:
                action = np.asarray(nominal_action, dtype=float).copy()
                target = DEFAULT_Q[:12]+action[:12]*SCALE
            tau = np.zeros(16)
        else:
            self.previous_force = force.copy()
            tau = bias[6:]-A.T[6:]@force
            target = desired[:12]+(tau[:12]+3*desired_v[:12])/80.
            wheel_base = qd[12:] if self.wheel_torque_control else rolling
            action = np.r_[(target-DEFAULT_Q[:12])/SCALE, (wheel_base+tau[12:]/2.)/5.]
        supported = measured_normal > 5.
        rolling_error = load_weighted_rolling_error(qd[12:], rolling, measured_normal)
        ready = (success and time >= self.duration and abs(root_pos@n-self.reference['height_m']) < .03
            and abs(q[:12]-np.asarray(self.reference['q'])[:12]).max() < .18
            and abs(error).max() < .08 and abs(omega_world).max() < .3
            and abs(root_com_velocity@t-command_velocity) < .08
            and supported.any() and rolling_error < .1 and bool(downhill['settling_allowed']))
        blend = self.advance_preparation(ready, time)
        action = (1.-blend)*action+blend*np.asarray(nominal_action)
        final_target = DEFAULT_Q[:12]+action[:12]*SCALE
        hip_active = bool(self.hold_hip_preload and prepared_at_start and valid)
        hip_clip = np.zeros(4)
        if hip_active:
            hips = self.hip_ids
            candidate = desired[hips]+(self.source_hip_preload+3.*desired_v[hips])/80.
            lower = q[hips]+(tl[hips]+3.*qd[hips])/80.
            upper = q[hips]+(tu[hips]+3.*qd[hips])/80.
            if np.any(lower > upper+1.e-12):
                raise RuntimeError('Hip preload target has no original position/96Nm intersection')
            selected = np.clip(candidate, lower, upper)
            hip_clip = 80.*(selected-candidate)
            final_target[hips] = selected
            action[hips] = (selected-DEFAULT_Q[hips])/SCALE[hips]
        rate_active = bool(self.continue_control and prepared_at_start and self.motion_target_rate)
        limiter = dict(rate_adjustment=np.zeros(12), physical_adjustment=np.zeros(12),
            conflict=np.zeros(12), rate_window_excess=np.zeros(12),
            actual_torque=80.*(final_target-q[:12])-3.*qd[:12])
        if rate_active:
            previous = q[:12] if self.previous_output_leg_target is None else self.previous_output_leg_target
            final_target, limiter = limit_motion_leg_target(final_target, previous, q, qd,
                self.model.limits, self.motion_target_rate)
            action[:12] = (final_target-DEFAULT_Q[:12])/SCALE
        self.previous_output_leg_target = final_target.copy()
        final_wheel_request = 2.*(action[12:]*5.-qd[12:])
        if budget_active and (np.any(final_wheel_request < tl[12:]-1.e-4)
                or np.any(final_wheel_request > tu[12:]+1.e-4)):
            raise RuntimeError('Ordinary wheel request violated measured traction/servo intersection')
        return action, dict(ordinary_entry_active=float(self.continue_control or not self.finished), ordinary_entry_success=float(success),
            ordinary_entry_path_time_s=min(time, self.duration), ordinary_entry_ready_s=self.ready_time,
            ordinary_entry_prepared=float(self.finished), ordinary_entry_continue_control=float(self.continue_control),
            ordinary_entry_rolling_load_weighted_error_mps=rolling_error,
            ordinary_entry_positive_support_N=float(np.maximum(measured_normal, 0.).sum()),
            ordinary_entry_current_state_ready=float(ready),
            ordinary_entry_blend=blend, ordinary_entry_target_q=target,
            ordinary_entry_force_w_N=force.reshape(4, 3), ordinary_entry_wrench_residual=E@force-rhs,
            ordinary_entry_gap_m=gaps, ordinary_entry_mask=mask.astype(float),
            ordinary_entry_tau_feedforward_Nm=tau, ordinary_entry_rolling_rad_s=rolling,
            ordinary_entry_rolling_joint_velocity=rolling_qd.copy(),
            ordinary_entry_rolling_velocity_override=float(rolling_joint_velocity is not None),
            ordinary_entry_rolling_residual_mps=residual, ordinary_entry_attitude_error_rad=error,
            ordinary_entry_root_link_velocity_mps=root_link_velocity,
            ordinary_entry_motion_feedback_active=float(prepared_at_start),
            ordinary_entry_motion_velocity_gain=velocity_gain,
            ordinary_entry_motion_acceleration_limit_mps2=acceleration_limit,
            ordinary_entry_route_velocity_reference_mps=route_velocity_reference,
            ordinary_entry_route_acceleration_feedforward_mps2=route_acceleration_feedforward,
            ordinary_entry_route_acceleration_feedback_mps2=route_acceleration_feedback,
            ordinary_entry_route_acceleration_requested_mps2=route_acceleration,
            ordinary_entry_route_acceleration_bounds_mps2=np.array([acceleration_lower, acceleration_upper]),
            ordinary_entry_downhill_rolling_entry=float(self.downhill_rolling_entry),
            ordinary_entry_downhill_settled_entry=float(self.downhill_settled_entry),
            ordinary_entry_downhill_settled_errors_rad_s=settled_entry_errors,
            ordinary_entry_downhill_settled_thresholds_rad_s=settled_entry_thresholds,
            ordinary_entry_downhill_settled_checks=settled_entry_checks,
            ordinary_entry_downhill_brake_eligible=float(entry_brake_eligible),
            ordinary_entry_downhill_phase=float(downhill['phase']),
            ordinary_entry_downhill_phase_time_s=downhill['phase_time_s'],
            ordinary_entry_downhill_transition_reason=float(downhill['transition_reason']),
            ordinary_entry_downhill_reference_velocity_mps=downhill['velocity_reference_mps'],
            ordinary_entry_downhill_next_reference_velocity_mps=downhill['next_velocity_reference_mps'],
            ordinary_entry_downhill_acceleration_feedforward_mps2=downhill['acceleration_feedforward_mps2'],
            ordinary_entry_downhill_low_pose_ready_s=downhill['low_pose_ready_s'],
            ordinary_entry_downhill_velocity_cap_active=downhill['velocity_cap_active'],
            ordinary_entry_downhill_settling_allowed=downhill['settling_allowed'],
            ordinary_entry_downhill_active=downhill['active'],
            ordinary_entry_downhill_low_pose_checks=low_pose_checks,
            ordinary_entry_downhill_low_pose_errors=low_pose_errors,
            ordinary_entry_root_link_acceleration_mps2=acc, ordinary_entry_root_com_acceleration_mps2=com_acc,
            ordinary_entry_reference_joint_velocity=desired_v, ordinary_entry_reference_joint_acceleration=desired_a,
            ordinary_entry_reference_root_velocity=planned_root_v, ordinary_entry_reference_root_acceleration=planned_root_a,
            ordinary_entry_explicit_plan_derivatives=self.explicit_derivatives,
            ordinary_entry_native_bias_override=float(native_bias22 is not None), ordinary_entry_dynamics_bias=dynamics_bias,
            ordinary_entry_final_target_q=final_target, ordinary_entry_motion_target_rate_active=float(rate_active),
            ordinary_entry_motion_target_rate_rad_s=self.motion_target_rate,
            ordinary_entry_motion_rate_adjustment_rad=limiter['rate_adjustment'],
            ordinary_entry_motion_physical_adjustment_rad=limiter['physical_adjustment'],
            ordinary_entry_motion_rate_conflict=limiter['conflict'],
            ordinary_entry_motion_rate_excess_rad=limiter['rate_window_excess'],
            ordinary_entry_final_leg_request_Nm=limiter['actual_torque'],
            ordinary_entry_hip_preload_active=float(hip_active),
            ordinary_entry_source_hip_preload_Nm=self.source_hip_preload,
            ordinary_entry_qp_hip_preload_Nm=tau[self.hip_ids],
            ordinary_entry_selected_hip_preload_Nm=80.*(final_target[self.hip_ids]-desired[self.hip_ids])-3.*desired_v[self.hip_ids],
            ordinary_entry_hip_preload_clipping_Nm=hip_clip,
            ordinary_entry_wheel_torque_control=float(self.wheel_torque_control),
            ordinary_entry_friction_mu=self.friction_mu,
            ordinary_entry_measured_traction_mu=self.measured_traction_mu,
            ordinary_entry_traction_budget_active=float(budget_active),
            ordinary_entry_traction_budget_contact_only=float(self.traction_budget_contact_only),
            ordinary_entry_traction_budget_contact_only_active=float(budget_active and self.traction_budget_contact_only),
            ordinary_entry_traction_budget_center_Nm=budget_center,
            ordinary_entry_wheel_inertial_bias_Nm=bias[18:].copy(),
            ordinary_entry_contact_wheel_request_Nm=final_wheel_request-bias[18:],
            ordinary_entry_measured_normal_N=measured_normal.copy(),
            ordinary_entry_allowed_wheel_tau_Nm=allowed_tau,
            ordinary_entry_wheel_torque_lower_Nm=tl[12:].copy(),
            ordinary_entry_wheel_torque_upper_Nm=tu[12:].copy(),
            ordinary_entry_final_wheel_request_Nm=final_wheel_request,
            ordinary_entry_final_wheel_target_rad_s=action[12:]*5.)
