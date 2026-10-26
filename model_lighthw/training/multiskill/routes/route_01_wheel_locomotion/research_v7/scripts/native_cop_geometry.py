"""隔离的原生CoP几何接口；尚未接入任何在线控制器。

输入上一20个1kHz实测样本，用去除轮自转的轮架坐标平均接触力臂，
再映射到当前root-relative world模型。native_valid只表示历史几何可用；
force_allowed是显式latest_point策略的结果，不是现有controller mask。
无接触A块为零，不回退理想圆点认定支撑。零A的滚动残差不代表成功。
"""
import numpy as np
from scipy.spatial.transform import Rotation

from model import pose_matrix
from research_common import LEGS


def _carrier_rotations(model, state):
    """wheel joint的父坐标乘固定origin；不含轮自身q，避免搬运材料点。"""
    result = []
    for leg in LEGS:
        joint = model.joints[leg+'_wheel_joint']
        parent = joint.find('parent').get('link')
        result.append(state['trans'][parent][:3, :3]
                      @pose_matrix(joint.find('origin'))[:3, :3])
    return np.asarray(result)


def _centers(state):
    return np.array([state['trans'][leg+'_wheel_link'][:3, 3] for leg in LEGS])


def build_native_cop_geometry(model, root_position, root_quaternion, joint_position,
                              history, normal, tangent, *, current_time_s,
                              env_origin_w, force_policy):
    """构造同一个12x22 A供将来的力分配与滚动计算使用。

history使用physics trace同名字段，另需time_s=(20,)，间隔严格1ms。
root_position为已减env_origin的当前根link位置，其他位置历史为world。
当前时间、root、quat、q必须与最后物理样本对齐，错配会报错。

force_policy必须显式传'latest_point'：有加权历史，但最新count=0时仍
保留native_valid，force_allowed=False。这个候选模式没有修改当前控制器
的接触许可逻辑，未来接入后必须独立验证。没有全四轮同时支撑门。
"""
    if force_policy != 'latest_point':
        raise ValueError('Explicit helper force_policy must be latest_point')
    root_position, root_quaternion, joint_position, normal, tangent, origin = [
        np.asarray(x, dtype=float) for x in
        (root_position, root_quaternion, joint_position, normal, tangent, env_origin_w)]
    shapes = ((3,), (4,), (16,), (3,), (3,), (3,))
    for value, shape in zip((root_position, root_quaternion, joint_position, normal, tangent, origin), shapes):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Expected finite current state and world basis')
    if (abs(np.linalg.norm(normal)-1.) > 1.e-7 or abs(np.linalg.norm(tangent)-1.) > 1.e-7
            or abs(normal@tangent) > 1.e-7):
        raise ValueError('Expected orthonormal normal and tangent')
    required = dict(time_s=(20,), body_pos_w=(20, 3), body_quat_w=(20, 4),
        joint_pos=(20, 16), wheel_pos_w=(20, 4, 3), native_ground_normal_w=(20, 4, 3),
        native_contact_count=(20, 4), native_contact_min_separation_m=(20, 4),
        native_contact_cop_w=(20, 4, 3))
    h = {key: np.asarray(history[key], dtype=float) for key in required}
    for key, shape in required.items():
        if h[key].shape != shape or not np.isfinite(h[key]).all():
            raise ValueError(f'Expected finite20-sample native history: {key}')
    if (not np.isfinite(current_time_s) or abs(h['time_s'][-1]-current_time_s) > 1.e-8
            or np.max(abs(np.diff(h['time_s'])-.001)) > 1.e-8):
        raise ValueError('Native contact history time is not aligned to current state')
    if np.any(h['native_contact_count'] < 0):
        raise ValueError('Native contact count cannot be negative')
    if np.max(abs(root_position-(h['body_pos_w'][-1]-origin))) > 2.e-4:
        raise ValueError('Native contact history root position is not aligned')
    current_R = Rotation.from_quat(root_quaternion[[1, 2, 3, 0]])
    last_R = Rotation.from_quat(h['body_quat_w'][-1, [1, 2, 3, 0]])
    if (current_R*last_R.inv()).magnitude() > 2.e-4:
        raise ValueError('Native contact history root orientation is not aligned')
    dq = joint_position-h['joint_pos'][-1]
    dq[12:] = (dq[12:]+np.pi)%(2*np.pi)-np.pi
    if np.max(abs(dq)) > 2.e-4:
        raise ValueError('Native contact history joints are not aligned')

    carrier_history = []; maximum_center_fk_error = 0.
    for i in range(20):
        state = model.state(h['body_pos_w'][i]-origin, h['body_quat_w'][i], h['joint_pos'][i])
        maximum_center_fk_error = max(maximum_center_fk_error,
            float(abs(_centers(state)-(h['wheel_pos_w'][i]-origin)).max()))
        carrier_history.append(_carrier_rotations(model, state))
    if maximum_center_fk_error > 2.e-4:
        raise ValueError('Native wheel centers do not match the actual URDF/root/q history')
    carrier_history = np.asarray(carrier_history)
    normal_load = np.maximum(h['native_ground_normal_w']@normal, 0.)
    weight = normal_load*(h['native_contact_count'] > 0)
    lever_world_history = h['native_contact_cop_w']-h['wheel_pos_w']
    # 每个样本先去掉本样本轮心，避免运动时平均world点产生滞后力臂。
    lever_carrier_history = np.einsum('hkji,hkj->hki', carrier_history, lever_world_history)
    total = weight.sum(0)
    native_valid = total > 1.e-8
    lever_carrier = np.zeros((4, 3))
    lever_carrier[native_valid] = (weight[:, :, None]*lever_carrier_history).sum(0)[native_valid]/total[native_valid, None]
    current = model.state(root_position, root_quaternion, joint_position)
    carrier = _carrier_rotations(model, current)
    lever = np.einsum('kij,kj->ki', carrier, lever_carrier)
    centers = _centers(current)
    points = centers+lever
    force_allowed = native_valid & (h['native_contact_count'][-1] > 0)
    A = np.zeros((12, 22))
    for k, leg in enumerate(LEGS):
        if force_allowed[k]:
            A[3*k:3*k+3] = current['jac'](leg+'_wheel_link', points[k])[:3]
    basis = np.column_stack((tangent, np.cross(normal, tangent), normal))
    coefficients = np.array([A[3*k:3*k+3, 18+k]@basis for k in range(4)])
    return dict(A=A, points=points, wheel_centers=centers, lever_world=lever,
        lever_carrier=lever_carrier, native_valid=native_valid, force_allowed=force_allowed,
        mean_normal_N=total/20., latest_count=h['native_contact_count'][-1].copy(),
        latest_gap_m=h['native_contact_min_separation_m'][-1].copy(),
        mapped_point_height_m=points@normal,
        force_policy=force_policy, source_time_s=h['time_s'].copy(),
        maximum_center_fk_error_m=maximum_center_fk_error,
        tangent=tangent.copy(), normal=normal.copy(),
        wheel_axis_contact_torque_per_N=coefficients,
        contact_torque_basis=('tangent', 'side', 'normal'),
        note='Isolated, not online. native_valid is geometry history; force_allowed is an explicit candidate policy. '
             'Filtered mapped points can differ from the latest physical contact plane; their height is diagnostic, not a measured gap. '
             'Constrain tangential contact force separately: normal eccentric moment and inertial bias are not friction budgets.')


def rolling_from_native_geometry(geometry, root_com_velocity, omega_world, joint_velocity):
    """同一A和实际速度求滚动轮速，不把命令速度代入实际根速度。

无接触/无可解切向轮系数时保持当前轮速，返回rolling_constrained=False。
该轮零残差只因没有约束，不允许解读成已经无滑移。
"""
    base_v, omega, qd = [np.asarray(x, dtype=float) for x in
                          (root_com_velocity, omega_world, joint_velocity)]
    if (base_v.shape != (3,) or omega.shape != (3,) or qd.shape != (16,)
            or not all(np.isfinite(x).all() for x in (base_v, omega, qd))):
        raise ValueError('Expected finite actual generalized velocity')
    A = geometry['A']; tangent = geometry['tangent']
    v = np.r_[base_v, omega, qd].copy()
    constrained = np.zeros(4, dtype=bool)
    for k in range(4):
        row = tangent@A[3*k:3*k+3]
        coefficient = row[18+k]
        if geometry['force_allowed'][k] and abs(coefficient) > 1.e-8:
            v[18+k] = 0.
            v[18+k] = -(row@v)/coefficient
            constrained[k] = True
    residual = (A@v).reshape(4, 3)@tangent
    return dict(wheel_rad_s=v[18:].copy(), tangent_residual_mps=residual,
                rolling_constrained=constrained)
