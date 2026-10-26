"""真实CPU/GPU窗口的离线接口检查，不生成恢复标签或接入在线控制。"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from model import Model
from native_cop_geometry import build_native_cop_geometry, rolling_from_native_geometry
from research_common import LEGS


FIXTURE = Path(__file__).parent/'fixtures/native_cop_windows.npz'
ANGLE = np.deg2rad(25.)
NORMAL = np.array([-np.sin(ANGLE), 0., np.cos(ANGLE)])
TANGENT = np.array([np.cos(ANGLE), 0., np.sin(ANGLE)])


def sample(index):
    with np.load(FIXTURE) as z:
        return {k: z[k][index].copy() for k in z.files}


def geometry(history):
    return build_native_cop_geometry(Model(), history['body_pos_w'][-1]-history['env_origin_w'],
        history['body_quat_w'][-1], history['joint_pos'][-1], history, NORMAL, TANGENT,
        current_time_s=history['time_s'][-1], env_origin_w=history['env_origin_w'], force_policy='latest_point')


def velocity(history):
    return dict(root_com_velocity=history['momentum_body_com_velocity_w'][-1, 0],
        omega_world=history['body_angular_velocity_w'][-1, 0], joint_velocity=history['joint_velocity'][-1])


def test_fixture_contains_executed_windows_with_provenance_and_no_success_labels():
    meta = json.loads(FIXTURE.with_suffix('.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == meta['fixture_sha256']
    assert len(meta['samples']) == 4
    assert meta['successful_teacher_labels'] == 0
    assert [s['time_s'] for s in meta['samples']] == [20., 20., 16.52, 16.54]


@pytest.mark.parametrize('index', range(4))
def test_actual_history_maps_to_current_model_with_one_shared_force_and_rolling_jacobian(index):
    h = sample(index); result = geometry(h); v = velocity(h)
    model = Model(); state = model.state(h['body_pos_w'][-1]-h['env_origin_w'], h['body_quat_w'][-1], h['joint_pos'][-1])
    for k, leg in enumerate(LEGS):
        block = result['A'][3*k:3*k+3]
        if result['force_allowed'][k]:
            expected = state['jac'](leg+'_wheel_link', result['points'][k])[:3]
            np.testing.assert_allclose(block, expected, atol=1.e-12)
        else:
            np.testing.assert_array_equal(block, np.zeros((3, 22)))
    roll = rolling_from_native_geometry(result, **v)
    qd = v['joint_velocity'].astype(float).copy(); qd[12:] = roll['wheel_rad_s']
    generalized = np.r_[v['root_com_velocity'], v['omega_world'], qd]
    np.testing.assert_allclose((result['A']@generalized).reshape(4, 3)@TANGENT, 0., atol=1.e-10)
    np.testing.assert_array_equal(roll['rolling_constrained'], result['force_allowed'])
    assert result['maximum_center_fk_error_m'] < 3.e-5
    assert result['force_policy'] == 'latest_point'


def test_actual_recent_rl_contact_does_not_reintroduce_support_after_latest_point_disappears():
    # CPUgain16.52s: prior20ms RL meanN42.38，但最新count=0。
    h = sample(2); g = geometry(h)
    assert g['native_valid'][3]
    assert g['mean_normal_N'][3] > 40.
    assert g['latest_count'][3] == 0
    assert not g['force_allowed'][3]
    assert g['force_allowed'].any()  # RR有效，不要求四轮。
    np.testing.assert_array_equal(g['A'][9:12], 0.)
    roll = rolling_from_native_geometry(g, **velocity(h))
    assert not roll['rolling_constrained'][3]
    assert roll['wheel_rad_s'][3] == h['joint_velocity'][-1, 15]


def test_actual_fully_unloaded_rl_has_no_cop_constraint_and_keeps_current_wheel_velocity():
    # CPUgain16.54s: RL整个20ms法向载荷0；不构造圆形接触替代。
    h = sample(3); g = geometry(h)
    assert not g['native_valid'][3]
    assert not g['force_allowed'][3]
    assert g['mean_normal_N'][3] == 0.
    np.testing.assert_array_equal(g['A'][9:12], 0.)
    roll = rolling_from_native_geometry(g, **velocity(h))
    assert roll['wheel_rad_s'][3] == h['joint_velocity'][-1, 15]
    assert not roll['rolling_constrained'][3]


def test_world_translation_and_wheel_phase_do_not_move_the_estimated_carrier_lever():
    h = sample(0); expected = geometry(h)
    shifted = {k: np.asarray(v, dtype=float).copy() for k, v in h.items()}
    delta = np.array([15., -23., 8.])
    for key in ('body_pos_w', 'wheel_pos_w', 'native_contact_cop_w'):
        shifted[key] += delta
    shifted['env_origin_w'] += delta
    shifted['joint_pos'][:, 12:] += np.array([.7, -1.1, 13., -24.])
    actual = geometry(shifted)
    for key in ('lever_carrier', 'lever_world', 'points', 'A'):
        np.testing.assert_allclose(actual[key], expected[key], atol=2.e-12)
    # 使用当前轮心+轮架力臂，不能直接平均world CoP后减最新轮心。
    moving = sample(1); result = geometry(moving)
    weight = np.maximum(moving['native_ground_normal_w']@NORMAL, 0.)*(moving['native_contact_count']>0)
    naive = (weight[:, :, None]*moving['native_contact_cop_w']).sum(0)/weight.sum(0)[:, None]-moving['env_origin_w']
    assert np.max(abs(naive-result['points'])) > 1.e-5


def test_actual_eccentric_normal_load_generates_wheel_axis_moment_and_is_not_tangential_force():
    h = sample(0); g = geometry(h); model = Model()
    state = model.state(h['body_pos_w'][-1]-h['env_origin_w'], h['body_quat_w'][-1], h['joint_pos'][-1])
    original_A = model.contacts(state, NORMAL)[0]
    old_normal_coefficient = np.array([NORMAL@original_A[3*k:3*k+3, 18+k] for k in range(4)])
    np.testing.assert_allclose(old_normal_coefficient, 0., atol=1.e-12)
    coefficients = g['wheel_axis_contact_torque_per_N']
    assert abs(coefficients[2, 2]) > .01  # 真实RR法向偏心，不应扣作切向摩擦预算。
    normal_only_force = g['mean_normal_N'][:, None]*NORMAL
    general_contact = g['A'].T@normal_only_force.reshape(-1)
    np.testing.assert_allclose(general_contact[18:], coefficients[:, 2]*g['mean_normal_N'], atol=1.e-12)
    np.testing.assert_allclose(normal_only_force@TANGENT, 0., atol=1.e-12)
    assert abs(general_contact[20]) > .4
    # 与实际逐点法向力矩对照；无摩擦力输入，不会误把偏心矩叫摩擦。
    measured = []
    for i in range(20):
        st = model.state(h['body_pos_w'][i]-h['env_origin_w'], h['body_quat_w'][i], h['joint_pos'][i])
        axes = np.array([st['axes'][leg+'_wheel_joint'] for leg in LEGS])
        moment = h['native_contact_moment_com_w'][i]-np.cross(
            h['wheel_pos_w'][i]-h['whole_com_w'][i], h['native_ground_normal_w'][i])
        measured.append(np.sum(axes*moment, axis=1))
    np.testing.assert_allclose(general_contact[18:], np.mean(measured, axis=0), atol=.02)


@pytest.mark.parametrize('kind', ['time', 'position', 'orientation', 'joint', 'wheel_center'])
def test_misaligned_last_state_or_wheel_history_is_explicitly_rejected(kind):
    h = sample(0); p = h['body_pos_w'][-1]-h['env_origin_w']; quat = h['body_quat_w'][-1].copy()
    q = h['joint_pos'][-1].copy(); time = h['time_s'][-1]
    if kind == 'time': time += .001
    if kind == 'position': p += [.001, 0., 0.]
    if kind == 'orientation': quat = (Rotation.from_rotvec([.001, 0., 0.])*Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    if kind == 'joint': q[2] += .001
    if kind == 'wheel_center': h['wheel_pos_w'][5, 2, 0] += .001
    with pytest.raises(ValueError):
        build_native_cop_geometry(Model(), p, quat, q, h, NORMAL, TANGENT,
            current_time_s=time, env_origin_w=h['env_origin_w'], force_policy='latest_point')


def test_force_policy_is_explicit_and_helper_does_not_define_motor_torque_budget():
    h = sample(0)
    with pytest.raises(ValueError, match='force_policy'):
        build_native_cop_geometry(Model(), h['body_pos_w'][-1]-h['env_origin_w'],
            h['body_quat_w'][-1], h['joint_pos'][-1], h, NORMAL, TANGENT,
            current_time_s=h['time_s'][-1], env_origin_w=h['env_origin_w'], force_policy='geometric_fallback')
    g = geometry(h)
    assert 'torque_budget' not in g and 'motor_limits' not in g
