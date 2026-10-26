"""CPU 几何与导数回归；不代表真实接触/力矩可行或教师成功。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from entry_path import DEFAULT_Q, build_path
from model import Model
from partial_support_control import solve_foot_pose
from rear_entry_path import DEFAULT_HEIGHT_M, build_rear_path
from research_common import LEGS


@pytest.fixture(scope='module')
def reference():
    with np.load(Path(__file__).parent/'fixtures/coord_real_states.npz') as fixture:
        value = json.loads(fixture['reference_json'].item())
    value['provenance'] = {'kind': 'source posture, retained unchanged'}
    value['force_body_N'] = [[0., 0., 30.]]*4
    return value


@pytest.fixture(scope='module')
def path(reference):
    return build_rear_path(reference, 25., .16, .24)


def basis():
    slope = np.deg2rad(25.)
    return np.array([-np.sin(slope), 0., np.cos(slope)]), np.array([np.cos(slope), 0., np.sin(slope)])


def centers(model, position, quaternion, q):
    state = model.state(position, quaternion, q)
    return np.array([state['trans'][leg+'_wheel_link'][:3, 3] for leg in LEGS])


def test_rear_phase_preserves_front_wheels_and_root_then_lowers_without_forward_motion(path, reference):
    model = Model(); normal, tangent = basis(); split = path['stage_boundary_index']
    p = np.array(path['body_position_plan_m']); quats = np.array(path['body_quaternion_plan'])
    q = np.array(path['q'])
    actual = np.array([centers(model, p[i], quats[i], q[i]) for i in range(len(q))])
    np.testing.assert_allclose(p[:split+1], np.broadcast_to(p[0], (split+1, 3)), atol=1.e-12)
    np.testing.assert_allclose(quats[:split+1], np.broadcast_to(quats[0], (split+1, 4)), atol=1.e-12)
    np.testing.assert_allclose(actual[:split+1, :2], np.broadcast_to(actual[0, :2], (split+1, 2, 3)), atol=1.e-7)
    np.testing.assert_allclose(actual[split, 2:]-actual[0, 2:], np.broadcast_to(-.2*tangent, (2, 3)), atol=1.e-7)
    np.testing.assert_allclose(actual[split:], np.broadcast_to(actual[split], actual[split:].shape), atol=1.e-7)
    np.testing.assert_allclose(p@tangent, 0., atol=1.e-12)
    np.testing.assert_allclose(p@np.array([0., 1., 0.]), 0., atol=1.e-12)
    np.testing.assert_allclose((p@normal)[[0, -1]], [DEFAULT_HEIGHT_M, reference['height_m']], atol=1.e-12)
    np.testing.assert_allclose(Rotation.from_quat(quats[-1, [1, 2, 3, 0]]).as_matrix(),
        Rotation.from_quat(np.array(reference['quat'])[[1, 2, 3, 0]]).as_matrix(), atol=1.e-12)
    np.testing.assert_allclose(q[0], DEFAULT_Q, atol=1.e-10)
    assert np.all(q[:, :12] >= model.limits[:12, 0])
    assert np.all(q[:, :12] <= model.limits[:12, 1])
    assert path['maximum_ik_error_m'] < 1.e-7
    assert path['planned_root_forward_m'] == 0.


def test_endpoint_is_new_geometry_with_preserved_but_uncalibrated_source(reference, path):
    original = deepcopy(reference)
    fresh_path = build_rear_path(reference, 25., .16, .24)
    endpoint = fresh_path['endpoint_reference']
    np.testing.assert_array_equal(endpoint['q'], path['q'][-1])
    np.testing.assert_array_equal(endpoint['quat'], path['body_quaternion_plan'][-1])
    assert max(abs(np.array(endpoint['q'])[:12]-np.array(reference['q'])[:12])) > .1
    assert endpoint['provenance'] == reference['provenance']
    assert endpoint['force_body_N'] == reference['force_body_N']
    assert endpoint['height_m'] == reference['height_m']
    assert endpoint['posture_variant']['physically_validated'] is False
    assert endpoint['posture_variant']['requires_static_rebase'] is True
    assert reference == original
    endpoint['provenance']['kind'] = 'changed'
    assert endpoint['provenance'] != reference['provenance']
    endpoint['q'][0] += .1
    assert endpoint['q'][0] != fresh_path['q'][-1][0]
    assert set(build_path(reference, 25., .4)) <= set(path)
    json.dumps(path, allow_nan=False)


def test_first_point_four_eight_second_candidate_reports_analytic_normal_acceleration(reference, path):
    safer_duration = build_rear_path(reference, 25., .16, .32)
    fast = path['planned_normal_acceleration_diagnostic']
    slow = safer_duration['planned_normal_acceleration_diagnostic']
    drop = DEFAULT_HEIGHT_M-reference['height_m']
    assert fast['analytic_min_mps2'] == pytest.approx(-drop*10*np.sqrt(3)/3/.24**2)
    assert slow['analytic_min_mps2'] == pytest.approx(-drop*10*np.sqrt(3)/3/.32**2)
    assert fast['analytic_min_mps2'] < -fast['gravity_normal_mps2']
    assert fast['faster_downward_than_freefall_frame_fraction'] > 0.
    assert slow['analytic_min_mps2'] > -slow['gravity_normal_mps2']
    assert slow['faster_downward_than_freefall_frame_fraction'] == 0.
    assert slow['sampled_min_mps2'] >= slow['analytic_min_mps2']-1.e-12
    assert slow['sampled_max_mps2'] <= slow['analytic_max_mps2']+1.e-12
    assert safer_duration['endpoint_reference']['posture_variant']['physically_validated'] is False


def test_both_stages_stop_with_zero_velocity_and_acceleration(path):
    indices = [0, path['stage_boundary_index'], len(path['time_s'])-1]
    for key in ('joint_velocity_rad_s', 'joint_acceleration_rad_s2', 'wheel_rad_s',
                'wheel_acceleration_rad_s2', 'body_velocity_plan_mps', 'body_acceleration_plan_mps2',
                'body_angular_velocity_plan_rad_s', 'body_angular_acceleration_plan_rad_s2',
                'wheel_center_velocity_plan_mps', 'wheel_center_acceleration_plan_mps2'):
        actual = np.array(path[key])[indices]
        np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_full_jacobian_rates_realize_center_motion_and_tangent_rolling(path):
    model = Model(); normal, tangent = basis()
    for i, q in enumerate(path['q']):
        R = Rotation.from_quat(np.array(path['body_quaternion_plan'][i])[[1, 2, 3, 0]]).as_matrix()
        omega = np.array(path['body_angular_velocity_plan_rad_s'][i])
        root_v = np.array(path['body_velocity_plan_mps'][i])
        joint_v = np.array(path['joint_velocity_rad_s'][i]); joint_v[12:] = path['wheel_rad_s'][i]
        v = np.r_[root_v+np.cross(omega, R@model.base_offset), omega, joint_v]
        state = model.state(path['body_position_plan_m'][i], path['body_quaternion_plan'][i], q)
        measured = np.array([state['jac'](leg+'_wheel_link', state['trans'][leg+'_wheel_link'][:3, 3])[:3]@v for leg in LEGS])
        np.testing.assert_allclose(measured, path['wheel_center_velocity_plan_mps'][i], atol=1.e-10)
        A, _, _ = model.contacts(state, normal)
        np.testing.assert_allclose((A@v).reshape(4, 3)@tangent, 0., atol=1.e-10)
    assert path['maximum_tangent_rolling_residual_mps'] < 1.e-10
    assert path['maximum_center_velocity_residual_mps'] < 1.e-10


@pytest.mark.parametrize('index', [3, 6, 11, 16])
def test_rates_match_independent_local_ik_and_rotation_differences(path, reference, index):
    # 不对输出数组再做同一种 gradient：在非栅格 t±h 独立生成几何，重解 IK。
    model = Model(); normal, tangent = basis(); t = path['time_s'][index]
    R0 = Rotation.from_euler('y', -np.deg2rad(25.))
    R1 = Rotation.from_quat(np.array(reference['quat'])[[1, 2, 3, 0]])
    rv = (R1*R0.inv()).as_rotvec(); p0 = DEFAULT_HEIGHT_M*normal
    start = centers(model, p0, R0.as_quat()[[3, 0, 1, 2]], DEFAULT_Q)
    seed = np.array(path['q'][index]); h = 2.e-5
    samples = []
    for time in (t-h, t, t+h):
        if time < .16:
            x = time/.16; blend = 6*x**5-15*x**4+10*x**3
            p, R = p0, R0; target = start.copy(); target[2:] -= .2*blend*tangent
        else:
            x = (time-.16)/.24; blend = 6*x**5-15*x**4+10*x**3
            p = p0+blend*(reference['height_m']-DEFAULT_HEIGHT_M)*normal
            R = Rotation.from_rotvec(blend*rv)*R0; target = start.copy(); target[2:] -= .2*tangent
        q = solve_foot_pose(model, seed, (target-p)@R.as_matrix(), np.zeros((4, 3)))
        samples.append((p, R, q))
    before, now, after = samples
    np.testing.assert_allclose((after[2][:12]-before[2][:12])/(2*h), path['joint_velocity_rad_s'][index][:12], atol=2.e-5)
    np.testing.assert_allclose((after[2][:12]-2*now[2][:12]+before[2][:12])/h**2, path['joint_acceleration_rad_s2'][index][:12], atol=.02)
    np.testing.assert_allclose((after[0]-before[0])/(2*h), path['body_velocity_plan_mps'][index], atol=2.e-7)
    np.testing.assert_allclose((after[0]-2*now[0]+before[0])/h**2, path['body_acceleration_plan_mps2'][index], atol=2.e-5)
    np.testing.assert_allclose((after[1]*before[1].inv()).as_rotvec()/(2*h), path['body_angular_velocity_plan_rad_s'][index], atol=2.e-7)
    # 独立用邻域 IK 的腿导数计算滚动速率，比较中心轮加速度。
    wheels = []
    for p, R, q in (before, after):
        state = model.state(p, R.as_quat()[[3, 0, 1, 2]], q)
        delta = -h if len(wheels) == 0 else h
        qd = np.array(path['joint_velocity_rad_s'][index])+delta*np.array(path['joint_acceleration_rad_s2'][index])
        omega = np.array(path['body_angular_velocity_plan_rad_s'][index])+delta*np.array(path['body_angular_acceleration_plan_rad_s2'][index])
        velocity = np.array(path['body_velocity_plan_mps'][index])+delta*np.array(path['body_acceleration_plan_mps2'][index])
        v = np.r_[velocity+np.cross(omega, R.as_matrix()@model.base_offset), omega, qd]; v[18:] = 0.
        A, _, _ = model.contacts(state, normal)
        for k in range(4):
            row = tangent@A[3*k:3*k+3]; v[18+k] = -(row@v)/row[18+k]
        wheels.append(v[18:].copy())
    np.testing.assert_allclose((wheels[1]-wheels[0])/(2*h), path['wheel_acceleration_rad_s2'][index], atol=.02)


@pytest.mark.parametrize('options', [dict(dt=0.), dict(rear_distance=-.1), dict(slope_deg=np.nan),
    dict(rear_duration=.15), dict(lower_duration=.01), dict(rear_distance=2.), dict(final_centers='unknown')])
def test_invalid_or_unreachable_candidate_fails_explicitly(reference, options):
    args = dict(slope_deg=25., rear_duration=.16, lower_duration=.24); args.update(options)
    with pytest.raises(ValueError):
        build_rear_path(reference, **args)


def test_explicit_fixed_mode_keeps_default_candidate_unchanged(reference, path):
    assert build_rear_path(reference, 25., .16, .24, final_centers='fixed') == path


def test_reference_mode_recovers_original_endpoint_without_root_forward_or_wheel_phase_jump(reference):
    original = deepcopy(reference)
    plan = build_rear_path(reference, 25., .16, .32, final_centers='reference')
    fixed = build_rear_path(reference, 25., .16, .32)
    model = Model(); normal, tangent = basis(); split = plan['stage_boundary_index']
    q = np.array(plan['q']); p = np.array(plan['body_position_plan_m'])
    quats = np.array(plan['body_quaternion_plan'])
    actual = np.array([centers(model, p[i], quats[i], q[i]) for i in range(len(q))])
    np.testing.assert_array_equal(q[-1], reference['q'])
    np.testing.assert_array_equal(quats[-1], reference['quat'])
    np.testing.assert_array_equal(plan['endpoint_reference']['q'], reference['q'])
    np.testing.assert_array_equal(plan['endpoint_reference']['quat'], reference['quat'])
    assert plan['endpoint_reference']['height_m'] == reference['height_m']
    np.testing.assert_allclose(q[:split+1, :12], np.array(fixed['q'])[:split+1, :12], atol=1.e-12)
    np.testing.assert_array_equal(q[:, 12:], np.broadcast_to(reference['q'][12:], (len(q), 4)))
    np.testing.assert_array_equal(np.array(plan['joint_velocity_rad_s'])[:, 12:], np.zeros((len(q), 4)))
    np.testing.assert_allclose(p@tangent, 0., atol=1.e-12)
    source_centers = centers(model, normal*reference['height_m'], reference['quat'], reference['q'])
    for i in range(split, len(q)):
        x = (plan['time_s'][i]-.16)/.32
        fraction = 10*x**3-15*x**4+6*x**5
        np.testing.assert_allclose(actual[i], actual[split]+fraction*(source_centers-actual[split]), atol=1.e-7)
    assert np.all(q[:, :12] >= model.limits[:12, 0])
    assert np.all(q[:, :12] <= model.limits[:12, 1])
    assert plan['preserve_wheel_phase'] is True
    assert plan['endpoint_reference']['posture_variant']['physically_validated'] is False
    assert plan['endpoint_reference']['posture_variant']['endpoint_matches_source_geometry'] is True
    assert reference == original
    test_both_stages_stop_with_zero_velocity_and_acceleration(plan)
    test_full_jacobian_rates_realize_center_motion_and_tangent_rolling(plan)


def test_reference_mode_derivatives_match_dense_independent_path_sampling(reference):
    # 细时间网格独立求解IK；只核对远离边界的中心差分，边界另验严格零。
    coarse = build_rear_path(reference, 25., .16, .32, final_centers='reference')
    fine = build_rear_path(reference, 25., .16, .32, dt=.002, final_centers='reference')
    for index in (11, 15, 20):
        j = index*10
        q = np.array(fine['q']); w = np.array(fine['wheel_rad_s'])
        np.testing.assert_allclose((q[j+1]-q[j-1])/.004, coarse['joint_velocity_rad_s'][index], atol=.004)
        np.testing.assert_allclose((q[j+1]-2*q[j]+q[j-1])/.002**2, coarse['joint_acceleration_rad_s2'][index], atol=.04)
        np.testing.assert_allclose((w[j+1]-w[j-1])/.004, coarse['wheel_acceleration_rad_s2'][index], atol=.04)
