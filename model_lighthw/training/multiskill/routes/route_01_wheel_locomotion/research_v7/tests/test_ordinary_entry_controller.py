"""普通进入接口核查，数值解或测试通过不能算完整行为通过。"""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from ordinary_entry_controller import OrdinaryEntryController, DownhillRollingEntry, rolling_reference, root_link_acceleration_to_com, load_weighted_rolling_error, limit_motion_leg_target
from entry_path import build_path, DEFAULT_Q
from partial_support_control import SCALE


def fixture():
    with np.load(Path(__file__).parent/'fixtures/coord_real_states.npz') as p:
        ref = json.loads(p['reference_json'].item())
    plan = build_path(ref, 25., .6)
    controller = OrdinaryEntryController(plan, ref, 25.)
    quat = Rotation.from_euler('y', -25., degrees=True).as_quat()[[3, 0, 1, 2]]
    n, t = controller.B[:, 2], controller.B[:, 0]
    return controller, ref, dict(time=0., q=DEFAULT_Q.copy(), qd=np.zeros(16), quat=quat,
        omega_world=np.zeros(3), root_pos=n*.416223, root_com_velocity=np.zeros(3),
        external_world=-38.*t-7.6*n, hitch_local=np.array([-.38, 0., -.176223]),
        measured_normal=np.zeros(4), nominal_action=np.zeros(16))


@pytest.mark.parametrize('mu', [0., -1., np.inf, -np.inf, np.nan])
def test_ground_friction_requires_positive_finite_coefficient(mu):
    c, ref, _ = fixture()
    with pytest.raises(ValueError, match='friction_mu must be finite and positive'):
        OrdinaryEntryController(c.plan, ref, 25., friction_mu=mu)


def test_default_ground_friction_preserves_actions_and_explicit_value_reaches_constraints(monkeypatch):
    import ordinary_entry_controller as module
    c, ref, state = fixture()
    default_action, default_trace = c.step(**state)
    explicit = OrdinaryEntryController(c.plan, ref, 25., friction_mu=3.)
    action, trace = explicit.step(**state)
    np.testing.assert_array_equal(action, default_action)
    assert default_trace['ordinary_entry_friction_mu'] == trace['ordinary_entry_friction_mu'] == 3.
    actual = module.friction_rows
    calls = []
    def checked(B, mu, minimum_normal):
        calls.append((mu, minimum_normal))
        return actual(B, mu, minimum_normal)
    monkeypatch.setattr(module, 'friction_rows', checked)
    higher = OrdinaryEntryController(c.plan, ref, 25., friction_mu=4.)
    _, higher_trace = higher.step(**state)
    assert calls == [(4., 0.)]
    assert higher_trace['ordinary_entry_friction_mu'] == 4.


def test_ordinary_initial_target_is_independent_of_loaded_leg_action_bias():
    c, ref, z = fixture()
    a, trace = c.step(**z)
    assert trace['ordinary_entry_success'] == 1.
    # 故意污染旧低姿态动作/轮载，不应进入新的初态支撑计算。
    poisoned = dict(ref, leg_action=[100.]*12, force_body_N=np.full((4, 3), 1000.).tolist())
    other = OrdinaryEntryController(c.plan, poisoned, 25.)
    b, _ = other.step(**z)
    np.testing.assert_allclose(a, b, atol=0, rtol=0)
    tau = np.r_[80*(DEFAULT_Q[:12]+a[:12]*SCALE-z['q'][:12])-3*z['qd'][:12],
                2*(a[12:]*5-z['qd'][12:])]
    np.testing.assert_allclose(tau, trace['ordinary_entry_tau_feedforward_Nm'], atol=1.e-10)
    assert np.all(abs(tau) <= np.r_[np.full(12, 96.), np.full(4, 48.)]+1.e-4)
    target = trace['ordinary_entry_target_q']
    assert np.all(target >= c.model.limits[:12, 0]-1.e-6)
    assert np.all(target <= c.model.limits[:12, 1]+1.e-6)
    assert not c.finished


def test_live_rolling_uses_actual_leg_velocities_and_root_command():
    c, _, z = fixture()
    state = c.model.state(z['root_pos'], z['quat'], z['q'])
    n, t = c.B[:, 2], c.B[:, 0]
    base = np.array([-.3, .02, .1]); omega = np.array([.1, -.2, .05])
    qd = np.linspace(-.3, .4, 16)
    wheel, residual = rolling_reference(c.model, state, n, t, base, omega, qd, .15)
    np.testing.assert_allclose(residual, 0., atol=1.e-12)
    changed, _ = rolling_reference(c.model, state, n, t, base, omega, qd+np.r_[np.ones(12)*.5, np.zeros(4)], .15)
    assert np.max(abs(changed-wheel)) > .1


def test_uniform_lateral_translation_does_not_request_return_to_global_origin():
    c, _, z = fixture(); a, t = c.step(**z)
    d, _, other = fixture(); other['root_pos'] = other['root_pos']+np.array([0., .1, 0.])
    b, u = d.step(**other)
    np.testing.assert_allclose(a, b, atol=1.e-7)
    np.testing.assert_allclose(t['ordinary_entry_force_w_N'], u['ordinary_entry_force_w_N'], atol=1.e-6)


def test_airborne_wheels_cannot_receive_planned_ground_force():
    c, _, z = fixture(); z['root_pos'] = z['root_pos']+c.B[:, 2]*.1
    a, t = c.step(**z)
    np.testing.assert_array_equal(t['ordinary_entry_mask'], np.zeros(4))
    np.testing.assert_allclose(t['ordinary_entry_force_w_N'], 0., atol=1.e-4)
    assert not c.finished


def test_finished_geometric_path_has_zero_reference_velocity_and_acceleration():
    c, _, _ = fixture()
    assert abs(c.pa[-1]).max() > .01  # 实际差分端点原先确有非零残留。
    assert abs(c.apath[-1]).max() > .01
    for time in (c.duration, c.duration+.02, c.duration+5.):
        idx, q, qd, qa, pv, pa = c.reference_kinematics(time)
        np.testing.assert_array_equal(q, c.qpath[-1])
        for value in (qd, qa, pv, pa):
            np.testing.assert_array_equal(value, np.zeros_like(value))


def test_link_to_com_acceleration_matches_independent_position_second_difference():
    # 根link作已知平移和固定轴变角速转动；对COM位置独立做中心二阶差分。
    axis = np.array([.2, -.6, .4]); axis /= np.linalg.norm(axis)
    omega = 1.3*axis; alpha = -.7*axis
    offset = np.array([.04137, -.001475, .011995])
    acceleration = np.array([.3, -.2, .1]); h = 1.e-4
    def point(time):
        rotation = Rotation.from_rotvec(axis*(1.3*time-.35*time*time)).as_matrix()
        return .5*acceleration*time*time+rotation@offset
    measured = (point(h)-2*point(0)+point(-h))/h**2
    expected = root_link_acceleration_to_com(acceleration, alpha, omega, offset)
    np.testing.assert_allclose(measured, expected, atol=1.e-8)


def test_rolling_preparation_uses_actual_load_without_ignoring_loaded_rear_error():
    # 轻载前轮6N短暂轮误差0.2m/s；后轮90N滚动合格，旧max判据误拒。
    weights = np.array([6., -2., 45., 45.])
    errors = np.array([.2, 20., 0., 0.])
    value = load_weighted_rolling_error(errors/.0875, np.zeros(4), weights)
    np.testing.assert_allclose(value, .2*6/96)
    assert value < .1
    # 承重轮本身的滚动失败仍被同0.1m/s阈值拒绝。
    errors[2:] = .12
    assert load_weighted_rolling_error(errors/.0875, np.zeros(4), weights) > .1
    assert load_weighted_rolling_error(np.ones(4), np.zeros(4), np.zeros(4)) == 0.


def test_continuous_mode_needs_point_four_seconds_and_remains_active_after_command_release():
    c, ref, z = fixture()
    c = OrdinaryEntryController(c.plan, ref, 25., continue_control=True)
    for i in range(19):
        assert c.advance_preparation(True, i*.02) == 0.
        assert not c.finished
    c.advance_preparation(False, .38)
    for i in range(19):
        c.advance_preparation(True, .4+i*.02)
        assert not c.finished
    assert c.advance_preparation(True, .78) == 0.
    assert c.finished
    # 命令开启后失去原静止准备条件不能重新冻结命令或切回旧教师。
    assert c.advance_preparation(False, .8) == 0. and c.finished
    z.update(time=1., q=np.asarray(ref['q']), quat=np.asarray(ref['quat']),
        root_pos=c.B[:, 2]*ref['height_m'], external_world=-165.*c.B[:, 0]-15.*c.B[:, 2],
        measured_normal=np.array([14., 14., 46., 43.]), nominal_action=np.full(16, 100.), command_velocity=.2)
    action, trace = c.step(**z)
    assert trace['ordinary_entry_success'] == 1.
    assert trace['ordinary_entry_continue_control'] == 1.
    assert trace['ordinary_entry_active'] == 1.
    assert trace['ordinary_entry_prepared'] == 1.
    assert trace['ordinary_entry_blend'] == 0.
    assert np.max(abs(action)) < 20.
    np.testing.assert_allclose(action[12:]*5,
        trace['ordinary_entry_rolling_rad_s']+trace['ordinary_entry_tau_feedforward_Nm'][12:]/2., atol=1.e-12)


def test_continuous_mode_infeasible_fallback_cannot_take_old_teacher_action(monkeypatch):
    from types import SimpleNamespace
    import ordinary_entry_controller as module
    c, ref, z = fixture()
    c = OrdinaryEntryController(c.plan, ref, 25., continue_control=True)
    z['nominal_action'] = np.full(16, 100.)
    z['qd'][12:] = [1., -2., 3., -4.]
    monkeypatch.setattr(module, 'linprog', lambda *args, **kwargs: SimpleNamespace(success=False))
    action, trace = c.step(**z)
    assert trace['ordinary_entry_success'] == 0.
    np.testing.assert_allclose(DEFAULT_Q[:12]+action[:12]*SCALE, z['q'][:12])
    np.testing.assert_allclose(action[12:]*5, z['qd'][12:])
    assert not c.finished


def test_rolling_velocity_override_preserves_instantaneous_pd_and_torque_bounds():
    c, _, z = fixture()
    z['qd'] = np.r_[np.linspace(-.2, .3, 12), [1., -1., .5, -.5]]
    averaged = np.r_[np.linspace(.1, -.2, 12), np.zeros(4)]
    action, trace = c.step(**z, rolling_joint_velocity=averaged)
    assert trace['ordinary_entry_success'] == 1.
    state = c.model.state(z['root_pos'], z['quat'], z['q'])
    expected_wheel, _ = rolling_reference(c.model, state, c.B[:, 2], c.B[:, 0],
        z['root_com_velocity'], z['omega_world'], averaged, 0.)
    np.testing.assert_allclose(trace['ordinary_entry_rolling_rad_s'], expected_wheel, atol=1.e-12)
    np.testing.assert_array_equal(trace['ordinary_entry_rolling_joint_velocity'], averaged)
    actual_tau = np.r_[80*(DEFAULT_Q[:12]+action[:12]*SCALE-z['q'][:12])-3*z['qd'][:12],
        2*(action[12:]*5-z['qd'][12:])]
    _, desired, desired_v, _, _, _ = c.reference_kinematics(z['time'])
    expected_tau = trace['ordinary_entry_tau_feedforward_Nm']+np.r_[
        80*(desired[:12]-z['q'][:12])+3*(desired_v[:12]-z['qd'][:12]),
        2*(expected_wheel-z['qd'][12:])]
    np.testing.assert_allclose(actual_tau, expected_tau, atol=1.e-10)
    assert np.all(abs(actual_tau) <= np.r_[np.full(12, 96.), np.full(4, 48.)]+1.e-4)
    assert np.max(abs(action[12:]*5)) <= 50.+1.e-6
    # 若误将实际PD也替换为平均qd，复算会出现明显误差。
    wrong_tau = np.r_[80*(DEFAULT_Q[:12]+action[:12]*SCALE-z['q'][:12])-3*averaged[:12],
        2*(action[12:]*5-averaged[12:])]
    assert np.max(abs(wrong_tau-actual_tau)) > 1.


@pytest.mark.parametrize('bad', [np.zeros(12), np.full(16, np.nan)])
def test_invalid_rolling_velocity_override_is_rejected(bad):
    c, _, z = fixture()
    with pytest.raises(ValueError, match='finite16D'):
        c.step(**z, rolling_joint_velocity=bad)


def test_motion_limiter_declares_rate_conflict_and_preserves_hard_motor_position_bounds():
    c, _, z = fixture(); q=z['q'].copy(); qd=z['qd'].copy()
    previous=q[:12].copy(); desired=previous+.2
    final, report=limit_motion_leg_target(desired, previous, q, qd, c.model.limits, .1)
    assert np.max(abs(final-previous)) <= .002+1.e-12
    assert not report['conflict'].any()
    # 高瞬时关节速度使原力矩可行区离开上一输出速率窗；必须公开放弃软速率窗。
    qd[0]=40.
    final, report=limit_motion_leg_target(desired, previous, q, qd, c.model.limits, .1)
    assert report['conflict'][0] == 1.
    assert report['rate_window_excess'][0] > .2
    assert np.max(abs(report['actual_torque'])) <= 96.+1.e-7
    assert np.all(final >= c.model.limits[:12,0]-1.e-12)
    assert np.all(final <= c.model.limits[:12,1]+1.e-12)
    qd[0]=200.
    with pytest.raises(RuntimeError, match='no original position/96Nm intersection'):
        limit_motion_leg_target(desired, previous, q, qd, c.model.limits, .1)


def test_motion_limiter_leaves_preparation_and_wheels_unchanged():
    c, ref, z=fixture()
    limited=OrdinaryEntryController(c.plan, ref, 25., continue_control=True, motion_target_rate=.1)
    baseline=OrdinaryEntryController(c.plan, ref, 25., continue_control=True)
    a, initial=limited.step(**z); b, _=baseline.step(**z)
    np.testing.assert_allclose(a,b,atol=1.e-10)
    assert initial['ordinary_entry_motion_target_rate_active'] == 0.
    previous=initial['ordinary_entry_final_target_q'].copy()
    limited.finished=baseline.finished=True
    z['time']=.02; z['external_world']*=1.4
    a, report=limited.step(**z); b, _=baseline.step(**z)
    assert report['ordinary_entry_motion_target_rate_active'] == 1.
    np.testing.assert_allclose(a[12:],b[12:],atol=1.e-10)
    np.testing.assert_allclose(DEFAULT_Q[:12]+a[:12]*SCALE,report['ordinary_entry_final_target_q'],atol=1.e-12)
    actual=80*(report['ordinary_entry_final_target_q']-z['q'][:12])-3*z['qd'][:12]
    np.testing.assert_allclose(actual,report['ordinary_entry_final_leg_request_Nm'],atol=1.e-10)
    assert np.max(abs(actual)) <= 96.+1.e-7
    assert np.all(abs(report['ordinary_entry_final_target_q']-previous) <= .002+report['ordinary_entry_motion_rate_excess_rad']+1.e-10)


def test_native_bias_override_can_match_gravity_and_rejects_bad_coordinates():
    c, ref, z=fixture()
    baseline,_=c.step(**z)
    fresh=OrdinaryEntryController(c.plan,ref,25.)
    _,gravity=fresh.model.dynamics(fresh.model.state(z['root_pos'],z['quat'],z['q']))
    action,trace=fresh.step(**z,native_bias22=gravity)
    np.testing.assert_allclose(action,baseline,atol=1.e-10)
    assert trace['ordinary_entry_native_bias_override'] == 1.
    np.testing.assert_array_equal(trace['ordinary_entry_dynamics_bias'],gravity)
    nonzero=OrdinaryEntryController(c.plan,ref,25.)
    bias=gravity.copy();bias[2]+=10.;bias[4]+=.5
    changed,newtrace=nonzero.step(**z,native_bias22=bias)
    assert np.max(abs(changed-action)) > .001
    np.testing.assert_array_equal(newtrace['ordinary_entry_dynamics_bias'],bias)
    for bad in (np.zeros(12),np.full(22,np.nan)):
        with pytest.raises(ValueError,match='finite22D'):
            fresh.step(**z,native_bias22=bad)


def test_nonzero_motion_rate_requires_continuous_controller():
    c,ref,_=fixture()
    with pytest.raises(ValueError,match='requires continuous control'):
        OrdinaryEntryController(c.plan,ref,25.,motion_target_rate=.1)


def test_explicit_plan_derivatives_are_consumed_and_old_paths_keep_original_differences():
    c,ref,_=fixture()
    np.testing.assert_array_equal(c.apath,np.gradient(c.vpath,.02,axis=0))
    np.testing.assert_array_equal(c.pv,np.gradient(c.ppath,.02,axis=0))
    np.testing.assert_array_equal(c.pa,np.gradient(np.gradient(c.ppath,.02,axis=0),.02,axis=0))
    qa=np.full_like(c.qpath,1.25);pv=np.full_like(c.ppath,.2);pa=np.full_like(c.ppath,-.4)
    plan=dict(c.plan,joint_acceleration_rad_s2=qa,body_velocity_plan_mps=pv,body_acceleration_plan_mps2=pa)
    explicit=OrdinaryEntryController(plan,ref,25.)
    _,_,_,actual_qa,actual_pv,actual_pa=explicit.reference_kinematics(.1)
    np.testing.assert_array_equal(actual_qa,qa[5])
    np.testing.assert_array_equal(actual_pv,pv[5])
    np.testing.assert_array_equal(actual_pa,pa[5])
    np.testing.assert_array_equal(explicit.explicit_derivatives,np.ones(3))
    # 路径结束后仍遵循静止端点合同，不无限重复非零显式尾导数。
    _,_,qd,end_qa,end_pv,end_pa=explicit.reference_kinematics(explicit.duration)
    for value in (qd,end_qa,end_pv,end_pa):
        np.testing.assert_array_equal(value,np.zeros_like(value))


@pytest.mark.parametrize('key', ['joint_acceleration_rad_s2','body_velocity_plan_mps','body_acceleration_plan_mps2'])
@pytest.mark.parametrize('invalid', ['shape','nonfinite'])
def test_invalid_explicit_plan_derivative_is_rejected(key,invalid):
    c,ref,_=fixture()
    value=np.zeros((len(c.qpath),16 if key=='joint_acceleration_rad_s2' else 3))
    if invalid=='shape':value=value[:-1]
    else:value[0,0]=np.nan
    with pytest.raises(ValueError,match=key):
        OrdinaryEntryController(dict(c.plan,**{key:value}),ref,25.)


def test_hip_preload_anchor_preserves_preparation_other_legs_and_wheels():
    c,ref,z=fixture()
    source=json.loads((Path(__file__).parent/'fixtures/loaded_entry_reference.json').read_text())
    ref=dict(ref,leg_action=source['leg_action'])
    baseline=OrdinaryEntryController(c.plan,ref,25.,continue_control=True)
    anchored=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,hold_hip_preload=True)
    a,_=baseline.step(**z);b,tr=anchored.step(**z)
    np.testing.assert_allclose(a,b,atol=1.e-10)
    assert tr['ordinary_entry_hip_preload_active'] == 0.
    baseline.finished=anchored.finished=True
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        external_world=-165.*c.B[:,0]-15.*c.B[:,2],measured_normal=np.array([14.,14.,46.,43.]))
    a,_=baseline.step(**z);b,tr=anchored.step(**z)
    hips=anchored.hip_ids;others=np.array([i for i in range(16) if i not in hips])
    np.testing.assert_allclose(a[others],b[others],atol=1.e-9)
    assert tr['ordinary_entry_hip_preload_active'] == 1.
    wanted=80.*(DEFAULT_Q[:12]+np.array(source['leg_action'])*SCALE-np.asarray(ref['q'])[:12])[hips]
    np.testing.assert_allclose(tr['ordinary_entry_selected_hip_preload_Nm'],wanted,atol=1.e-10)
    np.testing.assert_allclose(DEFAULT_Q[hips]+b[hips]*SCALE[hips],np.asarray(ref['q'])[hips]+wanted/80.,atol=1.e-10)
    actual=80.*(DEFAULT_Q[:12]+b[:12]*SCALE-z['q'][:12])-3*z['qd'][:12]
    np.testing.assert_allclose(actual,tr['ordinary_entry_final_leg_request_Nm'],atol=1.e-10)
    assert np.max(abs(actual)) <= 96.+1.e-7


def test_hip_preload_is_clipped_to_original_position_and_actual_torque_window():
    c,ref,z=fixture()
    action=np.zeros(12);action[[0,3,6,9]]=100.
    ref=dict(ref,leg_action=action.tolist())
    anchored=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,hold_hip_preload=True)
    anchored.finished=True
    a,tr=anchored.step(**z)
    assert tr['ordinary_entry_hip_preload_active'] == 1.
    assert np.max(abs(tr['ordinary_entry_hip_preload_clipping_Nm'])) > 100.
    actual=80.*(DEFAULT_Q[:12]+a[:12]*SCALE-z['q'][:12])-3*z['qd'][:12]
    assert np.max(abs(actual)) <= 96.+1.e-7
    target=DEFAULT_Q[:12]+a[:12]*SCALE
    assert np.all(target >= anchored.model.limits[:12,0]-1.e-10)
    assert np.all(target <= anchored.model.limits[:12,1]+1.e-10)


def test_wheel_torque_mode_exactly_realizes_allocated_instantaneous_torque():
    c,ref,z=fixture()
    z['qd']=np.r_[np.linspace(-.2,.3,12),[2.,-1.,3.,-.5]]
    controller=OrdinaryEntryController(c.plan,ref,25.,wheel_torque_control=True)
    action,trace=controller.step(**z)
    assert trace['ordinary_entry_success'] == 1.
    actual=2.*(action[12:]*5.-z['qd'][12:])
    np.testing.assert_allclose(actual,trace['ordinary_entry_tau_feedforward_Nm'][12:],atol=1.e-10)
    assert np.max(abs(actual)) <= 48.+1.e-7
    assert np.max(abs(action[12:]*5.)) <= 50.+1.e-7
    other=OrdinaryEntryController(c.plan,ref,25.,wheel_torque_control=True)
    changed,_=other.step(**z,rolling_joint_velocity=np.r_[np.ones(12)*4.,np.zeros(4)])
    # 此模式滚动参考仅用于诊断/内部准备，不再向执行力矩叠加隐藏速度项。
    np.testing.assert_allclose(action,changed,atol=1.e-9)
    commanded=OrdinaryEntryController(c.plan,ref,25.,wheel_torque_control=True)
    moving,_=commanded.step(**z,command_velocity=.2)
    assert np.max(abs(moving[12:]-action[12:])) > 1.e-4


def test_hip_anchor_requires_continuous_control_and_valid_source_actions():
    c,ref,_=fixture()
    with pytest.raises(ValueError,match='continuous control'):
        OrdinaryEntryController(c.plan,ref,25.,hold_hip_preload=True)
    with pytest.raises(ValueError,match='source leg action'):
        OrdinaryEntryController(c.plan,ref,25.,continue_control=True,hold_hip_preload=True)


def test_measured_traction_budget_keeps_preparation_actions_exactly_unchanged():
    c,ref,z=fixture()
    baseline=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True)
    budgeted=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4)
    a,old=baseline.step(**z);b,report=budgeted.step(**z)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(old['ordinary_entry_force_w_N'],report['ordinary_entry_force_w_N'])
    assert report['ordinary_entry_traction_budget_active'] == 0.
    # 初始轮载全零没有被当作有效预算；inactive时这里只记录原48Nm界。
    np.testing.assert_array_equal(report['ordinary_entry_allowed_wheel_tau_Nm'],np.full(4,48.))


def test_measured_traction_budget_reallocates_before_actual_pd_without_final_clipping():
    c,ref,z=fixture()
    baseline=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True)
    budgeted=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4)
    baseline.finished=budgeted.finished=True
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        external_world=-165.*c.B[:,0]-15.*c.B[:,2],measured_normal=np.array([-2.,2.,50.,50.]),
        command_velocity=.2)
    z['qd'][12:]=[1.,-2.,3.,-.5]
    _,old=baseline.step(**z);action,report=budgeted.step(**z)
    assert report['ordinary_entry_success'] == 1.
    assert report['ordinary_entry_traction_budget_active'] == 1.
    allowed=2.4*np.maximum(z['measured_normal'],0.)*.0875
    np.testing.assert_allclose(report['ordinary_entry_allowed_wheel_tau_Nm'],allowed,atol=0.)
    actual=2.*(action[12:]*5.-z['qd'][12:])
    np.testing.assert_allclose(actual,report['ordinary_entry_tau_feedforward_Nm'][12:],atol=1.e-10)
    np.testing.assert_allclose(actual,report['ordinary_entry_final_wheel_request_Nm'],atol=1.e-10)
    assert np.all(abs(actual) <= allowed+1.e-4)
    assert abs(actual[0]) < 1.e-4  # 负测量轮载按零载荷处理。
    assert np.max(abs(actual)) <= 48.+1.e-4
    assert np.max(abs(action[12:]*5.)) <= 50.+1.e-4
    assert np.max(abs(report['ordinary_entry_force_w_N']-old['ordinary_entry_force_w_N'])) > 1.


def test_measured_traction_budget_starts_only_after_previously_finished_step():
    c,ref,z=fixture()
    budgeted=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4)
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        external_world=-165.*c.B[:,0]-15.*c.B[:,2],measured_normal=np.array([14.,14.,46.,43.]))
    budgeted.ready_time=.38
    _,transition=budgeted.step(**z)
    assert transition['ordinary_entry_current_state_ready'] == 1.
    assert budgeted.finished
    assert transition['ordinary_entry_traction_budget_active'] == 0.
    _,following=budgeted.step(**z)
    assert following['ordinary_entry_traction_budget_active'] == 1.


@pytest.mark.parametrize('command,phase',[(.3,'prepared_nonzero_command'),(0.,'prepared_zero_command')])
def test_empty_measured_traction_servo_intersection_stops_before_qp(monkeypatch,command,phase):
    import ordinary_entry_controller as module
    c,ref,z=fixture()
    budgeted=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4)
    budgeted.finished=True
    z.update(time=20.42,command_velocity=command)
    z['qd'][12]=60.  # 原±50rad/s目标要求至少−20Nm，但零实测载荷预算只能请求0Nm。
    def forbidden_solve(*args,**kwargs):
        raise AssertionError('Empty physical intersection must stop before force allocation')
    monkeypatch.setattr(module,'linprog',forbidden_solve)
    with pytest.raises(RuntimeError,match='no original wheel speed/48Nm intersection') as failure:
        budgeted.step(**z)
    details=json.loads(str(failure.value).partition(': ')[2])
    assert details['time_s'] == 20.42
    assert details['controller_phase'] == phase
    assert details['prepared_at_start'] is True
    assert details['command_velocity_mps'] == command
    assert details['wheel_order'] == ['FR','FL','RR','RL']
    np.testing.assert_array_equal(details['actual_wheel_qd_rad_s'],[60.,0.,0.,0.])
    np.testing.assert_array_equal(details['measured_normal_N'],np.zeros(4))
    np.testing.assert_array_equal(details['allowed_wheel_tau_Nm'],np.zeros(4))
    np.testing.assert_array_equal(details['original_feasible_wheel_lower_Nm'],[-48.,-48.,-48.,-48.])
    np.testing.assert_array_equal(details['original_feasible_wheel_upper_Nm'],[-20.,48.,48.,48.])
    np.testing.assert_array_equal(details['intersection_wheel_lower_Nm'],np.zeros(4))
    np.testing.assert_array_equal(details['intersection_wheel_upper_Nm'],[-20.,0.,0.,0.])


def test_measured_traction_solver_failure_fallback_reports_failure_and_obeys_interval(monkeypatch):
    from types import SimpleNamespace
    import ordinary_entry_controller as module
    c,ref,z=fixture()
    budgeted=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4)
    budgeted.finished=True
    z['qd'][12:]=[52.,-2.,3.,-.5]
    z['measured_normal']=np.array([20.,0.,0.,0.])
    monkeypatch.setattr(module,'linprog',lambda *args,**kwargs: SimpleNamespace(success=False))
    action,report=budgeted.step(**z)
    assert report['ordinary_entry_success'] == 0.
    actual=2.*(action[12:]*5.-z['qd'][12:])
    np.testing.assert_allclose(actual,[-4.,0.,0.,0.],atol=1.e-10)
    assert np.all(actual >= report['ordinary_entry_wheel_torque_lower_Nm']-1.e-10)
    assert np.all(actual <= report['ordinary_entry_wheel_torque_upper_Nm']+1.e-10)


@pytest.mark.parametrize('mu,continuous,torque', [(-1.,True,True),(np.nan,True,True),
    (np.inf,True,True),(2.4,False,True),(2.4,True,False),(2.4,False,False)])
def test_measured_traction_option_requires_finite_mu_and_both_control_modes(mu,continuous,torque):
    c,ref,_=fixture()
    with pytest.raises(ValueError,match='requires continuous wheel torque control'):
        OrdinaryEntryController(c.plan,ref,25.,continue_control=continuous,wheel_torque_control=torque,
            measured_traction_mu=mu)


def test_motion_feedback_configuration_preserves_preparation_actions_exactly():
    c,ref,z=fixture()
    baseline=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True)
    stronger=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        motion_velocity_gain=6.,motion_acceleration_limit=1.5)
    z['root_com_velocity']=c.B[:,0]*-.4
    a,old=baseline.step(**z);b,report=stronger.step(**z)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(old['ordinary_entry_force_w_N'],report['ordinary_entry_force_w_N'])
    np.testing.assert_array_equal(old['ordinary_entry_root_link_acceleration_mps2'],
        report['ordinary_entry_root_link_acceleration_mps2'])
    assert report['ordinary_entry_motion_feedback_active'] == 0.
    assert report['ordinary_entry_motion_velocity_gain'] == 1.5
    assert report['ordinary_entry_motion_acceleration_limit_mps2'] == .5


@pytest.mark.parametrize('command,expected',[(.1,.6),(.3,1.5),(-.3,-1.5)])
def test_motion_feedback_uses_configured_gain_and_signed_acceleration_limit(command,expected):
    c,ref,z=fixture()
    stronger=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4,motion_velocity_gain=6.,motion_acceleration_limit=1.5)
    stronger.finished=True
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        external_world=-165.*c.B[:,0]-15.*c.B[:,2],measured_normal=np.array([14.,14.,46.,43.]),
        command_velocity=command)
    action,report=stronger.step(**z)
    assert report['ordinary_entry_motion_feedback_active'] == 1.
    assert report['ordinary_entry_motion_velocity_gain'] == 6.
    assert report['ordinary_entry_motion_acceleration_limit_mps2'] == 1.5
    np.testing.assert_allclose(report['ordinary_entry_root_link_acceleration_mps2']@c.B[:,0],expected,atol=1.e-12)
    actual=2.*(action[12:]*5.-z['qd'][12:])
    np.testing.assert_allclose(actual,report['ordinary_entry_tau_feedforward_Nm'][12:],atol=1.e-10)
    assert np.all(abs(actual) <= report['ordinary_entry_allowed_wheel_tau_Nm']+1.e-4)


@pytest.mark.parametrize('name',['motion_velocity_gain','motion_acceleration_limit'])
@pytest.mark.parametrize('invalid',[-1.,np.nan,np.inf])
def test_motion_feedback_parameters_reject_nonfinite_and_negative(name,invalid):
    c,ref,_=fixture()
    with pytest.raises(ValueError,match=f'{name} must be finite and nonnegative'):
        OrdinaryEntryController(c.plan,ref,25.,**{name:invalid})


@pytest.mark.parametrize('gain,cap',[(0.,1.5),(6.,0.)])
def test_zero_motion_feedback_parameter_disables_along_route_acceleration(gain,cap):
    c,ref,z=fixture()
    controller=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,
        motion_velocity_gain=gain,motion_acceleration_limit=cap)
    controller.finished=True
    _,report=controller.step(**z,command_velocity=.3)
    np.testing.assert_allclose(report['ordinary_entry_root_link_acceleration_mps2']@c.B[:,0],0.,atol=1.e-12)


def test_contact_traction_budget_keeps_preparation_actions_exactly_unchanged():
    c,ref,z=fixture()
    options=dict(continue_control=True,wheel_torque_control=True,measured_traction_mu=2.4)
    old=OrdinaryEntryController(c.plan,ref,25.,**options)
    contact=OrdinaryEntryController(c.plan,ref,25.,**options,traction_budget_contact_only=True)
    a,_=old.step(**z);b,report=contact.step(**z)
    np.testing.assert_array_equal(a,b)
    assert report['ordinary_entry_traction_budget_contact_only_active'] == 0.
    np.testing.assert_array_equal(report['ordinary_entry_traction_budget_center_Nm'],np.zeros(4))


def test_airborne_zero_load_contact_budget_keeps_inertial_bias_without_inventing_ground_force():
    c,ref,z=fixture()
    z['root_pos']=z['root_pos']+.1*c.B[:,2]
    options=dict(continue_control=True,wheel_torque_control=True,measured_traction_mu=2.4)
    old=OrdinaryEntryController(c.plan,ref,25.,**options)
    contact=OrdinaryEntryController(c.plan,ref,25.,**options,traction_budget_contact_only=True)
    old.finished=contact.finished=True
    # 显式非零wheel bias使原“离地F=0、总τ=0、轮加速度=0”矛盾可独立复现。
    _,bias=contact.model.dynamics(contact.model.state(z['root_pos'],z['quat'],z['q']))
    bias[18:]+=[.1,-.2,.3,-.4]
    _,old_report=old.step(**z,native_bias22=bias)
    action,report=contact.step(**z,native_bias22=bias)
    assert old_report['ordinary_entry_success'] == 0.
    assert report['ordinary_entry_success'] == 1.
    np.testing.assert_array_equal(report['ordinary_entry_mask'],np.zeros(4))
    np.testing.assert_allclose(report['ordinary_entry_force_w_N'],0.,atol=1.e-8)
    np.testing.assert_allclose(report['ordinary_entry_contact_wheel_request_Nm'],0.,atol=1.e-8)
    actual=2.*(action[12:]*5.-z['qd'][12:])
    np.testing.assert_allclose(actual,report['ordinary_entry_wheel_inertial_bias_Nm'],atol=1.e-8)
    np.testing.assert_allclose(actual,report['ordinary_entry_traction_budget_center_Nm'],atol=1.e-8)
    assert np.max(abs(actual)) > .1
    assert np.max(abs(actual)) <= 48.+1.e-7


def test_contact_budget_matches_circle_contact_jacobian_and_preserves_original_motor_limits():
    c,ref,z=fixture()
    controller=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        measured_traction_mu=2.4,traction_budget_contact_only=True)
    controller.finished=True
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        external_world=-165.*c.B[:,0]-15.*c.B[:,2],measured_normal=np.array([1000.,1000.,1000.,1000.]))
    z['qd'][12:]=[45.,-45.,0.,0.]
    action,report=controller.step(**z)
    assert report['ordinary_entry_success'] == 1.
    state=controller.model.state(z['root_pos'],z['quat'],z['q'])
    A,_,_=controller.model.contacts(state,controller.B[:,2])
    contact_tau=-A.T[18:]@report['ordinary_entry_force_w_N'].reshape(12)
    np.testing.assert_allclose(report['ordinary_entry_contact_wheel_request_Nm'],contact_tau,atol=1.e-10)
    actual=2.*(action[12:]*5.-z['qd'][12:])
    np.testing.assert_allclose(actual,report['ordinary_entry_tau_feedforward_Nm'][12:],atol=1.e-10)
    np.testing.assert_allclose(report['ordinary_entry_wheel_torque_lower_Nm'],[-48.,-10.,-48.,-48.],atol=1.e-10)
    np.testing.assert_allclose(report['ordinary_entry_wheel_torque_upper_Nm'],[10.,48.,48.,48.],atol=1.e-10)
    assert np.max(abs(actual)) <= 48.+1.e-7
    assert np.max(abs(action[12:]*5.)) <= 50.+1.e-7


def downhill_fixture():
    _,ref,z=fixture()
    ref=dict(ref,quat=(Rotation.from_euler('y',45.,degrees=True)*
        Rotation.from_quat(np.asarray(ref['quat'])[[1,2,3,0]])).as_quat()[[3,0,1,2]].tolist())
    plan=build_path(ref,-20.,.6)
    controller=OrdinaryEntryController(plan,ref,-20.,continue_control=True,wheel_torque_control=True,
        downhill_rolling_entry=True)
    z.update(root_pos=controller.B[:,2]*.416223,quat=Rotation.from_euler('y',20.,degrees=True).as_quat()[[3,0,1,2]],
        external_world=-38.*controller.B[:,0]-7.6*controller.B[:,2])
    return controller,ref,z


def test_downhill_settled_gate_requires_rolling_entry_mode():
    c,ref,_=downhill_fixture()
    with pytest.raises(ValueError,match='Downhill settled entry requires downhill rolling entry'):
        OrdinaryEntryController(c.plan,ref,-20.,continue_control=True,wheel_torque_control=True,
            downhill_settled_entry=True)


@pytest.mark.parametrize('omega,leg_velocity,expected',[(.3,0.,False),(.299,0.,True),
    (0.,.5,False),(0.,.499,True)])
def test_downhill_settled_gate_checks_strict_rate_bounds_and_three_consecutive_steps(omega,leg_velocity,expected):
    base,ref,z=downhill_fixture()
    c=OrdinaryEntryController(base.plan,ref,-20.,continue_control=True,wheel_torque_control=True,
        downhill_rolling_entry=True,downhill_settled_entry=True)
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        measured_normal=np.full(4,30.),omega_world=np.array([omega,0.,0.]))
    z['qd'][0]=leg_velocity
    for k in range(3):
        _,report=c.step(**z)
        np.testing.assert_allclose(report['ordinary_entry_downhill_settled_errors_rad_s'],[omega,leg_velocity])
        np.testing.assert_array_equal(report['ordinary_entry_downhill_settled_thresholds_rad_s'],[.3,.5])
        assert bool(report['ordinary_entry_downhill_brake_eligible']) == expected
        assert report['ordinary_entry_downhill_phase'] == (2. if expected and k == 2 else 1.)
        assert not c.finished
    if expected:
        assert report['ordinary_entry_downhill_transition_reason'] == 1.
    else:
        assert report['ordinary_entry_downhill_low_pose_ready_s'] == 0.


def test_downhill_settled_gate_tracks_leg_reference_velocity_and_ignores_wheel_spin():
    base,ref,z=downhill_fixture()
    c=OrdinaryEntryController(base.plan,ref,-20.,continue_control=True,wheel_torque_control=True,
        downhill_rolling_entry=True,downhill_settled_entry=True)
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        measured_normal=np.full(4,30.))
    reference=list(c.reference_kinematics(z['time']))
    reference[2]=np.r_[np.full(12,.6),np.zeros(4)]
    c.reference_kinematics=lambda time: tuple(reference)
    z['qd']=np.r_[np.full(12,.7),np.full(4,4.)]
    _,report=c.step(**z)
    np.testing.assert_allclose(report['ordinary_entry_downhill_settled_errors_rad_s'],[0.,.1],atol=1.e-12)
    assert report['ordinary_entry_downhill_brake_eligible'] == 1.


def test_disabled_downhill_settled_gate_preserves_existing_actions_exactly():
    old,ref,z=downhill_fixture()
    explicit=OrdinaryEntryController(old.plan,ref,-20.,continue_control=True,wheel_torque_control=True,
        downhill_rolling_entry=True,downhill_settled_entry=False)
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=old.B[:,2]*ref['height_m'],
        measured_normal=np.full(4,30.),omega_world=np.array([.4,0.,0.]))
    z['qd'][0]=.7
    for _ in range(3):
        a,baseline=old.step(**z);b,report=explicit.step(**z)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(baseline['ordinary_entry_force_w_N'],report['ordinary_entry_force_w_N'])
    assert report['ordinary_entry_downhill_phase'] == 2.
    assert report['ordinary_entry_downhill_settled_entry'] == 0.


def test_downhill_reference_starts_zero_integrates_acceleration_and_stops_at_speed_cap():
    state=DownhillRollingEntry()
    for i in range(20):
        report=state.step(False)
        assert report['phase'] == 1 and not report['settling_allowed']
        np.testing.assert_allclose(report['velocity_reference_mps'],i*.04,atol=1.e-12)
        np.testing.assert_allclose(report['acceleration_feedforward_mps2'],2.,atol=1.e-12)
        assert report['next_velocity_reference_mps'] <= .8
    for _ in range(5):
        report=state.step(False)
        assert report['phase'] == 1 and report['transition_reason'] == 0
        assert report['velocity_reference_mps'] == .8
        assert report['acceleration_feedforward_mps2'] == 0.
        assert report['velocity_cap_active'] == 1.
        assert not report['settling_allowed']  # 到速度上限不能冒充姿态已建立。


def test_downhill_low_pose_must_persist_then_braking_reaches_zero_without_reverse_reference():
    state=DownhillRollingEntry()
    for _ in range(20):state.step(False)
    for _ in range(2):assert state.step(True)['phase'] == 1
    assert state.step(False)['low_pose_ready_s'] == 0.
    for _ in range(2):assert state.step(True)['phase'] == 1
    report=state.step(True)
    assert report['phase'] == 2 and report['transition_reason'] == 1
    assert report['acceleration_feedforward_mps2'] == -1.
    assert report['low_pose_ready_s'] >= .06-1.e-12
    for _ in range(45):
        previous=state.velocity_reference
        report=state.step(False)
        assert 0. <= report['next_velocity_reference_mps'] <= previous
        if report['phase'] == 3:
            assert report['transition_reason'] == 2
            assert report['velocity_reference_mps'] == 0.
            assert report['acceleration_feedforward_mps2'] == 0.
            break
        assert report['phase'] == 2 and not report['settling_allowed']
        assert -1. <= report['acceleration_feedforward_mps2'] <= 0.
    else:pytest.fail('Brake reference did not settle')
    report=state.step(False,prepared=True)
    assert report['phase'] == 4 and report['active'] == 0. and report['transition_reason'] == 3


@pytest.mark.parametrize('speed,expected',[(0.,2.),(.3,.2),(-1.,3.),(1.,-2.)])
def test_downhill_preparation_uses_acceleration_feedforward_and_bounded_velocity_feedback(speed,expected):
    c,_,z=downhill_fixture()
    z['root_com_velocity']=c.B[:,0]*speed
    _,report=c.step(**z)
    assert report['ordinary_entry_downhill_phase'] == 1.
    assert report['ordinary_entry_route_velocity_reference_mps'] == 0.
    assert report['ordinary_entry_route_acceleration_feedforward_mps2'] == 2.
    np.testing.assert_allclose(report['ordinary_entry_route_acceleration_requested_mps2'],expected,atol=1.e-12)
    np.testing.assert_allclose(report['ordinary_entry_root_link_acceleration_mps2']@c.B[:,0],expected,atol=1.e-12)
    np.testing.assert_array_equal(report['ordinary_entry_route_acceleration_bounds_mps2'],[-2.,3.])
    assert not c.finished


def test_downhill_controller_waits_for_low_pose_then_original_point_four_second_ready():
    c,ref,z=downhill_fixture()
    z.update(time=1.,q=np.asarray(ref['q']),quat=np.asarray(ref['quat']),root_pos=c.B[:,2]*ref['height_m'],
        measured_normal=np.array([30.,30.,30.,30.]))
    first_settle=None
    for k in range(60):
        _,report=c.step(**z)
        if report['ordinary_entry_downhill_phase'] < 3:
            assert report['ordinary_entry_current_state_ready'] == 0.
            assert c.ready_time == 0. and not c.finished
        elif first_settle is None:first_settle=k
        if c.finished:
            assert first_settle is not None
            assert k-first_settle == 19
            assert c.ready_time >= .4-1.e-12
            break
    else:pytest.fail('Original ready gate never completed at repeated stable endpoint state')
    _,following=c.step(**z)
    assert following['ordinary_entry_downhill_phase'] == 4.
    assert following['ordinary_entry_downhill_active'] == 0.
    assert following['ordinary_entry_route_acceleration_feedforward_mps2'] == 0.


def test_disabled_downhill_mode_leaves_default_uphill_actions_exactly_unchanged():
    c,ref,z=fixture()
    baseline=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True)
    explicit=OrdinaryEntryController(c.plan,ref,25.,continue_control=True,wheel_torque_control=True,
        downhill_rolling_entry=False)
    for ready in [False,True]:
        baseline.finished=explicit.finished=ready
        a,old=baseline.step(**z);b,report=explicit.step(**z)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(old['ordinary_entry_force_w_N'],report['ordinary_entry_force_w_N'])
        assert report['ordinary_entry_downhill_active'] == 0.
        assert report['ordinary_entry_downhill_phase'] == 0.


@pytest.mark.parametrize('slope,continuous,torque',[(25.,True,True),(0.,True,True),(-20.,False,True),(-20.,True,False)])
def test_downhill_entry_rejects_wrong_slope_or_modes(slope,continuous,torque):
    c,ref,_=fixture()
    with pytest.raises(ValueError,match='negative slope and continuous wheel torque control'):
        OrdinaryEntryController(c.plan,ref,slope,continue_control=continuous,wheel_torque_control=torque,
            downhill_rolling_entry=True)
