import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from partial_support_control import BalanceOptions,PartialSupportControl,attitude,foot_adjustments,solve_foot_pose,support_joint_torque
from model import Model
from research_common import LEGS


def test_attitude_signs_and_foot_correction_lower_raised_side():
    n=np.array([0.,0.,1.]);q=np.roll(Rotation.from_euler('xyz',[.1,-.2,0]).as_quat(),1)
    angles,rates=attitude(q,n,np.array([.3,-.4,0.]))
    np.testing.assert_allclose(angles,[.2,.1]);np.testing.assert_allclose(rates,[.4,.3])
    centers=np.array([[.3,-.2,-.3],[.3,.2,-.3],[-.3,-.2,-.3],[-.3,.2,-.3]])
    correction=foot_adjustments(centers,angles,np.zeros(2),BalanceOptions(pitch_gain=.5,roll_gain=.5))
    assert correction[:2,2].mean()>0 and correction[2:,2].mean()<0
    assert correction[[1,3],2].mean()>correction[[0,2],2].mean()


def test_full_attitude_foot_shift_matches_inverse_body_rotation():
    centers=np.array([[.3,-.2,-.3],[.3,.2,-.3],[-.3,-.2,-.3],[-.3,.2,-.3]])
    error=np.array([.02,.03])
    correction=foot_adjustments(centers,error,np.zeros(2),BalanceOptions(pitch_gain=1,roll_gain=1,horizontal_attitude_gain=1))
    # Desired body rotation is opposite the measured roll/pitch error.
    np.testing.assert_allclose(correction,np.cross([error[1],-error[0],0.],centers),atol=1.e-12)


def test_diagonal_load_admittance_preserves_body_plane_and_uses_world_normal():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    quat=np.roll(Rotation.from_euler('y',-.4).as_quat(),1);normal=Rotation.from_quat(quat[[1,2,3,0]]).apply([0,0,1])
    c=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4,diagonal_admittance_m_Ns=.005))
    force=np.array([40,10,10,40])[:,None]*normal
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,quat,normal,[0,0,0],force)
    for t in np.arange(4,5,.02):_,trace=c.step(t,np.zeros(12),q,quat,normal,[0,0,0],force)
    z=trace['balance_foot_delta_m'][:,2]
    assert 0<z[0]<=.02 and z[1]<0 and z[2]<0 and z[3]>0
    np.testing.assert_allclose([z.sum(),z[:2].sum(),z[[0,2]].sum()],0,atol=1.e-12)


def test_rate_filter_rejects_an_impulse_without_flipping_the_correction():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    c=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4,rate_gain=.12,rate_filter_s=.1))
    for t in np.arange(3,4.02,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0])
    c.step(4.02,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,1])
    # Yaw is not used as pitch/roll rate.
    np.testing.assert_allclose(c.filtered_rates,0)
    c.step(4.04,np.zeros(12),q,[1,0,0,0],[0,0,1],[1,0,0])
    assert c.filtered_rates[1]==pytest.approx(1-np.exp(-.2))


def test_zero_feedback_holds_reference_and_caps_target_rate():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    controller=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4))
    for t in np.arange(3,4,.02):controller.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0])
    previous=np.ones(12)
    for t in np.arange(4,7,.02):
        a,trace=controller.step(t,np.ones(12),q,[1,0,0,0],[0,0,1],[0,0,0])
        assert np.max(abs((a-previous)*np.tile([.08,.14,.14],4)))<=.01200001
        previous=a
    np.testing.assert_allclose(a,0,atol=1.e-12)


def test_invalid_gain_and_missing_capture_fail_explicitly():
    with pytest.raises(ValueError):BalanceOptions(pitch_gain=float('nan'))
    c=PartialSupportControl(BalanceOptions())
    with pytest.raises(RuntimeError):c.step(16,np.zeros(12),np.zeros(16),[1,0,0,0],[0,0,1],[0,0,0])


def test_contact_feedforward_compensates_added_leg_load_with_original_servo():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    options=BalanceOptions(capture_start_s=3,capture_end_s=4,leg_force_gain=1,max_joint_delta_rad=.5)
    c=PartialSupportControl(options);zero=np.zeros((4,3));force=zero.copy();force[2:]=[10,0,8]
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],zero)
    for t in np.arange(4,7,.02):a,trace=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],force)
    # Actual target offset through the original servo counteracts added ground wrench.
    tau=80*a*np.tile([.08,.14,.14],4)
    external=np.concatenate([J.T@f for J,f in zip(c.jacobians,force)])
    np.testing.assert_allclose(tau+external,0,atol=1.e-8)
    assert np.linalg.norm(tau)>1


def test_full_wrench_matches_ground_contact_jacobian_instead_of_wheel_center_only():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    options=BalanceOptions(capture_start_s=3,capture_end_s=4,leg_force_gain=1,contact_moment_gain=1,max_joint_delta_rad=.5)
    c=PartialSupportControl(options);zero=np.zeros((4,3));force=zero.copy();force[2:]=[10,0,8]
    state=c.model.state(np.zeros(3),[1,0,0,0],q)
    A,points,centers=c.model.contacts(state,np.array([0.,0.,1.]))
    moment=np.cross(points-centers,force)
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],zero,zero)
    for t in np.arange(4,7,.02):action,_=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],force,moment)
    tau=80*action*np.tile([.08,.14,.14],4)
    np.testing.assert_allclose(tau+A[:,6:18].T@force.ravel(),0,atol=1.e-8)


def test_actor_reference_retains_live_leg_response_with_zero_compensation():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    c=PartialSupportControl(BalanceOptions(reference_mode='actor',capture_start_s=3,capture_end_s=4))
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0])
    for t in np.arange(4,7,.02):action,_=c.step(t,np.full(12,.25),q,[1,0,0,0],[0,0,1],[0,0,0])
    np.testing.assert_allclose(action,.25)


def test_hip_force_can_be_disabled_without_disabling_sagittal_compensation():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    c=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4,
        leg_force_gain=1,hip_force_scale=0,max_joint_delta_rad=.5))
    zero=np.zeros((4,3));force=np.tile([10.,5.,8.],(4,1))
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],zero)
    for t in np.arange(4,7,.02):action,trace=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],force)
    np.testing.assert_allclose(action[::3],0,atol=1.e-12)
    assert np.linalg.norm(trace['balance_force_delta_rad'])>0


def test_joint_feedback_corrects_deflection_and_integral_is_bounded():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    c=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4,
        joint_feedback_gain=1,joint_integral_gain=3,max_joint_delta_rad=.5))
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0])
    moved=q.copy();moved[7]-=.1
    for t in np.arange(4,12,.02):action,trace=c.step(t,np.zeros(12),moved,[1,0,0,0],[0,0,1],[0,0,0])
    assert action[7]>0
    assert abs(c.integral).max()<=.4
    assert abs(trace['balance_leg_delta_rad']).max()<=.5+1.e-12
    for args in [dict(hip_force_scale=1.1),dict(joint_integral_gain=-1),dict(joint_feedback_gain=5)]:
        with pytest.raises(ValueError):BalanceOptions(**args)


def test_lowering_and_hip_damping_have_distinct_signed_effects():
    centers=np.array([[.3,-.2,-.3],[.3,.2,-.3],[-.3,-.2,-.3],[-.3,.2,-.3]])
    shift=foot_adjustments(centers,np.zeros(2),np.zeros(2),BalanceOptions(body_lower_m=.03))
    np.testing.assert_allclose(shift[:,2],.03)
    np.testing.assert_allclose(shift[:,:2],0)
    wide=foot_adjustments(centers,np.zeros(2),np.zeros(2),BalanceOptions(stance_widen_m=.04))
    np.testing.assert_allclose(wide[:,1],[-.04,.04,-.04,.04])
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    c=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4,hip_damping_s=.08))
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0])
    qd=np.zeros(16);qd[[0,3,6,9]]=[.1,-.1,.2,-.2]
    for t in np.arange(4,6,.02):action,_=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],joint_velocity=qd)
    np.testing.assert_allclose(action[::3]*.08,-.08*qd[:12:3],atol=1.e-10)
    np.testing.assert_allclose(action.reshape(4,3)[:,1:],0,atol=1.e-10)


def test_paired_feedforward_retains_common_load_without_differential_sagittal_targets():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    c=PartialSupportControl(BalanceOptions(capture_start_s=3,capture_end_s=4,
        leg_force_gain=1,hip_force_scale=0,force_pair_average=1,max_joint_delta_rad=.5))
    zero=np.zeros((4,3));force=zero.copy();force[[0,2]]=[20,0,10]
    for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],zero)
    for t in np.arange(4,6,.02):action,trace=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],force)
    delta=trace['balance_force_delta_rad'].reshape(4,3)
    np.testing.assert_allclose(delta[0,1:],delta[1,1:])
    np.testing.assert_allclose(delta[2,1:],delta[3,1:])
    assert abs(delta).max()>.01


def test_calibrated_load_feedforward_does_not_follow_contact_pulses():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    opt=BalanceOptions(capture_start_s=3,capture_end_s=4,leg_force_gain=1,hip_force_scale=0)
    slope=np.tile([0,.001,.0005],4)
    calibration=dict(joint_delta_rad_per_N=slope,joint_delta_intercept_rad=np.zeros(12))
    controllers=[PartialSupportControl(opt,calibration) for _ in range(2)]
    for c in controllers:
        for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],np.zeros((4,3)))
    actions=[]
    for index,c in enumerate(controllers):
        for t in np.arange(4,6,.02):
            a,trace=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],
                np.full((4,3),100*index),known_load_N=100)
        actions.append(a)
        np.testing.assert_allclose(trace['balance_force_delta_rad'],100*slope)
    np.testing.assert_allclose(actions[0],actions[1])
    with pytest.raises(ValueError):PartialSupportControl(opt,dict(joint_delta_rad_per_N=[1],joint_delta_intercept_rad=[0]))


def test_force_feedforward_inverts_actual_stiffness_without_changing_requested_torque():
    q=np.array([0,.6,-1.5,0,.6,-1.5,0,-.6,1.5,0,-.6,1.5,0,0,0,0.])
    opt=BalanceOptions(capture_start_s=3,capture_end_s=4,leg_force_gain=1,max_joint_delta_rad=.5)
    values=[];force=np.tile([10.,0.,10.],(4,1))
    for gain in [80,160]:
        c=PartialSupportControl(opt,servo_stiffness=np.full(12,gain))
        for t in np.arange(3,4,.02):c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],np.zeros((4,3)))
        for t in np.arange(4,7,.02):action,_=c.step(t,np.zeros(12),q,[1,0,0,0],[0,0,1],[0,0,0],force)
        values.append(gain*action*np.tile([.08,.14,.14],4))
    np.testing.assert_allclose(values[0],values[1],atol=1.e-8)


def test_large_posture_ik_matches_requested_wheel_centers_inside_original_joint_bounds():
    m=Model();q=np.array([0,.57,-1.63,0,.56,-1.64,0,-.61,1.35,0,-.61,1.35,0,0,0,0.])
    s=m.state(np.zeros(3),[1,0,0,0],q)
    centers=np.array([s['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS])
    offset=np.tile([-.1,0,.12],(4,1));goal=solve_foot_pose(m,q,centers,offset)
    s=m.state(np.zeros(3),[1,0,0,0],goal)
    actual=np.array([s['trans'][leg+'_wheel_link'][:3,3] for leg in LEGS])
    np.testing.assert_allclose(actual,centers+offset,atol=1.e-6)
    assert np.all(goal[:12]>=m.limits[:12,0]) and np.all(goal[:12]<=m.limits[:12,1])


def test_desired_geometry_support_torque_matches_world_contact_virtual_work():
    m=Model();q=np.array([0,.57,-1.63,0,.56,-1.64,0,-.61,1.35,0,-.61,1.35,0,0,0,0.])
    quat=np.roll(Rotation.from_euler('xyz',[.04,-.4,.03]).as_quat(),1)
    s=m.state(np.zeros(3),quat,q);R=s['trans']['Base_link'][:3,:3]
    A,points,centers=m.contacts(s,np.array([0.,0.,1.]))
    world_force=np.array([[30,1,20],[28,-1,22],[50,2,40],[52,-2,42.]])
    world_moment=np.cross(points-centers,world_force)
    tau=support_joint_torque(m,s,world_force@R,world_moment@R)
    _,g=m.dynamics(s)
    np.testing.assert_allclose(tau,g[6:18]-A[:,6:18].T@world_force.ravel(),atol=1.e-10)
