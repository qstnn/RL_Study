import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loaded_cart_worker import entry_fraction,cart_root_from_preload,entry_wheel_velocity
from wheel_load_control import allocated_targets
from model import Model


def test_entry_starts_at_default_target_and_enters_without_endpoint_jump():
    assert entry_fraction(0,1)==0
    assert entry_fraction(1,1)==1
    values=np.array([entry_fraction(t,1) for t in np.linspace(0,1,1001)])
    assert np.all(np.diff(values)>=0)
    assert values[1]<1.e-7 and 1-values[-2]<1.e-7


def test_downhill_cart_push_requires_braking_and_compressed_spring():
    slope=np.deg2rad(-20);load=40*9.81*np.sin(slope)
    out=allocated_targets(.2,.2,load,[30.,30.,20.,20.],11.668111,slope,3.,uniform=True,signed_load=True)
    expected=(40+11.668111)*9.81*np.sin(slope)
    assert out['force_N'].sum()==pytest.approx(expected)
    assert np.all(out['torque_feedforward_Nm']<0)
    with pytest.raises(ValueError):allocated_targets(.2,.2,load,[30.,30.,20.,20.],11.668111,slope,3.)
    tangent=np.array([np.cos(slope),0,np.sin(slope)]);normal=np.array([-np.sin(slope),0,np.cos(slope)])
    robot=-.38*tangent+.137*normal;cart_local=.35*tangent-.26*normal
    center=cart_root_from_preload(robot,cart_local,tangent,normal,.3,load/2000,np.zeros(3))
    assert 2000*(np.linalg.norm(robot-center-cart_local)-1)==pytest.approx(load)


def test_entry_leg_and_wheel_velocity_satisfy_tangent_rolling_constraint():
    model=Model();default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5,0.,0.,0.,0.])
    reference=default.copy();reference[:12]+=np.tile([0.,.15,-.1],4)
    time=.2;duration=.6;quat=np.array([1.,0,0,0]);tangent=np.array([1.,0,0]);normal=np.array([0.,0,1.])
    wheels=entry_wheel_velocity(model,time,duration,reference,quat,tangent,normal)
    q=default+entry_fraction(time,duration)*(reference-default)
    epsilon=1.e-6;qd=(entry_fraction(time+epsilon,duration)-entry_fraction(time-epsilon,duration))/(2*epsilon)*(reference-default)
    qd[12:]=wheels;A,_,_=model.contacts(model.state(np.zeros(3),quat,q),normal)
    np.testing.assert_allclose((A@np.r_[np.zeros(6),qd]).reshape(4,3)@tangent,0,atol=1.e-9)
    np.testing.assert_allclose(entry_wheel_velocity(model,0,duration,reference,quat,tangent,normal),0)
    np.testing.assert_allclose(entry_wheel_velocity(model,duration,duration,reference,quat,tangent,normal),0)


def test_three_dimensional_attitude_correction_has_consistent_rotation_sign():
    from partial_support_control import foot_adjustments,BalanceOptions
    centers=np.array([[.2,-.13,-.25],[.2,.13,-.25],[-.2,-.13,-.25],[-.2,.13,-.25]])
    error=np.array([.08,.06])
    options=BalanceOptions(pitch_gain=1.,roll_gain=1.,horizontal_attitude_gain=1.)
    actual=foot_adjustments(centers,error,np.zeros(2),options)
    # Positive pitch means nose up (-world y); positive roll means left up (+x).
    # Restore body angle by [-roll,+pitch,0]; fixed ground feet move oppositely in body frame.
    expected=-np.cross(np.array([-error[1],error[0],0.]),centers)
    np.testing.assert_allclose(actual,expected,atol=1.e-12)


def reference_fixture():
    import json
    return json.loads((Path(__file__).parent/'fixtures/loaded_entry_reference.json').read_text())


def test_cartesian_entry_stays_reachable_and_wheel_targets_match_ground_path():
    from entry_path import build_path,DEFAULT_Q
    from research_common import LEGS
    reference=reference_fixture();path=build_path(reference,25.,.6,forward_m=.16)
    q=np.array(path['q']);qd=np.array(path['joint_velocity_rad_s'])
    np.testing.assert_allclose(q[0,:12],DEFAULT_Q[:12],atol=1.e-6)
    np.testing.assert_allclose(q[-1,:12],reference['q'][:12],atol=1.e-6)
    np.testing.assert_allclose(qd[[0,-1]],0)
    assert path['maximum_ik_error_m']<1.e-6 and path['maximum_tangent_rolling_residual_mps']<1.e-9
    model=Model();normal=np.array([-np.sin(np.deg2rad(25)),0,np.cos(np.deg2rad(25))])
    for i in range(len(q)):
        assert np.all(q[i,:12]>=model.limits[:12,0]) and np.all(q[i,:12]<=model.limits[:12,1])
        state=model.state(path['body_position_plan_m'][i],path['body_quaternion_plan'][i],q[i])
        height=np.array([state['trans'][leg+'_wheel_link'][:3,3]@normal for leg in LEGS])
        assert height.min()>.08 and height.max()<.10


def test_static_preload_matches_wheel_share_and_preserves_original_torque_limits():
    from static_preload import rebase,mirror_posture
    from scipy.spatial.transform import Rotation
    reference=reference_fixture()
    result=rebase(reference,25.,40.,.04,.2,np.full(12,80.),3.,front_share=.3)
    audit=result['static_rebase'];tangent=np.array(audit['tangent_N'])
    np.testing.assert_allclose(tangent/tangent.sum(),[.15,.15,.35,.35],atol=1.e-8)
    assert audit['residual_N_Nm']<1.e-6
    assert np.all(np.abs(audit['joint_torque_Nm'])<=np.r_[np.full(12,96),np.full(4,48)])
    reference['quat']=(Rotation.from_euler('y',45,degrees=True)*Rotation.from_quat(np.array(reference['quat'])[[1,2,3,0]])).as_quat()[[3,0,1,2]].tolist()
    with pytest.raises(ValueError,match='static wrench feasibility'):rebase(reference,-20.,40.,.04,.2,np.full(12,80.),3.)
    downhill=rebase(mirror_posture(reference,-20.),-20.,40.,.04,.2,np.full(12,80.),3.)
    assert sum(downhill['static_rebase']['tangent_N'])<0
    assert downhill['static_rebase']['residual_N_Nm']<1.e-6
