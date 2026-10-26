import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rolling_load_audit import rolling_metrics
from real_cart_audit import cart_rolling_with_partial_contact,drawbar_errors


def test_intermittent_cart_contact_allowed_but_robot_protocol_unchanged():
    normal=np.zeros((1000,4));normal[::2,2:]=50.
    center=np.full_like(normal,.2)
    metrics=rolling_metrics(normal,normal>0,center,center,np.zeros_like(normal))
    assert not metrics['rolling_dominates']
    assert metrics['any_loaded_wheel_fraction']==.5
    assert cart_rolling_with_partial_contact(metrics)


def test_cart_still_requires_observed_rolling_under_load():
    normal=np.ones((1000,4))*20.;center=np.full_like(normal,.2)
    for spin in [np.zeros_like(normal),np.full_like(normal,.6)]:
        assert not cart_rolling_with_partial_contact(rolling_metrics(normal,normal>0,center,spin,abs(center-spin)))
    assert not cart_rolling_with_partial_contact(rolling_metrics(normal*0,normal*0,center,center,normal*0))


def test_independent_spring_audit_detects_extra_assistance_and_wrong_moment():
    # 0.1m extension and 0.2m/s extension rate require 200+30N.
    force=np.array([[230.,0.,0.]])
    robot_arm=np.array([[-.38,0.,-.15]]);cart_arm=np.array([[.35,0.,-.26]])
    values=dict(robot_hitch_pre_w=[[1.1,0,0]],cart_hitch_pre_w=[[0,0,0]],
        robot_hitch_velocity_pre_w=[[.2,0,0]],cart_hitch_velocity_pre_w=[[0,0,0]],
        robot_arm_pre_w=robot_arm,cart_arm_pre_w=cart_arm,
        robot_force_w_N=-force,robot_torque_w_Nm=np.cross(robot_arm,-force),
        cart_torques_w_Nm=np.cross(cart_arm,force)[:,None,:])
    z={'real_cart_'+k:np.array(v) for k,v in values.items()}
    assert max(drawbar_errors(z).values())<1.e-10
    z['real_cart_robot_force_w_N'][0,0]+=10
    assert drawbar_errors(z)['spring_force_max_N']>9.99
    z['real_cart_robot_torque_w_Nm']*=0
    assert drawbar_errors(z)['robot_arm_torque_max_Nm']>30
