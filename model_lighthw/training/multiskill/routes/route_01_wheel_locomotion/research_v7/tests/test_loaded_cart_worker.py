import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loaded_cart_worker import cart_root_from_preload,retarget_preload,forwarded_ground_friction


@pytest.mark.parametrize('arguments,expected',[
    ([],3.),(['--device','cuda:0','--ground-friction','4'],4.),
    (['--ground-friction=4'],4.),(['--ground-friction','3','--ground-friction=4'],4.),
    (['--ground-fric','4'],4.)])
def test_forwarded_friction_keeps_controller_and_physics_parser_in_sync(arguments,expected):
    assert forwarded_ground_friction(arguments)==expected


def test_cart_preload_sets_physical_spring_extension_without_changing_support_height():
    angle=np.deg2rad(25);tangent=np.array([np.cos(angle),0,np.sin(angle)])
    normal=np.array([-np.sin(angle),0,np.cos(angle)]);origin=np.array([0.,70.,0.])
    robot_point=origin-.4*tangent+.065*normal;local=.35*tangent-.23*normal
    tension=40*9.81*np.sin(angle);center=cart_root_from_preload(robot_point,local,tangent,normal,.3,tension/2000,origin)
    delta=robot_point-(center+local)
    assert center@normal==pytest.approx(.3)
    assert delta@tangent>0
    assert 2000*(np.linalg.norm(delta)-1)==pytest.approx(tension)
    assert .5*2000*(np.linalg.norm(delta)-1)**2<7.
    # The initial spring stores far less than the gravitational work of lifting the cart 1m uphill.
    assert 40*9.81*np.sin(angle)>160


def test_unreachable_hitch_height_is_rejected():
    with pytest.raises(ValueError):cart_root_from_preload(np.array([0,0,3.]),np.zeros(3),np.array([1,0,0]),np.array([0,0,1]),.3,.1,np.zeros(3))


def test_preload_retarget_preserves_torque_and_recovers_front_joint_target_margin():
    default=np.array([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5]);scale=np.tile([.08,.14,.14],4)
    q=default.copy();q[[1,4]]=1.36
    old=np.full(12,80.);new=np.tile([80.,160.,160.],4)
    target=q.copy();target[[1,4]]=1.56;action=(target-default)/scale
    corrected=retarget_preload(action,q,old,new);new_target=default+corrected*scale
    np.testing.assert_allclose(new*(new_target-q),old*(target-q),atol=1.e-12)
    np.testing.assert_allclose(new_target[[1,4]],[1.46,1.46])
    with pytest.raises(ValueError):retarget_preload(action,q,old,np.zeros(12))
