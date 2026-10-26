import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wheel_load_control import allocated_targets,heading_velocity_offset


def test_rolling_servo_delivers_requested_gravity_and_load_balance():
    mass=11.66811;slope=np.deg2rad(25.)
    normals=np.array([10.,5.,45.,43.7])
    result=allocated_targets(.2,.2,60.,normals,mass,slope,3.)
    # Independent replay through the original velocity servo at pure rolling speed.
    torque=2*(result['target_rad_s']-.2/.0875)
    np.testing.assert_allclose(torque/.0875,result['force_N'],atol=1.e-12)
    assert (torque/.0875).sum()==pytest.approx(mass*9.81*np.sin(slope)+60)
    np.testing.assert_allclose(result['force_N']/normals,np.repeat(result['force_N'].sum()/normals.sum(),4))


def test_unloaded_wheel_gets_no_force_and_friction_margin_is_respected():
    normal=np.array([0.,5.,45.,43.7])
    r=allocated_targets(.2,-2.,250.,normal,11.66811,np.deg2rad(25.),3.)
    assert r['force_N'][0]==0
    assert np.all(abs(r['force_N'])<=.85*3.*normal+1.e-10)
    assert np.all(abs(r['target_rad_s'])<=50.)


def test_no_support_or_nonfinite_state_is_rejected():
    for n in ([0,0,0,0],[float('nan'),20,30,40]):
        with pytest.raises(ValueError):allocated_targets(.2,.2,30,n,11.66811,np.deg2rad(25.),3.)


def test_fixed_axle_distribution_is_independent_of_left_right_contact_pulses():
    for normals in ([10,15,40,40],[15,10,50,30]):
        r=allocated_targets(.2,.2,90,normals,11.66811,np.deg2rad(25.),3.,uniform=True,front_share=.3)
        np.testing.assert_allclose(r['weights'],[.15,.15,.35,.35])
        assert r['force_N'][0]==r['force_N'][1]
        assert r['force_N'][2]==r['force_N'][3]
    with pytest.raises(ValueError):allocated_targets(.2,.2,90,[25]*4,11.66811,.4,3.,front_share=1.5)


def test_heading_feedback_requests_a_restoring_turn_and_preserves_mean_speed():
    y=np.array([-.18,.18,-.18,.18])
    offset=heading_velocity_offset(.1,0,y,2.)
    assert offset[0]<0<offset[1]
    assert offset.sum()==pytest.approx(0)
    assert (offset[0]-offset[1])*.0875/.36==pytest.approx(-.2)
    np.testing.assert_allclose(heading_velocity_offset(0,.4,y,2),offset)
