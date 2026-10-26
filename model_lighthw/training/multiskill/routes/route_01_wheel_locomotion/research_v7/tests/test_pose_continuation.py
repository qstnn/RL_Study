import json
from pathlib import Path
import numpy as np
import pytest
from entry_path import build_path
from model import Model
from research_common import LEGS


def reference():
    return json.loads((Path(__file__).parent/'fixtures/loaded_entry_reference.json').read_text())


@pytest.mark.parametrize('progress',[0.,.25,.5,.75,.9,1.])
def test_continuation_keeps_ground_geometry_and_finishes_at_loaded_pose(progress):
    r=reference();path=build_path(r,25.,1.,start_progress=progress,preserve_wheel_phase=True)
    normal=np.array([-np.sin(np.deg2rad(25)),0,np.cos(np.deg2rad(25))])
    height=np.asarray(path['body_position_plan_m'])@normal
    assert height[0]==pytest.approx(.416223+progress*(r['height_m']-.416223))
    assert height[-1]==pytest.approx(r['height_m'])
    np.testing.assert_allclose(np.asarray(path['q'])[-1,:12],r['q'][:12],atol=1.e-6)
    np.testing.assert_array_equal(np.asarray(path['q'])[:,12:],np.tile(r['q'][12:],(len(height),1)))
    np.testing.assert_array_equal(np.asarray(path['joint_velocity_rad_s'])[[0,-1]],0)
    model=Model()
    for p,quat,q in zip(path['body_position_plan_m'],path['body_quaternion_plan'],path['q']):
        s=model.state(p,quat,q)
        gaps=np.array([s['trans'][leg+'_wheel_link'][:3,3]@normal-.0875 for leg in LEGS])
        assert gaps.min()>-.005 and gaps.max()<.01
    assert path['maximum_tangent_rolling_residual_mps']<1.e-8


def test_loaded_endpoint_has_no_requested_motion_and_invalid_start_is_rejected():
    r=reference();path=build_path(r,25.,1.,start_progress=1.)
    np.testing.assert_array_equal(path['q'],np.tile(r['q'],(51,1)))
    np.testing.assert_array_equal(path['wheel_rad_s'],0)
    for progress in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError):build_path(r,25.,1.,start_progress=progress)


def test_successful_intermediate_resets_never_satisfy_ordinary_entry_gate():
    from review_entry_gate import readiness
    rows=[dict(run='continuation',slope_deg=s,device='cuda:0',initial_state='continuation',
        complete_teacher_pass=True,source_valid=True) for s in [25.,-20.]]
    result=readiness(rows)
    assert not result['ready_for_teacher_collection']
    assert result['missing']==['uphill_cuda_ordinary','downhill_cuda_ordinary']


def test_initial_velocity_is_rolling_and_does_not_move_the_leg_joints():
    from entry_initial_velocity import rolling_joint_velocity
    r=reference();slope=np.deg2rad(25);normal=np.array([-np.sin(slope),0,np.cos(slope)]);tangent=np.cross([0,1,0],normal)
    model=Model();A,_,_=model.contacts(model.state(normal*r['height_m'],r['quat'],r['q']),normal)
    for speed in [0.,-.5,-1.5]:
        qd=rolling_joint_velocity(r,25.,speed)
        np.testing.assert_array_equal(qd[:12],0.)
        np.testing.assert_allclose((A@np.r_[speed*tangent,np.zeros(3),qd]).reshape(4,3)@tangent,0.,atol=1.e-10)
