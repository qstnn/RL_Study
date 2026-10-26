import json
from pathlib import Path
import numpy as np
from entry_path import build_path,DEFAULT_Q
from real_cart_audit import movement_mask


def test_preparation_past_three_seconds_never_becomes_a_move_window():
    time=np.arange(1,5101)*.001
    selected=movement_mask(time,None,None)
    position=-time**2
    assert selected.sum()==1
    assert np.ptp(position[selected])==0.
    moving=movement_mask(time,3.5,4.5)
    assert time[moving][0]==3.5 and time[moving][-1]<4.5


def test_crouch_first_path_lowers_without_initial_horizontal_support_shift():
    from model import Model
    from research_common import LEGS
    r=json.loads((Path(__file__).parent/'fixtures/loaded_entry_reference.json').read_text())
    plan=build_path(r,25.,1.2,crouch_first=True);model=Model()
    a=np.deg2rad(25.);tangent=np.array([np.cos(a),0,np.sin(a)]);normal=np.array([-np.sin(a),0,np.cos(a)])
    assert abs(np.array(plan['body_position_plan_m'])[15]@normal-.26)<1.e-9
    wheels=[]
    for i in [0,5,10,15]:
        state=model.state(plan['body_position_plan_m'][i],plan['body_quaternion_plan'][i],plan['q'][i])
        wheels.append([state['trans'][leg+'_wheel_link'][:3,3]@tangent for leg in LEGS])
    assert np.max(np.ptp(wheels,axis=0))<1.e-6
    np.testing.assert_allclose(plan['q'][0][:12],DEFAULT_Q[:12],atol=1.e-6)
    np.testing.assert_allclose(plan['q'][-1][:12],r['q'][:12],atol=1.e-6)
    assert plan['maximum_ik_error_m']<1.e-6 and plan['maximum_tangent_rolling_residual_mps']<1.e-8
