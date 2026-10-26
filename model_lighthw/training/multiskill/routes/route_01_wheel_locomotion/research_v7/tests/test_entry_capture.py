import numpy as np
from entry_capture import TorqueCapture


def call(controller,time,speed,normal_target=None):
    return controller.step(time,np.full(4,-20. if speed else 0.),np.full(4,5.),np.zeros(4) if normal_target is None else normal_target,
        speed,.282,.282,np.zeros(2),np.zeros(2),np.zeros(3),0.,np.full(4,30.))


def test_fast_downhill_wheels_get_declared_torque_and_do_not_switch_to_zero_speed():
    c=TorqueCapture(.6,.2)
    target,trace=call(c,.6,-1.7)
    np.testing.assert_array_equal(target,-20.)
    for time in np.arange(.8,5.,.02):
        target,trace=call(c,time,-1.7)
        np.testing.assert_allclose(2*(target+20.),5.)
        assert not c.finished and trace['entry_capture_handoff']==0.


def test_handoff_requires_continuous_settling_and_reaches_nominal_without_jump():
    c=TorqueCapture(.6,.2)
    for time in np.arange(.8,1.1,.02):call(c,time,0.)
    call(c,1.1,.2)  # 破坏连续稳定窗，不能累计间断的短暂过零。
    for time in np.arange(1.12,1.5,.02):
        target,trace=call(c,time,0.)
        assert trace['entry_capture_handoff']==0.
    values=[]
    for time in np.arange(1.5,2.7,.02):
        target,trace=call(c,time,0.);values.append(target[0])
    assert c.finished and values[-1]==0.
    assert np.max(np.abs(np.diff(values)))<.7


def test_loaded_spinning_wheels_cannot_complete_capture_but_partial_support_is_allowed():
    c=TorqueCapture(.6,.2)
    for time in np.arange(.8,2.,.02):
        _,trace=c.step(time,np.full(4,-20.),np.full(4,5.),np.zeros(4),0.,.282,.282,
            np.zeros(2),np.zeros(2),np.zeros(3),0.,np.full(4,30.))
        assert not c.finished and trace['entry_capture_ready_s']==0.
    for time in np.arange(2.,4.,.02):
        _,trace=c.step(time,np.array([-20.,-20.,0.,0.]),np.full(4,5.),np.zeros(4),0.,.282,.282,
            np.zeros(2),np.zeros(2),np.zeros(3),0.,np.array([0.,0.,60.,60.]))
    assert c.finished  # 只要求实际承载轮已捕获，没有增加四轮接地门。


def test_body_anchor_does_not_recirculate_a_lifted_wheel_runaway():
    c=TorqueCapture(0.,.2,velocity_reference='body')
    outputs=[]
    for spin in (0.,35.,-35.):
        wheel=np.array([spin,spin,-.5/.0875,-.5/.0875])
        target,_=c.step(.2,wheel,np.array([0.,0.,8.,8.]),np.zeros(4),-.5,.282,.282,
            np.zeros(2),np.zeros(2),np.zeros(3),0.,np.array([0.,0.,60.,60.]))
        outputs.append(target)
    np.testing.assert_array_equal(outputs[0],outputs[1])
    np.testing.assert_array_equal(outputs[0],outputs[2])
    np.testing.assert_allclose(outputs[0],-.5/.0875+np.array([0.,0.,4.,4.]))
    assert not c.finished


def test_ramp_reference_does_not_follow_growing_rollback():
    c=TorqueCapture(0.,.02,velocity_reference='ramp',capture_acceleration=.1)
    def sample(time,speed):
        return c.step(time,np.full(4,speed/.0875),np.zeros(4),np.zeros(4),speed,.282,.282,
            np.zeros(2),np.zeros(2),np.zeros(3),0.,np.full(4,30.))[0]
    np.testing.assert_allclose(sample(0.,-.5),-.5/.0875)
    np.testing.assert_allclose(sample(1.,-1.),-.4/.0875)
    np.testing.assert_allclose(sample(5.,-2.),0.)
    assert not c.finished
