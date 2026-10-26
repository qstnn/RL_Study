"""真实失败前态上的接口/约束核查；不把离线可解当成成功教师。"""
import json
from pathlib import Path
import numpy as np
import pytest
from coordinated_capture import CoordinatedCapture, CaptureHandoff
from discrete_servo_qp import predict
from entry_path import DEFAULT_Q


def fixture(index):
    with np.load(Path(__file__).parent/'fixtures/coord_real_states.npz') as p:
        return ({k:p[k][index] for k in p.files if k not in ('reference_json','nominal')},
                p['nominal'][index],json.loads(p['reference_json'].item()))


@pytest.mark.parametrize('index',[0,1,2])
@pytest.mark.parametrize('options',[{},dict(servo_discrete=True),dict(nominal_hips=True),dict(leg_trust_rad=.05)])
def test_executed_interface_and_physical_constraints_at_real_failure_states(index,options):
    z,nominal,ref=fixture(index);c=CoordinatedCapture(ref,25.,**options)
    action,r,d=c.compute(z,nominal)
    assert r['success'],r
    scale=np.tile([.08,.14,.14],4)
    target=DEFAULT_Q[:12]+action[:12]*scale
    tau=np.r_[80*(target-z['q'][:12])-3*z['v'][6:18],2*(action[12:]*5-z['v'][18:])]
    np.testing.assert_allclose(tau,r['initial_servo_torque'],atol=1.e-9)
    assert np.all(target>=c.model.limits[:12,0]-1.e-6)
    assert np.all(target<=c.model.limits[:12,1]+1.e-6)
    assert np.all(abs(tau)<=np.r_[np.full(12,96.),np.full(4,48.)]+1.e-4)
    assert np.max(abs(action[12:]*5))<=50.+1.e-6
    forces=r['contact_forces']@c.B
    assert np.min(forces[:,2])>=-1.e-4
    assert np.all(np.linalg.norm(forces[:,:2],axis=1)<=3*forces[:,2]+1.e-4)
    np.testing.assert_allclose(forces[~d['mask']],0.,atol=1.e-4)
    if options.get('nominal_hips'):
        np.testing.assert_allclose(action[[0,3,6,9]],nominal[[0,3,6,9]],atol=1.e-7)
    if options.get('leg_trust_rad'):
        assert np.max(abs((action[:12]-nominal[:12])*scale))<=.05+1.e-6


def test_discrete_affine_prediction_matches_direct_constrained_servo_execution():
    # 独立KKT逐步计算：固定M/接触的诊断模型，原始PD每1ms更新。
    z,nominal,ref=fixture(0);c=CoordinatedCapture(ref,25.)
    state=c.model.state(z['pos'],z['quat'],z['q']);A,_,_=c.model.contacts(state,c.B[:,2])
    adot=c.model.contact_derivative(z['pos'],z['quat'],z['q'],z['v'],c.B[:,2])@z['v']
    mask=np.ones(4,dtype=bool);p=predict(z['M'],z['h'],A,adot,z['v'],z['external'],A,mask)
    _,r,_=c.compute(z,nominal);tau0=r['initial_servo_torque'];ax=np.r_[tau0,1.]
    target_q=z['q'][:12]+(tau0[:12]+3*z['v'][6:18])/80
    target_w=z['v'][18:]+tau0[12:]/2
    dx=np.zeros(22);dv=np.zeros(22)
    KKT=np.block([[z['M'],-A.T],[A,np.zeros((12,12))]])
    for step in range(20):
        tau=np.r_[80*(target_q-z['q'][:12]-dx[6:18])-3*(z['v'][6:18]+dv[6:18]),2*(target_w-z['v'][18:]-dv[18:])]
        rhs=np.r_[np.r_[np.zeros(6),tau]+z['external']-z['h'],-adot-8*A@(z['v']+dv)]
        solved=np.linalg.solve(KKT,rhs);dv+=.001*solved[:22];dx+=.001*(z['v']+dv)
        for key,actual in [('tau',tau),('force',solved[22:]),('acc',solved[:22]),('dx',dx),('dv',dv)]:
            np.testing.assert_allclose(p[key][step]@ax,actual,atol=1.e-8)


def test_discrete_horizon_rejects_incompatible_configuration():
    _,_,ref=fixture(0)
    with pytest.raises(ValueError,match='20ms'):
        CoordinatedCapture(ref,25.,servo_horizon=0.,servo_discrete=True)


def test_failed_capture_state_during_handoff_cannot_finish_by_elapsed_time():
    # 实际CPU倒滑在1.22~1.80s合格，1.82s后失稳；旧混合仍在2.60s结束。
    h=CaptureHandoff();blends=[]
    for i in range(150):
        time=i*.02;blends.append(h.advance(1.22<=time<=1.80,time))
    assert max(blends)>0.
    assert not h.finished
    assert blends[-1]==0.


def test_handoff_needs_requalification_and_hold_only_never_moves():
    h=CaptureHandoff();hold=CaptureHandoff(disabled=True)
    for i in range(30):h.advance(True,i*.02)
    for i in range(50):h.advance(False,(i+30)*.02)
    assert h.progress==0. and not h.armed
    for i in range(19):assert h.advance(True,(i+80)*.02)==0.
    for i in range(75):
        h.advance(True,(i+99)*.02)
        assert hold.advance(True,i*.02)==0.
    assert h.finished and not hold.finished
