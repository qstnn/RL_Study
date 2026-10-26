"""筛查反例：站立、出生跌落、假腾空和落地后失败不能算成功。"""
from pathlib import Path
import sys
import math
import pytest
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'offline/routes.json').exists())
sys.path.insert(0,str(ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw'))
from r5a.protocol import JumpState,reference_pose
from r5a.spec import CONTRACT


def simulate(kind,dt=.02,state=None):
    s=state or JumpState(1); s.reset([0],torch.tensor([.33]))
    v0=.4 if kind=='short_hop' else 1.4
    flight=2*v0/9.81
    scalar=lambda x:torch.tensor([x],dtype=torch.float32)
    for k in range(round(4/dt)+1):
        t=k*dt; u=t-1.
        flying=0<u<flight and kind!='stand'
        h=max(0.,v0*u-4.905*u*u) if flying else 0.
        vz=v0-9.81*u if flying else 0.
        if kind=='fake_lift': h=vz=0.
        if kind=='spawn_drop':
            flying=t<.3; h=max(0.,.04-.5*9.81*t*t); vz=-9.81*t if flying else 0.
        force=[0.]*4 if flying else [30.]*4
        if kind=='one_wheel_landing' and t>1+flight: force=[120.,0.,0.,0.]
        contact=10. if kind=='late_collision' and t>=3. else 0.
        if kind=='between_policy_steps' and 2.005<=t<2.007: contact=10.
        s.update(0. if k==0 else dt,scalar(t),scalar(.33+h),scalar(vz),torch.tensor([force]),
            torch.tensor([[h+.02]*4 if flying else [0.]*4]),scalar(0.),scalar(.416+h),scalar(0.),
            scalar(0.),scalar(contact),scalar(36.))
    return s


@pytest.mark.parametrize('dt',[.02,.001])
def test_complete_jump_and_hold(dt):
    s=simulate('valid',dt)
    assert s.success.item() and not s.failed.item()
    assert .095<float(s.peak_rise)<.105
    assert .20<float(s.flight_time)<.29


@pytest.mark.parametrize('kind',['stand','spawn_drop','fake_lift','short_hop','one_wheel_landing','late_collision'])
def test_incomplete_or_failed_behavior_rejected(kind):
    assert not simulate(kind).success.item()


def test_physics_trace_catches_contact_between_policy_steps():
    assert not simulate('between_policy_steps',.001).success.item()


def test_reset_cannot_inherit_previous_success():
    s=simulate('valid'); assert s.success.item()
    s=simulate('stand',state=s)
    assert not s.success.item() and not s.launched.item() and s.peak_rise.item()==0.


def test_reference_is_continuous_at_phase_boundaries():
    for edge in CONTRACT['reference_times_s'][1:-1]:
        values=reference_pose(torch.tensor([edge-1e-6,edge+1e-6]))
        assert torch.max(torch.abs(values[1]-values[0]))<1e-4


def test_batch_environments_do_not_share_success_state():
    s=JumpState(2)
    s.success[0]=True; s.launched[0]=True; s.peak_rise[0]=.1
    s.reset([1],torch.tensor([.3]))
    assert s.success.tolist()==[True,False] and s.peak_rise.tolist()[1]==0.
