"""验证重置初始接触与真实物理步接触的区别。"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from capture import judge


def trajectory():
    t=np.arange(4001)*.001; u=t-1.; duration=2*1.4/9.81
    air=(u>0)&(u<duration)
    h=np.where(air,1.4*u-4.905*u*u,0.)
    vz=np.where(air,1.4-9.81*u,0.)
    com=np.zeros((len(t),3)); com[:,2]=.33+h
    vel=np.zeros_like(com); vel[:,2]=vz
    force=np.repeat(np.where(air,0.,30.)[:,None],4,axis=1)
    clear=np.repeat(np.where(air,h+.02,0.)[:,None],4,axis=1)
    return dict(t=t,com=com,com_velocity=vel,wheel_force_z=force,wheel_clearance=clear,
        tilt=np.zeros(len(t)),base_height=.416+h,body_force=np.zeros(len(t)),
        applied_torque=np.ones((len(t),16))*20.)


def test_pre_reset_contact_buffer_does_not_count_as_new_episode_collision():
    a=trajectory(); a['body_force'][0]=100.
    assert judge(a,'timeout')['success']


def test_contact_in_first_completed_physics_step_is_rejected():
    a=trajectory(); a['body_force'][1]=10.
    result=judge(a,'timeout')
    assert result['failed'] and not result['success']


def test_late_nonfinite_sample_is_rejected():
    a=trajectory(); a['com'][-1,2]=np.nan
    assert not judge(a,'timeout')['success']
