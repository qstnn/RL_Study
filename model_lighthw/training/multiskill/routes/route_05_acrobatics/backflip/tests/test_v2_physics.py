"""真实失败轨迹回归：每个物理步推进一次，去重、重置和终末保持均覆盖。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'offline/routes.json').is_file())
SOURCE=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/r5a'
sys.path.insert(0,str(SOURCE.parent))
from r5a.protocol import JumpState


def method():
    tree=ast.parse((SOURCE/'env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JumpLandingEnv')
    fn=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_advance_protocol'),None)
    assert fn is not None, 'Missing production per-physics-step state update'
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE/'env.py'),'exec'),namespace)
    return namespace['_advance_protocol']


def fake_env(a,n=1):
    env=SimpleNamespace(_protocol_tick=0,physics_dt=.001,physics_finite=torch.tensor(True),
        episode_physics_steps=torch.zeros(n,dtype=torch.long),
        origin_xy=torch.zeros(n,2),state=JumpState(n),
        policy_airborne_steps=torch.zeros(n),policy_peak_support_force=torch.zeros(n),
        mass=torch.ones(n)*11.6681,index=0)
    env.state.reset(list(range(n)),torch.ones(n)*float(a['com'][0,2]))
    def metrics():
        i=env.index
        t=lambda v:torch.as_tensor(np.asarray(v).copy(),dtype=torch.float32).expand(n,*np.shape(v)).clone()
        return dict(com=t(a['com'][i]),com_vel=t(a['com_velocity'][i]),force_z=t(a['wheel_force_z'][i]),
            clearance=t(a['wheel_clearance'][i]),tilt=t(a['tilt'][i]),base_z=t(a['base_height'][i]),
            speed=t(np.linalg.norm(a['com_velocity'][i,:2])),body_force=t(a['body_force'][i]),
            torque_peak=t(np.abs(a['applied_torque'][i]).max()))
    env.metrics=metrics
    return env


def test_real_nine_millisecond_takeoff_is_not_terminated_at_deadline():
    a=dict(np.load(Path(__file__).parent/'fixtures/missed_takeoff.npz',allow_pickle=False))
    env=fake_env(a); advance=method()
    for i in range(len(a['t'])):
        env.index=i
        advance(env,i)
        # _get_dones plus the following _apply_action see the same terminal tick.
        if i%20==0:advance(env,i)
    assert env.episode_physics_steps.item()==1800
    assert env.state.launched.item() and env.state.landed.item()
    assert not env.state.failed.item()
    assert abs(env.state.takeoff_t.item()-1.124)<1e-5
    assert abs(env.state.flight_time.item()-.107)<1e-5


def test_completed_step_collision_is_not_skipped():
    a=dict(np.load(Path(__file__).parent/'fixtures/missed_takeoff.npz',allow_pickle=False))
    a['body_force'][5]=10.
    env=fake_env(a);advance=method()
    for i in range(6):env.index=i;advance(env,i)
    assert env.state.failed.item()


def test_duplicate_and_skipped_ticks():
    a=dict(np.load(Path(__file__).parent/'fixtures/missed_takeoff.npz',allow_pickle=False))
    env=fake_env(a);advance=method()
    advance(env,0);assert env.episode_physics_steps.item()==0
    env.index=1;advance(env,1);advance(env,1)
    assert env.episode_physics_steps.item()==1
    with pytest.raises(RuntimeError):advance(env,3)


def test_subset_reset_does_not_reuse_old_contact_or_reset_other_clock():
    a=dict(np.load(Path(__file__).parent/'fixtures/missed_takeoff.npz',allow_pickle=False))
    env=fake_env(a,2);advance=method()
    for i in range(21):env.index=i;advance(env,i)
    env.episode_physics_steps[0]=0;env.state.reset([0],torch.tensor([.33]))
    advance(env,20)
    assert env.episode_physics_steps.tolist()==[0,20]
    env.index=21;advance(env,21)
    assert env.episode_physics_steps.tolist()==[1,21]


def test_late_drift_revokes_current_success_but_retains_history():
    from test_protocol import simulate
    s=simulate('valid',dt=.001)
    assert s.success.item()
    scalar=lambda x:torch.tensor([x])
    s.update(.001,scalar(4.001),scalar(.33),scalar(0.),torch.tensor([[30.]*4]),
        torch.zeros(1,4),scalar(0.),scalar(.416),scalar(.09),scalar(.18),scalar(0.),scalar(0.))
    assert not s.success.item()
    assert s.ever_success.item()
