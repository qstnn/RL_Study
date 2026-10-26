"""验证物理子步尖峰不会被50Hz采样/窗口平均漏掉，重置写入不重复计数。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SRC=Path(__file__).resolve().parents[7]/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v4'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

torque=load('_torque_sampling_test',SRC/'torque.py')
protocol=load('_torque_protocol_test',ROOT/'scripts/protocol.py')

def test_one_millisecond_peak_reset_duplicates_and_window_rms():
    env=SimpleNamespace(num_envs=1,device='cpu',cfg=SimpleNamespace(decimation=20),_sim_step_counter=0)
    robot=SimpleNamespace(data=SimpleNamespace(applied_torque=torch.zeros(1,1),computed_torque=torch.zeros(1,1),joint_vel=torch.ones(1,1)))
    recorder=torque.PhysicsTorqueRecorder(env,robot,[0])
    window=torque.TorqueWindow(['wheel'],.001)
    limits=torch.tensor([[48.]])
    frames=[]
    for control_step in range(2):
        recorder.begin()
        for substep in range(20):
            env._sim_step_counter+=1
            peak=control_step==0 and substep==5
            robot.data.applied_torque.fill_(48. if peak else 0.)
            robot.data.computed_torque.fill_(80. if peak else 0.)
            recorder.sample()
        recorder.sample() # 自动reset重复写入同一物理计数，不能重复采样。
        recorder.finish()
        window.update(recorder,{'all':torch.tensor([True])},limits)
        frames.append(recorder.numpy_trace())
    row=window.drain()[0]
    assert row['physics_samples']==40 and row['peak_Nm']==48.
    assert row['rms_Nm']==pytest.approx(48/np.sqrt(40))
    assert row['saturation_fraction']==pytest.approx(1/40)
    assert row['clipped_fraction']==pytest.approx(1/40)
    assert window.drain()==[]
    trace={key:np.concatenate([f[key] for f in frames]) for key in frames[0]}
    result=torque.summarize_trace(trace,['wheel'],np.array([48.]),False)
    assert not result['requirement_evidence_valid']
    assert result['joints'][0]['peak_censored_by_limit']
    assert result['joints'][0]['longest_above90_s']==.001
    assert result['joints'][0]['request_peak_Nm']==80.

def test_float32_distance_stop_keeps_exact_five_second_hold():
    case=dict(slope_deg=0.,payload_kg=15.,towed_kg=0.,speed=.3,goal_distance_m=1.)
    stop=float(np.float32(23.18))
    stop_step=1159
    rows=[]
    for i in range(stop_step+350):
        rows.append(dict(t=(i+1)*.02,eval_phase=3.,eval_stop_time_s=stop,eval_progress_m=1.,
                         vx=.05,velocity_error=.25,speed=0.,yaw_rate_abs=0.,tilt_deg=0.,heading_rad=0.,
                         height=.416,four_wheel=1.,body_contact=0.,torque_abs_max=0.,drawbar_force_N=0.,x=0.,y=0.,z=.416,command_vx=0. if i>=stop_step else .3,target_vx=0. if i>=stop_step else .3))
    assert not protocol.finished(rows[:-1],case)
    assert protocol.finished(rows,case)
    result=protocol.summarize(rows,case,None)
    assert result['window_samples']['hold']==250
    assert result['success'],result


def test_training_window_can_flush_after_inference_rollout():
    window=torque.TorqueWindow(['wheel'],.001)
    with torch.inference_mode():
        recorder=SimpleNamespace(applied=torch.ones(2,20,1),requested=torch.ones(2,20,1),velocity=torch.ones(2,20,1))
        window.update(recorder,{'all':torch.tensor([True,True])},torch.full((2,1),48.))
    rows=window.drain()
    assert rows[0]['physics_samples']==40
    assert window.drain()==[]
