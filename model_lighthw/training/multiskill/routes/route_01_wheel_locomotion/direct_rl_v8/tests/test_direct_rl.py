"""V8 CPU边界测试；真实状态机/奖励函数，替身只代替Isaac场景与基类。"""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
REPO=next(p for p in ROOT.parents if (p/'source/rl_training').is_dir())
SRC=REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_direct_v8'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
core=module('_v8core',SRC/'core.py');spec=module('_v8spec',SRC/'spec.py');protocol=module('_v8protocol',ROOT/'scripts/protocol.py')

def fixture(n=1):
    class Parent:
        def __init__(self,cfg,env):
            self._env=env;self.robot=env.scene['robot'];self.num_envs=n;self.device='cpu'
            self.vel_command_b=torch.zeros(n,3);self.metrics={};self.is_standing_env=torch.zeros(n,dtype=torch.bool);self.is_heading_env=self.is_standing_env.clone()
        @property
        def command(self):return self.vel_command_b
        def _resample_command(self,ids):self.vel_command_b[ids,0]=.3
        def _resample(self,ids):self._resample_command(ids)
        def reset(self,ids):self._resample(ids);return {}
    def asset():return NS(data=NS(root_pos_w=torch.zeros(n,3),root_lin_vel_w=torch.zeros(n,3),root_ang_vel_w=torch.zeros(n,3)))
    forces=torch.zeros(n,5,3);forces[:,:4,2]=10.
    scene={'robot':asset(),'trailer':asset(),'contact_forces':NS(data=NS(net_forces_w=forces))}
    env=NS(scene=scene,step_dt=.02,episode_length_buf=torch.zeros(n,dtype=torch.long),
        cfg=NS(fixed_speed_mps=torch.full((n,),.3),loaded_speed_min=.2,loaded_speed_max=.4,eval_distance_m=1.,loaded_acceleration=.3,unloaded_acceleration=1.,command_profile='slew'),
        r1_payload_kg=torch.zeros(n),r1_towed_kg=torch.full((n,),40.),r1_slope_rad=torch.zeros(n),
        r1_eval_phase=torch.zeros(n,dtype=torch.long),r1_eval_stop_time=torch.full((n,),-1.),r1_eval_origin=torch.zeros(n,3),r1_cart_origin=torch.zeros(n,3),r1_eval_progress=torch.zeros(n),
        r1_tangent=torch.tensor([[1.,0.,0.]]).repeat(n,1),r1_normal=torch.tensor([[0.,0.,1.]]).repeat(n,1),r1_wheel_sensor_ids=[0,1,2,3],r1_body_sensor_ids=[4],r1_progress_before=torch.zeros(n))
    mdp=NS(state=lambda e:(e.scene['robot'],e.scene['robot'].data.root_lin_vel_w[:,:2],e.r1_normal),
           heading_error=lambda e:torch.zeros(n,1),hitch_clearance=lambda e:torch.full((n,),.07))
    cls=next(x for x in ast.parse((SRC/'commands.py').read_text()).body if isinstance(x,ast.ClassDef) and x.name=='UnifiedVelocityCommand')
    ns=dict(torch=torch,core=core,mdp=mdp,UniformVelocityCommand=Parent);exec(compile(ast.Module(body=[cls],type_ignores=[]),str(SRC/'commands.py'),'exec'),ns)
    cmd=ns['UnifiedVelocityCommand'](None,env);cmd.reset(torch.arange(n))
    env.command_manager=NS(get_term=lambda _:cmd,get_command=lambda _:cmd.command)
    env.termination_manager=NS(terminated=torch.zeros(n,dtype=torch.bool))
    return env,cmd

def step(env,cmd,k):env.episode_length_buf[:]=k;cmd._update_command()

def reward(name,env):
    node=next(x for x in ast.parse((SRC/'mdp.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)
    ns=dict(torch=torch);exec(compile(ast.Module(body=[node],type_ignores=[]),str(SRC/'mdp.py'),'exec'),ns);return ns[name](env)

@pytest.mark.parametrize('n',[256,512,1024])
def test_first_batch_all64_and_two_hard_anchors(n):
    rows=np.column_stack(core.assignments(n,spec.SLOPE_MIX,spec.PAYLOADS,spec.TOWED_MASSES))
    assert len(np.unique(rows,axis=0))==64
    assert not ((rows[:,0]==-25)&(rows[:,2]>0)).any()
    assert not ((rows[:,1]>0)&(rows[:,2]>0)).any()
    for angle in (25,-20):assert ((rows[:,0]==angle)&(rows[:,2]==40)).mean()>=.15
    assert (rows[:,2]>0).mean()==.5

def test_distance_requires_both_vehicles_and_excludes_wait_drift():
    e,c=fixture();e.scene['robot'].data.root_pos_w[:,0]=5.;e.scene['trailer'].data.root_pos_w[:,0]=2.
    step(e,c,150);assert c.joint_progress()==0
    e.scene['robot'].data.root_pos_w[:,0]=6.2;e.scene['trailer'].data.root_pos_w[:,0]=2.8
    step(e,c,151);assert e.r1_eval_phase==1 and not c.distance_reached_this_step
    e.scene['trailer'].data.root_pos_w[:,0]=3.01;step(e,c,152)
    assert e.r1_eval_phase==2 and c.distance_reached_this_step
    step(e,c,153);assert not c.distance_reached_this_step

def test_progress_is_signed_and_cannot_gain_from_spring_or_round_trip():
    e,c=fixture();step(e,c,150)
    e.scene['robot'].data.root_pos_w[:,0]=.2
    assert reward('towing_signed_progress',e)==0
    e.scene['trailer'].data.root_pos_w[:,0]=.1
    first=reward('towing_signed_progress',e)*.02
    e.r1_progress_before[:]=c.joint_progress()
    e.scene['trailer'].data.root_pos_w[:,0]=0.
    assert (first+reward('towing_signed_progress',e)*.02).abs()<1.e-6
    e.r1_eval_phase[:]=3;assert reward('towing_signed_progress',e)==0

def complete(bad_support=False,drift=False):
    e,c=fixture();step(e,c,150)
    e.scene['robot'].data.root_pos_w[:,0]=1.1;e.scene['trailer'].data.root_pos_w[:,0]=1.1
    step(e,c,151)
    stopped=round(float(e.r1_eval_stop_time[0])/.02) if e.r1_eval_phase[0]==3 else None;events=0
    for k in range(152,600):
        if bad_support:e.scene['contact_forces'].data.net_forces_w[:,0,2]=0.
        if drift and stopped is not None and k>stopped+110:e.scene['trailer'].data.root_pos_w[:,0]+=0.001
        step(e,c,k)
        if stopped is None and e.r1_eval_phase==3:stopped=k
        events+=int(c.successful_hold_this_step)
        if c.completed_hold_this_step:
            assert k-stopped==350
            assert c.hold_count==250
            assert bool(reward('towing_complete_hold',e))==(not bad_support and not drift)
            e.termination_manager.terminated[:]=True
            assert reward('towing_complete_hold',e)==0
    return events

def test_complete_hold_pulse_requires_full_window_once():assert complete()==1
@pytest.mark.parametrize('kind',['support','drift'])
def test_elapsed_time_alone_is_not_success(kind):assert complete(bad_support=kind=='support',drift=kind=='drift')==0

def test_partial_reset_clears_new_task_buffers():
    e,c=fixture(2);c.hold_count[:]=200;c.hold_support_sum[:]=123.;c.successful_hold_this_step[:]=True
    e.episode_length_buf[:]=500;c.reset(torch.tensor([0]))
    assert c.hold_count.tolist()==[0,200] and c.hold_support_sum.tolist()==[0.,123.]
    assert c.successful_hold_this_step.tolist()==[False,True]
    assert c.cycle_start_step[0]==0

def test_protocol_has_downhill_braking_and_full_coverage():
    assert len(protocol.cases())==18 and len(protocol.cases(suite='full'))==128
    assert any(c['slope_deg']==-20 and c['towed_kg']==40 and c['speed']>0 for c in protocol.cases())

def test_conditional_exploration_has_consistent_log_probability():
    cls=module('_v8policy',SRC/'policy.py').MotionExplorationActorCritic
    p=cls(300,268,16,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],noise_std_type='log')
    with torch.no_grad():p.log_std.copy_(torch.tensor([.12]*12+[.24]*4).log())
    obs=torch.zeros(3,300);obs[1,42]=.3;obs[1,299]=1.;obs[2,42]=3.1
    sigma=p.std_for_observations(obs)
    torch.testing.assert_close(sigma[0],torch.tensor([.005]*12+[.01]*4))
    torch.testing.assert_close(sigma[2],sigma[0]);torch.testing.assert_close(sigma[1],sigma[0]*24)
    p.update_distribution(obs);a=p.distribution.sample()
    torch.testing.assert_close(p.get_actions_log_prob(a),torch.distributions.Normal(p.actor(obs),sigma).log_prob(a).sum(-1))

def test_old_actor_transfer_is_exact_and_critic_new_columns_zero():
    path=ROOT/'initialization/candidate_fixed.pt';saved=spec.checkpoint(path)
    original=spec.initialization_checkpoint(ROOT.parent/'unified_v5/initialization/selected/model_init.pt')
    for key,value in original['model_state_dict'].items():
        if key.startswith('actor.'):assert torch.equal(value,saved['model_state_dict'][key])
    dim=original['model_state_dict']['critic.0.weight'].shape[1]
    assert not torch.count_nonzero(saved['model_state_dict']['critic.0.weight'][:,dim:])


def test_training_task_sequence_is_seed_independent_and_has_no_velocity_randomization():
    e,c=fixture(20);e.cfg.fixed_speed_mps=None;e.r1_slope_rad[:]=25.*torch.pi/180.
    e.r1_towed_kg[:]=40.;c.task_sequence[:]=-1
    torch.manual_seed(1);c._resample_command(torch.arange(20));first=c.requested_vx.clone()
    assert set(first.tolist()).issubset({0.,float(torch.tensor(.3))})
    assert int((first==0).sum())==2
    c.task_sequence[:]=-1;torch.manual_seed(999);c._resample_command(torch.arange(20))
    assert torch.equal(first,c.requested_vx)
    c._resample_command(torch.arange(20));assert not torch.equal(first,c.requested_vx)


def test_evaluation_creates_unloaded_robot_template_before_case_mass_updates():
    cfg=NS(fixed_payload_kg=15.,fixed_towed_kg=0.,fixed_slope_deg=0.,fixed_speed_mps=.3)
    setup=module('_v8configuration',ROOT/'scripts/configuration.py').evaluation_template
    setup(cfg,dict(mode='evaluate',slope_deg=0.,payload_kg=15.,towed_kg=0.,speed=.3))
    assert (cfg.fixed_payload_kg,cfg.fixed_towed_kg,cfg.fixed_slope_deg,cfg.fixed_speed_mps)==(0.,40.,25.,0.)
    setup(cfg,dict(mode='play',slope_deg=0.,payload_kg=15.,towed_kg=0.,speed=.3))
    assert cfg.fixed_payload_kg==15. and cfg.fixed_towed_kg==0.
