"""CPU回归：真实奖励调用点、指令状态机与固定验收，不声称GPU行为通过。"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
REPO=next(p for p in ROOT.parents if (p/'source/rl_training').is_dir())
SOURCE=REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
core=module('_v4_core_tests',SOURCE/'core.py')
protocol=module('_v4_protocol_tests',ROOT/'scripts/protocol.py')
def functions(*names,**extra):
    tree=ast.parse((SOURCE/'mdp.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    ns=dict(torch=torch,math=math,core=core,**extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-mdp-functions>','exec'),ns)
    return ns

def test_real_state_uses_task_direction_when_body_yaws():
    angle=math.pi/6
    def rotate(_,v):
        result=v.clone();result[:,0]=v[:,0]*math.cos(angle)-v[:,1]*math.sin(angle)
        result[:,1]=v[:,0]*math.sin(angle)+v[:,1]*math.cos(angle);return result
    robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=None,root_lin_vel_w=torch.tensor([[3.1*math.cos(angle),3.1*math.sin(angle),0.]])))
    env=SimpleNamespace(scene={'robot':robot},r1_axis_x=torch.tensor([[1.,0.,0.]]),r1_axis_z=torch.tensor([[0.,0.,1.]]),
                        r1_tangent=torch.tensor([[1.,0.,0.]]),r1_normal=torch.tensor([[0.,0.,1.]]))
    state=functions('state',quat_apply=rotate)['state']
    _,velocity,_=state(env)
    assert velocity[0,0].item()==pytest.approx(3.1*math.cos(angle),abs=1e-6)
    assert velocity[0,1].item()==pytest.approx(3.1*math.sin(angle),abs=1e-6)

def test_real_hitch_penalty_warns_at_six_centimeters():
    env=SimpleNamespace(r1_towed_kg=torch.tensor([40.,0.]),cfg=SimpleNamespace(hitch_margin_m=.08))
    f=functions('hitch_clearance_cost',hitch_clearance=lambda e:torch.tensor([.06,.06]))['hitch_clearance_cost']
    cost=f(env)
    assert cost[0]>0
    assert cost[1]==0

def test_slew_reaches_target_without_overshoot_and_respects_loaded_rate():
    current=torch.zeros(2);target=torch.tensor([3.1,.3]);rate=torch.tensor([1.,.3])
    for _ in range(200):
        previous=current.clone();current=core.slew(current,target,rate,.02)
        assert bool((abs(current-previous)<=rate*.02+1e-6).all())
        assert bool((current<=target+1e-6).all())
    torch.testing.assert_close(current,target)
    for _ in range(200):current=core.slew(current,torch.zeros(2),rate,.02)
    torch.testing.assert_close(current,torch.zeros(2))

def test_all_conditions_and_mutual_exclusion_are_preserved():
    s,p,t=core.assignments(1024,(0,0,0,0,-5,5,-10,10,-15,15,-20,20,-25,25),(0,5,10,15),(0,20,40))
    assert len(set(zip(s,p,t)))==64
    assert not np.any((s==-25)&(t>0))
    assert not np.any((p>0)&(t>0))
    for case in ((-25,0,0),(-25,15,0),(25,0,40),(0,0,40),(-20,0,40)):
        assert case in set(zip(s,p,t))

def test_real_actor_initialization_preserves_function_and_new_optimizer_contract(tmp_path):
    from rsl_rl.modules import ActorCritic
    spec=module('_v5_spec_tests',SOURCE/'spec.py')
    torch.set_num_threads(2)
    kwargs=dict(actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',noise_std_type='log',init_noise_std=.005)
    old=ActorCritic(300,242,16,**kwargs);spec.install_policy_input_mask(old)
    source_path=tmp_path/'source.pt'
    torch.save(dict(model_state_dict=old.state_dict(),infos={'r1_unified_contract':{'version':'lighthw_r1_unified_v4'}}),source_path)
    saved=spec.initialization_checkpoint(source_path)
    new=ActorCritic(300,252,16,**kwargs);noise=new.log_std.clone()
    new.load_state_dict(spec.transfer_policy(saved,new.state_dict()))
    x=torch.randn(32,300);cx=torch.randn(32,242);extra=torch.randn(32,10)
    with torch.no_grad():
        torch.testing.assert_close(old.act_inference(x),new.act_inference(x),rtol=0,atol=0)
        torch.testing.assert_close(old.critic(cx),new.critic(torch.cat((cx,extra),dim=-1)),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(new.log_std,noise,rtol=0,atol=0)
    assert not torch.count_nonzero(new.critic[0].weight[:,242:])
    saved['infos']['r1_unified_contract']['version']='foreign_policy'
    torch.save(saved,source_path)
    with pytest.raises(ValueError):spec.initialization_checkpoint(source_path)


def test_actor_context_uses_only_nominal_job_information():
    f=functions('task_context')['task_context']
    env=SimpleNamespace(r1_payload_kg=torch.tensor([0.,15.,0.]),r1_towed_kg=torch.tensor([0.,0.,40.]))
    torch.testing.assert_close(f(env),torch.tensor([[0.,0.],[1.,0.],[0.,1.]]))
    spec=module('_v4_context_spec',SOURCE/'spec.py')
    assert sum(spec.POLICY_TERMS.values())*spec.HISTORY==300
    assert not spec.CONTRACT['policy_base_linear_velocity']
    assert not spec.CONTRACT['policy_height_scan']
    assert 'support_teacher' not in (ROOT/'scripts/worker.py').read_text()


def command_fixture(fixed):
    """使用真实子类方法；只替换Isaac命令基类和物理状态，不仿造待测斜率逻辑。"""
    class Parent:
        def __init__(self,cfg,env):
            self._env=env;self.robot=env.scene['robot'];self.num_envs=2;self.device='cpu'
            self.vel_command_b=torch.zeros(2,3);self.metrics={'error_vel_xy':torch.zeros(2),'error_vel_yaw':torch.zeros(2)}
            self.is_standing_env=torch.zeros(2,dtype=torch.bool);self.is_heading_env=self.is_standing_env.clone()
        @property
        def command(self):return self.vel_command_b
        def _resample_command(self,ids):self.vel_command_b[ids,0]=3.1
        def reset(self,ids=None):self._resample_command(torch.arange(2) if ids is None else ids);return {}
    robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2,3)))
    env=SimpleNamespace(scene={'robot':robot},step_dt=.02,episode_length_buf=torch.zeros(2,dtype=torch.long),
        cfg=SimpleNamespace(fixed_speed_mps=torch.tensor([3.1,.3]) if fixed else None,eval_distance_m=1.,
                            unloaded_acceleration=1.,loaded_acceleration=.3,command_profile='slew',loaded_speed_min=.2,loaded_speed_max=.6),
        r1_payload_kg=torch.tensor([0.,15.]),r1_towed_kg=torch.zeros(2),r1_eval_phase=torch.zeros(2,dtype=torch.long),
        r1_eval_origin=torch.zeros(2,3),r1_eval_progress=torch.zeros(2),r1_eval_stop_time=torch.full((2,),-1.),
        r1_tangent=torch.tensor([[1.,0.,0.],[1.,0.,0.]]))
    node=next(n for n in ast.parse((SOURCE/'commands.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedVelocityCommand')
    ns=dict(torch=torch,core=core,UniformVelocityCommand=Parent)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<real-command-state-machine>','exec'),ns)
    cmd=ns['UnifiedVelocityCommand'](None,env);cmd.reset();return cmd,env


def test_real_command_reset_resample_slew_and_loaded_mode():
    cmd,env=command_fixture(False)
    assert not cmd.command.any()
    cmd.requested_vx[:]=torch.tensor([3.1,.3])
    rate=torch.tensor([1.,.3])
    for step in range(250):
        env.episode_length_buf[:]=step
        previous=cmd.command[:,0].clone();cmd._update_command()
        assert bool(((cmd.command[:,0]-previous).abs()<=rate*.02+1e-6).all())
        if step<50:assert not cmd.command.any()
    before=cmd.command.clone();cmd._resample_command(torch.arange(2))
    torch.testing.assert_close(cmd.command,before)
    cmd.requested_vx.zero_();cmd._update_command()
    assert cmd.command[0,0]>0 and cmd.hold_elapsed[0]==0
    cmd.reset(torch.tensor([0]));assert cmd.command[0,0]==0
    assert cmd.command[1,0]>0


def test_real_fixed_command_waits_until_applied_zero_before_hold_clock():
    cmd,env=command_fixture(True)
    phases=[]
    for step in range(1100):
        env.episode_length_buf[:]=step
        if step>=400:env.scene['robot'].data.root_pos_w[1,0]=1.01
        cmd._update_command();phases.append((step,cmd.command.clone(),env.r1_eval_phase.clone()))
    assert 14.0<float(env.r1_eval_stop_time[0])<14.2
    assert 8.9<float(env.r1_eval_stop_time[1])<9.1
    assert any(p[2][0]==2 and p[1][0,0]>0 for p in phases)
    assert all(p[1][0,0]==0 for p in phases if p[2][0]==3)


def synthetic_rows(case,stop=14.1):
    stop=float(np.float32(stop));n=1000 if case['speed']==0 else round(stop/.02)+350
    rows=[]
    for step in range(1,n+1):
        t=step*.02;moving=case['speed']>0 and 3<=t<stop
        v=.05 if case['payload_kg'] or case['towed_kg'] else case['speed']
        rows.append(dict(t=t,eval_phase=1. if moving else 3. if case['speed'] else 0.,
            eval_stop_time_s=stop if t>=stop else -1.,eval_progress_m=1. if t>=stop else 0.,
            command_vx=case['speed'] if moving else 0.,target_vx=case['speed'] if moving else 0.,
            vx=v if moving else 0.,velocity_error=abs(v-case['speed']) if moving else 0.,speed=abs(v) if moving else 0.,
            yaw_rate_abs=0.,tilt_deg=0.,heading_rad=0.,height=.416,four_wheel=1.,body_contact=0.,torque_abs_max=10.,
            drawbar_force_N=100.,hitch_clearance_m=.1,x=0.,y=0.,z=.416))
    return rows


@pytest.mark.parametrize('payload,tow,speed',[(0.,0.,0.),(0.,0.,3.1),(15.,0.,.3),(0.,40.,.3)])
def test_protocol_full_window_and_no_loaded_speed_requirement(payload,tow,speed):
    case=dict(slope_deg=25.,payload_kg=payload,towed_kg=tow,speed=speed,goal_distance_m=1.)
    rows=synthetic_rows(case)
    assert not protocol.finished(rows[:-1],case)
    result=protocol.summarize(rows,case,None)
    assert result['success'],result
    assert result['window_samples']['hold']==250
    if payload or tow:assert result['metrics']['cruise_vx_mps']<.1
    assert not protocol.summarize(rows[:-30],case,'terminated')['success']
    for r in rows[-100:]:r['y']=.06
    failed=protocol.summarize(rows,case,None)
    assert 'hold_drift' in failed['failed_checks']


def test_protocol_never_accepts_braking_as_stopped_or_zero_motion_as_distance():
    case=dict(slope_deg=0.,payload_kg=15.,towed_kg=0.,speed=.3,goal_distance_m=1.)
    rows=synthetic_rows(case)
    for r in rows:r['eval_phase']=2.
    assert not protocol.finished(rows,case)
    rows=synthetic_rows(case)
    for r in rows:r['eval_progress_m']=0.
    assert 'loaded_distance' in protocol.summarize(rows,case,None)['failed_checks']


def test_slope_posture_allows_uphill_front_short_rear_long_without_relaxing_roll():
    env=SimpleNamespace(r1_slope_rad=torch.tensor([math.radians(25.)]),r1_axis_x=torch.tensor([[1.,0.,0.]]),
        r1_tangent=torch.tensor([[1.,0.,0.]]),r1_normal=torch.tensor([[0.,0.,1.]]))
    def cost(pitch,roll=0.):
        pitch,roll=math.radians(pitch),math.radians(roll)
        forward=torch.tensor([[math.cos(pitch),0.,math.sin(pitch)]])
        up=torch.tensor([[-math.sin(pitch),math.sin(roll),math.cos(pitch)*math.cos(roll)]])
        robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=None))
        f=functions('posture_cost',state=lambda e:(robot,None,up),quat_apply=lambda q,v:forward)['posture_cost']
        return float(f(env))
    assert cost(0)==0 and cost(-5)==0 and cost(-9.9)==0
    assert cost(-20)>0 and cost(10)>0 and cost(-5,10)>0
    env.r1_slope_rad.zero_();assert cost(-5)>0


def test_default_leg_pose_penalty_is_disabled_on_slope():
    robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.ones(2,12),default_joint_pos=torch.zeros(2,12)))
    env=SimpleNamespace(scene={'robot':robot},r1_slope_rad=torch.tensor([0.,math.radians(25.)]))
    f=functions('stand_pose',standing=lambda e:torch.ones(2,dtype=torch.bool))['stand_pose']
    result=f(env,SimpleNamespace(name='robot',joint_ids=list(range(12))))
    assert result[0]>0 and result[1]==0


@pytest.mark.parametrize('tow',[20.,40.])
def test_fixed_minus25_towing_is_rejected(tow):
    with pytest.raises(ValueError,match='No permitted'):
        core.assignments(1,(-25.,0.,25.),(0.,15.),(0.,20.,40.),-25.,0.,tow)


def test_fixed_minus25_payload_and_unloaded_are_retained():
    for payload in (0.,15.):
        slope,mass,tow=core.assignments(8,(-25.,0.,25.),(0.,15.),(0.,20.,40.),-25.,payload,0.)
        assert np.all(slope==-25.) and np.all(mass==payload) and np.all(tow==0.)


def test_evaluation_scope_preserves_downhill_speed_and_payload_but_no_trailer():
    cases=protocol.cases()
    assert len(cases)==16
    downhill=[c for c in cases if c['slope_deg']==-25.]
    assert len(downhill)==4 and all(c['towed_kg']==0 for c in downhill)
    assert any(c['payload_kg']==0 and c['speed']==3.1 for c in downhill)
    assert any(c['payload_kg']==15. and c['speed']>.0 for c in downhill)
    assert all(any(c['slope_deg']==s and c['towed_kg']==40. and c['speed']>0 for c in cases) for s in (0.,25.))
    spec=module('_v4_scope_spec',SOURCE/'spec.py')
    assert spec.CONTRACT['evaluation_case_count']==len(cases)
    assert spec.CONTRACT['training_condition_count']==64


def test_mask_survives_real_adam_updates_and_rejects_bad_momentum():
    from rsl_rl.modules import ActorCritic
    spec=module('_v4_mask_spec',SOURCE/'spec.py')
    policy=ActorCritic(300,242,16,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu')
    spec.install_policy_input_mask(policy)
    opt=torch.optim.Adam(policy.parameters(),lr=1e-3)
    old=policy.actor[0].weight.detach().clone()
    for _ in range(3):
        opt.zero_grad();policy.actor(torch.randn(32,300)).square().mean().backward();opt.step()
        spec.validate_policy_input_mask(policy,opt)
    assert not torch.equal(old[:,:205],policy.actor[0].weight[:,:205])
    opt.state[policy.actor[0].weight]['exp_avg'][:,205]=1.
    with pytest.raises(RuntimeError,match='Adam'):spec.validate_policy_input_mask(policy,opt)
