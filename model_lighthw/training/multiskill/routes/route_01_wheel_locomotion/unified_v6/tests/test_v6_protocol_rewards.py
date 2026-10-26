"""V6真实命令/奖励函数CPU回归；状态机替身只替换Isaac基类，不宣称GPU行为验收。"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import math
from typing import Sequence
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
REPO=next(p for p in ROOT.parents if (p/'source/rl_training').is_dir())
SOURCE=REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v6'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


core=module('_v6_cycle_core',SOURCE/'core.py')
protocol=module('_v6_cycle_protocol',ROOT/'scripts/protocol.py')


def functions(*names,**extra):
    nodes=[n for n in ast.parse((SOURCE/'mdp.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace=dict(torch=torch,math=math,core=core,**extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<V6-real-mdp>','exec'),namespace)
    return namespace


def fixture(speeds=(3.1,.3),loaded=(False,True),fixed=False,profile='slew'):
    count=len(speeds)
    class Parent:
        def __init__(self,cfg,env):
            self._env=env;self.robot=env.scene['robot'];self.num_envs=count;self.device='cpu'
            self.vel_command_b=torch.zeros(count,3)
            self.metrics={'error_vel_xy':torch.zeros(count),'error_vel_yaw':torch.zeros(count)}
            self.is_standing_env=torch.zeros(count,dtype=torch.bool)
            self.is_heading_env=self.is_standing_env.clone()
            self.resample_count=torch.zeros(count,dtype=torch.long)
        @property
        def command(self):return self.vel_command_b
        def _resample_command(self,ids):
            self.vel_command_b[ids,0]=torch.tensor(speeds)[ids]
            self.is_standing_env[ids]=torch.tensor(speeds)[ids]==0
        def _resample(self,ids):
            self.resample_count[ids]+=1;self._resample_command(ids)
        def reset(self,ids=None):self._resample(torch.arange(count) if ids is None else ids);return {}
    robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(count,3),root_quat_w=None))
    env=SimpleNamespace(scene={'robot':robot},step_dt=.02,episode_length_buf=torch.zeros(count,dtype=torch.long),
        cfg=SimpleNamespace(fixed_speed_mps=torch.tensor(speeds) if fixed else None,eval_distance_m=1.,
             unloaded_acceleration=1.,loaded_acceleration=.3,command_profile=profile,loaded_speed_min=.3,loaded_speed_max=.3),
        r1_payload_kg=torch.tensor(loaded)*15.,r1_towed_kg=torch.zeros(count),
        r1_eval_phase=torch.zeros(count,dtype=torch.long),r1_eval_origin=torch.zeros(count,3),
        r1_eval_progress=torch.zeros(count),r1_eval_stop_time=torch.full((count,),-1.),
        r1_tangent=torch.tensor([[1.,0.,0.]]).repeat(count,1),r1_normal=torch.tensor([[0.,0.,1.]]).repeat(count,1))
    cls=next(n for n in ast.parse((SOURCE/'commands.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedVelocityCommand')
    namespace=dict(torch=torch,core=core,UniformVelocityCommand=Parent)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<V6-real-command>','exec'),namespace)
    command=namespace['UnifiedVelocityCommand'](None,env);command.reset()
    command.requested_vx[:]=torch.tensor(speeds)
    env.command_manager=SimpleNamespace(get_term=lambda name:command,get_command=lambda name:command.command)
    return command,env



def framework_episode_reset(command, env, ids):
    """执行当前Isaac真实_reset_idx，只替换无物理的manager/scene对象。"""
    package=importlib.util.find_spec('isaaclab')
    assert package is not None, 'The installed Isaac Lab framework is required for reset-order regression'
    framework=Path(next(iter(package.submodule_search_locations)))/'envs/manager_based_rl_env.py'
    cls=next(n for n in ast.parse(framework.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManagerBasedRLEnv')
    reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx')
    namespace={'Sequence':Sequence}
    exec(compile(ast.Module(body=[reset],type_ignores=[]),str(framework),'exec'),namespace)
    class Scene(dict):
        def reset(self,env_ids): pass
    env.scene=Scene(env.scene)
    noop=SimpleNamespace(reset=lambda ids:{},compute=lambda **kw:None)
    for key in ('curriculum_manager','observation_manager','action_manager','reward_manager','termination_manager','recorder_manager'):
        setattr(env,key,noop)
    env.event_manager=SimpleNamespace(available_modes=[],reset=lambda ids:{})
    env.command_manager.reset=command.reset
    env.extras={}
    namespace['_reset_idx'](env,ids)


def advance(command,env,step):
    env.episode_length_buf[:]=step;command._update_command()


def test_fast_training_task_keeps_complete_brake_settle_and_hold_before_resampling():
    command,env=fixture(speeds=(3.1,),loaded=(False,))
    stop_step=None;complete_step=None
    for step in range(1300):
        previous=command.command[:,0].clone();advance(command,env,step)
        assert torch.all(abs(command.command[:,0]-previous)<=.020001)
        if step<150:assert not command.command.any()
        if env.r1_eval_phase[0]==3 and stop_step is None:
            stop_step=step
            assert command.hold_elapsed[0]==0
        if stop_step is not None:
            assert float(command.hold_elapsed[0])==pytest.approx((step-stop_step)*.02,abs=1e-6)
        if command.completed_cycle_this_step[0]:complete_step=step;break
        assert command.resample_count[0]==1
    assert stop_step is not None and 700<=stop_step<=710
    assert complete_step-stop_step==350
    assert command.hold_windows_completed[0]==1
    assert command.cycles_completed[0]==1
    assert command.resample_count[0]==1
    advance(command,env,complete_step+1)
    assert command.resample_count[0]==2 and env.r1_eval_phase[0]==0
    assert command.phase_elapsed[0]==0 and command.cycle_elapsed[0]==0


def test_static_task_runs_twenty_seconds_and_only_then_resamples():
    command,env=fixture(speeds=(0.,),loaded=(True,))
    for step in range(1001):
        advance(command,env,step)
        assert not command.command.any()
        assert command.resample_count[0]==1
        assert command.cycles_completed[0]==int(step==1000)
    assert command.static_cycles_completed[0]==1
    assert command.hold_windows_completed[0]==0
    advance(command,env,1001)
    assert command.resample_count[0]==2
    assert command.cycles_completed[0]==1


def test_loaded_distance_excludes_wait_drift_and_requires_net_one_meter():
    command,env=fixture(speeds=(.3,),loaded=(True,))
    for step in range(150):
        env.scene['robot'].data.root_pos_w[0,0]=step*.04
        advance(command,env,step)
        assert env.r1_eval_progress[0]==0
    advance(command,env,150)
    origin=float(env.r1_eval_origin[0,0])
    assert env.r1_eval_progress[0]==0 and env.r1_eval_phase[0]==1
    env.scene['robot'].data.root_pos_w[0,0]=origin-.4
    advance(command,env,151)
    assert env.r1_eval_progress[0]==pytest.approx(-.4,abs=1e-6)
    for step in range(152,301):advance(command,env,step)
    env.scene['robot'].data.root_pos_w[0,0]=origin+.99
    advance(command,env,301)
    assert env.r1_eval_phase[0]==1
    env.scene['robot'].data.root_pos_w[0,0]=origin+1.01
    advance(command,env,302)
    assert env.r1_eval_phase[0]==2 and command.command[0,0]>0
    stopped=None;completed=None
    for step in range(303,800):
        advance(command,env,step)
        if env.r1_eval_phase[0]==3 and stopped is None:stopped=step
        if command.completed_hold_this_step[0]:completed=step;break
    assert completed-stopped==350
    assert command.requested_vx[0]==pytest.approx(.3)
    assert command.target_vx[0]==0


def test_fixed_cycle_completes_once_without_resampling_or_restarting():
    command,env=fixture(speeds=(3.1,),loaded=(False,),fixed=True)
    for step in range(1300):advance(command,env,step)
    assert command.cycles_completed[0]==1 and command.resample_count[0]==1
    assert env.r1_eval_phase[0]==3 and command.command[0,0]==0
    assert not command.completed_cycle_this_step[0]


def test_step_profile_shares_complete_task_clock():
    command,env=fixture(speeds=(3.1,),loaded=(False,),fixed=True,profile='step')
    for step in range(901):advance(command,env,step)
    assert env.r1_eval_stop_time[0]==11.
    assert command.completed_hold_this_step[0]
    assert command.cycles_completed[0]==1


def test_partial_episode_reset_clears_only_reset_environment_task_state():
    command,env=fixture(speeds=(3.1,.3),loaded=(False,True),fixed=True)
    for step in range(400):advance(command,env,step)
    command.cycles_completed[:]=torch.tensor([2,3])
    command.hold_windows_completed[:]=torch.tensor([2,3])
    other=command.command[1].clone();other_phase=env.r1_eval_phase[1].clone()
    framework_episode_reset(command,env,torch.tensor([0]))
    assert command.command[0,0]==0 and command.hold_elapsed[0]==0
    assert env.r1_eval_progress[0]==0 and env.r1_eval_phase[0]==0
    assert env.r1_eval_stop_time[0]==-1 and command.cycles_completed[0]==0
    assert command.hold_windows_completed[0]==0 and command.cycle_start_step[0]==0
    torch.testing.assert_close(command.command[1],other)
    assert env.r1_eval_phase[1]==other_phase and command.cycles_completed[1]==3



@pytest.mark.parametrize('old_steps',[200,3250])
@pytest.mark.parametrize('speed',[0.,3.1])
def test_native_episode_reset_starts_task_from_zero_after_early_failure_and_timeout(old_steps,speed):
    command,env=fixture(speeds=(speed,),loaded=(False,))
    env.episode_length_buf.fill_(old_steps)
    framework_episode_reset(command,env,torch.tensor([0]))
    assert command.cycle_elapsed[0]==0 and command.phase_elapsed[0]==0
    first_motion=None;first_complete=None
    for step in range(1101):
        advance(command,env,step)
        if command.command[0,0]>0 and first_motion is None:first_motion=step
        if command.completed_cycle_this_step[0]:first_complete=step;break
    assert first_motion==(150 if speed else None)
    assert first_complete==(1055 if speed else 1000)


def test_native_reset_keeps_movement_cycles_after_two_full_timeouts():
    command,env=fixture(speeds=(3.1,),loaded=(False,))
    completed=[]
    for episode in range(3):
        if episode:
            env.episode_length_buf.fill_(3250)
            framework_episode_reset(command,env,torch.tensor([0]))
        first_motion=None;events=0
        for step in range(3250):
            advance(command,env,step)
            assert command.cycle_elapsed[0]>=0 and command.phase_elapsed[0]>=0
            if first_motion is None and command.command[0,0]>0:first_motion=step
            events+=int(command.completed_hold_this_step[0])
        assert first_motion==150
        completed.append(events)
    assert completed==[3,3,3]


def reward_fixture(commands,velocity=(.03,0.)):
    commands=torch.tensor(commands,dtype=torch.float32)
    count=len(commands);vel=torch.tensor(velocity).expand(count,2)
    env=SimpleNamespace(r1_payload_kg=torch.full((count,),15.),r1_towed_kg=torch.zeros(count),
        command_manager=SimpleNamespace(get_command=lambda name:torch.stack((commands,commands*0,commands*0),dim=-1)))
    namespace=functions('loaded_motion_weight','loaded_progress_score','tracking','tracking_l1','loaded_progress',
        state=lambda e:(None,vel,None),tracking_gate=lambda e:torch.ones(count))
    return env,namespace


def test_actual_tracking_and_l1_are_continuous_across_previous_point_zero_five_threshold():
    env,ns=reward_fixture([.05-1e-6,.05,.05+1e-6])
    for function in ('tracking','tracking_l1','loaded_progress'):
        scores=ns[function](env)
        assert torch.isfinite(scores).all()
        assert float((scores[1:]-scores[:-1]).abs().max())<3e-5


def test_loaded_progress_is_positive_at_low_speed_and_independent_of_reference_speed():
    ns=functions('loaded_progress_score')
    velocity=torch.tensor([[-.2,0.],[0.,0.],[.001,0.],[.02,0.],[.2,0.],[.6,0.]])
    scores=ns['loaded_progress_score'](velocity)
    assert scores[0]==0 and scores[1]==0
    assert torch.all(scores[2:]>0) and torch.all(scores[3:]>scores[2:-1])
    assert scores[-1]<1
    env,ns=reward_fixture([.2,.3,.6],velocity=(.1,0.))
    scores=ns['loaded_progress'](env)
    assert (scores>0).all()
    assert float(scores.max()-scores.min())<1e-6
    env,ns=reward_fixture([.3],velocity=(0.,0.))
    assert (4*ns['tracking'](env)+2*ns['tracking'](env,fine=True)+6*ns['loaded_progress'](env))[0]<.001


def test_split_progress_preserves_combined_weighted_reward():
    env,ns=reward_fixture([0.,.02,.05,.051,.1,.3,.6],velocity=(.03,.02))
    command=env.command_manager.get_command('base_velocity')[:,0]
    velocity=torch.tensor([[.03,.02]]).expand(len(command),-1)
    blend=ns['loaded_motion_weight'](command)
    delta=velocity-torch.stack((command,torch.zeros_like(command)),dim=-1)
    old=4*torch.lerp(core.velocity_score(delta,command,False),ns['loaded_progress_score'](velocity),blend)
    old+=2*torch.lerp(core.velocity_score(delta,command,True),ns['loaded_progress_score'](velocity),blend)
    new=4*ns['tracking'](env)+2*ns['tracking'](env,fine=True)+6*ns['loaded_progress'](env)
    torch.testing.assert_close(new,old)


def test_zero_command_keeps_original_velocity_tracking_and_lateral_cost():
    env,ns=reward_fixture([0.],velocity=(.03,.1))
    expected=core.velocity_score(torch.tensor([[.03,.1]]),torch.zeros(1),False)
    torch.testing.assert_close(ns['tracking'](env),expected)
    assert ns['tracking_l1'](env)[0]>0


def test_critic_context_adds_ten_columns_and_exposes_reward_relevant_hidden_state():
    command,env=fixture(speeds=(.3,.3),loaded=(True,True),fixed=True)
    ns=functions('privileged_context',quat_apply_inverse=lambda q,v:v,
                 heading_error=lambda e:torch.zeros(2,1))
    initial=ns['privileged_context'](env)
    assert initial.shape==(2,16)
    assert initial.shape[-1]-6==10  # existing critic 242 + ten = 252; Actor unchanged.
    command.hold_origin[1,0]=-.05;command.hold_origin[1,1]=-.10
    command.hold_elapsed[1]=7.
    env.r1_eval_phase[1]=3;env.episode_length_buf[1]=1500
    changed=ns['privileged_context'](env)
    torch.testing.assert_close(changed[0,:6],changed[1,:6])
    assert changed[1,6]==1 and changed[1,7]==2 and changed[1,8]==1
    torch.testing.assert_close(changed[0,10:14],torch.tensor([1.,0.,0.,0.]))
    torch.testing.assert_close(changed[1,10:14],torch.tensor([0.,0.,0.,1.]))
    assert changed[1,14]==1 and changed[1,15]==0 and changed[0,15]==1
    assert not torch.equal(changed[0],changed[1])


def test_summary_ignores_arbitrary_prestart_distance():
    case=dict(slope_deg=25.,payload_kg=0.,towed_kg=40.,speed=.3,goal_distance_m=1.)
    base=dict(t=1.,eval_phase=0.,eval_progress_m=100.,eval_stop_time_s=-1.,
        tilt_deg=0.,drawbar_force_N=0.,body_contact=0.,hitch_clearance_m=.07,target_vx=0.)
    rows=[base,dict(base,t=3.,eval_phase=1.,eval_progress_m=0.),
          dict(base,t=4.,eval_phase=1.,eval_progress_m=.005)]
    result=protocol.summarize(rows,case,'drawbar_ground')
    assert result['observed']['progress_m']==.005 and not result['success']


def test_task_scope_still_contains_all_sixty_four_conditions_and_no_downhill_towing():
    slopes,payloads,tows=core.assignments(1024,(0,0,0,0,-5,5,-10,10,-15,15,-20,20,-25,25),(0,5,10,15),(0,20,40))
    assert len(set(zip(slopes,payloads,tows)))==64
    assert not np.any((payloads>0)&(tows>0))
    assert not np.any((slopes==-25)&(tows>0))


def monitor_fixture():
    """执行真实UnifiedEnv.step/_reset_idx/监控方法，替身按Isaac终止→reset→command顺序运行。"""
    command,base=fixture(speeds=(.3,.3),loaded=(True,True),fixed=True)
    tree=ast.parse((SOURCE/'env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UnifiedEnv')
    accumulate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_accumulate')
    selected=next(n.value for n in accumulate.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='selected' for t in n.targets))
    keys=ast.literal_eval(selected)
    class Parent:
        def _reset_idx(self,ids):
            self.episode_length_buf[ids]=0
            self.command_manager.get_term('base_velocity').reset(ids)
        def step(self,action):
            self.episode_length_buf+=1
            ids=(self.reset_terminated|self.reset_time_outs).nonzero().flatten()
            if len(ids):self._reset_idx(ids)
            self.command_manager.get_term('base_velocity')._update_command()
            return None,torch.zeros(2),self.reset_terminated,self.reset_time_outs,{}
    def measurements(env,applied):
        term=env.command_manager.get_term('base_velocity')
        values={key:torch.zeros(2) for key in keys}
        values.update(command_vx=applied[:,0].clone(),target_vx=env.r1_transition_target.clone(),
            requested_vx=term.requested_vx.clone(),task_phase=env.r1_eval_phase.float(),
            phase_elapsed_s=term.phase_elapsed.clone(),cycle_elapsed_s=term.cycle_elapsed.clone(),
            cycle_complete=term.completed_cycle_this_step.float(),hold_complete=term.completed_hold_this_step.float(),
            cycles_completed=term.cycles_completed.float(),hold_windows_completed=term.hold_windows_completed.float(),
            static_cycles_completed=term.static_cycles_completed.float(),hold_elapsed_s=term.hold_elapsed.clone())
        return values
    namespace=dict(torch=torch,ManagerBasedRLEnv=Parent,mdp=SimpleNamespace(measurements=measurements))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<V6-real-env-monitor>','exec'),namespace)
    env=object.__new__(namespace['UnifiedEnv']);env.__dict__.update(base.__dict__);command._env=env
    env.num_envs=2;env.device='cpu';env.r1_slope_rad=torch.full((2,),math.radians(25.))
    env.r1_drawbar_guard=torch.zeros(2,dtype=torch.bool);env.r1_drawbar_peak=torch.zeros(2)
    env.r1_torque_recorder=SimpleNamespace(begin=lambda:None,finish=lambda:None,enabled=False)
    env.r1_torque_window=SimpleNamespace(update=lambda *args:None);env.r1_motor_limits=torch.ones(2,16)
    env._monitor_sums=None;env._monitor_count=None;env._monitor_keys=None
    env._monitor_event_sums=None;env._monitor_event_totals=None
    env.reset_terminated=torch.zeros(2,dtype=torch.bool);env.reset_time_outs=torch.zeros(2,dtype=torch.bool)
    env._collect_transition=False;env._terminal_metrics=None
    return env,command


def prime_hold_completion(env,command):
    env.episode_length_buf[:]=499
    env.r1_eval_phase[:]=3
    command.phase_start_step[:]=150  # next physical frame =350 held steps exactly.
    command.cycle_start_step[:]=0
    command.vel_command_b.zero_();command.target_vx.zero_()
    command.holding_active[:]=True;command.hold_steps[:]=349
    command.hold_elapsed[:]=6.98
    command.cycle_completed.zero_()


def test_monitor_completion_pulse_is_not_counted_again_when_next_step_terminates():
    env,command=monitor_fixture();prime_hold_completion(env,command)
    env.step(None)
    assert env.r1_last_metrics['cycle_complete'].tolist()==[1.,1.]
    saved=env.r1_last_metrics['cycle_complete'].clone()
    env.reset_terminated[0]=True
    env.step(None)
    assert env.r1_last_metrics['cycle_complete'].tolist()==[0.,0.]
    assert env.r1_last_metrics['cycles_completed'].tolist()==[1.,1.]
    assert command.cycles_completed.tolist()==[0,1]  # terminal count retained in metrics before reset.
    torch.testing.assert_close(saved,torch.ones(2))
    rows={row['group']:row for row in env.drain_metrics()}
    assert rows['all']['cycle_complete_events']==2 and rows['all']['hold_complete_events']==2
    assert rows['all']['cycle_complete_total']==2
    phase_samples=sum(rows[name]['samples'] for name in ('phase_wait_or_static','phase_moving','phase_braking','phase_hold'))
    assert phase_samples==rows['all']['samples']==4
    assert rows['all']['cycles_completed_episode_mean']==1.
    drained={row['group']:row for row in env.drain_metrics()}
    assert drained['all']['cycle_complete_events']==0 and drained['all']['cycle_complete_total']==2


def test_monitor_timeout_boundary_retains_full_hold_event_before_reset_but_not_failure():
    env,command=monitor_fixture();prime_hold_completion(env,command)
    env.reset_time_outs[:]=True
    env.reset_terminated[1]=True
    env.step(None)
    assert env.r1_last_metrics['cycle_complete'].tolist()==[1.,0.]
    assert env.r1_last_metrics['hold_complete'].tolist()==[1.,0.]
    assert env.r1_last_metrics['cycles_completed'].tolist()==[1.,0.]
    assert command.cycles_completed.tolist()==[0,0]
    rows={row['group']:row for row in env.drain_metrics()}
    assert rows['all']['cycle_complete_events']==1 and rows['all']['hold_complete_events']==1
    assert rows['all']['static_cycle_complete_events']==0


def test_monitor_static_timeout_event_is_separate_from_moving_hold_completion():
    env,command=monitor_fixture()
    command.requested_vx.zero_();env.episode_length_buf[:]=999
    env.reset_time_outs[:]=True
    env.step(None)
    rows={row['group']:row for row in env.drain_metrics()}
    assert rows['all']['cycle_complete_events']==2
    assert rows['all']['hold_complete_events']==0
    assert rows['all']['static_cycle_complete_events']==2
    assert rows['phase_wait_or_static']['cycle_complete_events']==2
    assert rows['all']['static_cycles_completed_episode_mean']==1.


def test_towing_distance_bonus_is_one_step_and_clears_on_partial_reset():
    command,env=fixture(speeds=(.3,.3),loaded=(True,True),fixed=True)
    env.r1_towed_kg.fill_(40.);env.r1_payload_kg.zero_()
    advance(command,env,150)
    env.scene["robot"].data.root_pos_w[0,0]=1.01
    advance(command,env,200)
    reward=functions("towing_distance_reached")["towing_distance_reached"]
    assert reward(env).tolist()==[True,False]
    advance(command,env,201)
    assert not reward(env).any()
    command.distance_reached_this_step[:]=True
    framework_episode_reset(command,env,torch.tensor([0]))
    assert command.distance_reached_this_step.tolist()==[False,True]
