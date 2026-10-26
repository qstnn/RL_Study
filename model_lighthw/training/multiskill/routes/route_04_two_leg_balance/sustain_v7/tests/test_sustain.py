"""有意义的持续任务保护：跨旧截止、慢漂移、毫秒中断、显式返回及末状态价值。"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_sustain import check_legacy, read, BASELINE, candidate
from two_leg_balance_sustain_v7.core import SustainState, desired_gravity
from two_leg_balance_sustain_v7.spec import STAND, WEIGHTS, contract
from two_leg_balance_sustain_v7.rewards import task_rewards
from teacher_sustain import SustainTeacher
from continuing_ppo import timeout_rewards, ContinuingPPO


def metrics(n=1):
    wheels = torch.tensor([[[.4,-.2,.2],[.4,.2,.2],[0.,-.2,.0875],[0.,.2,.0875]]]).repeat(n,1,1)
    return dict(com=torch.zeros(n,3), com_vel=torch.zeros(n,3), ang_vel=torch.zeros(n,3),
        gravity=desired_gravity(torch.full((n,),-.9794692)), base_z=torch.full((n,),.6),
        line_error=torch.zeros(n), body_force=torch.zeros(n), torque_peak=torch.zeros(n),
        force_z=torch.tensor([[0.,0.,55.,55.]]).repeat(n,1),
        clearance=torch.tensor([[.1,.1,0.,0.]]).repeat(n,1), wheel_pos=wheels,
        leg_position=torch.tensor([STAND]).repeat(n,1), wheel_velocity=torch.zeros(n,4))


def state(n=1, observation_s=10.):
    s = SustainState(n, 'cpu', observation_s)
    s.entered[:] = True
    s.start_time[:] = 1.
    return s


class SustainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_standing_target_has_no_7_12_or_300_second_expiry(self):
        s = state(4)
        phase, amp = s.schedule(torch.tensor([7.,12.,60.,300.]))
        self.assertEqual(phase.tolist(), [2]*4)
        self.assertEqual(amp.tolist(), [1.]*4)
        self.assertTrue(s.stand_request.all())

    def test_capture_then_extra_time_and_first_break_cannot_be_erased(self):
        s = state(observation_s=.1); m = metrics(); mode=torch.ones(1,dtype=torch.long)
        for k in range(2200):
            s.update(.001,torch.tensor([4.+k*.001]),m,mode,m['gravity'])
        self.assertTrue(s.held[0]); self.assertTrue(s.success[0])
        self.assertAlmostEqual(float(s.post_capture_s[0]),.2,places=7)
        m['force_z'][0,0]=3.
        s.update(.001,torch.tensor([6.2]),m,mode,m['gravity'])
        self.assertFalse(s.success[0]); self.assertTrue(s.interrupted[0])
        self.assertEqual(int(s.break_reason[0]),1)
        m['force_z'][0,0]=0.
        for k in range(2100):
            s.update(.001,torch.tensor([6.201+k*.001]),m,mode,m['gravity'])
        self.assertFalse(s.success[0])
        self.assertAlmostEqual(float(s.first_break_time[0]),6.2,places=5)
        self.assertAlmostEqual(float(s.post_capture_s[0]),.2,places=7)
        self.assertGreater(float(s.hold_time[0]),2.)

    def test_millisecond_support_loss_reduces_integral_even_if_control_endpoint_good(self):
        s=state(); m=metrics(); mode=torch.ones(1,dtype=torch.long)
        for k in range(20):
            m['force_z'][0,0]=3. if k==10 else 0.
            s.update(.001,torch.tensor([4.+k*.001]),m,mode,m['gravity'])
        self.assertAlmostEqual(float(s.stable_step_s[0]),.019,places=8)
        self.assertGreater(float(s.hold_time[0]),0.)

    def test_slow_drift_beyond_old_boundary_not_fall_real_ground_edge_is_truncation(self):
        s=state(2); m=metrics(2)
        m['com'][:,0]=torch.tensor([2.,198.01]); m['wheel_pos'][:,:,0]+=m['com'][:,0,None]
        m['com_vel'][:,0]=.1
        s.update(.001,torch.tensor([4.,4.]),m,torch.ones(2,dtype=torch.long),m['gravity'])
        self.assertFalse(s.failed.any())
        self.assertEqual(s.boundary.tolist(),[False,True])
        self.assertGreater(float(s.stable_step_s[0]),0.)
        self.assertEqual(float(s.stable_step_s[1]),0.)

    def test_collision_and_capture_deadline_remain_failures(self):
        s=state(2); m=metrics(2)
        m['body_force'][0]=6.; m['force_z'][1,:]=28.
        s.update(.001,torch.tensor([9.,9.]),m,torch.ones(2,dtype=torch.long),m['gravity'])
        self.assertEqual(s.reason.tolist(),[1,6])

    def test_explicit_return_is_local_idempotent_and_does_not_reset_filter(self):
        s=state(2); m=metrics(2)
        s.update(.001,torch.tensor([4.,4.]),m,torch.ones(2,dtype=torch.long),m['gravity'])
        old=s.jitter.offset.clone()
        s.request_return(torch.tensor([0]),torch.tensor([30.,30.]))
        s.request_return(torch.tensor([0]),torch.tensor([31.,31.]))
        self.assertEqual(s.return_time.tolist(),[30.,-1.])
        phase,amp=s.schedule(torch.tensor([31.5,31.5]))
        self.assertEqual(phase.tolist(),[3,2]); self.assertAlmostEqual(float(amp[0]),.5)
        self.assertEqual(s.schedule(torch.tensor([33.,33.]))[0].tolist(),[4,2])
        torch.testing.assert_close(s.jitter.offset,old)

    def test_teacher_keeps_feedback_after_old_return_time(self):
        teacher=SustainTeacher([candidate(.5)],read(BASELINE/'geometry.json'),'cpu')
        m=metrics(); s=state(); b=SimpleNamespace(metrics=lambda:m,state=s,physics_dt=.001,physics_steps=torch.tensor([4000]))
        teacher(b,torch.tensor([4.]))
        b.physics_steps[:]=6000; six=teacher(b,torch.tensor([6.]))
        b.physics_steps[:]=300000; later=teacher(b,torch.tensor([300.]))
        torch.testing.assert_close(six,later,atol=0,rtol=0)
        self.assertFalse(teacher.return_started[0]); self.assertGreater(float(later.abs().max()),.001)
        s.request_return(torch.tensor([0]),torch.tensor([300.]))
        teacher(b,torch.tensor([300.]))
        self.assertTrue(teacher.return_started[0])

    def test_reward_does_not_pay_capture_timeout_or_passive_return(self):
        s=state(); m=metrics(); z=torch.zeros(1,16)
        s.held[:]=True; s.hold_event[:]=True; s.success[:]=True; s.stable_step_s[:]=.019
        def reward():
            return task_rewards(m,s,torch.tensor([2]),torch.tensor([1]),m['gravity'],torch.tensor([11.668]),.02,
                m['leg_position'],z[:,:12],z[:,:12],z[:,:4],z,z,torch.tensor([True]),torch.tensor([True]),older_actions=z)
        r=reward()
        self.assertAlmostEqual(float(r['balance'][0])*WEIGHTS['balance'],.228,places=6)
        for name in ('hold_once','success_once','missed_hold','drift','return_pose','return_attitude','recovery','time'):
            self.assertEqual(float(r[name][0]),0.,name)
        m['com'][0,0]=50.
        for name,value in reward().items():
            torch.testing.assert_close(value,r[name])
        s.jitter.rms[0,2]=.02
        self.assertLess(float(reward()['capture'][0]),float(r['capture'][0]))

    def test_timeout_uses_pre_reset_value_and_failure_is_never_bootstrapped(self):
        info=dict(time_outs=torch.tensor([1,1,0]), physical_terminated=torch.tensor([0,1,0]),
            terminal_observation_valid=torch.ones(3,dtype=torch.bool),
            terminal_critic_observation=torch.tensor([[7.],[50.],[90.]]))
        r,new=timeout_rewards(torch.ones(3),torch.tensor([1,1,0]),info,lambda o:2*o,.9)
        torch.testing.assert_close(r,torch.tensor([13.6,1.,1.]))
        self.assertNotIn('time_outs',new); self.assertIn('time_outs',info)
        info['terminal_observation_valid'][0]=False
        with self.assertRaises(ValueError):
            timeout_rewards(torch.ones(3),torch.ones(3),info,lambda o:o,.9)

    def test_real_rsl_storage_has_exactly_one_timeout_bootstrap(self):
        from rsl_rl.modules import ActorCritic
        policy=ActorCritic(3,3,1,actor_hidden_dims=[8],critic_hidden_dims=[8])
        alg=ContinuingPPO(policy,device='cpu',gamma=.9,num_learning_epochs=1,num_mini_batches=1)
        alg.init_storage('rl',2,2,[3],[3],[1])
        obs=torch.ones(2,3); terminal=torch.full((2,3),3.)
        alg.act(obs,obs)
        expected=1.+.9*policy.evaluate(terminal[:1]).detach().flatten()
        info=dict(time_outs=torch.tensor([1,0]),physical_terminated=torch.tensor([0,1]),
            terminal_observation_valid=torch.ones(2,dtype=torch.bool),terminal_critic_observation=terminal,
            observations=dict(critic=torch.zeros_like(obs)))
        alg.process_env_step(torch.ones(2),torch.ones(2),info)
        torch.testing.assert_close(alg.storage.rewards[0,0],expected)
        self.assertEqual(float(alg.storage.rewards[0,1]),1.)
        alg.act(obs,obs); alg.process_env_step(torch.ones(2),torch.zeros(2),{})
        alg.compute_returns(obs)
        self.assertTrue(torch.isfinite(alg.storage.returns).all())

    def test_contract_and_old_sources_remain_explicit(self):
        check_legacy()
        c=contract()
        self.assertFalse(c['old_checkpoint_compatible']); self.assertFalse(c['automatic_return'])
        self.assertFalse(c['ppo_admission']); self.assertIsNone(c['max_episode_displacement_m'])
        for key in ('hold_seconds','episode_seconds','hold_exploration','optimizer'):
            self.assertNotIn(key,c)
        self.assertFalse(c['recovery_reward_requires_held'])


if __name__=='__main__':
    unittest.main()
