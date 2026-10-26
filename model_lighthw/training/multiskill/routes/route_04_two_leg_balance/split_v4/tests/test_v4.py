"""R4拆分v4入场事件、条件分布、反例与隔离契约。"""
import unittest
import math
import torch
from test_protocol import BalanceState, desired_gravity, schedule, SOURCE
from two_leg_balance_split_v4.policy import PhaseActorCritic
from two_leg_balance_split_v4.spec import contract


def metrics(n):
    return dict(com=torch.tensor([[0., 0., .35]]).repeat(n, 1), com_vel=torch.zeros(n,3),
                force_z=torch.full((n,4),28.62),clearance=torch.zeros(n,4),
                gravity=torch.tensor([[0.,0.,-1.]]).repeat(n,1),ang_vel=torch.zeros(n,3),
                base_z=torch.full((n,),.416),line_error=torch.full((n,),.35),
                body_force=torch.zeros(n),torque_peak=torch.zeros(n))


class V4Tests(unittest.TestCase):
    def test_delayed_entry_shifts_all_task_phases_and_deadline(self):
        s=BalanceState(2,'cpu',2.)
        m=metrics(2)
        for k in range(1,302):
            t=torch.full((2,),k*.01)
            # env0直到1.2s才安静；env1始终速度过大，必须在3s失败。
            m['com_vel'][0,0]=.3 if k<120 else 0.
            m['com_vel'][1,0]=.3
            s.update(.01,t,m,torch.zeros(2,dtype=torch.long),m['gravity'])
        self.assertAlmostEqual(float(s.start_time[0]),1.69,places=5)
        self.assertTrue(bool(s.entered[0]));self.assertFalse(bool(s.entered[1]))
        self.assertEqual(s.reason.tolist(),[0,5])
        phase,_=schedule(s.clock(torch.tensor([4.68,4.68])),2.)
        self.assertEqual(int(phase[0]),1)
        phase,_=schedule(s.clock(torch.tensor([4.70,4.70])),2.)
        self.assertEqual(int(phase[0]),2)
        self.assertFalse(bool(s.finished(torch.tensor([12.68,12.68]))[0]))
        self.assertTrue(bool(s.finished(torch.tensor([12.70,12.70]))[0]))
        self.assertFalse(bool(s.finished(torch.tensor([20.,20.]))[1]))
        self.assertGreater(float(s.entry_linear[0]/s.entry_samples[0]),0)
        self.assertEqual(float(s.entry_linear[1]),0.)

    def test_conditions_do_not_weaken_strict_contract(self):
        c=contract()
        self.assertEqual(c['version'],'lighthw_r4_split_v4')
        for k,v in dict(entry_seconds=.5,hold_seconds=2.,recovery_seconds=1.,min_clearance_m=.03,
                        max_line_error_m=.06,max_com_speed_m_s=.12,max_ang_speed_rad_s=.25,
                        max_target_tilt_deg=12.,max_hold_drift_m=.1,max_recovery_drift_m=.1).items():
            self.assertEqual(c[k],v)

    def test_reward_profile_is_part_of_checkpoint_contract(self):
        baseline = contract()
        diagnostic = contract(reward_profile='unload_diag_v1')
        self.assertEqual(baseline['reward_profile'], 'baseline_v4')
        self.assertEqual(diagnostic['reward_profile'], 'unload_diag_v1')
        self.assertNotEqual(baseline['weights'], diagnostic['weights'])

    def test_noise_probability_entropy_and_mean_match_phase_distribution(self):
        torch.manual_seed(4)
        p=PhaseActorCritic(305,88,16,actor_hidden_dims=[8],critic_hidden_dims=[8],noise_std_type='log')
        obs=torch.zeros(6,305)
        for i,phase in enumerate([0,0,1,2,3,4]):obs[i,-6+phase]=1.
        p.update_distribution(obs)
        mean=p.actor(obs)
        expected=p.log_std.exp().expand_as(mean)*torch.tensor([[.03],[.03],[1.],[.25],[1.],[.03]])
        torch.testing.assert_close(p.action_std,expected)
        torch.testing.assert_close(p.act_inference(obs),mean)
        a=p.distribution.sample()
        manual=(-.5*((a-mean)/expected).square()-expected.log()-.5*math.log(2*math.pi)).sum(-1)
        torch.testing.assert_close(p.get_actions_log_prob(a),manual)
        entropy=(expected.log()+.5*math.log(2*math.pi*math.e)).sum(-1)
        torch.testing.assert_close(p.entropy,entropy)
        old=p.get_actions_log_prob(a).detach().clone()
        p.update_distribution(obs)
        torch.testing.assert_close((p.get_actions_log_prob(a)-old).exp(),torch.ones(6))
        p.get_actions_log_prob(a.detach()).sum().backward()
        self.assertTrue(torch.isfinite(p.log_std.grad).all())

    def test_hold_exploration_uses_only_recorded_observation_and_preserves_density(self):
        p=PhaseActorCritic(305,88,16,actor_hidden_dims=[8],critic_hidden_dims=[8],noise_std_type='log')
        obs=torch.zeros(3,305);frame=obs[:,-61:]
        frame[:,57]=1.;frame[:,54]=-1.;frame[:,5]=-1.
        frame[1,3:6]=desired_gravity(torch.tensor([.4]))[0]
        frame[2,0]=.2*.3
        p.update_distribution(obs)
        expected=p.log_std.exp()[None,:]*torch.tensor([[.03],[.25],[.25]])
        torch.testing.assert_close(p.action_std,expected)
        actions=p.distribution.sample();old=p.get_actions_log_prob(actions).clone()
        p.update_distribution(obs.clone())
        torch.testing.assert_close(p.get_actions_log_prob(actions),old)
        torch.testing.assert_close(p.action_mean,p.act_inference(obs))


if __name__=='__main__':unittest.main()
