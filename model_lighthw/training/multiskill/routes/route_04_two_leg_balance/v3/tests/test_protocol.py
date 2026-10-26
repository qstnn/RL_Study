"""R4验收反例：四轮/腾空/错腿/碰撞/漂移/短保持/末段失稳均不能算成功。"""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ROUTE = Path(__file__).resolve().parents[1]
SOURCE = ROUTE.parents[5] / "source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/two_leg_balance_v3"
sys.path.insert(0, str(SOURCE.parent))
from two_leg_balance_v3.core import BalanceState, targets, schedule, desired_gravity, support_geometry, masks
from two_leg_balance_v3.spec import STAND, LOW, HIGH


class ProtocolTests(unittest.TestCase):
    def test_action_neutral_and_limits(self):
        a = torch.zeros(1, 16)
        torch.testing.assert_close(targets(a)[0], torch.tensor(STAND))
        torch.testing.assert_close(targets(a - 5)[0], torch.tensor(LOW))
        torch.testing.assert_close(targets(a + 5)[0], torch.tensor(HIGH))

    def test_phase_and_target_sign(self):
        t = torch.tensor([0., 1., 4., 6., 7., 10., 12.])
        p, scale = schedule(t, 2.)
        self.assertEqual(p.tolist(), [0, 1, 2, 2, 3, 4, 4])
        torch.testing.assert_close(scale, torch.tensor([0., 0., 1., 1., 1., 0., 0.]))
        self.assertGreater(float(desired_gravity(torch.tensor([.91]))[0, 0]), 0)
        self.assertLess(float(desired_gravity(torch.tensor([-.98]))[0, 0]), 0)

    def test_line_is_finite_segment_not_infinite(self):
        wheels = torch.tensor([[[0., -.2, 0.], [0., .2, 0.], [-.6, -.2, 0.], [-.6, .2, 0.]]])
        self.assertAlmostEqual(float(support_geometry(torch.tensor([[0., .5, .5]]), wheels, torch.tensor([0]))), .3, places=6)
        self.assertAlmostEqual(float(support_geometry(torch.tensor([[-.6, 0., .5]]), wheels, torch.tensor([1]))), 0, places=6)

    def test_complete_task_and_rejection_cases(self):
        # 0完整前/1完整后/2始终四轮/3腾空/4错支撑/5超速/6漂移/7碰撞/
        # 8保持频繁中断/9仅保持不返回/10末段失稳/11未进入四轮稳定态。
        n = 12
        state = BalanceState(n, "cpu", 2.)
        mode = torch.zeros(n, dtype=torch.long)
        mode[1] = 1
        origin = torch.zeros(n, 2)
        state.reset(torch.arange(n), origin)
        for step in range(1, 1201):
            time = step * .01
            t = torch.full((n,), time)
            phase, amplitude = schedule(t, 2.)
            target = desired_gravity(torch.where(mode == 0, .91, -.98) * amplitude)
            hold = 4 <= time < 7
            force = torch.full((n, 4), 28.)
            clear = torch.zeros(n, 4)
            if hold:
                force[:] = torch.tensor([57., 57., 0., 0.])
                clear[:, 2:] = .1
                force[1] = torch.tensor([0., 0., 57., 57.])
                clear[1] = torch.tensor([.1, .1, 0., 0.])
                force[2] = 28.
                clear[2] = 0.
                force[3] = 0.
                force[4] = torch.tensor([0., 0., 57., 57.])
                if step % 50 == 0:
                    force[8, 2] = 28.
            com = torch.zeros(n, 3)
            com[:, 2] = .5
            com[6, 0] = max(0, (time - 4)) * .1
            velocity = torch.zeros(n, 3)
            if hold:
                velocity[5, 0] = .5
            if time > 11.8:
                velocity[10, 0] = .5
            if time < 3.1:
                velocity[11, 0] = .5
            if time >= 10:
                force[9, 2:] = 0.
            body_force = torch.zeros(n)
            if 5 < time < 5.1:
                body_force[7] = 20.
            m = dict(com=com, com_vel=velocity, force_z=force, clearance=clear, gravity=target,
                     ang_vel=torch.zeros(n, 3), base_z=torch.full((n,), .416),
                     line_error=torch.zeros(n), body_force=body_force, torque_peak=torch.zeros(n))
            state.update(.01, t, m, mode, target)
        self.assertEqual(state.success.tolist(), [True, True] + [False] * 10)
        self.assertEqual(int(state.reason[7]), 1)
        self.assertEqual(int(state.reason[11]), 5)
        self.assertGreater(float(state.best_hold[9]), 2.)
        self.assertEqual(float(state.recovery_time[10]), 0.)

    def test_one_physics_sample_breaks_continuity(self):
        state = BalanceState(1, "cpu", 2.)
        state.entered[:] = True
        state.start_time[:] = 1.
        state.hold_time[:] = 1.999
        state.best_hold[:] = 1.999
        target = desired_gravity(torch.tensor([.91]))
        m = dict(com=torch.tensor([[0., 0., .5]]), com_vel=torch.zeros(1, 3),
                 force_z=torch.tensor([[57., 57., 3., 0.]]), clearance=torch.tensor([[0., 0., .1, .1]]),
                 gravity=target, ang_vel=torch.zeros(1, 3), base_z=torch.tensor([.5]),
                 line_error=torch.zeros(1), body_force=torch.zeros(1), torque_peak=torch.zeros(1))
        state.update(.001, torch.tensor([6.]), m, torch.tensor([0]), target)
        self.assertEqual(float(state.hold_time[0]), 0.)
        self.assertFalse(bool(state.held[0]))

    def test_reward_prefers_supported_balance_over_four_wheel_and_airborne(self):
        import ast
        from types import SimpleNamespace as NS
        from two_leg_balance_v3.spec import WEIGHTS
        tree = ast.parse((SOURCE / "env.py").read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "TwoLegBalanceEnv")
        function = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_get_rewards")
        scope = dict(torch=torch, WEIGHTS=WEIGHTS, STAND=STAND, masks=masks)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "reward_test", "exec"), scope)
        n = 3
        target = desired_gravity(torch.full((n,), .91))
        state = BalanceState(n, "cpu", 2.)
        state.entered[:] = True
        state.hold_time[0] = .1
        metric = dict(gravity=target, com=torch.tensor([[0., 0., .5]]).repeat(n, 1), com_vel=torch.zeros(n, 3),
                      ang_vel=torch.zeros(n, 3), force_z=torch.tensor([[57.,57.,0.,0.],[28.,28.,28.,28.],[0.,0.,0.,0.]]),
                      clearance=torch.tensor([[0.,0.,.1,.1],[0.,0.,0.,0.],[.1,.1,.1,.1]]),
                      line_error=torch.zeros(n), base_z=torch.full((n,), .5))
        data = NS(joint_pos=torch.tensor(STAND).repeat(n,1), applied_torque=torch.zeros(n,16), joint_vel=torch.zeros(n,16))
        env = NS(current_metrics=metric,state=state,robot=NS(data=data),mode=torch.zeros(n,dtype=torch.long),
                 num_envs=n, device="cpu", mass=torch.full((n,), 11.668111), reset_time_outs=torch.zeros(n,dtype=torch.bool),
                 step_dt=.02,command=lambda:(torch.full((n,),2),target),actions=torch.zeros(n,16),previous_actions=torch.zeros(n,16),
                 leg_ids=list(range(12)),wheel_ids=list(range(12,16)),episode_length_buf=torch.full((n,),250),max_episode_length=600,
                 episode_rewards={k:torch.zeros(n) for k in WEIGHTS},extras={"log":{}})
        reward = scope["_get_rewards"](env)
        self.assertGreater(float(reward[0]),float(reward[1]))
        self.assertGreater(float(reward[0]),float(reward[2]))
        self.assertEqual(env.reward_terms["clearance"][1:].tolist(),[0.,0.])
        self.assertEqual(env.reward_terms["balance"][1:].tolist(),[0.,0.])
        self.assertEqual(env.reward_terms["success_once"].tolist(),[0.,0.,0.])

    def test_reset_only_selected_environment(self):
        state = BalanceState(2, "cpu", 2.)
        state.start_time[:] = 1.4
        state.held[:] = True
        state.best_hold[:] = 2.
        state.reset(torch.tensor([0]), torch.tensor([[.2, .3]]))
        self.assertEqual(state.held.tolist(), [False, True])
        self.assertEqual(float(state.start_time[0]), -1.)
        self.assertAlmostEqual(float(state.start_time[1]), 1.4, places=5)
        self.assertEqual(state.best_hold.tolist(), [0., 2.])


if __name__ == "__main__":
    unittest.main()
