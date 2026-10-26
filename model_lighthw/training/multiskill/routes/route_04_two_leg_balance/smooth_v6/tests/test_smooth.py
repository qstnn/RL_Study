"""允许慢漂移但拒绝实际振荡；独立滤波复算及数据/预算边界。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from scipy.signal import butter, lfilter

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_smooth import legacy_sources, read, GUIDED, contract
from two_leg_balance_smooth_v6.jitter import Smoothness, LIMITS
from two_leg_balance_smooth_v6.core import BalanceState, desired_gravity
from two_leg_balance_smooth_v6.spec import STAND, WEIGHTS
from two_leg_balance_smooth_v6.rewards import task_rewards
from learning_smooth import temporal_indices
from run import parser, preflight


def metrics(n=2):
    return dict(com=torch.zeros(n, 3), com_vel=torch.zeros(n, 3), ang_vel=torch.zeros(n, 3),
        gravity=desired_gravity(torch.full((n,), -.9794692)), base_z=torch.full((n,), .6),
        line_error=torch.zeros(n), body_force=torch.zeros(n), torque_peak=torch.zeros(n),
        force_z=torch.tensor([[0., 0., 55., 55.]]).expand(n, -1).clone(),
        clearance=torch.tensor([[.1, .1, 0., 0.]]).expand(n, -1).clone(),
        leg_position=torch.tensor([STAND]).expand(n, -1).clone(), wheel_velocity=torch.zeros(n, 4))


class SmoothChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_slow_translation_holds_beyond_ten_cm_but_fast_oscillation_does_not(self):
        state = BalanceState(2, 'cpu', 2.)
        state.entered[:] = True
        state.start_time[:] = 1.
        m = metrics()
        for j in range(2400):
            t = 4. + j * .001
            m['com'][:, 0] = .1 * (t - 4.)
            m['com_vel'][:, 0] = .1
            # 瞬时速度始终在0.12内，但15Hz往复应被抖动门拒绝。
            m['com_vel'][1, 0] = .03 + .08 * np.sin(2. * np.pi * 15. * (t - 4.))
            state.update(.001, torch.full((2,), t), m, torch.ones(2, dtype=torch.long), m['gravity'])
        self.assertTrue(state.held[0])
        self.assertGreater(float(m['com'][0, 0]), .2)
        self.assertFalse(state.held[1])
        self.assertLess(float(state.best_hold[1]), .2)

    def test_each_actual_channel_rejects_high_frequency(self):
        state = Smoothness(4, 'cpu')
        m = metrics(4)
        for j in range(2000):
            v = np.sin(2 * np.pi * 20 * j * .001)
            m['com_vel'][0, 0] = .04 * v
            m['ang_vel'][1, 0] = .12 * v
            m['leg_position'][2, 0] = .01 * v
            m['wheel_velocity'][3, 0] = 2. * v
            state.update(.001, m)
        self.assertFalse(state.accepted().any())
        for i in range(4):
            self.assertGreater(float(state.rms[i, i]), LIMITS[i])

    def test_physics_sampling_detects_leg_vibration_above_policy_nyquist(self):
        state = Smoothness(1, 'cpu')
        m = metrics(1)
        for j in range(1000):
            m['leg_position'][0, 0] = .01 * np.sin(2 * np.pi * 50 * j * .001)
            state.update(.001, m)
        self.assertGreater(float(state.rms[0, 2]), LIMITS[2])

    def test_filter_matches_independent_scipy_and_constant_speed_has_no_jitter(self):
        state = Smoothness(2, 'cpu')
        m = metrics()
        raw, actual = [], []
        for j in range(1200):
            m['com_vel'][0, 0] = .04 + .03 * np.sin(2 * np.pi * 12 * j * .001)
            m['com_vel'][1, 0] = .1
            raw.append(float(m['com_vel'][0, 0]))
            actual.append(float(state.update(.001, m)[0, 0]))
        b, a = butter(2, 5, fs=1000, btype='highpass')
        signal = np.array(raw)
        hp = lfilter(b, a, signal - signal[0])
        alpha = 1 - np.exp(-.001 / .1)
        expected = np.sqrt(lfilter([alpha], [1, -(1 - alpha)], hp * hp))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=2e-3)
        self.assertEqual(float(state.rms[1].max()), 0.)
        old = state.rms[1].clone()
        state.reset(torch.tensor([0]))
        torch.testing.assert_close(state.rms[1], old)
        self.assertFalse(state.initialized[0])

    def test_global_escape_and_collision_remain_failures(self):
        m = metrics()
        m['com'][0, 0] = 1.51
        m['body_force'][1] = 6.
        s = BalanceState(2, 'cpu', 2.)
        s.update(.001, torch.ones(2), m, torch.ones(2, dtype=torch.long), m['gravity'])
        self.assertEqual(s.reason.tolist(), [3, 1])

    def test_recovery_allows_slow_drift_without_old_ten_cm_gate(self):
        s = BalanceState(1, 'cpu', 2.)
        s.entered[:] = s.held[:] = True
        s.start_time[:] = 1.
        m = metrics(1)
        m['gravity'][:] = torch.tensor([0., 0., -1.])
        m['base_z'][:] = .416
        m['force_z'][:] = 29.
        m['com_vel'][0, 0] = .12
        for j in range(1200):
            m['com'][0, 0] = .12 * j * .001
            s.update(.001, torch.tensor([10. + j * .001]), m, torch.tensor([1]), m['gravity'])
        self.assertGreater(float(m['com'][0, 0]), .1)
        self.assertTrue(s.success[0])
        self.assertGreater(float(s.recovery_time[0]), 1.)

    def test_constant_overspeed_is_not_mistaken_for_allowed_smooth_translation(self):
        from two_leg_balance_smooth_v6.core import masks
        m = metrics(1)
        m['com_vel'][0, 0] = .1201
        self.assertFalse(masks(m, torch.tensor([1]), m['gravity'])['balance'][0])

    def test_rewards_allow_position_shift_and_charge_actual_jitter(self):
        m = metrics(1)
        s = BalanceState(1, 'cpu', 2.)
        s.held[:] = True
        s.hold_time[:] = 2.
        z = torch.zeros(1, 16)
        def reward():
            return task_rewards(m, s, torch.tensor([2]), torch.tensor([1]), m['gravity'],
                torch.tensor([11.668]), .02, m['leg_position'], z[:, :12], z[:, :12], z[:, :4],
                z, z, torch.tensor([False]), torch.tensor([False]), older_actions=z)
        before = reward()
        m['com'][0, 0] = .3
        m['com_vel'][0, 0] = .10
        after = reward()
        self.assertEqual(set(after), set(WEIGHTS))
        for k in before:
            torch.testing.assert_close(before[k], after[k])
        s.jitter.rms[0, 2] = 2 * LIMITS[2]
        vibrating = reward()
        self.assertGreater(float(vibrating['jitter']), 0.)
        self.assertLess(float(vibrating['capture']), float(after['capture']))
        m['com_vel'][0, 0] = .2
        self.assertGreater(float(reward()['slow_speed']), 0.)

    def test_temporal_loss_never_crosses_reset_phase_or_counterfactual(self):
        d = dict(phase=np.array([2, 2, 2, 2, 2, 4, 4, 4]), episode=np.array([0, 0, 0, 1, 1, 1, 1, 1]),
            time=np.arange(8) * .02, sample_kind=np.array([2, 2, 2, 2, 2, 2, 1, 2]))
        self.assertEqual(temporal_indices(d).tolist(), [2])

    def test_contract_and_legacy_sources(self):
        c = contract()
        self.assertIsNone(c['max_hold_drift_m'])
        self.assertIsNone(c['max_recovery_drift_m'])
        self.assertEqual(c['max_com_speed_m_s'], .12)
        self.assertEqual(c['hold_seconds'], 2.)
        self.assertFalse(c['automatic_frozen_evaluation'])
        self.assertEqual(legacy_sources(), read(GUIDED / 'takeover_v3/runs/2026-10-22_bc_corrected_03/source_manifest.json'))

    def test_budget_and_missing_input_fail_before_execution(self):
        for args in [['pretrain'], ['collect'], ['pretrain', '--epochs', '201'], ['evaluate', '--seeds', '43', '43']]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                preflight(parser().parse_args(args))


if __name__ == '__main__':
    unittest.main()
