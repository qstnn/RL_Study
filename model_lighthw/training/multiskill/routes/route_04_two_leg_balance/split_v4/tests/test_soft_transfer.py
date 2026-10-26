"""软门控诊断的行为反例：带载转移、动态低载、过速/漂移与旧档案隔离。"""
import unittest
import torch
from test_split_rewards import example, weighted
from two_leg_balance_split_v4.core import desired_gravity
from two_leg_balance_split_v4.spec import contract, REWARD_PROFILES


PROFILE = "soft_transfer_v2"


def supported(n):
    a = example(n)
    m = a['m']
    m['gravity'] = a['full_target'].clone()
    m['line_error'].zero_()
    m['force_z'][:, :2] = 0.
    m['force_z'][:, 2:] = (a['mass'] * 9.81 / 2)[:, None]
    m['clearance'][:, :2] = .08
    return a


class SoftTransferTests(unittest.TestCase):
    def test_progress_survives_unchanged_four_wheel_load(self):
        a = example(3)
        a['m']['gravity'] = desired_gravity(torch.tensor([0., -.15, -.30]))
        a['m']['line_error'] = torch.tensor([.35, .30, .25])
        old, new = weighted(a, 'unload_diag_v1'), weighted(a, PROFILE)
        for key in ('attitude', 'com_line'):
            self.assertEqual(old[key].tolist(), [0., 0., 0.])
            self.assertTrue(torch.all(new[key][1:] > new[key][:-1]), key)
        for key in ('support', 'capture', 'clearance', 'pair', 'balance'):
            self.assertEqual(new[key].tolist(), [0., 0., 0.], key)
        total = sum(new.values())
        self.assertLess(float(total[0]), 0.)
        self.assertTrue(torch.all(total[1:] > total[:-1]))

    def test_dynamic_low_total_load_has_continuous_reward_but_flight_does_not(self):
        a = supported(4)
        w = a['mass'] * 9.81
        a['m']['force_z'][:, 2:] = (w * torch.tensor([0., .24, .324, .326]))[:, None]
        new, old = weighted(a, PROFILE), weighted(a, 'unload_diag_v1')
        self.assertEqual(float(old['attitude'][1]), 0.)
        self.assertEqual(float(new['attitude'][0]), 0.)
        self.assertGreater(float(new['attitude'][1]), 0.)
        self.assertGreater(float(new['attitude'][2]), float(new['attitude'][1]))
        self.assertLess(float((new['attitude'][3] - new['attitude'][2]) / a['dt']), .02)

    def test_airborne_wrong_pair_and_single_stance_have_no_guidance(self):
        a = supported(3)
        w = a['mass'] * 9.81
        a['m']['force_z'][0] = 0.
        a['m']['force_z'][1] = w[1] * torch.tensor([.5, .5, 0., 0.])
        a['m']['force_z'][2] = w[2] * torch.tensor([0., 0., 1., 0.])
        terms = weighted(a, PROFILE)
        for key in ('attitude', 'com_line', 'support', 'capture', 'clearance', 'pair'):
            self.assertEqual(terms[key].tolist(), [0., 0., 0.], key)

    def test_target_capture_prefers_low_speed_and_low_drift_before_strict_boundary(self):
        a = supported(5)
        a['m']['ang_vel'][1, 1] = .20
        a['m']['com_vel'][2, 0] = .10
        a['m']['com'][3, 0] = .05
        a['m']['com'][4, 0] = .09
        terms = weighted(a, PROFILE)
        total = sum(terms.values())
        self.assertTrue(torch.all(total[0] > total[1:]))
        self.assertGreater(float(total[3]), float(total[4]))
        self.assertLess(float(terms['drift'][3]), 0.)
        self.assertLess(float(terms['capture'][3]), float(terms['capture'][0]))
        self.assertEqual(float(weighted(a, 'unload_diag_v1')['drift'][3]), 0.)

    def test_entry_and_return_terms_remain_identical_to_d1(self):
        a = example(5)
        a['phase'] = torch.tensor([0, 3, 4, 3, 4])
        a['state'].held[3:] = True
        old, new = weighted(a, 'unload_diag_v1'), weighted(a, PROFILE)
        for key in old:
            torch.testing.assert_close(new[key], old[key], atol=0, rtol=0, msg=key)

    def test_ideal_balance_outscores_loaded_target_and_return_requires_hold(self):
        a = supported(2)
        a['m']['force_z'][1] = a['mass'][1] * 9.81 / 4
        a['m']['clearance'][1] = 0.
        terms = weighted(a, PROFILE)
        self.assertGreater(float(sum(terms.values())[0]), float(sum(terms.values())[1]))
        self.assertLess(float(sum(terms.values())[1]), 0.)
        a['phase'][:] = 4
        terms = weighted(a, PROFILE)
        for key in ('return_pose', 'return_attitude', 'quiet', 'recovery', 'success_once'):
            self.assertEqual(terms[key].tolist(), [0., 0.], key)

    def test_new_profile_preserves_physics_success_and_optimizer_contract(self):
        self.assertIn(PROFILE, REWARD_PROFILES)
        old, new = contract(reward_profile='unload_diag_v1'), contract(reward_profile=PROFILE)
        self.assertEqual(new['transfer_gate_floor'], .25)
        for key in old:
            if key not in ('reward_profile', 'support_reward'):
                self.assertEqual(new[key], old[key], key)


if __name__ == '__main__':
    unittest.main()
