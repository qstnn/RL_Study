"""实验隔离、负向请求与对照重放的保护检查，无需Isaac Sim。"""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import compare


class ComparisonGuards(unittest.TestCase):
    def test_fixed_budget(self):
        compare.validate_gains([.20, .25, .30])
        for gains in ([], [.20], [.20, .20, .30], [.20, .25, .35], [.20, .25, float('nan')], [.30, .25, .20]):
            with self.assertRaises(ValueError):
                compare.validate_gains(gains)

    def test_only_pitch_rate_changes(self):
        base = compare.feedback_candidate('strong', compare.teacher_evidence(compare.TEACHER)['candidate'])
        saved = copy.deepcopy(base)
        for gain in compare.GAINS:
            result = compare.candidate_for_gain(base, gain)
            self.assertEqual(result['pitch_rate_gain'], gain)
            self.assertEqual({k: v for k, v in result.items() if k != 'pitch_rate_gain'},
                             {k: v for k, v in base.items() if k != 'pitch_rate_gain'})
        self.assertEqual(base, saved)
        with self.assertRaises(ValueError):
            compare.candidate_for_gain(base, .35)

    def test_request_rejects_drift(self):
        request = compare.request_payload()
        compare.validate_request(request)
        for key, value in dict(switch_s=3.6, seed=44, ppo_updates=1, actor_sha256='wrong',
                               base_feedback='medium', control_trace_sha256='wrong').items():
            bad = {**request, key: value}
            with self.assertRaises(ValueError):
                compare.validate_request(bad)

    def test_exact_control_detects_physical_difference(self):
        sample = {'actions': np.zeros((5, 2, 16)), 'smooth_rms': np.zeros((100, 2, 4))}
        self.assertEqual(compare.exact_control_check(sample, copy.deepcopy(sample))['exact_array_count'], 2)
        changed = copy.deepcopy(sample)
        changed['smooth_rms'][50, 1, 2] = 1e-9
        with self.assertRaises(AssertionError):
            compare.exact_control_check(changed, sample)

    def test_prefix_detects_switch_state_mismatch(self):
        steps = round(compare.SWITCH / .02)
        baseline = dict(active=np.ones((steps * 20 + 20, 2), bool),
                        policy_observation=np.zeros((steps + 1, 2, 305)),
                        policy_actor_action=np.zeros((steps + 1, 2, 16)))
        changed = copy.deepcopy(baseline)
        changed['policy_observation'][steps, 0, 1] = .01
        with patch.object(compare.prior, 'prefix_check', return_value={'error': 0.}):
            with self.assertRaises(AssertionError):
                compare.prior.full_prefix_check(changed, baseline, compare.SWITCH)


if __name__ == '__main__':
    unittest.main()
