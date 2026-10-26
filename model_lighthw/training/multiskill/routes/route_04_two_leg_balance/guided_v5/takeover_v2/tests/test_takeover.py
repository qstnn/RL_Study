"""接管纠偏边界：只收成功教师后缀、排除reset、保持原合同及来源。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_takeover import successful_suffix, prefix_check, contract, sources, original_sources, GUIDED, read
from run_takeover import parser, preflight


def trace():
    x = np.zeros((6, 2, 305), np.float32)
    x[:, :, 300] = 1.
    active = np.ones((6, 2), bool)
    active[5:, 0] = False
    takeover = np.zeros((6, 2), bool)
    takeover[2:] = True
    y = np.zeros((6, 2, 16), np.float32)
    y[:, :, 0] = np.arange(6, dtype=np.float32)[:, None] * .1
    return dict(policy_observation=x, policy_action=y, policy_teacher_action=y.copy(),
        policy_active=active, policy_takeover=takeover,
        policy_time=np.broadcast_to(np.arange(6, dtype=np.float32)[:, None] * .02, (6, 2)).copy())


def case(**changes):
    return dict(dict(held=True, success=True, timeout=True, failed=False, trace_env=0), **changes)


class TakeoverChecks(unittest.TestCase):
    def test_feedback_changes_only_declared_capture_parameters(self):
        from common_takeover import feedback_candidate, FEEDBACK
        from teacher import candidates
        base = candidates('return')[13]
        for name, overrides in FEEDBACK.items():
            actual = feedback_candidate(name, base)
            for key in base:
                self.assertEqual(actual[key], overrides.get(key, base[key]))
        self.assertEqual(base, candidates('return')[13])
        with self.assertRaises(ValueError):
            feedback_candidate('unbounded', base)

    def test_only_applied_teacher_suffix_and_no_reset_frame(self):
        d = trace()
        r = successful_suffix(d, case(), 18)
        np.testing.assert_array_equal(r['actions'], d['policy_action'][2:5, 0])
        self.assertEqual(len(r['phase']), 3)
        self.assertTrue((r['sample_kind'] == 2).all())
        self.assertTrue((r['episode'] == 18).all())
        self.assertEqual(r['observations'].shape, (3, 305))

    def test_failure_and_partial_success_rejected(self):
        for changes in ({'held': False}, {'success': False}, {'timeout': False}, {'failed': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                successful_suffix(trace(), case(**changes), 0)

    def test_teacher_label_must_equal_applied_action(self):
        d = trace()
        d['policy_teacher_action'][2, 0, 0] += .01
        with self.assertRaises(AssertionError):
            successful_suffix(d, case(), 0)

    def test_takeover_cannot_unlatch(self):
        d = trace()
        d['policy_takeover'][3, 0] = False
        with self.assertRaises(ValueError):
            successful_suffix(d, case(), 0)

    def test_no_teacher_frames_rejected(self):
        d = trace()
        d['policy_takeover'][:] = False
        with self.assertRaises(ValueError):
            successful_suffix(d, case(), 0)

    def test_prefix_excludes_post_switch_but_detects_pre_switch_change(self):
        keys = ('t', 'task_t', 'gravity', 'com', 'com_vel', 'wheel_pos', 'force_z', 'actions',
                'policy_observation', 'policy_action')
        baseline = {k: np.zeros((200 if k.startswith('policy_') else 4000, 2), np.float32) for k in keys}
        d = {k: v.copy() for k, v in baseline.items()}
        d['policy_action'][180:] = 1.
        d['gravity'][3600:] = 1.
        self.assertEqual(max(prefix_check(d, baseline, 3.6).values()), 0.)
        d['com'][3599] = .001
        with self.assertRaises(ValueError):
            prefix_check(d, baseline, 3.6)

    def test_no_changed_original_sources(self):
        self.assertEqual(original_sources(True), read(GUIDED / 'runs/2026-10-22_bc_corrected_02/source_manifest.json'))
        self.assertEqual(len(sources()), 26)

    def test_no_automatic_evaluation_or_ppo(self):
        c = contract()
        self.assertFalse(c['automatic_evaluation'])
        self.assertFalse(c['ppo_enabled'])
        self.assertFalse(c['physics_reset_at_takeover'])
        self.assertEqual(c['actor_dim'], 305)
        self.assertEqual(c['physical_contract']['hold_seconds'], 2.)

    def test_budget_and_missing_lineage_rejected(self):
        for arguments in (['pretrain'], ['diagnose'], ['collect'], ['evaluate'], ['pretrain', '--epochs', '401']):
            with self.subTest(args=arguments), self.assertRaises(ValueError):
                preflight(parser().parse_args(arguments))

    def test_no_success_no_collection(self):
        a = parser().parse_args(['collect', '--dataset', '/tmp/parent', '--diagnosis', '/tmp/diagnosis'])
        with patch('run_takeover.original_dataset'), patch('run_takeover.diagnosis_check', return_value={'full_successes': 0}):
            with self.assertRaises(ValueError):
                preflight(a)


if __name__ == '__main__':
    unittest.main()
