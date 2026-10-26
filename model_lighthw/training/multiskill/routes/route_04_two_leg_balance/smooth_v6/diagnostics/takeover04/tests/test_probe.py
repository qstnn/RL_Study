"""新诊断入口的预算、身份与连续状态保护；不重复测试旧环境公式。"""
import unittest
from unittest.mock import patch
import numpy as np
import probe


class ProbeGuards(unittest.TestCase):
    def test_geometry_roundoff_and_real_changes(self):
        # 首次真实preflight数据：几何求和最后一位变化，候选角和物理未变。
        before = probe.read(probe.BASELINE / 'geometry.json')
        recomputed = probe.read(probe.HERE / 'runs/2026-10-22_fourth_actor_takeover_6/geometry.json')
        self.assertNotEqual(before, recomputed)
        probe.check_geometry(before, recomputed)
        import copy
        changed = copy.deepcopy(recomputed)
        changed['candidates'][1]['pitch_rad'] += 1e-6
        with self.assertRaises(AssertionError):
            probe.check_geometry(before, changed)

    def test_budget_rejects_unplanned_switches(self):
        for value in ([], [2.8, 2.8], [2.8, 3.2, 3.6, 4.], [2.81], [4.], [float('nan')]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.validate_switches(value)
        probe.validate_switches([2.8, 3.2, 3.6])

    def test_request_rejects_model_or_feedback_substitution(self):
        good = dict(actor=str(probe.ACTOR), actor_sha256=probe.ACTOR_SHA,
                    baseline=str(probe.BASELINE), teacher=str(probe.TEACHER), seed=43,
                    feedback='strong', ppo_updates=0, switch_times=[2.8, 3.2, 3.6],
                    baseline_result_sha256='baseline', teacher_result_sha256='teacher')
        hashes = {probe.ACTOR: probe.ACTOR_SHA, probe.BASELINE / 'result.json': 'baseline',
                  probe.TEACHER / 'result.json': 'teacher'}
        with patch.object(probe, 'sha', side_effect=lambda p: hashes[p]):
            probe.validate_request(good)
            for key, value in (('actor', str(probe.SMOOTH / 'runs/wrong/actor.pt')),
                               ('actor_sha256', 'wrong'), ('feedback', 'base'), ('seed', 44),
                               ('ppo_updates', 1), ('baseline', '/tmp/wrong')):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    probe.validate_request({**good, key: value})
            with self.assertRaises(AssertionError):
                probe.validate_request({**good, 'baseline_result_sha256': 'changed'})

    def fixture(self):
        physical = ('t', 'task_t', 'gravity', 'com', 'com_vel', 'wheel_pos', 'force_z',
                    'actions', 'leg_position', 'wheel_velocity', 'smooth_rms')
        d = {k: np.zeros((2820, 2, 1), dtype=np.float32) for k in physical}
        d['active'] = np.ones((2820, 2), dtype=bool)
        for k in ('observation', 'action', 'actor_action', 'teacher_action'):
            d['policy_' + k] = np.zeros((141, 2, 1), dtype=np.float32)
        return d

    def test_actual_state_and_actor_history_at_switch_are_checked(self):
        baseline = self.fixture()
        data = {k: v.copy() for k, v in baseline.items()}
        data['policy_teacher_action'][:] = 1  # 影子教师不执行，允许与自主基线不同。
        probe.full_prefix_check(data, baseline, 2.8)
        for key, idx in (('leg_position', 2799), ('smooth_rms', 2799), ('policy_observation', 140)):
            altered = {k: v.copy() for k, v in data.items()}
            altered[key][idx] = .01
            with self.subTest(key=key), self.assertRaises(AssertionError):
                probe.full_prefix_check(altered, baseline, 2.8)

    def test_teacher_suffix_is_not_compared_to_autonomous_future(self):
        baseline = self.fixture()
        data = {k: v.copy() for k, v in baseline.items()}
        data['actions'][2800:] = .2
        data['policy_action'][140:] = .2
        probe.full_prefix_check(data, baseline, 2.8)


if __name__ == '__main__':
    unittest.main()
