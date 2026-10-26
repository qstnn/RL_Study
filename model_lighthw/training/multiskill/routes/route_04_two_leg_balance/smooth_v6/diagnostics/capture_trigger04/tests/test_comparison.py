"""捕获阈值对照的来源/预算保护，以及真实Teacher触发和过渡检查。"""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import compare
from teacher import Teacher


class CaptureTriggerGuards(unittest.TestCase):
    def base(self):
        return compare.feedback_candidate('strong', compare.teacher_evidence(compare.TEACHER)['candidate'])

    def test_fixed_budget(self):
        compare.validate_triggers([-20., -10., -5.])
        for values in ([], [-20.], [-20., -20., -5.], [-20., -10., 0.],
                       [-20., -10., float('nan')], [-5., -10., -20.]):
            with self.assertRaises(ValueError):
                compare.validate_triggers(values)

    def test_only_trigger_changes(self):
        base = self.base()
        saved = copy.deepcopy(base)
        for threshold in compare.TRIGGERS:
            candidate = compare.candidate_for_trigger(base, threshold)
            self.assertEqual(candidate['trigger_pitch_deg'], threshold)
            self.assertEqual({k: v for k, v in candidate.items() if k != 'trigger_pitch_deg'},
                             {k: v for k, v in base.items() if k != 'trigger_pitch_deg'})
        self.assertEqual(base, saved)
        with self.assertRaises(ValueError):
            compare.candidate_for_trigger(base, 0.)
        with self.assertRaises(AssertionError):
            compare.candidate_for_trigger({**base, 'pitch_rate_gain': .30}, -10.)

    def test_request_rejects_drift(self):
        request = compare.request_payload()
        compare.validate_request(request)
        for key, value in dict(switch_s=3.6, seed=44, ppo_updates=1, actor_sha256='wrong',
                               base_feedback='medium', control_trace_sha256='wrong').items():
            with self.assertRaises(ValueError):
                compare.validate_request({**request, key: value})

    def test_control_difference_is_not_ignored(self):
        sample = {'actions': np.zeros((5, 2, 16)), 'smooth_rms': np.zeros((100, 2, 4))}
        self.assertEqual(compare.exact_control_check(sample, copy.deepcopy(sample))['exact_array_count'], 2)
        changed = copy.deepcopy(sample)
        changed['smooth_rms'][50, 1, 2] = 1e-9
        with self.assertRaises(AssertionError):
            compare.exact_control_check(changed, sample)

    def test_actual_teacher_threshold_and_continuous_blend(self):
        base = self.base()
        teacher = Teacher([compare.candidate_for_trigger(base, x) for x in compare.TRIGGERS],
                          compare.read(compare.BASELINE / 'geometry.json'), 'cpu')
        wheels = torch.tensor([[[.4, -.15, .0875], [.4, .15, .0875],
                                [0., -.15, .0875], [0., .15, .0875]]]).repeat(3, 1, 1)
        metrics = dict(gravity=torch.zeros(3, 3), ang_vel=torch.zeros(3, 3), wheel_pos=wheels,
                       com=torch.tensor([[.01, 0., .6]]).repeat(3, 1), com_vel=torch.zeros(3, 3),
                       force_z=torch.full((3, 4), 20.))
        robot = SimpleNamespace(metrics=lambda: metrics, state=SimpleNamespace(hold_origin=torch.zeros(3, 2)))
        for t, pitch_deg, flags, newly_fired in (
                (3.2, 6., [False, False, False], None),
                (3.92, -5.1, [False, False, True], 2),
                (4.22, -10.1, [False, True, True], 1),
                (4.62, -20.1, [True, True, True], 0)):
            angle = torch.deg2rad(torch.tensor(pitch_deg))
            metrics['gravity'][:, 0] = angle.sin()
            metrics['gravity'][:, 2] = -angle.cos()
            output = teacher(robot, torch.full((3,), t))
            self.assertEqual(teacher.triggered.tolist(), flags)
            if newly_fired is not None:
                torch.testing.assert_close(output[newly_fired], teacher.start[newly_fired], rtol=0, atol=0)
        torch.testing.assert_close(teacher.trigger_t, torch.tensor([4.62, 4.22, 3.92]))


if __name__ == '__main__':
    unittest.main()
