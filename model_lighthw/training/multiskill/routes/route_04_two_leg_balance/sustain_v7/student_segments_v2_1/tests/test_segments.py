"""控制交接边界与失败示范隔离，防止接管被误报为自主。"""
import unittest
import torch
from programs import takeover,validate,PROGRAMS
from learning_segments import selected_cases


class SegmentsTest(unittest.TestCase):
    def test_actual_release_boundaries(self):
        t=torch.tensor([2.48,2.5,2.88,2.9,3.18,3.2,3.58,3.6,3.78,3.8,200.])
        self.assertEqual(takeover(t,PROGRAMS['early']).tolist(),[False,True,True,False,False,True,True,False,False,True,True])

    def test_autonomous_teacher_never_executes(self):
        self.assertFalse(takeover(torch.arange(0,301,.02),None).any())

    def test_budget_and_overlap(self):
        for windows in PROGRAMS.values():validate(windows)
        for windows in [[[2.5,3.],[2.9,3.5],[3.8,None]],[[2.5,2.9],[3.2,3.6],[3.8,4.]],[[2.51,2.9],[3.2,3.6],[3.8,None]]]:
            with self.assertRaises(AssertionError):validate(windows)

    def test_failed_pair_cannot_be_cherry_picked(self):
        rows=[dict(trace='failed',success=True,return_time=-1),dict(trace='failed',success=False,return_time=-1),
              dict(trace='passed',success=True,return_time=-1),dict(trace='passed',success=True,return_time=-1)]
        self.assertEqual([r['trace'] for r in selected_cases(dict(cases=rows))],['passed','passed'])

    def test_return_and_single_slot_are_excluded(self):
        rows=[dict(trace='return',success=True,return_time=10),dict(trace='return',success=True,return_time=-1),
              dict(trace='single',success=True,return_time=-1)]
        self.assertEqual(selected_cases(dict(cases=rows)),[])


if __name__=='__main__':unittest.main()
