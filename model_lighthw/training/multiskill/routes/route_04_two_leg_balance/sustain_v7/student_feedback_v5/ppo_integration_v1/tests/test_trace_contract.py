"""CPU collector-shape test only; fabricated metrics are not physical evidence."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "structured_v1"))
from robot_worker import PhysicsEvidence


class TraceContractTests(unittest.TestCase):
    def test_all_independent_reconstruction_inputs_are_raw_and_vectorized(self):
        n = 16
        zeros = lambda *shape: torch.zeros(n, *shape)
        state = SimpleNamespace(
            jitter=SimpleNamespace(rms=zeros(4)),
            hold_time=zeros(), best_hold=zeros(), post_capture_s=zeros(),
            capture_time=zeros(), first_break_time=zeros(),
            held=torch.zeros(n, dtype=torch.bool),
            interrupted=torch.zeros(n, dtype=torch.bool),
            success=torch.zeros(n, dtype=torch.bool),
            failed=torch.zeros(n, dtype=torch.bool),
            boundary=torch.zeros(n, dtype=torch.bool),
            stand_request=torch.ones(n, dtype=torch.bool),
            start_time=zeros(), origin=zeros(2), hold_origin=zeros(2),
        )
        metrics = dict(com=zeros(3), wheel_pos=zeros(4, 3), base_z=zeros(),
            com_vel=zeros(3), line_error=zeros(), gravity=zeros(3),
            ang_vel=zeros(3), force_z=zeros(4), clearance=zeros(4),
            leg_position=zeros(12), wheel_velocity=zeros(4),
            body_force=zeros(), torque_peak=zeros())
        base = SimpleNamespace(num_envs=n, device="cpu", physics_dt=.001, step_dt=.02,
            state=state, current_metrics=metrics,
            physics_steps=torch.ones(n, dtype=torch.long),
            mode=torch.ones(n, dtype=torch.long), actions=zeros(16),
            command=lambda _: (torch.full((n,), 2, dtype=torch.long), zeros(3)))
        with tempfile.TemporaryDirectory() as directory:
            evidence = PhysicsEvidence(base, directory, trace_ticks=2)
            evidence.observe(base, 1)
            base.physics_steps += 1
            evidence.observe(base, 2)
            trace = evidence.save_trace()
            with np.load(trace) as data:
                expected = dict(com_w=3, wheel_pos_w=12, base_z_m=1,
                    target_gravity=3, start_time_s=1, phase=1,
                    leg_position_rad=12, wheel_velocity_rad_s=4,
                    stand_request=1, mode=1, com_vel_m_s=3,
                    ang_vel_rad_s=3, gravity=3, force_z_n=4,
                    origin_w_xy=2, hold_origin_w_xy=2)
                for name, width in expected.items():
                    with self.subTest(name=name):
                        self.assertEqual(data[name].shape, (2, n, width))
                        self.assertTrue(np.isfinite(data[name]).all())
                self.assertEqual(data["phase"][0, 0, 0], 2)
                self.assertEqual(data["mode"][0, 0, 0], 1)
                np.testing.assert_allclose(data["physics_time_s"][:, 0, 0], [.001, .002],
                    rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
