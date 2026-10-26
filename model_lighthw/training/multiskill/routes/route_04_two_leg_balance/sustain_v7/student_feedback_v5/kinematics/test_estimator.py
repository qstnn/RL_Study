"""无需仿真的 R4a 本体 FK/速度和可导接口核对。"""
from pathlib import Path
import unittest

import numpy as np
import torch

from body_estimator import BodyEstimator, TorchBodyEstimator, STAND


URDF = Path(__file__).resolve().parents[7] / "lighthw_urdf/urdf/lighthw.urdf"


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.estimator = BodyEstimator(URDF)

    def test_urdf_mass_joint_order_and_wheel_axis(self):
        self.assertAlmostEqual(self.estimator.mass, 22.0, places=5)
        self.assertEqual(list(self.estimator.index)[14:], ["RR_wheel_joint", "RL_wheel_joint"])
        *_, axis, angular_jac = self.estimator.geometry(STAND[None])
        np.testing.assert_allclose(axis[0], np.array(((0., 1., 0.), (0., 1., 0.))), atol=1e-12)
        np.testing.assert_allclose(angular_jac[0, 0, :, 14], (0., 1., 0.), atol=1e-12)
        np.testing.assert_allclose(angular_jac[0, 1, :, 15], (0., 1., 0.), atol=1e-12)

    def test_analytic_com_and_rear_wheel_jacobians(self):
        q = STAND.copy()
        q[[1, 4, 7, 10]] += [.10, -.07, .13, -.04]
        com, wheels, jac, wheel_jac, _, _ = self.estimator.geometry(q[None])
        h = 1e-6
        for j in range(12):
            plus, minus = q.copy(), q.copy()
            plus[j] += h
            minus[j] -= h
            c_plus, w_plus, *_ = self.estimator.geometry(plus[None])
            c_minus, w_minus, *_ = self.estimator.geometry(minus[None])
            np.testing.assert_allclose((c_plus[0] - c_minus[0]) / (2 * h), jac[0, :, j], atol=2e-9)
            np.testing.assert_allclose((w_plus[0] - w_minus[0]) / (2 * h), wheel_jac[0, :, :, j], atol=2e-9)

    def test_torch_matches_numpy_and_has_gradient(self):
        frame = np.zeros((3, 61), dtype=np.float64)
        frame[:, 3:6] = ((0., 0., -1.), (-.5, .04, -.865), (-.75, -.02, -.661))
        frame[:, 6:18] = ((0.,) * 12, (.1, -.2, .1) * 4, (-.1, .2, -.1) * 4)
        frame[:, 18:34] = ((0.,) * 16, (.04,) * 16, (-.02,) * 16)
        frame[:, :3] = ((0., 0., 0.), (.01, -.02, .03), (-.02, .01, -.01))
        raw = np.tile(frame[:, None, :], (1, 5, 1)).reshape(3, 305)
        expected = self.estimator.estimate(raw)
        x = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        actual = TorchBodyEstimator(URDF)(x)
        for key in ("gap_m", "velocity_m_s", "rear_velocity_m_s", "velocity_spin_only_m_s"):
            np.testing.assert_allclose(actual[key].detach().numpy(), expected[key], atol=1e-12)
        (actual["gap_m"].square().sum() + actual["velocity_m_s"].square().sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(torch.count_nonzero(x.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
