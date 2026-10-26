"""Static physical checks for the separate research A-frame drawbar."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/linkage_variant.py"
SPEC = importlib.util.spec_from_file_location("r1_linkage_variant", PATH)
linkage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linkage)


def state(*, robot_x=0.0, cart_yaw=0.0, robot_vx=0.0, cart_vx=0.0, active=True):
    c, s = math.cos(cart_yaw), math.sin(cart_yaw)
    cart_com = torch.tensor([[-1.35, 0.0, 0.0]], dtype=torch.float64)
    cart_center = cart_com + torch.tensor([[0.35 * c, 0.35 * s, 0.0]], dtype=torch.float64)
    robot = torch.tensor([[robot_x, 0.0, 0.0]], dtype=torch.float64)
    zero = torch.zeros_like(robot)
    return dict(
        robot_point_w=robot,
        cart_center_point_w=cart_center,
        robot_point_velocity_w=torch.tensor([[robot_vx, 0.0, 0.0]], dtype=torch.float64),
        cart_center_velocity_w=torch.tensor([[cart_vx, 0.0, 0.0]], dtype=torch.float64),
        cart_angular_velocity_w=zero,
        cart_side_axis_w=torch.tensor([[-s, c, 0.0]], dtype=torch.float64),
        robot_com_w=zero,
        cart_com_w=cart_com,
        active=torch.tensor([active]),
    )


class AFrameLinkageTest(unittest.TestCase):
    def test_nominal_rest_and_inactive_do_not_create_drive(self):
        rest = linkage.a_frame_forces(**state())
        self.assertAlmostEqual(rest["robot_force_w"].norm().item(), 0.0, places=10)
        self.assertAlmostEqual(rest["cart_force_w"].norm().item(), 0.0, places=10)
        self.assertFalse(rest["guard"].item())
        off = linkage.a_frame_forces(**state(robot_x=0.08, active=False))
        self.assertAlmostEqual(off["robot_force_w"].norm().item(), 0.0, places=10)
        self.assertAlmostEqual(off["cart_force_w"].norm().item(), 0.0, places=10)

    def test_longitudinal_linearization_matches_original_drawbar(self):
        delta = 1e-5
        result = linkage.a_frame_forces(**state(robot_x=delta))
        slope = result["cart_force_w"][0, 0].item() / delta
        self.assertAlmostEqual(slope, linkage.MATCHED_LONGITUDINAL_STIFFNESS_NPM, delta=0.04)
        self.assertAlmostEqual(result["equivalent_tension_n"].item(), result["cart_force_w"][0, 0].item(), places=10)
        self.assertAlmostEqual(result["center_length_m"].item(), 1.0 + delta, places=10)
        result = linkage.a_frame_forces(**state(robot_vx=delta))
        slope = result["cart_force_w"][0, 0].item() / delta
        self.assertAlmostEqual(slope, linkage.MATCHED_LONGITUDINAL_DAMPING_NSPM, delta=0.001)

    def test_internal_force_moment_balance_and_passivity(self):
        inputs = state(robot_x=0.08, cart_yaw=0.025, robot_vx=0.04, cart_vx=0.01)
        result = linkage.a_frame_forces(**inputs)
        self.assertFalse(result["guard"].item())
        torch.testing.assert_close(result["robot_force_w"] + result["cart_force_w"], torch.zeros((1, 3), dtype=torch.float64), atol=1e-10, rtol=0)
        origin_moment = (
            torch.cross(inputs["robot_com_w"], result["robot_force_w"], dim=-1)
            + result["robot_torque_w"]
            + torch.cross(inputs["cart_com_w"], result["cart_force_w"], dim=-1)
            + result["cart_torque_w"]
        )
        torch.testing.assert_close(origin_moment, torch.zeros((1, 3), dtype=torch.float64), atol=1e-10, rtol=0)
        mechanical_power = (
            (result["robot_force_w"] * inputs["robot_point_velocity_w"]).sum(-1)
            + (result["cart_rod_forces_w"] * result["cart_point_velocities_w"]).sum((1, 2))
        )
        _, rod_k, _ = linkage.matched_rod_parameters()
        energy_rate = (rod_k * (result["lengths_m"] - math.hypot(1.0, 0.20)) * result["length_rates_mps"]).sum(-1)
        torch.testing.assert_close(mechanical_power + energy_rate, -result["damping_dissipation_w"], atol=1e-10, rtol=1e-10)

    def test_small_cart_yaw_receives_restoring_moment(self):
        plus = linkage.a_frame_forces(**state(robot_x=0.08, cart_yaw=0.02))
        minus = linkage.a_frame_forces(**state(robot_x=0.08, cart_yaw=-0.02))
        self.assertLess(plus["cart_torque_w"][0, 2].item(), 0.0)
        self.assertGreater(minus["cart_torque_w"][0, 2].item(), 0.0)

    def test_force_guard_preserves_equal_opposite_reactions(self):
        result = linkage.a_frame_forces(**state(robot_x=1.0))
        self.assertTrue(result["guard"].item())
        self.assertLessEqual(result["cart_rod_forces_w"].norm(dim=-1).sum().item(), linkage.FORCE_GUARD_N + 1e-9)
        torch.testing.assert_close(result["robot_force_w"] + result["cart_force_w"], torch.zeros((1, 3), dtype=torch.float64), atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
