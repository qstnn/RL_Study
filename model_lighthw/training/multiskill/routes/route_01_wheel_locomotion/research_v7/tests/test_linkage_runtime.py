"""Minimal state-path check without launching Isaac Sim or changing V6 source."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_runtime():
    # The production module imports Isaac Lab's quaternion function. Identity
    # quaternions suffice here; physical stepping remains an Isaac Sim test.
    fake_math = types.ModuleType("isaaclab.utils.math")
    fake_math.quat_apply = lambda q, v: v
    mapping = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": fake_math,
    }
    sys.path.insert(0, str(SCRIPTS))
    with patch.dict(sys.modules, mapping):
        spec = importlib.util.spec_from_file_location("r1_linkage_runtime_test", SCRIPTS / "linkage_runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class AFrameRuntimeStateTest(unittest.TestCase):
    def test_state_returns_real_robot_and_cart_points_and_reactions(self):
        runtime = load_runtime()
        f = lambda row: torch.tensor([row], dtype=torch.float64)
        q = f([1, 0, 0, 0])
        zero = f([0, 0, 0])
        robot_root = f([0.08, 0, 0.416223])
        cart_root = f([-1.73, 0, 0.30])
        robot = types.SimpleNamespace(data=types.SimpleNamespace(
            root_quat_w=q, root_pos_w=robot_root,
            body_com_pos_w=robot_root[:, None, :], root_com_lin_vel_w=zero,
            root_ang_vel_w=zero,
        ))
        cart = types.SimpleNamespace(data=types.SimpleNamespace(
            root_quat_w=q, root_pos_w=cart_root,
            body_com_pos_w=cart_root[:, None, :], root_com_lin_vel_w=zero,
            root_ang_vel_w=zero,
        ))
        env = types.SimpleNamespace(
            scene={"robot": robot, "trailer": cart},
            num_envs=1, device="cpu", r1_base_id=0, r1_cart_base_id=0,
            r1_robot_hitch=f([-0.38, 0, 0.07 - 0.416223]),
            r1_cart_hitch=f([0.35, 0, 0.07 - 0.30]),
            r1_towed_kg=torch.tensor([40.0]),
            r1_normal=f([0, 0, 1]),
        )
        # Dict-like scene plus env_origins, as in InteractiveScene.
        class Scene(dict):
            env_origins = zero
        env.scene = Scene(env.scene)
        result = runtime.current_linkage_state(env)
        torch.testing.assert_close(result["robot_point_w"], f([-0.30, 0, 0.07]), atol=1e-10, rtol=0)
        torch.testing.assert_close(result["cart_center_point_w"], f([-1.38, 0, 0.07]), atol=1e-10, rtol=0)
        self.assertGreater(result["cart_force_w"][0, 0].item(), 0)
        torch.testing.assert_close(result["robot_force_w"] + result["cart_force_w"], zero, atol=1e-10, rtol=0)
        self.assertFalse(result["ground_guard"].item())
        self.assertAlmostEqual(result["tip_clearance_m"].item(), 0.07, places=10)


if __name__ == "__main__":
    unittest.main()
