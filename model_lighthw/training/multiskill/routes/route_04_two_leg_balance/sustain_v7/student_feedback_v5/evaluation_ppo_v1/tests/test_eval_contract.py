"""CPU-only contract tests for isolated PPO checkpoint evaluation."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_eval import V5, load_ppo_actor
from worker_eval import BufferedPhysicsTrace


TRAINING_RUN = V5 / "ppo_integration_v1/runs/2026-10-23_15-13-34_formal"
FINAL = TRAINING_RUN / "checkpoints/model_001034.pt"
FINAL_SHA = "83a66d45ad7cac3e74ea705d81033e49689e261871fbda93baad43eb73870e9d"


class EvaluationContractTests(unittest.TestCase):
    def test_final_ppo_actor_loads_strictly_as_325_mean_policy(self):
        actor, metadata = load_ppo_actor(FINAL, TRAINING_RUN)
        self.assertEqual(metadata["checkpoint_updates"], 1034)
        self.assertEqual(metadata["checkpoint_sha256"], FINAL_SHA)
        self.assertTrue(metadata["mean_action_only"])
        with torch.inference_mode():
            action = actor(torch.zeros(2, 325))
        self.assertEqual(tuple(action.shape), (2, 16))
        self.assertTrue(torch.isfinite(action).all())
        with self.assertRaisesRegex(ValueError, "325-dim"):
            actor(torch.zeros(2, 324))

    def test_checkpoint_must_belong_to_designated_training_run(self):
        with self.assertRaisesRegex(ValueError, "inside training-run/checkpoints"):
            load_ppo_actor(V5 / "structured_v1/runs/2026-10-23_full_01/actor.pt", TRAINING_RUN)

    def test_bounded_physics_buffer_preserves_dtype_and_sample_order(self):
        recorder = BufferedPhysicsTrace(capacity=2)
        for index in range(5):
            recorder.append(dict(time=torch.tensor([index + 1.], dtype=torch.float32),
                                 active=torch.tensor([index % 2 == 0], dtype=torch.bool)))
        arrays = recorder.arrays()
        self.assertEqual(recorder.samples, 5)
        self.assertEqual(recorder.used, 0)
        self.assertEqual(arrays["time"].dtype.name, "float32")
        self.assertEqual(arrays["active"].dtype.name, "bool")
        self.assertEqual(arrays["time"][:, 0].tolist(), [1., 2., 3., 4., 5.])
        self.assertEqual(arrays["active"][:, 0].tolist(), [True, False, True, False, True])
        with self.assertRaisesRegex(RuntimeError, "fields changed"):
            recorder.append(dict(other=torch.tensor([1.])))


if __name__ == "__main__":
    unittest.main()
