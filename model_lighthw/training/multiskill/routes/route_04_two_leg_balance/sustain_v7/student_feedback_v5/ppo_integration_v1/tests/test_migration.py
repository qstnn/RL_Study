"""CPU-only migration and preflight checks; fabricated checkpoints are not robot evidence."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ppo_design"))
sys.path.insert(0, str(HERE.parent / "structured_v1"))
from common_feedback import sha
from continuing_driver import DriverConfig
from gate import AdmissionError
from robot_worker import migrate_formal_checkpoint
from run_ppo import main


class MigrationTests(unittest.TestCase):
    def test_only_rollout_length_changes_and_state_is_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.pt", root / "migrated.pt"
            old = DriverConfig(actor_obs_dim=325, num_steps_per_env=512,
                initial_std=.0005, learning_rate=3e-6)
            new = DriverConfig(actor_obs_dim=325, num_steps_per_env=32,
                initial_std=.0005, learning_rate=3e-6)
            saved = dict(config=dict(old.__dict__), num_envs=16,
                model_state_dict=dict(actor=torch.tensor([1., 2.])),
                optimizer_state_dict=dict(state={0: dict(step=torch.tensor(4.))}),
                rng=dict(numpy=np.random.get_state(), torch_cpu=torch.get_rng_state()),
                counters=dict(completed_updates=2, control_steps=1024), learning_rate=3e-6)
            torch.save(saved, source)
            request = dict(mode="formal-smoke", checkpoint_migration=dict(
                source_checkpoint_sha256=sha(source), source_technical_limit_s=20.,
                destination_technical_limit_s=360., source_steps_per_env=512,
                destination_steps_per_env=32))
            migrate_formal_checkpoint(source, destination, request, new)
            migrated = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(migrated["config"]["num_steps_per_env"], 32)
            self.assertTrue(torch.equal(migrated["model_state_dict"]["actor"],
                                        saved["model_state_dict"]["actor"]))
            self.assertEqual(migrated["explicit_migration"]["source_checkpoint_sha256"], sha(source))
            with self.assertRaises(FileExistsError):
                migrate_formal_checkpoint(source, destination, request, new)
            bad = copy.deepcopy(request)
            bad["checkpoint_migration"]["destination_technical_limit_s"] = 20.
            with self.assertRaises(RuntimeError):
                migrate_formal_checkpoint(source, root / "bad.pt", bad, new)
            different_lr = DriverConfig(actor_obs_dim=325, num_steps_per_env=32,
                initial_std=.0005, learning_rate=1e-5)
            with self.assertRaises(RuntimeError):
                migrate_formal_checkpoint(source, root / "bad_lr.pt", request, different_lr)

    def test_missing_six_case_blocks_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "should_not_exist"
            with self.assertRaises(AdmissionError):
                main(["smoke", "--actor", str(root / "actor.pt"),
                    "--six-case-verification", str(root / "verification.json"),
                    "--output", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
