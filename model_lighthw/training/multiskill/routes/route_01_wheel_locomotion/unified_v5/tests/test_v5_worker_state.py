"""CPU integration of actual worker methods; no Isaac Sim or GPU execution.

Runner methods are AST-extracted because the production class is scoped inside
main after Isaac startup. The frozen child is replaced only at the process seam;
quality review, checkpoint I/O, hashes, cache validation and RSL are real.
"""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import rsl_rl.runners.on_policy_runner as runner_module

ROOT = Path(__file__).resolve().parents[1]
REPO = next(parent for parent in ROOT.parents if (parent / "source/rl_training").is_dir())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load_module("v5_worker_test_launch", ROOT / "scripts/launch.py")
quality = load_module("v5_worker_test_quality", ROOT / "scripts/quality.py")
with patch.dict(sys.modules, {"launch": launch, "quality": quality}):
    worker = load_module("v5_worker_test_worker", ROOT / "scripts/worker.py")
optimizer_tests = load_module("v5_worker_test_optimizer", ROOT / "tests/test_v5_optimizer.py")
V5PPO = optimizer_tests.V5PPO


class CpuObservationEnv:
    num_envs, num_actions = 8, 16

    def get_observations(self):
        return torch.zeros(8, 300), {"observations": {"critic": torch.zeros(8, 252)}}


def actual_runner_class():
    tree = ast.parse((ROOT / "scripts/worker.py").read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "UnifiedRunner")
    namespace = dict(worker.__dict__, OnPolicyRunner=runner_module.OnPolicyRunner, spec=worker.SPEC,
                     args={"environment_signature": {"cpu_test": True}}, torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(ROOT / "scripts/worker.py"), "exec"), namespace)
    return namespace["UnifiedRunner"]


class CheckpointRunner:
    def __init__(self):
        self.alg = SimpleNamespace(completed_updates=0)
        self.quality_state = None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"iter": self.alg.completed_updates - 1, "quality": deepcopy(self.quality_state)}, path)


class WorkerStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_runner_zero_iteration_and_complete_state_round_trip(self):
        runner_class = actual_runner_class()
        config = dict(num_steps_per_env=8, save_interval=1, empirical_normalization=False,
                      policy=dict(class_name="ActorCritic", actor_hidden_dims=[512, 256, 128],
                                  critic_hidden_dims=[512, 256, 128], init_noise_std=.005, noise_std_type="log"),
                      algorithm=dict(class_name="PPO", learning_rate=1.e-5, critic_learning_rate=1.e-4,
                                     critic_warmup_updates=1, num_learning_epochs=3, num_mini_batches=4))
        with TemporaryDirectory() as directory, patch.object(runner_module, "PPO", V5PPO), \
                patch.object(torch.cuda, "is_available", return_value=False):
            first = runner_class(CpuObservationEnv(), deepcopy(config), device="cpu")
            initial = Path(directory) / "initial.pt"
            first.save(initial)
            saved = torch.load(initial, weights_only=False)
            self.assertEqual(saved["iter"], -1)
            self.assertEqual(saved["infos"]["completed_updates"], 0)
            self.assertEqual(first.current_learning_iteration, 0)
            optimizer_tests.collect(first.alg)
            first.alg.update()
            first.quality_state = {"protected": ["example"], "last": {"completed_updates": 1}}
            first.initialization_record = {"mode": "cpu_test"}
            trained_path = Path(directory) / "trained.pt"
            first.save(trained_path)
            saved = torch.load(trained_path, weights_only=False)
            self.assertEqual(saved["iter"], 0)
            second = runner_class(CpuObservationEnv(), deepcopy(config), device="cpu")
            second.load(trained_path, load_optimizer=True)
            self.assertTrue(optimizer_tests.nested_equal(first.alg.policy.state_dict(), second.alg.policy.state_dict()))
            self.assertTrue(optimizer_tests.nested_equal(first.alg.optimizer.state_dict(), second.alg.optimizer.state_dict()))
            self.assertTrue(optimizer_tests.nested_equal(first.alg.training_state_dict(), second.alg.training_state_dict()))
            self.assertEqual(first.quality_state, second.quality_state)
            self.assertEqual(first.initialization_record, second.initialization_record)
            self.assertTrue(torch.equal(torch.get_rng_state(), saved["infos"]["rng_state"]["torch_cpu"]))
            self.assertTrue(np.array_equal(np.random.get_state()[1], saved["infos"]["rng_state"]["numpy"][1]))
            self.assertEqual(random.getstate(), saved["infos"]["rng_state"]["python"])

    def test_periodic_baseline_and_two_regressions_keep_original_checkpoint(self):
        args = dict(device="cpu", ground_friction=4., rolling_coefficient=.02, leg_torque_limit=96.,
                    wheel_torque_limit=48., hitch_height=.07, unloaded_acceleration=1., loaded_acceleration=.3,
                    eval_distance=1., critic_warmup_updates=1)
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene={
            "robot": SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(16, 13)))}))
        flags = [True] * 15 + [False]

        def frozen_child(cmd, **kwargs):
            output = Path(cmd[cmd.index("--output_dir") + 1])
            output.mkdir(parents=True)
            checkpoint = Path(cmd[cmd.index("--checkpoint") + 1])
            result = dict(sha256=worker.sha256(checkpoint),
                          cases=[{"case": case, "success": passed} for case, passed in
                                 zip(worker.protocol.cases(1.), flags)],
                          pass_count=sum(flags), complete_count=16)
            worker.write_json(output / "evaluation.json", result)
            self.assertEqual(cmd[cmd.index("--critic_warmup_updates") + 1], "1")
            return SimpleNamespace(returncode=0)

        with TemporaryDirectory() as directory, patch("subprocess.run", side_effect=frozen_child):
            run = Path(directory)
            runner = CheckpointRunner()
            baseline = worker.periodic_evaluation(runner, env, args, run, -1, baseline=True)
            original_path = run / "periodic_evaluations/checkpoints/model_-1.pt"
            original_hash = worker.sha256(original_path)
            self.assertTrue(baseline["quality"]["promoted"])
            self.assertEqual(runner.quality_state["baseline_updates"], 0)
            flags[0] = False
            runner.alg.completed_updates = 100
            one = worker.periodic_evaluation(runner, env, args, run, 99)
            self.assertFalse(one["quality"]["stop"])
            self.assertFalse(one["quality"]["promoted"])
            # Same-update resume verification records evidence without another strike.
            resumed_run = run / "resumed"
            resumed_run.mkdir()
            worker.write_json(resumed_run / "quality/quality_state.json", runner.quality_state)
            repeated = worker.periodic_evaluation(runner, env, args, resumed_run, 99, baseline=True)
            self.assertTrue(repeated["quality"]["repeat_preflight_only"])
            self.assertFalse(repeated["quality"]["stop"])
            self.assertEqual(max(runner.quality_state["streaks"].values()), 1)
            runner.alg.completed_updates = 200
            with self.assertRaises(quality.QualityRegression):
                worker.periodic_evaluation(runner, env, args, run, 199)
            self.assertEqual(worker.sha256(original_path), original_hash)
            self.assertTrue(runner.quality_state["last"]["stop"])
            self.assertEqual(runner.quality_state["best_updates"], 0)

    def test_required_initialization_baseline_cannot_lower_protected_capabilities(self):
        args = dict(device="cpu", ground_friction=4., rolling_coefficient=.02, leg_torque_limit=96.,
                    wheel_torque_limit=48., hitch_height=.07, unloaded_acceleration=1., loaded_acceleration=.3,
                    eval_distance=1., critic_warmup_updates=1)
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene={
            "robot": SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(16, 13)))}))
        cases = worker.protocol.cases(1.)
        required_indices = list(range(10)) + list(range(12, 16))
        required = [quality.case_key(cases[index]) for index in required_indices]
        self.assertEqual(len(required), 14)
        self.assertTrue(all(key.endswith("|1.0") for key in required))
        for lose_required_case in (False, True):
            with self.subTest(lose_required_case=lose_required_case), TemporaryDirectory() as directory:
                run = Path(directory)
                runner = CheckpointRunner()
                runner.initialization_record = {"required_initialization_cases": required}
                flags = [index in required_indices for index in range(16)]
                if lose_required_case:
                    flags[0] = False

                def frozen_child(cmd, **kwargs):
                    output = Path(cmd[cmd.index("--output_dir") + 1])
                    output.mkdir(parents=True)
                    checkpoint = Path(cmd[cmd.index("--checkpoint") + 1])
                    worker.write_json(output / "evaluation.json", dict(
                        sha256=worker.sha256(checkpoint),
                        cases=[{"case": case, "success": passed} for case, passed in zip(cases, flags)],
                        pass_count=sum(flags), complete_count=16))
                    return SimpleNamespace(returncode=0)

                with patch("subprocess.run", side_effect=frozen_child):
                    if lose_required_case:
                        with self.assertRaisesRegex(RuntimeError, "Initialization baseline regression"):
                            worker.periodic_evaluation(runner, env, args, run, -1, baseline=True)
                        self.assertIsNone(runner.quality_state)
                        self.assertFalse((run / "quality/quality_state.json").exists())
                        self.assertFalse((run / "quality/best_eligible.pt").exists())
                    else:
                        record = worker.periodic_evaluation(runner, env, args, run, -1, baseline=True)
                        self.assertTrue(record["quality"]["promoted"])
                        self.assertEqual(set(runner.quality_state["protected"]), set(required))
                evidence = json.loads((run / "initialization_baseline_check.json").read_text())
                self.assertEqual(evidence["required_count"], 14)
                self.assertEqual(evidence["passed"], not lose_required_case)
                self.assertEqual(evidence["failed_required_cases"], [required[0]] if lose_required_case else [])
                self.assertEqual(runner.alg.completed_updates, 0)
                self.assertEqual(worker.sha256(evidence["checkpoint"]), evidence["checkpoint_sha256"])

    def test_reference_cache_sha_success_filter_and_missing_cache_rejection(self):
        algorithm = optimizer_tests.make_algorithm(reference_loss_coef=.1,
                                                  reference_action_scale=[.005] * 12 + [.01] * 4)
        runner = SimpleNamespace(alg=algorithm)
        observations = torch.randn(10, 300)
        with torch.no_grad():
            actions = algorithm.policy.act_inference(observations)
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "reference_success.npz"
            np.savez(cache, observation=observations.numpy(), action=actions.numpy(),
                     successful=np.array([True] * 8 + [False] * 2))
            saved = {"infos": {"initialization": {"reference_cache": {
                "path": cache.name, "sha256": worker.sha256(cache)}}}}
            checkpoint = Path(directory) / "model_init.pt"
            record = worker.install_reference_cache(runner, checkpoint, saved)
            self.assertEqual(record["successful_samples"], 8)
            self.assertEqual(record["rejected_samples"], 2)
            self.assertLess(record["initial_normalized_mse"], 1.e-8)
            self.assertLess(record["initial_raw_max_error"], 1.e-5)
            saved["infos"]["initialization"]["reference_cache"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                worker.install_reference_cache(runner, checkpoint, saved)
            for missing in (None, "legacy V4 initialization description"):
                with self.assertRaisesRegex(ValueError, "success cache"):
                    worker.install_reference_cache(runner, checkpoint, {"infos": {"initialization": missing}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
