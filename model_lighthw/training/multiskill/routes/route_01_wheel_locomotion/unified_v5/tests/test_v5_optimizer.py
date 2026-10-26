"""CPU integration checks using the installed RSL ActorCritic and rollout storage."""
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import rsl_rl.runners.on_policy_runner as runner_module

import torch
from rsl_rl.modules import ActorCritic

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "source/rl_training").is_dir())
PPO_PATH = REPO / "source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/wheel_unified_v5/ppo.py"
SPEC = importlib.util.spec_from_file_location("r1_v5_optimizer_under_test", PPO_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
V5PPO = MODULE.V5PPO


def make_algorithm(**overrides):
    torch.manual_seed(23)
    policy = ActorCritic(300, 252, 16, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                         init_noise_std=.005, noise_std_type="log")
    options = dict(device="cpu", learning_rate=1.e-5, critic_learning_rate=1.e-4,
                   num_learning_epochs=3, num_mini_batches=4, critic_warmup_updates=0,
                   initial_std_by_action=[.005] * 12 + [.01] * 4,
                   min_std_by_action=[.001] * 12 + [.002] * 4,
                   max_std_by_action=[.01] * 12 + [.02] * 4,
                   entropy_coef=.001, schedule="fixed", gamma=.995, lam=.95,
                   kl_stop_threshold=.01)
    options.update(overrides)
    algorithm = V5PPO(policy, **options)
    algorithm.init_storage("rl", 8, 8, [300], [252], [16])
    return algorithm


def collect(algorithm, seed=11):
    torch.manual_seed(seed)
    with torch.inference_mode():
        for step in range(8):
            obs, critic_obs = torch.randn(8, 300), torch.randn(8, 252)
            algorithm.act(obs, critic_obs)
            reward = torch.sin(critic_obs[:, 0]) + .2 * step
            dones = torch.zeros(8, dtype=torch.bool)
            dones[step % 8] = step % 3 == 0
            algorithm.process_env_step(reward, dones, {})
        algorithm.compute_returns(torch.randn(8, 252))


def nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(nested_equal(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(nested_equal(a, b) for a, b in zip(first, second))
    return first == second


class V5OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_finite_metrics(self, metrics):
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_critic_warmup_preserves_actor_and_exploration_then_enables_actor(self):
        algorithm = make_algorithm(critic_warmup_updates=1)
        actor_before = deepcopy(algorithm.policy.actor.state_dict())
        std_before = algorithm.policy.log_std.clone()
        critic_before = deepcopy(algorithm.policy.critic.state_dict())
        collect(algorithm)
        metrics = algorithm.update()
        self.assert_finite_metrics(metrics)
        self.assertEqual(metrics["critic_only"], 1.)
        self.assertEqual(metrics["actor_optimizer_steps"], 0.)
        self.assertEqual(metrics["optimizer_steps"], 12.)
        self.assertTrue(nested_equal(actor_before, algorithm.policy.actor.state_dict()))
        self.assertTrue(torch.equal(std_before, algorithm.policy.log_std))
        self.assertFalse(nested_equal(critic_before, algorithm.policy.critic.state_dict()))
        self.assertFalse(any(parameter in algorithm.optimizer.state for parameter in algorithm.actor_parameters))
        collect(algorithm, seed=12)
        metrics = algorithm.update()
        self.assertEqual(metrics["critic_only"], 0.)
        self.assertGreater(metrics["actor_optimizer_steps"], 0.)
        self.assertFalse(nested_equal(actor_before, algorithm.policy.actor.state_dict()))

    def test_actual_rollout_has_fixed_group_rates_finite_losses_and_persistent_mask(self):
        algorithm = make_algorithm()
        collect(algorithm)
        metrics = algorithm.update()
        self.assert_finite_metrics(metrics)
        self.assertEqual([group["lr"] for group in algorithm.optimizer.param_groups], [1.e-5, 1.e-4])
        self.assertGreater(metrics["actor_gradient_norm"], 0.)
        self.assertGreater(metrics["critic_gradient_norm"], 0.)
        self.assertEqual(metrics["value_explained_variance_defined"], 1.)
        weight = algorithm.policy.actor[0].weight
        self.assertEqual(torch.count_nonzero(weight[:, 205:285]), 0)
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(torch.count_nonzero(algorithm.optimizer.state[weight][key][:, 205:285]), 0)
        self.assertEqual(algorithm.policy.act_inference(torch.randn(3, 300)).shape, (3, 16))

    def test_kl_rejection_restores_actor_and_keeps_all_critic_updates(self):
        algorithm = make_algorithm(learning_rate=.1, kl_stop_threshold=1.e-8)
        actor_before = deepcopy(algorithm.policy.actor.state_dict())
        noise_before = algorithm.policy.log_std.clone()
        critic_before = deepcopy(algorithm.policy.critic.state_dict())
        collect(algorithm)
        metrics = algorithm.update()
        self.assertEqual(metrics["kl_stopped"], 1.)
        self.assertEqual(metrics["optimizer_steps"], 12.)
        self.assertEqual(metrics["actor_optimizer_steps"], 0.)
        self.assertEqual(metrics["actor_rejected_steps"], 1.)
        self.assertEqual(metrics["actor_proposal_steps"], 1.)
        self.assertEqual(metrics["kl_rollout"], 0.)
        self.assertTrue(nested_equal(actor_before, algorithm.policy.actor.state_dict()))
        self.assertTrue(torch.equal(noise_before, algorithm.policy.log_std))
        self.assertFalse(nested_equal(critic_before, algorithm.policy.critic.state_dict()))
        self.assertFalse(any(parameter in algorithm.optimizer.state for parameter in algorithm.actor_parameters))
        self.assertTrue(all(int(algorithm.optimizer.state[p]["step"]) == 12 for p in algorithm.critic_parameters))
        self.assertEqual([group["lr"] for group in algorithm.optimizer.param_groups], [.1, 1.e-4])
        self.assertEqual(algorithm.storage.step, 0)

    def test_backtracked_actor_step_satisfies_full_rollout_kl_and_keeps_adam_moments(self):
        algorithm = make_algorithm(learning_rate=1.e-4, num_learning_epochs=1, num_mini_batches=1)
        actor_before = deepcopy(algorithm.policy.actor.state_dict())
        collect(algorithm)
        metrics = algorithm.update()
        self.assertEqual(metrics["actor_optimizer_steps"], 1.)
        self.assertEqual(metrics["actor_rejected_steps"], 0.)
        self.assertGreater(metrics["actor_backtracks_total"], 0.)
        self.assertGreater(metrics["actor_accepted_fraction_min"], 0.)
        self.assertLess(metrics["actor_accepted_fraction_min"], 1.)
        self.assertLessEqual(metrics["kl_rollout"], algorithm.kl_stop_threshold)
        self.assertGreater(metrics["actor_proposal_kl_max"], algorithm.kl_stop_threshold)
        self.assertFalse(nested_equal(actor_before, algorithm.policy.actor.state_dict()))
        self.assertTrue(all(int(algorithm.optimizer.state[p]["step"]) == 1 for p in algorithm.actor_parameters))
        self.assertTrue(any(torch.count_nonzero(algorithm.optimizer.state[p]["exp_avg"]) for p in algorithm.actor_parameters))

    def test_rejected_step_restores_existing_actor_adam_state_exactly(self):
        algorithm = make_algorithm(num_learning_epochs=1, num_mini_batches=1)
        collect(algorithm)
        algorithm.update()
        algorithm.learning_rate = algorithm.optimizer.param_groups[0]["lr"] = .1
        algorithm.kl_stop_threshold = 1.e-8
        before_parameters = [p.clone() for p in algorithm.actor_parameters]
        before_adam = [deepcopy(algorithm.optimizer.state[p]) for p in algorithm.actor_parameters]
        collect(algorithm, seed=44)
        metrics = algorithm.update()
        self.assertEqual(metrics["actor_rejected_steps"], 1.)
        self.assertTrue(all(torch.equal(p, previous) for p, previous in zip(algorithm.actor_parameters, before_parameters)))
        self.assertTrue(all(nested_equal(algorithm.optimizer.state[p], previous)
                            for p, previous in zip(algorithm.actor_parameters, before_adam)))
        self.assertTrue(all(int(algorithm.optimizer.state[p]["step"]) == 2 for p in algorithm.critic_parameters))

    def test_checkpoint_restores_both_adam_groups_warmup_and_next_update_exactly(self):
        algorithm = make_algorithm(critic_warmup_updates=1, reference_loss_coef=.1)
        obs = torch.randn(7, 300)
        with torch.no_grad():
            actions = algorithm.policy.act_inference(obs)
        algorithm.set_reference_data(obs, actions, torch.tensor([True] * 5 + [False] * 2))
        collect(algorithm)
        algorithm.update()
        collect(algorithm, seed=12)
        algorithm.update()
        buffer = io.BytesIO()
        torch.save({"model": algorithm.policy.state_dict(), "optimizer": algorithm.optimizer.state_dict(),
                    "algorithm": algorithm.training_state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=False)
        restored = make_algorithm(critic_warmup_updates=1, reference_loss_coef=.1)
        restored.policy.load_state_dict(saved["model"])
        restored.optimizer.load_state_dict(saved["optimizer"])
        restored.load_training_state_dict(saved["algorithm"])
        self.assertTrue(nested_equal(algorithm.optimizer.state_dict(), restored.optimizer.state_dict()))
        self.assertEqual(restored.completed_updates, 2)
        collect(algorithm, seed=99)
        metrics_a = algorithm.update()
        collect(restored, seed=99)
        metrics_b = restored.update()
        self.assertEqual(metrics_a, metrics_b)
        self.assertTrue(nested_equal(algorithm.policy.state_dict(), restored.policy.state_dict()))
        self.assertTrue(nested_equal(algorithm.optimizer.state_dict(), restored.optimizer.state_dict()))
        self.assertTrue(nested_equal(algorithm.training_state_dict(), restored.training_state_dict()))

    def test_reference_requires_success_labels_and_does_not_admit_failed_samples(self):
        algorithm = make_algorithm(reference_loss_coef=.1)
        obs, targets = torch.randn(4, 300), torch.zeros(4, 16)
        obs[2:] = 12345.
        targets[2:] = -12345.
        algorithm.set_reference_data(obs, targets, torch.tensor([True, True, False, False]))
        self.assertEqual(len(algorithm.reference_observations), 2)
        self.assertTrue((algorithm.reference_actions == 0.).all())
        self.assertFalse((algorithm.reference_observations == 12345.).any())
        self.assertFalse(any("reference" in name for name in algorithm.policy.state_dict()))
        collect(algorithm)
        self.assertGreater(algorithm.update()["reference"], 0.)
        with self.assertRaisesRegex(ValueError, "boolean success"):
            algorithm.set_reference_data(obs, targets, torch.ones(4))
        algorithm.set_reference_data(obs, targets, torch.zeros(4, dtype=torch.bool))
        collect(algorithm)
        with self.assertRaisesRegex(RuntimeError, "verified-success"):
            algorithm.update()

    def test_entropy_cannot_expand_std_above_configured_limits(self):
        algorithm = make_algorithm(learning_rate=.1, entropy_coef=1000., kl_stop_threshold=1.e10)
        collect(algorithm)
        algorithm.update()
        std = algorithm.policy.log_std.exp()
        self.assertTrue((std >= algorithm.std_min * (1. - 1.e-6)).all())
        self.assertTrue((std <= algorithm.std_max * (1. + 1.e-6)).all())
        self.assertTrue(torch.isfinite(std).all())

    def test_installed_runner_constructs_and_restores_both_adam_groups(self):
        class ObservationEnv:
            num_envs, num_actions = 8, 16

            def get_observations(self):
                return torch.zeros(8, 300), {"observations": {"critic": torch.zeros(8, 252)}}

        config = dict(num_steps_per_env=8, save_interval=1, empirical_normalization=False,
                      policy=dict(class_name="ActorCritic", actor_hidden_dims=[512, 256, 128],
                                  critic_hidden_dims=[512, 256, 128], init_noise_std=.005, noise_std_type="log"),
                      algorithm=dict(class_name="PPO", learning_rate=1.e-5, critic_learning_rate=1.e-4,
                                     critic_warmup_updates=0, num_learning_epochs=3, num_mini_batches=4))
        with patch.object(runner_module, "PPO", V5PPO):
            runner = runner_module.OnPolicyRunner(ObservationEnv(), deepcopy(config), device="cpu")
            resumed = runner_module.OnPolicyRunner(ObservationEnv(), deepcopy(config), device="cpu")
        self.assertIsInstance(runner.alg, V5PPO)
        collect(runner.alg)
        runner.alg.update()
        runner.logger_type = "tensorboard"
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            runner.save(path, infos={"v5_optimizer_state": runner.alg.training_state_dict()})
            infos = resumed.load(path, load_optimizer=True)
            resumed.alg.load_training_state_dict(infos["v5_optimizer_state"])
        self.assertTrue(nested_equal(runner.alg.optimizer.state_dict(), resumed.alg.optimizer.state_dict()))
        self.assertTrue(nested_equal(runner.alg.training_state_dict(), resumed.alg.training_state_dict()))
        self.assertTrue(nested_equal(runner.alg.policy.state_dict(), resumed.alg.policy.state_dict()))

    def test_resume_rejects_changed_optimizer_configuration(self):
        algorithm = make_algorithm()
        changed = make_algorithm(kl_stop_threshold=.02)
        with self.assertRaisesRegex(ValueError, "Incompatible V5"):
            changed.load_training_state_dict(algorithm.training_state_dict())

    def test_resume_rejects_mask_corruption_and_incomplete_optimizer(self):
        algorithm = make_algorithm()
        collect(algorithm)
        algorithm.update()
        state = algorithm.training_state_dict()
        with torch.no_grad():
            algorithm.policy.actor[0].weight[0, 205] = 1.
        with self.assertRaisesRegex(ValueError, "masked Actor"):
            algorithm.load_training_state_dict(state)
        algorithm._project_constraints()
        algorithm.optimizer.param_groups[1]["lr"] = 1.e-5
        with self.assertRaisesRegex(ValueError, "learning rates"):
            algorithm.load_training_state_dict(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
