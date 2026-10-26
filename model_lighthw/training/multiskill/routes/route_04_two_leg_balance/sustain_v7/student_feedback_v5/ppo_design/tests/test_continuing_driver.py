"""CPU-only interface checks; synthetic transitions are not robot evidence."""

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuing_driver import ContinuingDriver, DriverConfig


class SyntheticVecEnv:
    """Auto-resetting RSL-shaped fake, with distinct pre/post-reset critic data."""

    def __init__(self, count, timeout_age=5, failure_age=7):
        self.num_envs = count
        self.num_actions = 16
        self.device = "cpu"
        self.timeout_age = timeout_age
        self.failure_age = failure_age
        self.age = torch.zeros(count, dtype=torch.long)
        self.total_steps = 0
        self.reset_calls = 0
        self.observation_reads = 0
        self.last_raw_reward = None
        self.last_terminal_critic = None
        self.last_post_reset_critic = None

    def _raw(self):
        raw = torch.zeros(self.num_envs, 305)
        raw[:, 0] = self.age.float() * 0.1
        raw[:, 3] = 0.2
        raw[:, 5] = -0.8
        raw[:, 60] = 1.0
        return raw

    def _critic(self):
        critic = torch.zeros(self.num_envs, 88)
        critic[:, 0] = self.age.float() + 10.0
        critic[:, 1] = self.age.float() * 0.25
        return critic

    def get_observations(self):
        self.observation_reads += 1
        return self._raw(), {"observations": {"critic": self._critic()}}

    def reset(self):
        self.reset_calls += 1
        self.age.zero_()
        return self.get_observations()

    def step(self, action):
        assert action.shape == (self.num_envs, 16)
        self.age += 1
        self.total_steps += 1
        # Even slots truncate, odd slots physically terminate. The synthetic
        # reason codes are only for testing the PPO transport contract.
        slot = torch.arange(self.num_envs)
        timeout = (slot % 2 == 0) & (self.age >= self.timeout_age)
        failure = (slot % 2 == 1) & (self.age >= self.failure_age)
        done = timeout | failure
        reward = 1.0 - 0.01 * action.square().sum(-1)
        terminal = self._critic().clone()
        self.last_raw_reward = reward.clone()
        self.last_terminal_critic = terminal.clone()
        self.age[done] = 0
        reset_critic = self._critic()
        self.last_post_reset_critic = reset_critic.clone()
        info = {
            "observations": {"critic": reset_critic},
            "time_outs": timeout,
            "physical_terminated": failure,
            "terminal_observation_valid": done,
            "terminal_critic_observation": terminal,
        }
        return self._raw(), reward, done.long(), info


class CausalFeatures:
    def __init__(self, count):
        self.age = torch.zeros(count, dtype=torch.long)
        self.calls = 0
        self.reset_masks = []

    def __call__(self, raw, done):
        self.calls += 1
        self.reset_masks.append(done.clone())
        self.age = torch.where(done, 0, self.age + 1)
        # Four illustrative causal inputs: memory latch/age plus two candidate
        # body-feature slots. Their values are fake and never used as labels.
        extra = torch.stack((self.age.float() / 10, done.float(),
                             raw[:, 0], raw[:, 3]), -1)
        return torch.cat((raw, extra), -1)


def learner(count=2, steps=2, timeout_age=5, failure_age=7,
            observation_contract="raw305_memory2_body2_draft", policy_contract="linear309_v1"):
    env = SyntheticVecEnv(count, timeout_age, failure_age)
    feature = CausalFeatures(count)
    actor = nn.Linear(309, 16)
    config = DriverConfig(actor_obs_dim=309, num_steps_per_env=steps,
                          num_learning_epochs=1, num_mini_batches=1,
                          initial_std=0.05, learning_rate=3e-4)
    driver = ContinuingDriver(env, actor, feature, config,
                              observation_contract=observation_contract,
                              policy_contract=policy_contract)
    return driver, env, feature


class PaddedFeatures(CausalFeatures):
    def __call__(self, raw, done):
        return torch.cat((super().__call__(raw, done), torch.zeros(len(raw), 16, device=raw.device)), -1)


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

    def test_actual_rsl_updates_work_for_2_16_and_64_slots(self):
        for count in (2, 16, 64):
            with self.subTest(count=count):
                driver, env, feature = learner(count=count)
                before = {k: v.clone() for k, v in driver.policy.state_dict().items()}
                first = driver.run_iteration()
                second = driver.run_iteration()
                self.assertEqual(first["completed_updates"], 1)
                self.assertEqual(second["completed_updates"], 2)
                self.assertEqual(second["optimizer_steps"], 2)
                self.assertEqual(second["control_steps"], 4)
                self.assertEqual(second["transition_samples"], 4 * count)
                self.assertEqual(driver.alg.storage.step, 0)
                self.assertEqual(env.observation_reads, 1)
                self.assertEqual(env.reset_calls, 0)
                self.assertEqual(feature.calls, 5)
                self.assertTrue(any(not torch.equal(v, before[k]) for k, v in driver.policy.state_dict().items()))
                self.assertFalse(torch.equal(driver.policy.log_std, before["log_std"]))
                self.assertTrue(all(torch.isfinite(v).all() for v in driver.policy.state_dict().values()))
                self.assertGreater(len(driver.alg.optimizer.state), 0)
                self.assertTrue(all(np.isfinite(v) for v in second["losses"].values()))

    def test_replaced_actor_is_on_optimizer_and_eval_loader_is_switched_to_train(self):
        env = SyntheticVecEnv(2)
        actor = nn.Linear(325, 16).eval()
        feature = PaddedFeatures(2)
        config = DriverConfig(actor_obs_dim=325, num_steps_per_env=2, num_learning_epochs=1)
        driver = ContinuingDriver(env, actor, feature, config,
                                  observation_contract="raw305_memory2_body2_start16_draft",
                                  policy_contract="linear325_v1")
        self.assertTrue(actor.training)
        self.assertIs(driver.policy.actor, actor)
        optimizer_parameters = {id(p) for group in driver.alg.optimizer.param_groups for p in group["params"]}
        self.assertTrue({id(p) for p in actor.parameters()} <= optimizer_parameters)
        before = actor.weight.detach().clone()
        with torch.no_grad():
            driver.policy.eval()
            mean_eval = driver.policy.act_inference(driver.obs)
            driver.policy.update_distribution(driver.obs)
            torch.testing.assert_close(driver.policy.action_std,
                                       driver.policy.log_std.exp().expand_as(mean_eval))
            driver.policy.train()
            torch.testing.assert_close(driver.policy.act_inference(driver.obs), mean_eval)
        driver.policy.eval()
        driver.run_iteration()
        self.assertTrue(actor.training)
        self.assertFalse(torch.equal(actor.weight, before))

    def test_device_and_terminal_shape_mismatch_fail_before_value_evaluation(self):
        env = SyntheticVecEnv(2)
        env.device = "cuda:0"
        with self.assertRaisesRegex(ValueError, "same device"):
            ContinuingDriver(env, nn.Linear(309, 16), CausalFeatures(2),
                             DriverConfig(actor_obs_dim=309),
                             observation_contract="cpu_test", policy_contract="linear")

        class BrokenTerminal(SyntheticVecEnv):
            def step(self, action):
                raw, reward, done, info = super().step(action)
                info["terminal_critic_observation"] = info["terminal_critic_observation"][:, :87]
                return raw, reward, done, info

        env = BrokenTerminal(2, timeout_age=1, failure_age=1)
        driver = ContinuingDriver(env, nn.Linear(309, 16), CausalFeatures(2),
                                  DriverConfig(actor_obs_dim=309, num_steps_per_env=2),
                                  observation_contract="cpu_test", policy_contract="linear")
        with self.assertRaisesRegex(ValueError, "Pre-reset terminal Critic"):
            driver.collect_rollout()

    def test_timeout_uses_pre_reset_value_once_and_failure_none(self):
        driver, env, feature = learner(steps=2, timeout_age=2, failure_age=2)
        report = driver.collect_rollout()
        self.assertEqual(report["timeouts"], 1)
        self.assertEqual(report["physical_terminations"], 1)
        self.assertEqual(driver.alg.storage.step, 2)
        with torch.no_grad():
            terminal_value = driver.policy.evaluate(env.last_terminal_critic[:1]).flatten()[0]
            reset_value = driver.policy.evaluate(env.last_post_reset_critic[:1]).flatten()[0]
        expected = env.last_raw_reward[0] + driver.config.gamma * terminal_value
        torch.testing.assert_close(driver.alg.storage.rewards[1, 0, 0], expected)
        torch.testing.assert_close(driver.alg.storage.rewards[1, 1, 0], env.last_raw_reward[1])
        self.assertGreater(float((terminal_value - reset_value).abs()), 0.001)
        self.assertTrue(feature.reset_masks[-1].all())
        final = driver.update_from_rollout()
        self.assertEqual(final["bootstrap_events"], 1)
        self.assertEqual(driver.alg.storage.step, 0)

    def test_memory_continues_over_iteration_boundary_and_resets_only_done_slots(self):
        driver, env, feature = learner(steps=2, timeout_age=5, failure_age=7)
        driver.run_iteration()
        self.assertEqual(feature.age.tolist(), [2, 2])
        driver.run_iteration()
        self.assertEqual(feature.age.tolist(), [4, 4])
        driver.run_iteration()
        self.assertEqual(feature.age.tolist(), [1, 6])
        self.assertEqual(feature.reset_masks[-2].tolist(), [True, False])
        self.assertEqual(env.observation_reads, 1)
        self.assertEqual(env.reset_calls, 0)
        self.assertEqual(driver.feature_build_calls, 7)

    def test_checkpoint_restores_model_adam_std_learning_rate_counts_and_rng(self):
        driver, _, _ = learner(steps=2, timeout_age=20, failure_age=20)
        driver.run_iteration()
        driver.run_iteration()
        self.assertFalse(torch.equal(driver.policy.log_std, torch.full_like(driver.policy.log_std, np.log(0.05))))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_000002.pt"
            driver.save_checkpoint(path)
            with self.assertRaises(FileExistsError):
                driver.save_checkpoint(path)
            expected_random = (random.random(), np.random.rand(), torch.rand(3))
            restored, env, feature = learner(steps=2, timeout_age=20, failure_age=20)
            meta = restored.load_checkpoint(path)
            self.assertEqual(meta["completed_updates"], 2)
            self.assertTrue(meta["new_physics_episode"])
            self.assertFalse(meta["standing_continuity_across_processes"])
            self.assertEqual(restored.completed_updates, 2)
            self.assertEqual(restored.control_steps, 4)
            self.assertEqual(restored.transition_samples, 8)
            self.assertEqual(restored.optimizer_steps, 2)
            self.assertEqual(restored.alg.learning_rate, driver.alg.learning_rate)
            for k, v in driver.policy.state_dict().items():
                torch.testing.assert_close(restored.policy.state_dict()[k], v, atol=0, rtol=0)
            self.assertEqual(restored.alg.optimizer.state_dict()["param_groups"],
                             driver.alg.optimizer.state_dict()["param_groups"])
            for key, original in driver.alg.optimizer.state_dict()["state"].items():
                recovered = restored.alg.optimizer.state_dict()["state"][key]
                for name, value in original.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(recovered[name], value, atol=0, rtol=0)
                    else:
                        self.assertEqual(recovered[name], value)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], atol=0, rtol=0)
            self.assertEqual(env.reset_calls, 0)
            self.assertEqual(feature.age.tolist(), [0, 0])
            self.assertEqual(restored.run_iteration()["completed_updates"], 3)

            wrong, _, _ = learner(steps=2, observation_contract="different")
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                wrong.load_checkpoint(path)

    def test_incomplete_rollout_cannot_be_saved_or_recollected(self):
        driver, _, _ = learner(steps=2)
        driver.collect_rollout()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                driver.save_checkpoint(Path(folder) / "invalid.pt")
        with self.assertRaises(RuntimeError):
            driver.collect_rollout()


if __name__ == "__main__":
    unittest.main()
