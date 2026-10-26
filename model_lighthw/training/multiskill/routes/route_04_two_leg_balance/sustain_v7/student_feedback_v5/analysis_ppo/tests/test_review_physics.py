"""CPU counterexamples for independent PPO physics review; no simulator claim."""

import math
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_physics import (RAW_WIDTHS, TELEMETRY_WIDTHS, TraceContractError,
                            analyse_episode, audit_training_artifacts,
                            episode_slices, sha256, trace_arrays)


PITCH = -.98


def stable_trace(ticks=16050):
    """Analytic 1 ms rear stand: entry at 1 s, hold begins 4 s, capture 5.999 s."""
    shape = {**RAW_WIDTHS, **TELEMETRY_WIDTHS}
    d = {key: np.zeros((ticks, 1, width), dtype=np.float32) for key, width in shape.items()}
    time = (np.arange(1, ticks + 1) * .001).astype(np.float32)
    d["physics_time_s"][:, 0, 0] = time
    d["mode"][:, 0, 0] = 1.
    d["stand_request"][:, 0, 0] = 1.
    d["base_z_m"][:, 0, 0] = .416
    d["com_w"][:, 0, 2] = .416
    wheel = d["wheel_pos_w"].reshape(ticks, 1, 4, 3)
    wheel[:, 0, 0, :2] = [.3, -.1]
    wheel[:, 0, 1, :2] = [.3, .1]
    wheel[:, 0, 2, :2] = [0., -.1]
    wheel[:, 0, 3, :2] = [0., .1]
    wheel[..., 2] = .087501
    hold = time >= 4.
    wheel[hold, 0, :2, 2] = .127501
    d["clearance_m"][:, 0] = wheel[:, 0, :, 2] - .087501
    d["force_z_n"][:, 0] = [8., 8., 8., 8.]
    d["force_z_n"][hold, 0, :2] = 0.
    entered = time >= 1.
    d["start_time_s"][:, 0, 0] = np.where(entered, 1., -1.)
    d["phase"][:, 0, 0] = np.where(hold, 2., np.where(entered, 1., 0.))
    x = np.clip((time.astype(np.float64) - 1.) / 3., 0., 1.)
    amplitude = x*x*(3.-2.*x)
    pitch = float(np.float32(PITCH)) * amplitude
    target = np.stack((np.sin(pitch), np.zeros(ticks), -np.cos(pitch)), -1).astype(np.float32)
    d["target_gravity"][:, 0] = target
    d["gravity"][:, 0] = target
    first_hold = int(np.flatnonzero(hold)[0])
    count = np.maximum(np.arange(ticks) - first_hold + 1, 0)
    capture = first_hold + 1999
    post = np.maximum(np.arange(ticks) - capture, 0)
    d["hold_time_s"][:, 0, 0] = count * .001
    d["best_hold_s"][:, 0, 0] = count * .001
    d["held"][capture:, 0, 0] = 1.
    d["capture_time_s"][:, 0, 0] = -1.
    d["capture_time_s"][capture:, 0, 0] = time[capture]
    d["first_break_time_s"][:, 0, 0] = -1.
    d["post_capture_s"][:, 0, 0] = post * .001
    d["success"][post >= 10000, 0, 0] = 1.
    return d


class ReviewTests(unittest.TestCase):
    def test_analytic_stable_capture_and_extra_ten_seconds(self):
        data = stable_trace()
        result = analyse_episode(data, 0, slice(0, len(data["physics_time_s"])), PITCH)
        self.assertTrue(result["gate_pass"], result["telemetry_comparison"])
        self.assertEqual(result["capture_time_s"], float(data["physics_time_s"][5998, 0, 0]))
        self.assertEqual(result["post_capture_uninterrupted_s"], 10.051)
        self.assertTrue(result["success_at_end"])

    def test_one_front_contact_break_is_irreversible_after_recovery(self):
        data = stable_trace()
        data["force_z_n"][9000, 0, 0] = 2.  # v7 requires |front force| < 2 N.
        result = analyse_episode(data, 0, slice(0, len(data["physics_time_s"])), PITCH)
        self.assertFalse(result["gate_pass"])
        self.assertTrue(result["interrupted"])
        self.assertIn("rear_pair", result["first_break_gates"])
        self.assertLess(result["post_capture_uninterrupted_s"], 10.)
        self.assertTrue(result["qualified_at_end"])

    def test_raw_hf_violation_is_not_hidden_by_zero_logged_rms(self):
        data = stable_trace()
        t = data["physics_time_s"][:, 0, 0]
        active = t >= 8.
        data["leg_position_rad"][active, 0, 0] = .02 * np.sin(2 * math.pi * 15 * t[active])
        result = analyse_episode(data, 0, slice(0, len(t)), PITCH)
        self.assertFalse(result["gate_pass"])
        self.assertTrue(result["interrupted"])
        self.assertGreater(result["telemetry_comparison"]["hf_limit_disagreement"], 0)
        self.assertFalse(result["reconstruction_verified"])

    def test_inactive_hf_component_rounding_is_reported_without_changing_decision(self):
        data = stable_trace()
        index = 3000  # phase 1; no capture has begun.
        limit = math.radians(.2)
        independent_hf = np.zeros((len(data["physics_time_s"]), 4))
        independent_hf[index, 1] = .1  # Both paths fail the aggregate smooth gate.
        independent_hf[index, 2] = limit + 4e-8
        data["jitter_rms"][index, 0, 1] = .1
        data["jitter_rms"][index, 0, 2] = limit - 4e-8
        with patch("review_physics.high_frequency_rms", return_value=independent_hf):
            result = analyse_episode(data, 0, slice(0, len(data["physics_time_s"])), PITCH)
        telemetry = result["telemetry_comparison"]
        self.assertTrue(result["gate_pass"])
        self.assertEqual(telemetry["hf_limit_disagreement"], 1)
        self.assertEqual(telemetry["smooth_gate_disagreement"], 0)
        self.assertFalse(telemetry["hf_limit_disagreement_details"][0]["capture_already_latched"])

    def test_episode_rollback_and_clock_gap(self):
        time = np.r_[np.arange(1, 101)*.001, np.arange(1, 51)*.001]
        pieces = episode_slices(time)
        self.assertEqual([(p.start, p.stop) for p in pieces], [(0, 100), (100, 150)])
        time[20] = time[19]
        with self.assertRaises(TraceContractError):
            episode_slices(time)

    def test_raw_fields_are_mandatory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.npz"
            data = {key: np.zeros((1, 1, width), dtype=np.float32)
                    for key, width in {**RAW_WIDTHS, **TELEMETRY_WIDTHS}.items()}
            data.pop("com_w")
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(TraceContractError, "Missing raw/telemetry fields: com_w"):
                trace_arrays(path)

    def test_checkpoint_lineage_and_real_adam_step_audit(self):
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            actor = run / "actor.pt"
            torch.save({"actor_state_dict": {
                "prefix.weight": torch.tensor([1.]),
                "coefficients": torch.zeros(16, 6),
            }}, actor)
            checkpoint = run / "model.pt"
            torch.save({
                "model_state_dict": {
                    "actor.prefix.weight": torch.tensor([1.]),
                    "actor.coefficients": torch.ones(16, 6),
                },
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(2)}}},
                "counters": {"completed_updates": 1, "control_steps": 1,
                             "transition_samples": 1, "optimizer_steps": 2},
            }, checkpoint)
            update = {"actual_critic_adam_steps": 2,
                      "collection": {"timeouts": 0, "physical_terminations": 0},
                      "bootstrap": {"pre_reset_bootstrap_once_verified": True,
                                    "timeout_count": 0, "physical_termination_count": 0}}
            (run / "updates.jsonl").write_text(json.dumps(update) + "\n")
            request = {"new_updates": 1, "steps_per_env": 1, "actor": str(actor),
                       "actor_sha256": sha256(actor), "resume_checkpoint": None}
            result = {"final_checkpoint": str(checkpoint),
                      "final_checkpoint_sha256": sha256(checkpoint),
                      "actor_sha256": sha256(actor),
                      "timeout_count": 0, "physical_termination_count": 0,
                      "actual_critic_adam_steps": 2, "initial_updates": 0,
                      "completed_updates": 1, "control_steps": 1,
                      "transition_samples": 1, "expected_optimizer_steps": 2,
                      "feature_build_calls": 2, "actor_coefficients_changed": True}
            evidence = audit_training_artifacts(run, request, result)
            self.assertTrue(evidence["frozen_prefix_exact"])
            self.assertTrue(evidence["coefficients_changed_this_run"])
            update["bootstrap"]["pre_reset_bootstrap_once_verified"] = False
            (run / "updates.jsonl").write_text(json.dumps(update) + "\n")
            with self.assertRaisesRegex(TraceContractError, "bootstrap audit is absent"):
                audit_training_artifacts(run, request, result)

    def test_formal_migration_changes_only_rollout_length(self):
        import copy
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "checkpoints").mkdir()
            actor = run / "actor.pt"
            torch.save({"actor_state_dict": {
                "prefix.weight": torch.tensor([1.]),
                "coefficients": torch.zeros(16, 6),
            }}, actor)
            source_path = run / "source.pt"
            source = {
                "config": {"num_steps_per_env": 512, "gamma": .995},
                "model_state_dict": {"actor.prefix.weight": torch.tensor([1.]),
                                     "actor.coefficients": torch.ones(16, 6)},
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(4)}}},
                "rng": {"numpy": np.arange(3), "torch_cpu": torch.arange(3)},
                "counters": {"completed_updates": 2, "control_steps": 1024,
                             "transition_samples": 16384, "optimizer_steps": 4},
            }
            torch.save(source, source_path)
            migrated_path = run / "checkpoints/migrated_input.pt"
            migrated = copy.deepcopy(source)
            migrated["config"]["num_steps_per_env"] = 32
            migrated["explicit_migration"] = {"source_checkpoint_sha256": sha256(source_path)}
            torch.save(migrated, migrated_path)
            final_path = run / "final.pt"
            final = copy.deepcopy(migrated)
            final["model_state_dict"]["actor.coefficients"] += .1
            final["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(68)
            final["counters"].update(completed_updates=34, control_steps=2048,
                                      transition_samples=32768, optimizer_steps=68)
            torch.save(final, final_path)
            rows = [{"actual_critic_adam_steps": 6 + 2*i,
                     "collection": {"timeouts": 0, "physical_terminations": 0},
                     "bootstrap": {"pre_reset_bootstrap_once_verified": True,
                                   "timeout_count": 0, "physical_termination_count": 0}}
                    for i in range(32)]
            (run / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            request = {"new_updates": 32, "steps_per_env": 32, "actor": str(actor),
                       "actor_sha256": sha256(actor), "resume_checkpoint": str(source_path),
                       "resume_checkpoint_sha256": sha256(source_path),
                       "checkpoint_migration": {"destination_steps_per_env": 32}}
            result = {"final_checkpoint": str(final_path),
                      "final_checkpoint_sha256": sha256(final_path),
                      "actor_sha256": sha256(actor),
                      "timeout_count": 0, "physical_termination_count": 0,
                      "actual_critic_adam_steps": 68, "initial_updates": 2,
                      "completed_updates": 34, "control_steps": 2048,
                      "transition_samples": 32768, "expected_optimizer_steps": 68,
                      "feature_build_calls": 1025, "actor_coefficients_changed": True,
                      "restored_state_verified": True, "checkpoint_migration_verified": True}
            evidence = audit_training_artifacts(run, request, result)
            self.assertTrue(evidence["checkpoint_migration_exact"])
            migrated["model_state_dict"]["actor.coefficients"][0, 0] = -9.
            torch.save(migrated, migrated_path)
            with self.assertRaisesRegex(TraceContractError, "migration changed"):
                audit_training_artifacts(run, request, result)


if __name__ == "__main__":
    unittest.main()
