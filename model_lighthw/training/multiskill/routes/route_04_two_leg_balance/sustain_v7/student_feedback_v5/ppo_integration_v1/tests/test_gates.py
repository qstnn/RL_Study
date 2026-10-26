"""CPU-only gate tests with fabricated dictionaries; no robot acceptance evidence."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gate import AdmissionError, formal_review_gate, validate_six_case_data
from common_feedback import sha


def six_case_fixture():
    request = dict(actor_sha256="a" * 64, autonomous=True, seeds=[43, 44, 45],
        hold_seconds=10., return_after_s=None,
        programs=[dict(id="autonomous", windows=None)])
    cases = [dict(seed=seed, trace_env=slot, success=True, post_capture_s=10.01,
        teacher=False, failed=False, interrupted=False, boundary=False,
        return_time=-1., stand_request=True)
        for seed in (43, 44, 45) for slot in (0, 1)]
    result = dict(actor_sha256=request["actor_sha256"], autonomous=True,
        cases=cases, observation_passes=6)
    verification = dict(autonomous_10s_gate=True, independent_physics_verified=True,
        independent_proprioception_verified=True, autonomous=True, teacher_constructed=False,
        cases=[dict(seed=seed, program="autonomous", observation_pass=True, returned=False)
               for seed in (43, 44, 45) for _ in (0, 1)], passes=6, physics_samples=120000,
        action_checks=[dict(teacher_frames=0, executed_action_error=0) for _ in range(3)])
    return request, result, verification, dict(status="completed")


class GateTests(unittest.TestCase):
    def test_six_case_exact_contract(self):
        args = six_case_fixture()
        validate_six_case_data(*args, args[0]["actor_sha256"])
        for name, mutate in (
            ("actor SHA", lambda r, x, v, s: r.update(actor_sha256="b" * 64)),
            ("seeds", lambda r, x, v, s: r.update(seeds=[43, 44, 44])),
            ("one slot", lambda r, x, v, s: x["cases"].pop()),
            ("short hold", lambda r, x, v, s: x["cases"][0].update(post_capture_s=9.9)),
            ("teacher", lambda r, x, v, s: x["cases"][0].update(teacher=True)),
            ("interruption", lambda r, x, v, s: x["cases"][0].update(interrupted=True)),
            ("physical verification", lambda r, x, v, s: v.update(independent_physics_verified=False)),
            ("unfinished", lambda r, x, v, s: s.update(status="running")),
        ):
            with self.subTest(name=name), self.assertRaises(AdmissionError):
                bad = copy.deepcopy(args)
                mutate(*bad)
                validate_six_case_data(*bad, args[0]["actor_sha256"])

    def test_formal_requires_named_review_n16_and_fresh_process_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke, resume, formal_smoke = root / "smoke", root / "resume", root / "formal_smoke"
            smoke.mkdir(); resume.mkdir(); formal_smoke.mkdir()
            actor_sha = "a" * 64
            payloads = (
                (smoke, dict(mode="smoke", num_envs=16, new_updates=2,
                    actor_sha256=actor_sha, technical_pass=True,
                    feature_build_calls=1025, process_control_steps=1024,
                    driver_reset_calls=0, audit_reset_calls=0, actual_critic_adam_steps=4,
                    control_steps=1024, completed_updates=2,
                    final_checkpoint_sha256="c" * 64), {}),
                (resume, dict(mode="resume-smoke", num_envs=16, new_updates=1,
                    actor_sha256=actor_sha, technical_pass=True,
                    feature_build_calls=513, process_control_steps=512,
                    driver_reset_calls=0, audit_reset_calls=0, actual_critic_adam_steps=6,
                    new_physics_episode=True, restored_state_verified=True),
                    dict(resume_checkpoint_sha256="c" * 64)),
                (formal_smoke, dict(mode="formal-smoke", num_envs=16, new_updates=32,
                    steps_per_env=32, technical_time_limit_s=360., control_steps=2048,
                    completed_updates=34, physics_trace_samples=20480,
                    physics=dict(physics_samples_per_slot=[20480] * 16),
                    feature_build_calls=1025, process_control_steps=1024,
                    driver_reset_calls=0, audit_reset_calls=0, actual_critic_adam_steps=68,
                    actor_sha256=actor_sha, technical_pass=True, behavior_candidate_pass=True,
                    checkpoint_migration_verified=True,
                    new_physics_episode=True, restored_state_verified=True),
                    dict(resume_checkpoint_sha256="c" * 64)),
            )
            for run, result, request in payloads:
                (run / "result.json").write_text(json.dumps(result))
                (run / "request.json").write_text(json.dumps(request))
                (run / "session.json").write_text(json.dumps(dict(status="completed")))
            review = root / "review.json"
            trace = smoke / "physics_trace.npz"
            trace.write_bytes(b"fabricated CPU test trace")
            smoke_result = json.loads((smoke / "result.json").read_text())
            smoke_result["physics_trace_sha256"] = sha(trace)
            smoke_result["worker_pid"] = 100
            (smoke / "result.json").write_text(json.dumps(smoke_result))
            resume_result = json.loads((resume / "result.json").read_text())
            resume_result.update(worker_pid=101, initial_physics_steps=[0] * 16,
                                 initial_episode_counts=[0] * 16)
            (resume / "result.json").write_text(json.dumps(resume_result))
            formal_trace = formal_smoke / "physics_trace.npz"
            formal_trace.write_bytes(b"fabricated formal-config CPU test trace")
            formal_result = json.loads((formal_smoke / "result.json").read_text())
            formal_result.update(worker_pid=102, initial_physics_steps=[0] * 16,
                initial_episode_counts=[0] * 16, physics_trace_sha256=sha(formal_trace))
            (formal_smoke / "result.json").write_text(json.dumps(formal_result))
            value = dict(decision="approved", reviewer="independent technical reviewer",
                reviewed_at="2026-10-23T00:00:00+08:00", smoke_run=str(smoke), resume_run=str(resume),
                formal_smoke_run=str(formal_smoke),
                technical_review_pass=True, behavior_review_pass=True,
                behavior_cases=[dict(slot=slot, extra10s_pass=True, velocity_pass=True,
                    support_pass=True, high_frequency_pass=True, uninterrupted=True)
                    for slot in range(16)],
                smoke_physics_trace_sha256=sha(trace), formal_smoke_physics_trace_sha256=sha(formal_trace),
                smoke_result_sha256=sha(smoke / "result.json"),
                resume_result_sha256=sha(resume / "result.json"),
                formal_smoke_result_sha256=sha(formal_smoke / "result.json"))
            review.write_text(json.dumps(value))
            formal_review_gate(review, actor_sha, lambda run: None)
            value["decision"] = "pending"
            review.write_text(json.dumps(value))
            with self.assertRaises(AdmissionError):
                formal_review_gate(review, actor_sha, lambda run: None)
            value["decision"] = "approved"
            value["behavior_review_pass"] = False
            review.write_text(json.dumps(value))
            with self.assertRaises(AdmissionError):
                formal_review_gate(review, actor_sha, lambda run: None)
            value["behavior_review_pass"] = True
            formal_result = json.loads((formal_smoke / "result.json").read_text())
            formal_result["control_steps"] = 1024  # inherited count alone proves no new rollout
            (formal_smoke / "result.json").write_text(json.dumps(formal_result))
            value["formal_smoke_result_sha256"] = sha(formal_smoke / "result.json")
            review.write_text(json.dumps(value))
            with self.assertRaises(AdmissionError):
                formal_review_gate(review, actor_sha, lambda run: None)
            formal_result["control_steps"] = 2048
            (formal_smoke / "result.json").write_text(json.dumps(formal_result))
            value["formal_smoke_result_sha256"] = sha(formal_smoke / "result.json")
            smoke_result = json.loads((smoke / "result.json").read_text())
            smoke_result["num_envs"] = 2
            (smoke / "result.json").write_text(json.dumps(smoke_result))
            value["smoke_result_sha256"] = sha(smoke / "result.json")
            review.write_text(json.dumps(value))
            with self.assertRaises(AdmissionError):
                formal_review_gate(review, actor_sha, lambda run: None)


if __name__ == "__main__":
    unittest.main()
