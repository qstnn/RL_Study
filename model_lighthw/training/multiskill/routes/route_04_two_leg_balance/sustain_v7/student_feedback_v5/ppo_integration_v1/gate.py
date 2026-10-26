"""Hard admission gates for isolated R4a robot PPO. No simulator imports here."""

from __future__ import annotations

from collections import Counter
import math
import numbers
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STRUCTURED = HERE.parent / "structured_v1"
if str(STRUCTURED) not in sys.path:
    sys.path.insert(0, str(STRUCTURED))
from common_feedback import load_actor, read, require, sha  # noqa: E402


class AdmissionError(RuntimeError):
    pass


def _need(condition, message):
    if not condition:
        raise AdmissionError(message)


def _same_reconstruction(actual, archived):
    """Exact structure/decisions, 1e-6 tolerance only for computed floats."""
    if isinstance(archived, bool):
        return isinstance(actual, bool) and actual == archived
    if isinstance(archived, int):
        return isinstance(actual, numbers.Integral) and not isinstance(actual, bool) and actual == archived
    if isinstance(archived, float):
        if isinstance(actual, (bool, str, bytes)):
            return False
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and math.isfinite(archived) and math.isclose(
            value, archived, rel_tol=1e-6, abs_tol=1e-6)
    if type(actual) is not type(archived):
        return False
    if isinstance(actual, dict):
        return actual.keys() == archived.keys() and all(
            _same_reconstruction(actual[key], archived[key]) for key in actual)
    if isinstance(actual, list):
        return len(actual) == len(archived) and all(
            _same_reconstruction(a, b) for a, b in zip(actual, archived))
    return actual == archived


def validate_six_case_data(request, result, verification, session, actor_sha):
    """Independently check the fixed two-slot, three-seed autonomous contract."""
    _need(session.get("status") == "completed", "Six-case session is not completed")
    _need(request.get("actor_sha256") == result.get("actor_sha256") == actor_sha,
          "Request/result/current Actor SHA differ")
    _need(request.get("autonomous") is True and result.get("autonomous") is True,
          "Evaluation was not autonomous")
    _need(request.get("seeds") == [43, 44, 45], "Seeds must be 43/44/45")
    _need(float(request.get("hold_seconds", 0)) >= 10., "Extra observation must be at least 10 s")
    _need(request.get("return_after_s") is None, "Automatic return is forbidden")
    _need(request.get("programs") == [{"id": "autonomous", "windows": None}],
          "Teacher-off autonomous program is required")
    cases = result.get("cases", [])
    _need(len(cases) == 6 and Counter((c.get("seed"), c.get("trace_env")) for c in cases)
          == Counter((seed, slot) for seed in (43, 44, 45) for slot in (0, 1)),
          "Result must contain exactly two slots at each of three seeds")
    for case in cases:
        _need(case.get("success") is True and float(case.get("post_capture_s", 0)) >= 10. - 1e-6,
              "A case did not hold autonomously for 10 s after capture")
        _need(not any(case.get(key) for key in ("teacher", "failed", "interrupted", "boundary")),
              "A case has teacher, failure, interruption or boundary")
        _need(case.get("return_time", -1) < 0 and case.get("stand_request") is True,
              "A case returned or disabled standing")
    _need(result.get("observation_passes") == 6, "Result has fewer than six passes")
    _need(verification.get("autonomous_10s_gate") is True
          and verification.get("independent_physics_verified") is True
          and verification.get("independent_proprioception_verified") is True
          and verification.get("autonomous") is True
          and verification.get("teacher_constructed") is False,
          "Independent verification did not pass")
    vcases = verification.get("cases", [])
    _need(len(vcases) == 6 and verification.get("passes") == 6
          and Counter((c.get("seed"), c.get("program")) for c in vcases)
          == Counter((seed, "autonomous") for seed in (43, 44, 45) for _ in range(2)),
          "Verification must cover all six cases")
    _need(all(c.get("observation_pass") is True and c.get("returned") is False for c in vcases),
          "An independently checked case failed")
    _need(verification.get("physics_samples", 0) > 0
          and len(verification.get("action_checks", [])) == 3,
          "Missing 1 kHz/action reconstruction evidence")
    _need(all(c.get("teacher_frames") == 0 and c.get("executed_action_error") == 0
              for c in verification["action_checks"]), "Teacher/action mismatch in trace")


def six_case_gate(verification_path: Path, actor_path: Path):
    """Read the designated artifact and re-run its independent full-trace verifier."""
    verification_path = Path(verification_path).resolve()
    actor_path = Path(actor_path).resolve()
    _need(verification_path.name == "verification.json", "Supply the designated verification.json")
    run = verification_path.parent
    for name in ("request.json", "result.json", "verification.json", "session.json"):
        _need((run / name).is_file(), f"Missing six-case {name}")
    request, result = read(run / "request.json"), read(run / "result.json")
    verification, session = read(verification_path), read(run / "session.json")
    _need(Path(request.get("actor", "")).resolve() == actor_path, "Designated Actor path differs")
    _need(actor_path.is_file(), "Actor file does not exist")
    actor_sha = sha(actor_path)
    validate_six_case_data(request, result, verification, session, actor_sha)
    # This also checks the exact frozen BC model, parent snapshots and source drift.
    require(run)
    actor, value = load_actor(actor_path)
    _need(value["variant"] == request["variant"], "Variant mismatch")
    import torch
    torch.set_num_threads(4)
    from verify_feedback import verify
    _need(_same_reconstruction(verify(run), verification),
          "Archived verification differs from a fresh trace reconstruction")
    return dict(actor=actor, actor_sha256=actor_sha, six_case_run=str(run),
                verification_sha256=sha(verification_path), result_sha256=sha(run / "result.json"))


def formal_review_gate(review_path: Path, actor_sha: str, provenance_check):
    """Require independent evidence review of n=16 smoke and resume."""
    review_path = Path(review_path).resolve()
    _need(review_path.name == "review.json" and review_path.is_file(), "Formal review.json required")
    review = read(review_path)
    _need(review.get("decision") == "approved" and review.get("reviewer") and review.get("reviewed_at")
          and review.get("technical_review_pass") is True
          and review.get("behavior_review_pass") is True,
          "Smoke has no independent technical and 1 kHz behavior review")
    behavior_cases = review.get("behavior_cases", [])
    _need(len(behavior_cases) == 16
          and {row.get("slot") for row in behavior_cases} == set(range(16))
          and all(all(row.get(key) is True for key in (
              "extra10s_pass", "velocity_pass", "support_pass",
              "high_frequency_pass", "uninterrupted")) for row in behavior_cases),
          "Independent review must document all 16 slots and physical gates")
    smoke = Path(review.get("smoke_run", "")).resolve()
    resume = Path(review.get("resume_run", "")).resolve()
    formal_smoke = Path(review.get("formal_smoke_run", "")).resolve()
    _need(len({smoke, resume, formal_smoke}) == 3,
          "Timeout smoke, resume and formal-config smoke must be separate runs")
    for run, expected_mode, expected_sha_key in (
        (smoke, "smoke", "smoke_result_sha256"),
        (resume, "resume-smoke", "resume_result_sha256"),
        (formal_smoke, "formal-smoke", "formal_smoke_result_sha256"),
    ):
        _need((run / "result.json").is_file() and (run / "session.json").is_file(),
              "Missing reviewed robot PPO run")
        _need(sha(run / "result.json") == review.get(expected_sha_key),
              "Reviewed result SHA mismatch")
        provenance_check(run)
        result = read(run / "result.json")
        _need(read(run / "session.json").get("status") == "completed"
              and result.get("mode") == expected_mode
              and result.get("actor_sha256") == actor_sha
              and result.get("technical_pass") is True,
              "Reviewed smoke/resume did not pass")
        _need(result.get("feature_build_calls") == result.get("process_control_steps", -2) + 1
              and result.get("driver_reset_calls") == 0
              and result.get("audit_reset_calls") == 0
              and result.get("actual_critic_adam_steps", 0) > 0,
              "Reviewed run lacks causal feature/update/no-reset evidence")
        if expected_mode == "smoke":
            _need(result.get("num_envs", 0) >= 16 and result.get("new_updates") == 2,
                  "Formal training requires actual n=16, two-update GPU smoke")
            trace = run / "physics_trace.npz"
            _need(trace.is_file() and result.get("physics_trace_sha256") == sha(trace)
                  and review.get("smoke_physics_trace_sha256") == sha(trace),
                  "Reviewed n=16 1 kHz physics trace differs")
        else:
            _need(result.get("new_updates") == (1 if expected_mode == "resume-smoke" else 32)
                  and result.get("new_physics_episode") is True
                  and result.get("restored_state_verified") is True
                  and not any(result.get("initial_physics_steps", [1]))
                  and not any(result.get("initial_episode_counts", [1])),
                  "Fresh-process one-update resume is unverified")
            _need(result.get("worker_pid") != read(smoke / "result.json").get("worker_pid"),
                  "Smoke and resume must run in separate worker processes")
            if expected_mode == "formal-smoke":
                _need(result.get("num_envs") == 16 and result.get("steps_per_env") == 32
                      and result.get("technical_time_limit_s") == 360.
                      and result.get("checkpoint_migration_verified") is True
                      and result.get("control_steps", 0) >= 1024
                      and result.get("behavior_candidate_pass") is True,
                      "360 s/32-step formal-config GPU smoke is unverified")
                trace = run / "physics_trace.npz"
                _need(trace.is_file() and result.get("physics_trace_sha256") == sha(trace)
                      and review.get("formal_smoke_physics_trace_sha256") == sha(trace),
                      "Formal-config 1 kHz trace differs")
    _need(read(resume / "request.json").get("resume_checkpoint_sha256")
          == read(smoke / "result.json").get("final_checkpoint_sha256"),
          "Resume did not load the reviewed smoke checkpoint")
    _need(read(formal_smoke / "request.json").get("resume_checkpoint_sha256")
          == read(smoke / "result.json").get("final_checkpoint_sha256"),
          "Formal-config smoke did not explicitly migrate reviewed timeout smoke")
    smoke_result = read(smoke / "result.json")
    formal_result = read(formal_smoke / "result.json")
    expected_ticks = 32 * 32 * 20
    _need(formal_result.get("control_steps") == smoke_result.get("control_steps", -1) + 1024
          and formal_result.get("completed_updates") == smoke_result.get("completed_updates", -1) + 32
          and formal_result.get("physics_trace_samples") == expected_ticks
          and formal_result.get("physics", {}).get("physics_samples_per_slot") == [expected_ticks] * 16,
          "Formal-config smoke lacks 32 real updates and all-slot 1 kHz evidence")
    return dict(review=str(review_path), review_sha256=sha(review_path),
                smoke_run=str(smoke), resume_run=str(resume), formal_smoke_run=str(formal_smoke))
