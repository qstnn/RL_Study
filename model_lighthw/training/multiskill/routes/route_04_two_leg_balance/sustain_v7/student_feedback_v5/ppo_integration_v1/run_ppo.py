"""User-invoked R4a PPO preflight, smoke/resume and manual formal launcher."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRUCTURED = HERE.parent / "structured_v1"
if str(STRUCTURED) not in sys.path:
    sys.path.insert(0, str(STRUCTURED))
from common_feedback import now, read, sha, write  # noqa: E402
from gate import AdmissionError, formal_review_gate, six_case_gate  # noqa: E402
from provenance import archive, check  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for mode in ("check", "smoke", "resume-smoke", "formal-smoke", "train"):
        q = sub.add_parser(mode)
        q.add_argument("--actor", type=Path, required=True)
        q.add_argument("--six-case-verification", type=Path, required=True)
        if mode != "check":
            q.add_argument("--output", type=Path, required=True)
            q.add_argument("--num-envs", type=int, choices=(2, 16, 64),
                           default=16 if mode in ("formal-smoke", "train") else 2)
        if mode in ("resume-smoke", "formal-smoke"):
            q.add_argument("--smoke-run", type=Path, required=True)
        if mode == "train":
            q.add_argument("--review", type=Path, required=True)
            q.add_argument("--max-updates", type=int, required=True,
                           help="Explicit NEW PPO update budget in this process")
    sub.add_parser("worker").add_argument("run", type=Path)
    return p


def prepare(a):
    admitted = six_case_gate(a.six_case_verification, a.actor)
    if a.command == "check":
        return admitted
    if a.command == "train" and a.max_updates <= 0:
        raise ValueError("Formal --max-updates must be positive")
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    request = dict(mode=a.command, actor=str(a.actor.resolve()),
        actor_sha256=admitted["actor_sha256"],
        six_case_verification=str(a.six_case_verification.resolve()),
        six_case_verification_sha256=admitted["verification_sha256"],
        six_case_result_sha256=admitted["result_sha256"],
        num_envs=a.num_envs, steps_per_env=32 if a.command in ("formal-smoke", "train") else 512,
        technical_time_limit_s=360. if a.command in ("formal-smoke", "train") else 20.,
        initial_std=.0005, learning_rate=3e-6,
        new_updates=2 if a.command == "smoke" else 32 if a.command == "formal-smoke"
                    else 1 if a.command == "resume-smoke" else a.max_updates,
        resume_checkpoint=None, resume_checkpoint_sha256=None,
        source_smoke_run=None, source_smoke_result_sha256=None,
        checkpoint_migration=None,
        no_observation_noise=True, no_reset_randomization=True,
        stop_on_observation=False, is_finite_horizon=False,
        automatic_frozen_evaluation=False, checkpoint_every=100)
    if a.command in ("resume-smoke", "formal-smoke"):
        smoke = a.smoke_run.resolve()
        check(smoke)
        source = read(smoke / "result.json")
        if read(smoke / "session.json")["status"] != "completed" or source["mode"] != "smoke":
            raise AdmissionError("Resume source is not completed smoke")
        if source["actor_sha256"] != admitted["actor_sha256"] or source["num_envs"] != a.num_envs:
            raise AdmissionError("Resume Actor or num_envs mismatch")
        if a.command == "formal-smoke" and (a.num_envs != 16 or source["technical_pass"] is not True):
            raise AdmissionError("Formal-config smoke requires passing actual n=16 timeout smoke")
        checkpoint = Path(source["final_checkpoint"]).resolve()
        if sha(checkpoint) != source["final_checkpoint_sha256"]:
            raise AdmissionError("Smoke checkpoint SHA mismatch")
        request["resume_checkpoint"] = str(checkpoint)
        request["resume_checkpoint_sha256"] = sha(checkpoint)
        request["source_smoke_run"] = str(smoke)
        request["source_smoke_result_sha256"] = sha(smoke / "result.json")
        if a.command == "formal-smoke":
            request["checkpoint_migration"] = dict(source_steps_per_env=512,
                destination_steps_per_env=32, source_technical_limit_s=20.,
                destination_technical_limit_s=360., source_checkpoint_sha256=sha(checkpoint),
                other_ppo_hyperparameters_unchanged=True,
                new_physics_episode=True)
    if a.command == "train":
        review = formal_review_gate(a.review, admitted["actor_sha256"], check)
        source = read(Path(review["formal_smoke_run"]) / "result.json")
        if a.num_envs != source["num_envs"]:
            raise AdmissionError("Formal env count must match reviewed resume checkpoint")
        checkpoint = Path(source["final_checkpoint"]).resolve()
        if sha(checkpoint) != source["final_checkpoint_sha256"]:
            raise AdmissionError("Reviewed resume checkpoint SHA mismatch")
        request.update(review=review, resume_checkpoint=str(checkpoint),
                       resume_checkpoint_sha256=sha(checkpoint))
    output.mkdir(parents=True, exist_ok=False)
    archive(output)
    write(output / "request.json", request)
    return output


def execute(run):
    started = now()
    write(run / "session.json", dict(status="running", started_at=started))
    process = None
    try:
        with (run / "console.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", str(HERE / "run_ppo.py"), "worker", str(run)],
                stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code:
            raise RuntimeError(f"Robot worker exited {code}; inspect {run / 'console.log'}")
        check(run)
        result = read(run / "result.json")
        if result["completed_updates"] <= result["initial_updates"]:
            raise RuntimeError("No actual PPO update")
        write(run / "session.json", dict(status="completed", started_at=started, finished_at=now()))
    except BaseException as exc:
        write(run / "session.json", dict(status="failed", started_at=started,
            finished_at=now(), error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "worker":
        # A direct worker invocation is not a bypass: source and six-case gates
        # are rechecked inside run_robot before Isaac AppLauncher is created.
        from robot_worker import run_robot
        run_robot(a.run)
        return
    prepared = prepare(a)
    if a.command == "check":
        print("Six-case admission passed:", prepared["six_case_run"])
        print("Actor SHA256:", prepared["actor_sha256"])
        return
    execute(prepared)
    print("Completed:", prepared)


if __name__ == "__main__":
    main()
