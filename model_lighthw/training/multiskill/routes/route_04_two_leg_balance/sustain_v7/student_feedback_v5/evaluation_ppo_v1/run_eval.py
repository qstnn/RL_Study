"""Run isolated teacher-off R4a PPO checkpoint behavior evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from common_eval import load_ppo_actor, now, read, require, snapshot, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--training-run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    evaluate.add_argument("--hold-seconds", type=int, choices=[10, 60, 300], required=True)
    for command in ("worker", "verify"):
        sub.add_parser(command).add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "worker":
        from worker_eval import worker
        worker(args.run)
        return
    if args.command == "verify":
        from verify_eval import verify
        print(verify(args.run))
        return
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or not set(args.seeds).issubset({43, 44, 45})):
        raise ValueError("Evaluation seeds must be distinct members of 43/44/45")
    checkpoint, training_run, run = (args.checkpoint.resolve(), args.training_run.resolve(),
                                     args.output.resolve())
    _, metadata = load_ppo_actor(checkpoint, training_run)
    run.mkdir(parents=True, exist_ok=False)
    started = now()
    worker_process = verify_process = None
    try:
        snapshot(run, metadata)
        request = dict(kind="ppo_checkpoint_mean_autonomous_evaluation", checkpoint=str(checkpoint),
            training_run=str(training_run), ppo_source=metadata, seeds=args.seeds,
            hold_seconds=float(args.hold_seconds), return_after_s=None,
            autonomous=True, teacher_off=True, mean_action_only=True,
            no_observation_noise=True, no_reset_randomization=True,
            raw_environment_observation_dim=305, actor_input_dim=325,
            programs=[dict(id="autonomous", windows=None)])
        write(run / "request.json", request)
        require(run)
        write(run / "session.json", dict(status="running", started_at=started))
        with (run / "console.log").open("w") as log:
            worker_process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "worker", str(run)],
                stdout=log, stderr=subprocess.STDOUT)
            worker_code = worker_process.wait()
        if worker_code or not (run / "result.json").is_file():
            raise RuntimeError(f"GPU worker failed ({worker_code}); see {run}/worker_error.json")
        with (run / "verification_console.log").open("w") as log:
            verify_process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "verify", str(run)],
                stdout=log, stderr=subprocess.STDOUT)
            verify_code = verify_process.wait()
        if verify_code or not (run / "verification.json").is_file():
            raise RuntimeError(f"Independent verification failed ({verify_code}); see {run}/verification_console.log")
        require(run)
        report = read(run / "verification.json")
        write(run / "session.json", dict(status="completed", started_at=started,
            finished_at=now(), independent_physics_verified=report["independent_physics_verified"],
            passes=report["passes"], cases=len(report["cases"])))
    except BaseException as exc:
        write(run / "session.json", dict(status="failed", started_at=started,
            finished_at=now(), error=str(exc)))
        raise
    finally:
        for process in (worker_process, verify_process):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
