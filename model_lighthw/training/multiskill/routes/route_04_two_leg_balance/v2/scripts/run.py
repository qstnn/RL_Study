"""R4单命令入口；训练按固定预算分段，每段独立冻结评估，绝不自动晋级。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import ROUTE, REPO, contract, now, read, write, sha, snapshot, checked_checkpoint
from geometry import audit


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def execute(run, request, geometry):
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    write(run / "request.json", request)
    write(run / "contract.json", contract(request["hold_seconds"], request["support_mode"]))
    write(run / "geometry.json", geometry)
    meta = dict(started_at=now(), status="running", behavior_accepted=False)
    write(run / "run.json", meta)
    command = [sys.executable, "-u", str(ROUTE / "scripts/worker.py"), "--run", str(run)]
    print(f"\n[R4 RUN] {run}", flush=True)
    process = None
    code = 1
    try:
        with (run / "console.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, TERM="xterm"),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"R4 worker exited {code}; inspect {run / 'console.log'}")
        if (run / "worker_error.json").exists():
            error = read(run / "worker_error.json")
            raise RuntimeError(f"R4 worker failed: {error['error']}; inspect {run / 'worker_error.json'}")
        result = read(run / "result.json")
        meta["status"] = "completed"
        return result
    except BaseException:
        meta["status"] = "interrupted" if code == 130 else "failed"
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        meta.update(finished_at=now(), returncode=code)
        write(run / "run.json", meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "smoke", "probe", "train", "evaluate", "play", "help"))
    p.add_argument("--num_envs", type=positive, default=512)
    p.add_argument("--max_iterations", type=positive, default=1000)
    p.add_argument("--eval_every", type=positive, default=100)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[43, 44, 45])
    p.add_argument("--probe_kind", choices=("phased", "unscaled", "mean"), default="phased")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hold_seconds", type=float, choices=(2., 5.), default=2.)
    p.add_argument("--support_mode", choices=("both", "front", "rear"), default="both")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--headless", action="store_true", help="play默认GUI，其他模式始终headless")
    a = p.parse_args()
    if a.mode == "help":
        p.print_help()
        return 0
    if a.num_envs > 4096:
        p.error("R4 finite ground and initial resource envelope support at most 4096 environments")
    c = contract(a.hold_seconds, a.support_mode)
    geometry = audit()
    if not all(row["candidate_pass"] for row in geometry["candidates"]):
        raise ValueError("R4 geometry candidate failed; review canonical asset before training")
    if a.mode == "check":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROUTE / "tests"), pattern="test_*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            return 1
        out = a.output or ROUTE / "validation/cpu"
        write(out / "geometry.json", geometry)
        write(out / "contract.json", c)
        write(out / "result.json", dict(checked_at=now(), tests=suite.countTestCases(),
                                      static_checks_passed=True, gpu_checked=False))
        print(f"[R4 CPU PASS] {out}\nActor305/Critic88/Action16; GPU/behavior not checked here.")
        return 0
    if a.mode in ("evaluate", "play") and a.checkpoint is None:
        p.error("evaluate/play require an explicit R4 checkpoint")
    if a.mode == "smoke" and a.checkpoint:
        p.error("smoke uses neutral actions and does not load a checkpoint")
    start = 0
    if a.checkpoint:
        a.checkpoint = a.checkpoint.resolve(strict=True)
        saved = checked_checkpoint(a.checkpoint, c)
        start = saved["infos"]["completed_updates"]
    run = (a.output or ROUTE / "runs" / now().replace(":", "-")).resolve()
    run.mkdir(parents=True, exist_ok=False)
    request = vars(a).copy()
    request["checkpoint"] = str(a.checkpoint) if a.checkpoint else None
    request["output"] = str(run)
    request["headless"] = a.headless or a.mode != "play"
    write(run / "session.json", dict(started_at=now(), request=request, contract=c, status="running"))
    try:
        if a.mode == "train":
            checkpoint = a.checkpoint
            completed = 0
            reviews = []
            while completed < a.max_iterations:
                steps = min(a.eval_every, a.max_iterations - completed)
                req = dict(request, updates=steps, checkpoint=str(checkpoint) if checkpoint else None,
                           num_envs=a.num_envs)
                folder = run / f"train_{start + completed + steps:06d}"
                result = execute(folder, req, geometry)
                checkpoint = folder / "final.pt"
                completed += steps
                import json
                ended = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()] if (folder / "episodes.jsonl").exists() else []
                if len(ended) >= 128 and not any(row["entered"] for row in ended):
                    raise RuntimeError("R4 v2 entry gate inactive in all ended training episodes; stopping before more PPO")
                if result["completed_updates"] != start + completed:
                    raise RuntimeError("R4 completed update count mismatch")
                review = run / f"eval_{start + completed:06d}"
                req = dict(request, mode="evaluate", checkpoint=str(checkpoint), num_envs=2,
                           updates=0, headless=True)
                report = execute(review, req, geometry)
                reviews.append(dict(updates=start + completed, checkpoint=str(checkpoint),
                                    checkpoint_sha256=sha(checkpoint), evaluation=report))
                write(run / "progress.json", reviews)
            result = dict(status="completed", completed_updates=start + completed, new_updates=completed,
                          checkpoint=str(checkpoint), frozen_reviews=reviews, behavior_accepted=False)
        else:
            req = dict(request, num_envs=(min(a.num_envs, 32) if a.mode == "smoke" else a.num_envs) if a.mode in ("smoke", "probe") else 2, updates=0)
            result = execute(run / a.mode, req, geometry)
        write(run / "result.json", result)
        write(run / "session.json", dict(started_at=read(run / "session.json")["started_at"],
                                        finished_at=now(), request=request, contract=c, status="completed"))
        print(f"\n[R4 V2 REVIEW DIRECTORY] {run}", flush=True)
        return 0
    except BaseException as error:
        write(run / "error.json", dict(at=now(), type=type(error).__name__, message=str(error)))
        info = read(run / "session.json")
        info.update(status="failed_or_interrupted", finished_at=now())
        write(run / "session.json", info)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
