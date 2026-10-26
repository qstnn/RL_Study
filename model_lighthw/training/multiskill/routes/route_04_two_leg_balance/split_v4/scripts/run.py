"""R4单命令入口；固定条件单进程训练；仅保存模型，评估由用户通知后显式执行。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import ROUTE, REPO, contract, now, read, write, sha, snapshot, checked_checkpoint
from geometry import audit
from two_leg_balance_split_v4.spec import SKILLS, REWARD_PROFILES


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def execute(run, request, geometry):
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    write(run / "request.json", request)
    write(run / "contract.json", contract(
        request["hold_seconds"], request["support_mode"], request["reward_profile"]))
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
    p.add_argument("--save_every", type=positive, default=100, help="每N次优化尝试保存模型；不评估、不重启环境")
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[43, 44, 45])
    p.add_argument("--probe_kind", choices=("phased", "unscaled", "mean"), default="phased")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hold_seconds", type=float, choices=(2., 5.), default=2.)
    p.add_argument("--skill", choices=tuple(SKILLS), default="R4a", help="R4a后腿支撑；R4b前腿支撑；分别训练模型")
    p.add_argument("--reward_profile", choices=REWARD_PROFILES, default="baseline_v4",
                   help="baseline_v4原奖励；unload_diag_v1卸载诊断；soft_transfer_v2软门控转移/稳定性诊断")
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
    a.support_mode = SKILLS[a.skill]
    c = contract(a.hold_seconds, a.support_mode, a.reward_profile)
    geometry = audit()
    if not all(row["candidate_pass"] for row in geometry["candidates"]):
        raise ValueError("R4 geometry candidate failed; review canonical asset before training")
    if a.mode == "check":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROUTE / "tests"), pattern="test_*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            return 1
        out = a.output or ROUTE / "validation/cpu" / a.skill
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
        if a.mode == "train" and saved["infos"].get("audit_only", False):
            p.error("audit-only checkpoint may be evaluated but cannot be used for training resume")
        start = saved["infos"]["completed_updates"]
    run = (a.output or ROUTE / a.skill / "runs" / now().replace(":", "-")).resolve()
    run.mkdir(parents=True, exist_ok=False)
    request = vars(a).copy()
    request["checkpoint"] = str(a.checkpoint) if a.checkpoint else None
    request["output"] = str(run)
    request["headless"] = a.headless or a.mode != "play"
    write(run / "session.json", dict(started_at=now(), request=request, contract=c, status="running"))
    try:
        if a.mode == "train":
            # 一次worker完成全部预算；不插入冻结、末尾评估或性能驱动的自动续训。
            total = start + a.max_iterations
            folder = run / f"train_{total:06d}"
            req = dict(request, updates=a.max_iterations)
            trained = execute(folder, req, geometry)
            checkpoint = folder / "final.pt"
            if trained["completed_updates"] != total:
                raise RuntimeError("R4 completed update count mismatch")
            digest = sha(checkpoint)
            if digest != trained["checkpoint_sha256"]:
                raise RuntimeError("R4 final checkpoint SHA mismatch")
            write(run / "progress.json", [dict(updates=total, checkpoint=str(checkpoint),
                  checkpoint_sha256=digest, review_status="awaiting_user_request")])
            result = dict(status="completed", completed_updates=total, new_updates=a.max_iterations,
                          checkpoint=str(checkpoint), checkpoint_sha256=digest, frozen_reviews=[],
                          review_status="awaiting_user_request", behavior_accepted=False)
        else:
            req = dict(request, num_envs=(min(a.num_envs, 32) if a.mode == "smoke" else a.num_envs) if a.mode in ("smoke", "probe") else 2, updates=0)
            result = execute(run / a.mode, req, geometry)
        write(run / "result.json", result)
        write(run / "session.json", dict(started_at=read(run / "session.json")["started_at"],
                                        finished_at=now(), request=request, contract=c, status="completed"))
        print(f"\n[R4 SPLIT V4 REVIEW DIRECTORY] {run}", flush=True)
        return 0
    except BaseException as error:
        write(run / "error.json", dict(at=now(), type=type(error).__name__, message=str(error)))
        info = read(run / "session.json")
        info.update(status="failed_or_interrupted", finished_at=now())
        write(run / "session.json", info)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
