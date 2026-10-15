#!/usr/bin/env python3
"""LightHW isolated route train/smoke/play/evaluate; check works without Isaac Sim."""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT, REPO, catalog, asset_manifest, new_run, now, sha256, snapshot, validate_checkpoint, write_json


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main(mode=None):
    # R6 owns a separate lifecycle/contract; never route it into the R1 worker.
    route_parser = argparse.ArgumentParser(add_help=False)
    route_parser.add_argument("--route")
    selected, _ = route_parser.parse_known_args()
    if selected.route == "06":
        from common import module_from_file
        module = module_from_file("_r6_launch", ROOT / "routes/route_06_heavy_payload/scripts/launch.py")
        return module.main(mode)
    p = argparse.ArgumentParser(description=__doc__)
    if mode is None:
        p.add_argument("mode", choices=("list", "check", "train", "smoke", "play", "evaluate"))
    else:
        p.set_defaults(mode=mode)
    p.add_argument("--route", choices=tuple(catalog.ROUTES), default="01")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num_envs", type=positive, default=1024)
    p.add_argument("--max_iterations", type=positive, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[43, 44, 45])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    if args.mode == "list":
        for key, record in catalog.ROUTES.items():
            print(key, record["status"], record["task_id"])
        return 0
    catalog.implemented_route(args.route)
    assets = asset_manifest()
    if args.mode == "check":
        catalog.register_tasks()
        print(f"PASS: explicit Gym registration; {assets['joint_count']} joints; {len(assets['meshes'])} meshes")
        print(f"Declared contract: {catalog.CONTRACT}")
        print("Isaac Sim config instantiation and GPU rollout have NOT been checked by this command.")
        return 0
    if args.mode in ("train", "smoke") and args.checkpoint is not None:
        p.error("Training is fresh only; checkpoint input is only for play/evaluate")
    if args.mode in ("play", "evaluate"):
        if args.checkpoint is None:
            p.error("An explicit Route01 checkpoint is required")
        args.checkpoint = args.checkpoint.resolve(strict=True)
        validate_checkpoint(args.checkpoint)
    # 先检查依赖，缺失时不创建伪装成成功的run或模型。
    missing = [name for name in ("isaacsim", "isaaclab") if importlib.util.find_spec(name) is None]
    if missing:
        p.error("Missing runtime modules: " + ", ".join(missing) + "; use the complete lab / Isaac Sim environment")
    if args.mode == "smoke":
        args.num_envs = min(args.num_envs, 32)
    if args.mode in ("play", "evaluate"):
        args.num_envs = 1
    run = new_run(args.route, args.mode)
    write_json(run / "asset_manifest.json", assets)
    write_json(run / "contract.json", catalog.CONTRACT)
    snapshot(run)
    command = [sys.executable, "-u", str(ROOT / "worker.py"), "--run", str(run)]
    request = vars(args).copy()
    request["checkpoint"] = str(args.checkpoint) if args.checkpoint is not None else None
    request["eval_seeds"] = list(dict.fromkeys(args.eval_seeds))
    write_json(run / "request.json", request)
    meta = {"started_at": now(), "mode": args.mode, "status": "running", "command": command,
            "python": sys.executable, "behavior_acceptance": "not_evaluated", "initialization": "fresh"}
    if args.checkpoint:
        meta.update(initialization="frozen_checkpoint", checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint))
    write_json(run / "run.json", meta)
    print(f"[MultiSkill] {run}", flush=True)
    process, code = None, 1
    try:
        with (run / "console.log").open("w", encoding="utf-8") as stream:
            env = dict(os.environ, TERM="xterm", PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            for line in process.stdout:
                stream.write(line); stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
        meta["status"] = "completed" if code == 0 else "failed"
    except KeyboardInterrupt:
        code, meta["status"] = 130, "interrupted"
    except BaseException:
        meta["status"] = "failed"
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if args.checkpoint and sha256(args.checkpoint) != meta["checkpoint_sha256"]:
            code, meta["status"] = 1, "checkpoint_changed"
        meta.update(finished_at=now(), returncode=code)
        write_json(run / "run.json", meta)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
