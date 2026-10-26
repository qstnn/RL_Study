"""R6独立进程入口；不导入Isaac Sim即可检查资产和载荷URDF。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common import REPO, ROOT, SOURCE, URDF, asset_manifest, module_from_file, now, sha256, snapshot, write_json

spec = module_from_file("_r6_spec", SOURCE / "payload_spec.py")
asset = module_from_file("_r6_asset", SOURCE / "payload_asset.py")
ROUTE = ROOT / "routes" / spec.DIRECTORY


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def playback_seconds(value):
    import math
    seconds = float(value)
    if not math.isfinite(seconds) or not 1 <= seconds <= 600:
        raise argparse.ArgumentTypeError('Interactive play duration must be between 1 and 600 seconds')
    return seconds


def failure_context(run):
    """读取子进程留下的诊断，不用猜测缺少运行结果的具体原因。"""
    details = [f"See console log: {run / 'console.log'}"]
    for name in ('worker_status.json', 'worker_error.json'):
        path = run / name
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
            if name == 'worker_status.json':
                details.append(f"Last worker stage: {record.get('stage', 'unknown')}")
            else:
                details.append(f"Original worker error: {record.get('exception_type')}: {record.get('error')}")
                details.append(f"Full traceback: {path}")
        except (OSError, ValueError, AttributeError):
            details.append(f"Unreadable diagnostic: {path}")
    return '; '.join(details)


def verify_completion(run, mode):
    # Kit可能提前exit(0)；进程返回零不能代替环境实例化和训练结果证据。
    if (run / 'worker_error.json').exists():
        raise RuntimeError('R6 worker reported an exception; ' + failure_context(run))
    if not (run / 'runtime_check.json').is_file():
        raise RuntimeError('R6 worker exited before completing runtime validation; '
                           'missing required output runtime_check.json; ' + failure_context(run))
    if not (run / 'result.json').is_file():
        raise RuntimeError('R6 worker exited before writing its final result; '
                           'missing required output result.json; ' + failure_context(run))
    runtime = json.loads((run / "runtime_check.json").read_text())
    result = json.loads((run / "result.json").read_text())
    if runtime.get("dimensions") != [58, 237, 16] or result.get("mode") != mode or result.get("status") != "completed":
        raise RuntimeError("R6 worker did not produce a valid completion record")
    if mode == "train":
        saved = spec.checkpoint(result["checkpoint"])
        if result.get("completed_updates") != result.get("requested_updates"):
            raise RuntimeError("PPO update budget did not complete")
        if sha256(result["checkpoint"]) != result["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint checksum mismatch")
    if mode in ('play','evaluate'):
        request=json.loads((run/'request.json').read_text()) if (run/'request.json').is_file() else {}
        if request.get('keyboard_payload'):
            if result.get('protocol') != 'interactive_payload_change_v1' or result.get('success_rate') is not None:
                raise RuntimeError('Interactive payload play must not claim fixed-load screening success')
            if not (run/'keyboard_payload_events.jsonl').is_file():
                raise RuntimeError('Keyboard payload event log missing')
        for case in result.get('cases',[]):
            label=f"seed{case['seed']}_episode{case['episode']}"
            trace_file=run/(label+'_trace.json')
            if request.get('keyboard_payload') and not (run/(label+'_payload.json')).is_file():
                raise RuntimeError('Keyboard payload episode metadata missing: '+label)
            if request.get('record_video') or request.get('physics_trace'):
                if not trace_file.is_file(): raise RuntimeError('Required review trace metadata missing: '+str(trace_file))
                trace=json.loads(trace_file.read_text())
                if request.get('physics_trace') and trace['physics_samples']!=20*trace['policy_samples']:
                    raise RuntimeError('Incomplete 1kHz physical trace: '+str(trace_file))
                if request.get('record_video'):
                    video=run/(label+'.mp4')
                    if not video.is_file() or video.stat().st_size<1024 or trace['video_frames']<1:
                        raise RuntimeError('Required playback video missing or empty: '+str(video))
    return result


def main(mode=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if mode is None:
        parser.add_argument("mode", choices=("check", "smoke", "train", "play", "evaluate"))
    else:
        parser.set_defaults(mode=mode)
    parser.add_argument("--route", choices=("06",), default="06")
    parser.add_argument("--payload-kg", type=spec.validate_mass, default=0.0)
    parser.add_argument("--num_envs", type=positive, default=1024)
    parser.add_argument("--max_iterations", type=positive, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--eval_episodes", type=positive, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, help="R6 only; train restores actor, critic and optimizer")
    parser.add_argument("--run-dir", type=Path, help="Explicit new run directory for the offline station")
    parser.add_argument("--record-video", action="store_true", help="Record offline MP4 during play")
    parser.add_argument("--physics-trace", action="store_true", help="Record 1 kHz physical samples during play/evaluate")
    parser.add_argument("--keyboard-payload", action="store_true", help="GUI play: change the physical payload from 0 to 65 kg with the keyboard")
    parser.add_argument("--play-seconds", type=playback_seconds, default=120., help="Simulation budget for keyboard-payload play (1 to 600 s, default 120)")
    args = parser.parse_args(argv)
    if args.keyboard_payload:
        if args.mode != 'play' or args.headless:
            parser.error('--keyboard-payload requires GUI play; remove --headless')
        args.physics_trace = args.record_video = True
    if args.record_video and args.mode != 'play':
        parser.error('--record-video is only supported by play')
    if args.physics_trace and args.mode not in ('play', 'evaluate'):
        parser.error('--physics-trace is only supported by play/evaluate')
    assets = asset_manifest()
    spec.register_task()
    if args.mode == "check":
        with tempfile.TemporaryDirectory(prefix="r6-asset-check-") as temp:
            report = asset.write_payload_urdf(URDF, Path(temp)/"payload.urdf", args.payload_kg)
        print(json.dumps({"status": "cpu_check_passed", "task": spec.TASK, "assets": {"joints": assets['joint_count'], "meshes": len(assets['meshes'])}, "payload": report, "gpu_verified": False}, ensure_ascii=False, indent=2))
        return 0
    if args.mode in ("play", "evaluate") and args.checkpoint is None:
        parser.error("play/evaluate require an explicit R6 --checkpoint")
    if args.mode == "smoke" and args.checkpoint is not None:
        parser.error("smoke uses zero actions; do not supply a checkpoint")
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve(strict=True)
        spec.checkpoint(args.checkpoint)
    missing = [name for name in ("isaacsim", "isaaclab", "isaaclab_rl", "rsl_rl") if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Missing runtime modules: " + ", ".join(missing) + "; activate the complete lab environment")
    if args.mode == "smoke":
        args.num_envs = min(args.num_envs, 32)
    if args.mode in ("play", "evaluate"):
        args.num_envs = 1
    from datetime import datetime
    from zoneinfo import ZoneInfo
    folder = {"train": "runs", "smoke": "validation", "play": "playbacks", "evaluate": "evaluations"}[args.mode]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run = args.run_dir.resolve() if args.run_dir else ROUTE / folder / f"{stamp}_payload{args.payload_kg:g}kg_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    payload = asset.write_payload_urdf(URDF, run/"input/robot_payload.urdf", args.payload_kg)
    payload["generated_urdf_sha256"] = sha256(run/"input/robot_payload.urdf")
    write_json(run/"payload.json", payload)
    write_json(run/"asset_manifest.json", assets)
    write_json(run/"contract.json", spec.CONTRACT)
    snapshot(run)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    request['checkpoint'] = str(args.checkpoint) if args.checkpoint else None
    request['eval_seeds'] = list(dict.fromkeys(args.eval_seeds))
    write_json(run/"request.json", request)
    command = [sys.executable, "-u", str(ROUTE/"scripts/worker.py"), "--run", str(run)]
    meta = {"started_at": now(), "status": "running", "mode": args.mode, "command": command,
            "python": sys.executable, "behavior_acceptance": "not_evaluated", "payload_kg": args.payload_kg,
            "initialization": "resume_r6" if args.checkpoint and args.mode == "train" else "frozen_r6" if args.checkpoint else "fresh"}
    if args.checkpoint:
        meta.update(checkpoint=str(args.checkpoint), source_sha256=sha256(args.checkpoint))
    write_json(run/"run.json", meta)
    print(f"[R6] {run}", flush=True)
    process, code = None, 1
    try:
        with (run/"console.log").open('w') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, TERM="xterm", PYTHONUNBUFFERED="1"),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
            code = process.wait()
        meta['worker_returncode'] = code
        if code != 0:
            raise RuntimeError(f'R6 worker exited with code {code}; ' + failure_context(run))
        verify_completion(run, args.mode)
        meta['status'] = 'completed' if code == 0 else 'failed'
    except KeyboardInterrupt:
        code, meta['status'] = 130, 'interrupted'
    except Exception as error:
        code, meta['status'] = 1, 'failed'
        meta['error'] = f"{type(error).__name__}: {error}"
        print(f"[R6 ERROR] {meta['error']}", file=sys.stderr)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if args.checkpoint and sha256(args.checkpoint) != meta['source_sha256']:
            code, meta['status'] = 1, 'source_checkpoint_changed'
        meta.update(finished_at=now(), returncode=code)
        write_json(run/"run.json", meta)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
