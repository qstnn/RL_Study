"""独立统一R1生命周期；Stage1入口、模型及R6工作区保持原状。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MULTISKILL=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(MULTISKILL))
from common import REPO,SOURCE,asset_manifest,module_from_file,now,sha256,snapshot,validate_checkpoint,write_json

SPEC=module_from_file("_r1_unified_spec",SOURCE/"wheel_unified/spec.py")
CART=module_from_file("_r1_cart_asset",SOURCE/"wheel_unified/cart_asset.py")
PROTOCOL=module_from_file("_r1_unified_protocol",ROOT/"scripts/protocol.py")
BASELINE=ROOT.parent/"runs/2026-10-15_15-43-42_148523/training/model_999.pt"


def positive(value):
    number=int(value)
    if number<1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def signature(args):
    return {"ground_friction":args.ground_friction,"rolling_coefficient":args.rolling_coefficient,
            "contract":SPEC.CONTRACT}


def main(mode=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if mode is None:
        parser.add_argument("mode",choices=("check","smoke","train","play","evaluate"))
    else:
        parser.set_defaults(mode=mode)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--validation",action="store_true",help="train-only <=5 updates, <=1024 environments; isolate under validation/")
    parser.add_argument("--num_envs",type=positive,default=1024)
    parser.add_argument("--max_iterations",type=positive,default=5000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda:0")
    initial=parser.add_mutually_exclusive_group()
    initial.add_argument("--warm_start",type=Path)
    initial.add_argument("--resume",type=Path)
    initial.add_argument("--fresh",action="store_true")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--ground_friction",type=float,default=1.3)
    parser.add_argument("--rolling_coefficient",type=float,default=.02)
    parser.add_argument("--slope_deg",type=float,choices=SPEC.ANGLES,default=25.)
    parser.add_argument("--payload_kg",type=float,choices=SPEC.PAYLOADS,default=15.)
    parser.add_argument("--towed_kg",type=float,choices=SPEC.TOWED_MASSES,default=40.)
    parser.add_argument("--speed",type=float,default=3.1)
    parser.add_argument("--eval_seeds",type=int,nargs="+",default=[43,44,45])
    args=parser.parse_args()
    if not 0.<args.ground_friction<=2. or not 0.<=args.rolling_coefficient<=.2:
        parser.error("Expected 0<ground_friction<=2 and 0<=rolling_coefficient<=0.2")
    if not 0.<=args.speed<=3.1:
        parser.error("Expected 0<=speed<=3.1 m/s")
    if args.validation and (args.mode!="train" or args.max_iterations>5 or args.num_envs>1024):
        parser.error("--validation permits train with <=5 updates and <=1024 environments only")
    assets=asset_manifest()
    if args.mode=="check":
        SPEC.register_task()
        print(json.dumps({"registered_task":SPEC.TASK,"contract":SPEC.CONTRACT,"assets":assets,
                          "gpu_checked":False},ensure_ascii=False,indent=2))
        return 0
    source=None
    if args.mode=="train":
        if args.checkpoint:
            parser.error("Use --resume for unified training or --warm_start for original Stage1")
        if args.resume:
            args.resume=args.resume.resolve(strict=True)
            source=SPEC.checkpoint(args.resume)
            if source["infos"].get("environment_signature")!=signature(args):
                parser.error("Resume settings differ from saved friction/rolling/contract")
        elif not args.fresh:
            args.warm_start=(args.warm_start or BASELINE).resolve(strict=True)
            source=validate_checkpoint(args.warm_start)
    elif args.resume or args.warm_start or args.fresh:
        parser.error("Initialization options apply only to train")
    if args.mode in ("play","evaluate"):
        if not args.checkpoint:
            parser.error("An explicit unified checkpoint is required")
        args.checkpoint=args.checkpoint.resolve(strict=True)
        source=SPEC.checkpoint(args.checkpoint)
        if source["infos"].get("environment_signature")!=signature(args):
            parser.error("Playback friction/rolling differ from training; pass the saved values")
        args.num_envs=1
    missing=[name for name in ("isaacsim","isaaclab") if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Activate the complete lab environment first; missing "+", ".join(missing))
    if args.mode=="smoke":
        args.num_envs=min(args.num_envs,168)
    category={"train":"runs","smoke":"validation","play":"playbacks","evaluate":"evaluations"}[args.mode]
    if args.validation:category="validation"
    stamp=now().replace(":","-").replace("T","_").split("+")[0]
    run=ROOT/category/stamp
    run.mkdir(parents=True,exist_ok=False)
    request=vars(args).copy()
    for key in ("warm_start","resume","checkpoint"):
        request[key]=str(request[key]) if request[key] else None
    request["environment_signature"]=signature(args)
    request["eval_seeds"]=list(dict.fromkeys(args.eval_seeds))
    write_json(run/"request.json",request)
    write_json(run/"contract.json",SPEC.CONTRACT)
    write_json(run/"asset_manifest.json",assets)
    CART.write_cart_urdf(run/"assets/cart.urdf")
    write_json(run/"cart_asset.json",{"file":"assets/cart.urdf","sha256":sha256(run/"assets/cart.urdf"),
               "nominal_mass_kg":40.,"passive_wheels":4,"driven_wheels":0,"rolling_coefficient":args.rolling_coefficient})
    snapshot(run)
    selected=args.resume or args.warm_start or args.checkpoint
    meta={"started_at":now(),"mode":args.mode,"status":"running","behavior_acceptance":"not_evaluated",
          "initialization":"resume" if args.resume else "stage1_network_transfer_new_optimizer" if args.warm_start else "frozen" if args.checkpoint else "fresh",
          "source_checkpoint":str(selected) if selected else None,"source_sha256":sha256(selected) if selected else None}
    write_json(run/"run.json",meta)
    print(f"[R1 Unified] {run}",flush=True)
    process=None;code=1
    try:
        with (run/"console.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen([sys.executable,"-u",str(ROOT/"scripts/worker.py"),"--run",str(run)],
                cwd=REPO,env=dict(os.environ,TERM="xterm",PYTHONUNBUFFERED="1"),stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,errors="replace",bufsize=1)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end="",flush=True)
            code=process.wait()
        expected={"smoke":"smoke.json","train":"training_result.json","play":"evaluation.json","evaluate":"evaluation.json"}[args.mode]
        if code==0 and ((run/"worker_error.json").exists() or not (run/expected).is_file()):
            code=1
            meta["failure_reason"]="worker_error_or_missing_completion_artifact"
        meta["status"]="completed" if code==0 else "failed"
    except KeyboardInterrupt:
        code=130;meta["status"]="interrupted"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if selected and sha256(selected)!=meta["source_sha256"]:
            code=1;meta["status"]="source_checkpoint_changed"
        meta.update(finished_at=now(),returncode=code)
        write_json(run/"run.json",meta)
    return code


if __name__=="__main__":
    raise SystemExit(main())
