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

SPEC=module_from_file("_r1_unified_v3_spec",SOURCE/"wheel_unified_v3/spec.py")
CART=module_from_file("_r1_cart_asset",SOURCE/"wheel_unified_v3/cart_asset.py")
PROTOCOL=module_from_file("_r1_unified_v3_protocol",ROOT/"scripts/protocol.py")
BASELINE=ROOT.parent/"unified_v2/runs/2026-10-16_10-19-29.659784/training/model_3900.pt"


def positive(value):
    number=int(value)
    if number<1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def signature(args):
    return {"ground_friction":args.ground_friction,"rolling_coefficient":args.rolling_coefficient,
            "leg_torque_limit_Nm":args.leg_torque_limit,"wheel_torque_limit_Nm":args.wheel_torque_limit,
            "hitch_height_m":args.hitch_height,"loaded_command_mps":[.2,.6],
            "unloaded_acceleration_mps2":args.unloaded_acceleration,"loaded_acceleration_mps2":args.loaded_acceleration,
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
    parser.add_argument("--max_iterations",type=positive,default=500)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda:0")
    initial=parser.add_mutually_exclusive_group()
    initial.add_argument("--initialize_from",type=Path)
    initial.add_argument("--resume",type=Path)
    initial.add_argument("--fresh",action="store_true")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--eval_interval",type=int,default=100,help=f"Frozen {len(PROTOCOL.cases())}-case evaluation every N updates; 0 disables")
    parser.add_argument("--output_dir",type=Path,help="Explicit new output directory")
    parser.add_argument("--command_profile",choices=("slew","step"),default="slew",help="step is playback/evaluation stress only")
    parser.add_argument("--unloaded_acceleration",type=float,default=1.)
    parser.add_argument("--loaded_acceleration",type=float,default=.3)
    parser.add_argument("--ground_friction",type=float,default=2.3)
    parser.add_argument("--leg_torque_limit",type=float,default=96.)
    parser.add_argument("--wheel_torque_limit",type=float,default=48.)
    parser.add_argument("--hitch_height",type=float,default=.10)
    parser.add_argument("--eval_distance",type=float,default=1.0)
    parser.add_argument("--rolling_coefficient",type=float,default=.02)
    parser.add_argument("--slope_deg",type=float,choices=SPEC.ANGLES,default=25.)
    parser.add_argument("--payload_kg",type=float,choices=SPEC.PAYLOADS,default=15.)
    parser.add_argument("--towed_kg",type=float,choices=SPEC.TOWED_MASSES,default=0.)
    parser.add_argument("--speed",type=float)
    parser.add_argument("--eval_seeds",type=int,nargs="+",default=[43,44,45])
    args=parser.parse_args()
    import math
    if not 0.<args.ground_friction<=4. or not 0.<=args.rolling_coefficient<=.2:
        parser.error("Expected 0<ground_friction<=4 and 0<=rolling_coefficient<=0.2")
    if any(not math.isfinite(v) or v<=0 for v in (args.leg_torque_limit,args.wheel_torque_limit,args.hitch_height,args.eval_distance,args.unloaded_acceleration,args.loaded_acceleration)):
        parser.error("Torque limits, initial hitch height and evaluation distance must be finite and positive")
    if args.eval_interval<0:
        parser.error("eval_interval must be nonnegative")
    if args.command_profile=="step" and args.mode not in ("play","evaluate"):
        parser.error("Step commands are evaluation stress tests only; training uses slew commands")
    if args.payload_kg>0 and args.towed_kg>0:
        parser.error("V3 payload and towing are mutually exclusive; set one mass to zero")
    if args.slope_deg==-25. and args.towed_kg>0:
        parser.error("The revised task excludes towing on -25 degree downhill; use towed_kg=0")
    if args.speed is None:
        args.speed=.3 if args.payload_kg>0 or args.towed_kg>0 else 3.1
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
            parser.error("Use --resume for exact V3 continuation or --initialize_from for a V2/V3 Actor")
        if args.resume:
            args.resume=args.resume.resolve(strict=True)
            source=SPEC.checkpoint(args.resume)
            if source["infos"].get("environment_signature")!=signature(args):
                parser.error("Resume settings differ from saved torque/friction/hitch/contract; use --initialize_from for a changed experiment")
        elif not args.fresh:
            args.initialize_from=(args.initialize_from or BASELINE).resolve(strict=True)
            source=SPEC.initialization_checkpoint(args.initialize_from)
    elif args.resume or args.initialize_from or args.fresh:
        parser.error("Initialization options apply only to train")
    if args.mode in ("play","evaluate"):
        if not args.checkpoint:
            parser.error("An explicit unified checkpoint is required")
        args.checkpoint=args.checkpoint.resolve(strict=True)
        source=SPEC.checkpoint(args.checkpoint)
        if source["infos"].get("environment_signature")!=signature(args):
            parser.error("Playback torque/friction/hitch differ from training; pass the saved values")
        args.num_envs=len(PROTOCOL.cases(args.eval_distance)) if args.mode=="evaluate" else 1
    missing=[name for name in ("isaacsim","isaaclab") if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Activate the complete lab environment first; missing "+", ".join(missing))
    if args.mode=="smoke":
        args.num_envs=min(args.num_envs,168)
    category={"train":"runs","smoke":"validation","play":"playbacks","evaluate":"evaluations"}[args.mode]
    if args.validation:category="validation"
    stamp=now().replace(":","-").replace("T","_").split("+")[0]
    run=args.output_dir.resolve() if args.output_dir else ROOT/category/stamp
    run.mkdir(parents=True,exist_ok=False)
    request=vars(args).copy()
    for key in ("initialize_from","resume","checkpoint","output_dir"):
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
    selected=args.resume or args.initialize_from or args.checkpoint
    meta={"started_at":now(),"mode":args.mode,"status":"running","behavior_acceptance":"not_evaluated",
          "initialization":"resume" if args.resume else "actor_transfer_new_critic_optimizer" if args.initialize_from else "frozen" if args.checkpoint else "fresh",
          "source_checkpoint":str(selected) if selected else None,"source_sha256":sha256(selected) if selected else None}
    write_json(run/"run.json",meta)
    print(f"[R1 Unified V3] {run}",flush=True)
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
