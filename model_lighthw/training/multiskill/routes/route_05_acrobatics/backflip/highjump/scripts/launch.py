"""R5A J1 离线进程入口；完整后空翻不自动启用。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

MULTISKILL=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(MULTISKILL))
from common import REPO,ROOT,SOURCE,URDF,asset_manifest,module_from_file,now,sha256,snapshot,write_json
sys.path.insert(0,str(SOURCE))
from r5a_highjump import spec
from static_audit import audit
ROUTE=spec.ROUTE


def implementation_hashes():
    paths=[SOURCE/'r5a_highjump'/name for name in ('spec.py','protocol.py','env.py','runner_cfg.py')]
    paths += [SOURCE/'r5a'/name for name in ('spec.py','protocol.py')]
    paths += [ROUTE/'scripts'/name for name in ('quality.py','worker.py','capture.py')]
    return {str(p.relative_to(REPO)):sha256(p) for p in paths}


def checkpoint(path):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=False)
    info=saved.get('infos') or {}
    if info.get('r5a_contract')!=spec.CONTRACT: raise ValueError('Checkpoint is not this R5A J1 contract; R1/R6 and full-backflip policies are rejected')
    if info.get('implementation_sha256')!=implementation_hashes(): raise ValueError('R5A runtime code differs from checkpoint; explicit migration is required')
    for name,tensor in saved['model_state_dict'].items():
        if not torch.isfinite(tensor).all(): raise ValueError('Non-finite checkpoint tensor: '+name)
    count=info.get('completed_updates')
    if not isinstance(count,int) or count<0 or (count==0 and info.get('initialization')!='actor_warm_start'):
        raise ValueError('Missing completed-update count')
    return saved


def check_completion(run,mode):
    if (run/"worker_error.json").is_file():
        failure=json.loads((run/"worker_error.json").read_text())
        raise RuntimeError(f"Worker stage {failure.get('stage')}: {failure.get('exception_type')}: {failure.get('error')}; see {run/'console.log'}")
    for name in ('runtime_check.json','result.json'):
        if not (run/name).is_file():
            raise RuntimeError(f'Missing {name}; see {run/"console.log"} and worker_error.json for the original error')
    if (run/'worker_error.json').exists(): raise RuntimeError(f'Worker failed; see {run/"worker_error.json"}')
    runtime=json.loads((run/'runtime_check.json').read_text()); result=json.loads((run/'result.json').read_text())
    if runtime.get('dimensions')!=[400,109,16] or result.get('mode')!=mode or result.get('status')!='completed':
        raise RuntimeError('R5A runtime/result contract mismatch')
    if mode=='train':
        checkpoint(result['checkpoint'])
        if result['completed_updates']!=result['requested_updates']: raise RuntimeError('Training budget incomplete')
        if sha256(result['checkpoint'])!=result['checkpoint_sha256']: raise RuntimeError('Saved checkpoint changed')
    if mode in ('evaluate','play'):
        if not result.get('cases'): raise RuntimeError('No replay cases saved')
        for case in result['cases']:
            if not (run/case['trace']).is_file(): raise RuntimeError('Missing raw replay trace')
    return result


def positive(value):
    value=int(value)
    if value<=0: raise argparse.ArgumentTypeError('Must be positive')
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','smoke','train','evaluate','play'))
    p.add_argument('--route',choices=('05a',),default='05a')
    p.add_argument('--num_envs','--num-envs',type=positive,default=512)
    p.add_argument('--max_iterations','--max-iterations',type=positive,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--eval_seeds',type=int,nargs='+',default=[43,44,45])
    p.add_argument('--eval_episodes',type=positive,default=4)
    p.add_argument('--play-episodes',type=positive,default=10)
    p.add_argument('--record-video',action='store_true')
    p.add_argument('--physics-trace',action='store_true',help='Evaluation and playback always retain 1 kHz traces')
    p.add_argument('--reference-profile',type=int,choices=range(len(spec.REFERENCE_PROFILES)),default=spec.CONTRACT['reference_profile'])
    p.add_argument('--probe-sweep',action='store_true')
    args=p.parse_args(argv)
    if (args.probe_sweep or args.reference_profile!=spec.CONTRACT['reference_profile']) and args.mode!='smoke':p.error('Reference overrides are diagnostic probes only')
    assets=asset_manifest(); static=audit(URDF,spec)
    if args.mode=='check':
        print(json.dumps({'task':spec.TASK,'contract':spec.CONTRACT,'audit':static,'assets':assets,'gpu_verified':False},ensure_ascii=False,indent=2)); return 0
    if args.mode in ('evaluate','play') and not args.checkpoint: p.error('An explicit R5A checkpoint is required')
    if args.mode=='smoke' and args.checkpoint: p.error('Smoke uses the reference controller and zero residual action')
    if args.checkpoint:
        args.checkpoint=args.checkpoint.resolve(strict=True); saved=checkpoint(args.checkpoint)
        if args.mode=='train' and saved['infos'].get('initialization')=='frozen_migration':
            p.error('Frozen migration is evaluation-only; use the explicit warm_start.pt for v2 refinement')
    if args.mode in ('evaluate','play'): args.num_envs=1
    if args.mode=='smoke': args.num_envs=min(args.num_envs,128)
    if args.mode!='play' and args.record_video: p.error('--record-video is for play')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S_%f')
    folder={'train':'runs','smoke':'validation','play':'playbacks','evaluate':'evaluations'}[args.mode]
    run=args.run_dir.resolve() if args.run_dir else ROUTE/folder/stamp
    run.mkdir(parents=True,exist_ok=False)
    asset=module_from_file('_r5a_zero_payload_asset',SOURCE/'payload_asset.py')
    model=asset.write_payload_urdf(URDF,run/'input/robot.urdf',0.)
    write_json(run/'model.json',model); write_json(run/'asset_manifest.json',assets)
    write_json(run/'feasibility_static.json',static); write_json(run/'contract.json',spec.CONTRACT)
    snapshot(run)
    hashes=json.loads((run/'source_manifest.json').read_text())
    for path in (REPO/'r5a.sh',REPO/'offline/r5a_refine.py',REPO/'offline/r5a_review_bundle.py',ROUTE/'compat/v1.json'):
        if path.is_file():
            rel=str(path.relative_to(REPO)); target=run/'source_snapshot'/rel
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target); hashes[rel]=sha256(target)
    write_json(run/'source_manifest.json',hashes)
    request={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    request['eval_seeds']=list(dict.fromkeys(args.eval_seeds)); write_json(run/'request.json',request)
    command=[sys.executable,'-u',str(ROUTE/'scripts/worker.py'),'--run',str(run)]
    meta=dict(started_at=now(),status='running',mode=args.mode,command=command,route='05a',stage=spec.CONTRACT['stage'],
        behavior_acceptance='not_evaluated',initialization='resume_r5a' if args.mode=='train' and args.checkpoint else 'frozen_r5a' if args.checkpoint else 'fresh')
    if args.checkpoint: meta.update(checkpoint=str(args.checkpoint),source_sha256=sha256(args.checkpoint))
    write_json(run/'run.json',meta); print('[R5A RUN]',run,flush=True)
    process=None; code=1
    try:
        with (run/'console.log').open('w') as stream:
            process=subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,errors='replace',bufsize=1,start_new_session=True)
            for line in process.stdout: stream.write(line); stream.flush(); print(line,end='',flush=True)
            code=process.wait()
        if code: raise RuntimeError(f'R5A worker exit {code}; see {run/"console.log"} and worker_error.json')
        check_completion(run,args.mode)
        meta['status']='completed'
    except KeyboardInterrupt:
        code=130; meta.update(status='interrupted',error='KeyboardInterrupt')
    except Exception as error:
        code=1; meta.update(status='failed',error=str(error),traceback=traceback.format_exc())
        print('[R5A ERROR]',error,file=sys.stderr,flush=True)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
        meta.update(finished_at=now(),returncode=code); write_json(run/'run.json',meta)
    return code


if __name__=='__main__': raise SystemExit(main())
