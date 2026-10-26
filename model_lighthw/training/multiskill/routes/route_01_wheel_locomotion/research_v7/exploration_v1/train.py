"""Manual R1 exploration commands; formal continuation requires matching real short evidence.

`--iterations` means ADDITIONAL PPO proposals, including rejected proposals.
Environment failures are ordinary RL transitions, not an early-training stop rule.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exploration_v1 import runtime
from migration_v1.dataset import EvidenceError


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','short','train','_execute'):
        sub=commands.add_parser(command,help=argparse.SUPPRESS if command=='_execute' else None)
        sub.add_argument('--output',required=True)
        sub.add_argument('--num-envs',type=int,default=64)
        sub.add_argument('--inference-mode',choices=('row','batch'),default='row')
        sub.add_argument('--towing-initialization',choices=('bc_preparation_9f406','zero'),default='bc_preparation_9f406')
        sub.add_argument('--terminal-timeout-run')
        sub.add_argument('--terminal-reference-run')
        sub.add_argument('--validated-short-run')
        sub.add_argument('--resume-checkpoint')
        sub.add_argument('--iterations',type=int)
        sub.add_argument('--save-interval',type=int,default=100)
        if command in ('prepare','_execute'):
            sub.add_argument('--kind',choices=('runtime_short_validation','formal_training'),default='runtime_short_validation')
    return parser


def prepare_args(args):
    kind=getattr(args,'kind','formal_training' if args.command=='train' else 'runtime_short_validation')
    return runtime.prepare(args.output,kind=kind,terminal_timeout_run=args.terminal_timeout_run,
        terminal_reference_run=args.terminal_reference_run,towing_initialization=args.towing_initialization,
        validated_short_run=args.validated_short_run,iterations=args.iterations,save_interval=args.save_interval,
        num_envs=args.num_envs,inference_mode=args.inference_mode,resume_checkpoint=args.resume_checkpoint)


def record_child_outcome(prepared,returncode):
    out=prepared['output']
    result=dict(returncode=int(returncode),signal=(-int(returncode) if returncode<0 else None),
        runtime_error_present=(out/'runtime_error.json').is_file(),
        runtime_review_present=(out/'runtime_review.json').is_file(),
        final_checkpoint_present=(out/'final.pt').is_file(),
        source_manifest_present=(out/'source_manifest.json').is_file(),
        artifact_role='subprocess_exit_record_before_independent_validation')
    if out.is_dir():
        (out/'child_process.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify_process_outcome(prepared,returncode):
    """Kit can exit zero after an exception; a success artifact is mandatory."""
    out=prepared['output']
    if returncode!=0 or (out/'runtime_error.json').exists():
        raise EvidenceError('Exploration subprocess failed; inspect runtime_error.json and console log')
    if not (out/'runtime_review.json').is_file():
        raise EvidenceError('Exploration subprocess exited without its required completion evidence')
    if prepared['kind']=='runtime_short_validation':
        return runtime.validate_short_run(out,expected_configuration=prepared['configuration'])
    return runtime.validate_formal_completion(prepared)


def main(argv=None):
    args=build_parser().parse_args(argv)
    prepared=prepare_args(args)  # CPU only; reject before an App or output directory.
    if args.command=='prepare':
        print(json.dumps(dict(version=runtime.VERSION,prepared=True,kind=prepared['kind'],
            output=str(prepared['output']),configuration=prepared['configuration'],
            source_count=len(prepared['sources']),terminal_preview_verified=True,
            app_started=False,robot_ppo_updates=0,final_behavior_pass=False),indent=2))
        return 0
    if args.command=='_execute':
        runtime.run(prepared)
        return 0
    command=[sys.executable,str(Path(__file__).resolve()),'_execute','--kind',prepared['kind'],
        '--output',str(prepared['output']),'--num-envs',str(args.num_envs),
        '--inference-mode',args.inference_mode,'--towing-initialization',args.towing_initialization,
        '--save-interval',str(args.save_interval)]
    for flag,value in (('--terminal-timeout-run',args.terminal_timeout_run),
            ('--terminal-reference-run',args.terminal_reference_run),
            ('--validated-short-run',args.validated_short_run),('--resume-checkpoint',args.resume_checkpoint),
            ('--iterations',args.iterations)):
        if value is not None: command.extend([flag,str(value)])
    result=subprocess.run(command,check=False)
    record_child_outcome(prepared,result.returncode)
    review=verify_process_outcome(prepared,result.returncode)
    print(json.dumps(review,indent=2))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
