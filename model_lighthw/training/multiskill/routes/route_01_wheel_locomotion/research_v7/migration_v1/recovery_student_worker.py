"""Teacher-off execution entry for the explicit recovery artifact schema.

Reuse the existing physical worker after independent strict loading. No global
loader replacement, teacher call, or action correction is installed.
"""
import argparse
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError
from migration_v1.recovery_student import load_recovery_student
from migration_v1.recovery_bc import source_signatures
from migration_v1.runtime_checks import select_case,check_source_preparation
from migration_v1.student_worker import dependency_sources,run_simulation


def prepare(checkpoint,slope_deg,output,preparation_seconds=3.):
    output=Path(output).resolve()
    if output.exists():raise EvidenceError('Use a fresh autonomous recovery-student output directory')
    policy,manifest=load_recovery_student(checkpoint,device='cpu',inference_mode='row')
    case,profile=select_case(manifest,slope_deg)
    clock=check_source_preparation(manifest,preparation_seconds,case_slope=case['slope_deg'])
    sources={str(path):sha256(path) for path in dependency_sources()}
    sources.update(source_signatures())
    return dict(policy=policy,manifest=manifest,case=case,profile=profile,output=output,
        checkpoint=str(Path(checkpoint).resolve()),sources=sources,preparation=clock)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--slope-deg',type=float,choices=(25.,-20.),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preparation-seconds',type=float,default=3.)
    args=parser.parse_args(argv)
    try:
        result=run_simulation(prepare(args.checkpoint,args.slope_deg,args.output,args.preparation_seconds))
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
        parser.exit(2,'Autonomous recovery student rejected: '+str(error)+'\n')
    if not result['autonomous_anchor_pass']:parser.exit(2,'Recovery student failed the complete physical protocol.\n')


if __name__=='__main__':main()
