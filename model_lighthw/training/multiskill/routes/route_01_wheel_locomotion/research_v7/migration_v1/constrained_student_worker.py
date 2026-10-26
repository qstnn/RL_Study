"""Isolated schema loader + unchanged original autonomous GPU execution worker."""
import argparse
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from migration_v1.constrained_student import load_student
from migration_v1.contract import sha256
from migration_v1.dataset import EvidenceError
from migration_v1.runtime_checks import select_case,check_source_preparation
from migration_v1.student_worker import dependency_sources,run_simulation


def prepare(checkpoint,slope_deg,output,preparation_seconds=3.):
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Use a fresh constrained autonomous output directory')
    policy,manifest = load_student(checkpoint,device='cpu',inference_mode='row')
    case,profile = select_case(manifest,slope_deg)
    preparation = check_source_preparation(manifest,preparation_seconds,case_slope=case['slope_deg'])
    files = set(dependency_sources()) | {Path(__file__).with_name(name).resolve() for name in
        ('constrained_refit.py','constrained_student.py','constrained_student_worker.py')}
    return dict(policy=policy,manifest=manifest,case=case,profile=profile,output=output,
        checkpoint=str(Path(checkpoint).resolve()),sources={str(path):sha256(path) for path in files},preparation=preparation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--slope-deg',type=float,choices=[25.,-20.],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preparation-seconds',type=float,default=3.)
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.checkpoint,args.slope_deg,args.output,args.preparation_seconds)
        review = run_simulation(prepared)
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
        parser.exit(2,'Constrained student rejected: '+str(error)+'\n')
    if not review.get('autonomous_anchor_pass'):
        parser.exit(2,'Constrained student did not pass the complete physical protocol.\n')


if __name__=='__main__':
    main()
