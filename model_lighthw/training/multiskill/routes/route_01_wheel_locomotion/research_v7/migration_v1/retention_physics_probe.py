"""Explicit diagnostic mu/solver contrasts; never current-profile retention admission."""
import argparse
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import sha256
from migration_v1 import retention_protocol as protocol
from migration_v1 import retention_worker as worker


def prepare(checkpoint,case_index,output,*,variant):
    if variant not in protocol.DIAGNOSTIC_PHYSICS_VARIANTS:
        raise ValueError('A named diagnostic physics variant is required')
    prepared = worker.prepare(checkpoint,case_index,output,diagnostic_physics_override=variant)
    source = Path(__file__).resolve()
    prepared['sources'][str(source)] = sha256(source)
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--case-index',type=int,choices=range(12),required=True)
    parser.add_argument('--variant',choices=protocol.DIAGNOSTIC_PHYSICS_VARIANTS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        result = worker.run_simulation(prepare(args.checkpoint,args.case_index,args.output,variant=args.variant))
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
        parser.exit(2,'Retention physics probe rejected: '+str(error)+'\n')
    if not result['success']:
        parser.exit(2,'Diagnostic configuration failed the original behavior protocol.\n')


if __name__=='__main__':
    main()
