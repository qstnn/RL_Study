"""Check the USB Python environment and source origins without starting Isaac Sim."""
from pathlib import Path
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from source_integrity import verify_sources

ROOT = Path(__file__).resolve().parent.parent
MODULES = {
    'warp-lang':'warp', 'onnx':'onnx', 'prettytable':'prettytable', 'hidapi':'hid',
    'pyglet':'pyglet', 'transformers':'transformers', 'einops':'einops', 'h5py':'h5py',
    'tensorboard':'tensorboard', 'pin-pink':'pink', 'dex-retargeting':'dex_retargeting',
    'pytest':'pytest', 'flatdict':'flatdict', 'flaky':'flaky', 'junitparser':'junitparser',
    'gymnasium':'gymnasium', 'rsl-rl-lib':'rsl_rl', 'hydra-core':'hydra', 'numpy':'numpy',
    'torch':'torch', 'torchvision':'torchvision', 'scipy':'scipy', 'pillow':'PIL', 'toml':'toml',
    'matplotlib':'matplotlib', 'starlette':'starlette',
}
SOURCES = {'rl_training':ROOT/'source/rl_training/rl_training',
           'isaaclab':ROOT/'Isaaclab/source/isaaclab/isaaclab',
           'isaaclab_assets':ROOT/'Isaaclab/source/isaaclab_assets/isaaclab_assets',
           'isaaclab_rl':ROOT/'Isaaclab/source/isaaclab_rl/isaaclab_rl',
           'isaaclab_tasks':ROOT/'Isaaclab/source/isaaclab_tasks/isaaclab_tasks'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brief', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'offline/checks/doctor-last.json')
    args=parser.parse_args()
    report={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'python':sys.executable,
            'python_version':sys.version, 'isaac_sim':os.environ.get('R6_ISAAC_SIM'), 'project':str(ROOT),
            'modules':{}, 'source_paths':{}, 'errors':[], 'gpu_verified':False}
    report['framework_sources'] = verify_sources(ROOT)
    report['errors'].extend(report['framework_sources']['errors'])
    if sys.version_info[:2] != (3,11) or sys.platform != 'linux' or platform.machine() != 'x86_64':
        report['errors'].append('Expected Python 3.11 / Linux x86_64')
    for name, expected in SOURCES.items():
        spec=importlib.util.find_spec(name)
        actual=Path(spec.origin).resolve() if spec and spec.origin else None
        report['source_paths'][name]=str(actual)
        if actual is None or not actual.is_relative_to(expected.resolve()):
            report['errors'].append(f'{name} resolves outside this USB project: {actual}')
    for name, module_name in MODULES.items():
        try:
            module=importlib.import_module(module_name)
            report['modules'][name]={'status':'import_ok','file':getattr(module,'__file__',None), 'version':str(getattr(module,'__version__','unknown'))}
        except Exception as error:
            report['modules'][name]={'status':'failed','error':f'{type(error).__name__}: {error}'}
            report['errors'].append(f'{name}: {type(error).__name__}: {error}')
    try:
        from isaaclab.app import AppLauncher  # import only; never instantiate here
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
        if importlib.metadata.version('rsl-rl-lib') != '2.3.3':
            raise RuntimeError('This R6 uses rsl-rl-lib 2.3.3; run bash r6.sh prepare and use the USB entry')
        report['runtime_entry_imports']='passed'
    except Exception as error:
        report['errors'].append(f'Runtime entry import: {type(error).__name__}: {error}')
        report['entry_traceback']=traceback.format_exc()
    spec=importlib.util.find_spec('isaacsim')
    if spec is None or not Path(spec.origin).resolve().is_relative_to(Path(os.environ['R6_ISAAC_SIM']).resolve()):
        report['errors'].append('isaacsim is missing or resolves to another installation')
    torch=sys.modules.get('torch')
    numpy=sys.modules.get('numpy')
    if torch and not torch.__version__.startswith('2.7.'):
        report['errors'].append(f'Expected tested PyTorch 2.7.x, found {torch.__version__}')
    if numpy and int(numpy.__version__.split('.')[0]) >= 2:
        report['errors'].append(f'Isaac Lab 2.2 requires NumPy < 2, found {numpy.__version__}')
    report['status']='passed' if not report['errors'] else 'failed'
    out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not args.brief or report['errors']: print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"[R6 USB] Environment {report['status']}; report: {out}; GPU not tested")
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
