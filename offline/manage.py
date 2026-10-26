"""R6 USB delivery: standard-library-only verification and isolated dependency setup."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
        return value.hexdigest()


def environment_path():
    return ROOT / 'envs' / ('cp311-' + digest(ROOT / 'requirements-hashed.txt')[:16]) / 'site-packages'


def verify():
    if sys.version_info[:2] != (3, 11) or sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This USB bundle requires Linux x86_64 / Python 3.11, as used by Isaac Sim 5.0. Set R6_PYTHON to the matching lab/bin/python.')
    record = json.loads((ROOT / 'wheels.json').read_text())
    for name, expected in record['sha256'].items():
        if Path(name).name != name or digest(ROOT / 'wheelhouse' / name) != expected:
            raise RuntimeError(f'Wheel checksum mismatch: {name}; recopy the USB bundle')
    if digest(ROOT / 'requirements-hashed.txt') != record['requirements_sha256']:
        raise RuntimeError('Offline requirements checksum mismatch')
    return record


def prepare():
    if (ROOT/'bundle_manifest.json').is_file():
        subprocess.run([sys.executable, '-I', str(ROOT/'verify_bundle.py')], check=True)
    record = verify()
    target = environment_path()
    marker = target.parent / 'ready.json'
    if marker.is_file() and json.loads(marker.read_text()).get('requirements_sha256') == record['requirements_sha256']:
        print(f'[R6 USB] Dependencies already prepared: {target}')
        return
    target.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.install-', dir=target.parent.parent) as temp:
        staging = Path(temp) / 'site-packages'
        subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
                        'install', '--no-index', '--no-deps', '--no-compile', '--require-hashes',
                        '--find-links', str(ROOT / 'wheelhouse'), '--target', str(staging),
                        '-r', str(ROOT / 'requirements-hashed.txt')], check=True)
        if target.exists():
            raise RuntimeError(f'Incomplete environment already exists at {target}; preserve it and extract a fresh bundle')
        target.parent.mkdir(exist_ok=True)
        staging.rename(target)
        marker.write_text(json.dumps({'requirements_sha256':record['requirements_sha256'], 'python':sys.version, 'packages':len(record['sha256'])}, indent=2)+'\n')
    print(f'[R6 USB] Offline dependencies prepared: {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('path', 'verify', 'prepare'))
    args = parser.parse_args()
    try:
        if args.command == 'path': print(environment_path())
        elif args.command == 'verify': print(json.dumps(verify(), indent=2))
        else: prepare()
    except Exception as error:
        print(f'[R6 USB ERROR] {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
