"""Check the complete framework source against an independently captured inventory.

The expected list comes from the working Isaac Lab installation, not git ls-files:
Isaac Lab's broad `datasets` ignore rule also hides a required Python package.
No simulator or third-party Python package is imported here.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent


def verify_sources(root=ROOT):
    root = Path(root).resolve()
    try:
        manifest = json.loads((root/'offline/isaaclab_source_manifest.json').read_text())
    except (OSError, ValueError) as error:
        return {'status':'failed', 'checked_files':0, 'errors':[f'Framework source manifest: {error}']}
    errors = []
    expected_version = manifest['isaaclab_version']
    version = root/'Isaaclab/VERSION'
    if not version.is_file() or version.read_text().strip() != expected_version:
        errors.append(f'Expected bundled Isaac Lab {expected_version}')
    for relative, expected in manifest['sha256'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f'{relative}: missing framework source')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'{relative}: framework source checksum mismatch')
    return {'status':'failed' if errors else 'passed',
            'checked_files':len(manifest['sha256']), 'errors':errors}


if __name__ == '__main__':
    report = verify_sources()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)
