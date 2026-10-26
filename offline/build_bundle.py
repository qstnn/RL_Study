"""Build a USB ZIP and optional overlay patch from the audited R6 worktree.

Framework files come from a separate, complete source inventory. Git ignore rules
must never decide whether a required Isaac Lab runtime module is shipped.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import subprocess
import zipfile

from source_integrity import verify_sources

ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL = 'rl_training_r6_usb_20261016'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect_files(root):
    report = verify_sources(root)
    if report['errors']:
        raise RuntimeError('\n'.join(report['errors']))
    files = set()
    for args in (['git', 'ls-files', '-z'], ['git', 'ls-files', '--others', '--exclude-standard', '-z']):
        files.update(subprocess.check_output(args, cwd=root).decode().split('\0'))
    files.discard('')
    excluded = {'.git', '__pycache__', '.pytest_cache', 'runs', 'logs', 'validation', 'evaluations', 'playbacks'}
    files = {f for f in files if (root/f).is_file() and not excluded.intersection(Path(f).parts)
             and not f.startswith(('offline/envs/', 'offline/checks/', 'offline_sessions/', 'usb_review/')) and not f.endswith(('.pyc', '.pyo'))}
    framework = json.loads((root/'offline/isaaclab_source_manifest.json').read_text())['sha256']
    # Apply the independent source list AFTER generated-output exclusions.
    files.update(framework)
    files.update(str(p.relative_to(root)) for p in (root/'offline/wheelhouse').glob('*.whl'))
    files.discard('offline/bundle_manifest.json')
    for relative in files:
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Invalid bundle input: {relative}')
    return files


def write_zip(root, files, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for relative in sorted(files):
            archive.write(root/relative, TOP_LEVEL+'/'+relative,
                          compress_type=zipfile.ZIP_STORED if relative.endswith('.whl') else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f'Corrupt archive: {temporary}')
    temporary.replace(output)
    sha = digest(output)
    Path(str(output)+'.sha256').write_text(f'{sha}  {output.name}\n')
    return {'archive':str(output), 'bytes':output.stat().st_size, 'sha256':sha, 'files':len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch-from', type=Path)
    parser.add_argument('--patch-output', type=Path)
    parser.add_argument('--revision', default='20261016-offline-station')
    args = parser.parse_args()
    if bool(args.patch_from) != bool(args.patch_output):
        parser.error('--patch-from and --patch-output must be supplied together')
    output = args.output.resolve()
    if args.patch_from and output == args.patch_from.resolve():
        parser.error('The baseline ZIP must be preserved')
    files = collect_files(ROOT)
    hashes = {f:digest(ROOT/f) for f in sorted(files)}
    assets = json.loads((ROOT/'model_lighthw/training/multiskill/records/portable_assets.json').read_text())
    for relative, expected in assets['meshes'].items():
        if hashes.get(relative) != expected:
            raise RuntimeError(f'Mesh checksum mismatch: {relative}')
    record = {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'delivery':'USB only',
              'revision':args.revision, 'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_changes_included':True, 'top_level_directory':TOP_LEVEL,
              'file_count':len(hashes), 'framework_source_files':len(json.loads((ROOT/'offline/isaaclab_source_manifest.json').read_text())['sha256']), 'sha256':hashes}
    (ROOT/'offline/bundle_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    files.add('offline/bundle_manifest.json')
    summary = {'full':write_zip(ROOT, files, output), 'real_meshes':len(assets['meshes']),
               'wheels':len(list((ROOT/'offline/wheelhouse').glob('*.whl')))}
    if args.patch_from:
        with zipfile.ZipFile(args.patch_from) as baseline:
            old = json.loads(baseline.read(TOP_LEVEL+'/offline/bundle_manifest.json'))
            if old['top_level_directory'] != TOP_LEVEL:
                raise RuntimeError('Patch baseline directory mismatch')
            removed = set(old['sha256'])-files
            if removed:
                raise RuntimeError(f'Overlay cannot remove files: {sorted(removed)}')
            changed = {f for f in hashes if old['sha256'].get(f) != hashes[f]}
            changed.add('offline/bundle_manifest.json')
        summary['patch'] = write_zip(ROOT, changed, args.patch_output.resolve())
        summary['patch'].update(baseline_sha256=digest(args.patch_from), changed_files=sorted(changed))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
