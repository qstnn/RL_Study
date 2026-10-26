"""Build the R5A USB archive from the verified base manifest and explicit R5A additions."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
TOP='rl_training_r5a_usb_20261019'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def build(output):
    manifest=ROOT/'offline/bundle_manifest.json'
    original=json.loads(manifest.read_text())
    names=set(original['sha256'])
    names.update(('r5a.sh','R5A_START_HERE.md','U盘启动说明.md',
                  'offline/r5a_review_bundle.py','offline/build_r5a_bundle.py','offline/R5A_VALIDATION.json'))
    source=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/r5a'
    route=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip'
    for folder in (source,route/'scripts',route/'tests'):
        names.update(str(p.relative_to(ROOT)) for p in folder.glob('*.py'))
    names.add(str((route/'IMPLEMENTATION_STATUS.md').relative_to(ROOT)))
    hashes={}
    excluded={'runs','validation','evaluations','playbacks','offline_sessions','usb_review','source_snapshot',
              '__pycache__','.git'}
    for name in sorted(names):
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or excluded.intersection(Path(name).parts) or name.startswith(('offline/envs/','offline/checks/')):
            raise ValueError('Unexpected delivery path: '+name)
        if not path.is_file() or (ROOT/name).is_symlink(): raise ValueError('Missing or linked input: '+name)
        hashes[name]=digest(path)
    record=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),delivery='USB only',
        revision='20261019-r5a-j1-v1',base_commit=original.get('base_commit'),working_tree_changes_included=True,
        top_level_directory=TOP,file_count=len(hashes),framework_source_files=873,
        stage='J1_jump_landing',full_backflip_implemented=False,sha256=hashes)
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.building')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for name in [*hashes,'offline/bundle_manifest.json']:
            path=ROOT/name
            archive.write(path,TOP+'/'+name,compress_type=zipfile.ZIP_STORED if path.suffix in {'.whl','.zip','.png','.mp4'} else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip()
        if bad: raise ValueError('ZIP CRC failure: '+bad)
    temporary.replace(output)
    sha=digest(output)
    output.with_suffix('.zip.sha256').write_text(sha+'  '+output.name+'\n')
    print(json.dumps(dict(archive=str(output),sha256=sha,bytes=output.stat().st_size,files=len(hashes)+1,top=TOP),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path)
    build(p.parse_args().output)
