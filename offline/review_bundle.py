"""Simulator-free review-folder collection and verification, shared by all routes."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {'asset_cache', '__pycache__', '.git', '.pytest_cache'}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root/relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {relative}')
    return path


def stable_copy(source, target, expected=None):
    source, target = Path(source), Path(target)
    if source.is_symlink():
        raise ValueError(f'Review inputs must be real files, not symlinks: {source}')
    before = digest(source)
    if expected and before != expected:
        raise ValueError(f'Source changed since run: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(target) != before or digest(source) != before:
        raise ValueError(f'File changed during collection: {source}')
    return before


def copy_tree(source, target):
    source=Path(source).resolve()
    for path in sorted(source.rglob('*')):
        relative=path.relative_to(source)
        if SKIP_PARTS.intersection(relative.parts) or path.suffix in ('.pyc','.tmp'):
            continue
        if path.is_symlink():
            raise ValueError(f'Unexpected linked run artifact: {path}')
        if path.is_file():
            stable_copy(path,target/relative)


def verify(folder):
    folder = Path(folder).resolve()
    manifest = read(folder/'manifest.json')
    if not manifest or manifest.get('schema_version') != 1:
        raise ValueError('Missing or unsupported review manifest.json')
    errors=[]
    for relative, expected in manifest['sha256'].items():
        path=safe_child(folder,relative)
        if not path.is_file() or path.is_symlink() or digest(path)!=expected:
            errors.append(relative)
    if errors:
        raise ValueError('Review checksum mismatch: '+', '.join(errors))
    return {'status':'passed','files':len(manifest['sha256']), 'bytes':manifest['bytes']}


def collect(runs, route, session=None, project=ROOT, output_root=None):
    """Freeze all checkpoints/logs/raw traces and exact run snapshots into one new folder."""
    project=Path(project).resolve()
    destination=Path(output_root or project/'usb_review').resolve()
    label=re.sub(r'[^A-Za-z0-9_-]+','_',str(route))
    if not label: raise ValueError('Empty route label')
    final=destination/f'route_{label}_{stamp()}'
    temporary=final.with_name(final.name+'.collecting')
    # Do not recursively include a return folder placed within a source run.
    for run in runs:
        if destination.is_relative_to(Path(run).resolve()):
            raise ValueError('Review output must be outside the input run')
    temporary.mkdir(parents=True,exist_ok=False)
    inventory=[]; warnings=[]; asset_hashes={}
    try:
        if session:
            session=Path(session).resolve()
            if not (session/'.station.lock').exists():
                warnings.append('Session has no station lock record; may be an older or imported run')
            session_status=read(session/'session.json',{}).get('status','unknown')
            if session_status!='completed': warnings.append('Session status: '+session_status)
            for path in session.iterdir():
                if path.is_file() and path.name not in ('.station.lock','latest_review.json'):
                    stable_copy(path,temporary/'session'/path.name)
            if (session/'environment').exists():
                copy_tree(session/'environment',temporary/'environment')
        if not runs: warnings.append('No child run artifacts; inspect session and environment logs')
        for index, run in enumerate(runs):
            run=Path(run).resolve(strict=True)
            if not run.is_dir(): raise ValueError(f'Not a run directory: {run}')
            status=read(run/'worker_status.json',{})
            process=Path('/proc')/str(status.get('pid',-1))/'cmdline'
            if process.is_file() and str(run).encode() in process.read_bytes().split(b'\0'):
                raise RuntimeError(f'Run is still active; collect after it stops: {run}')
            request=read(run/'request.json',{})
            meta=read(run/'run.json',{})
            result=read(run/'result.json',{})
            name=f'{index:02d}_{request.get("mode", "run")}_{run.name}'
            copied=temporary/'runs'/name
            copy_tree(run,copied)
            missing=[f for f in ('run.json','request.json','source_manifest.json','runtime_check.json') if not (copied/f).is_file()]
            if meta.get('status')!='completed': warnings.append(f'{name}: run status is {meta.get("status", "unknown")}')
            if missing: warnings.append(f'{name}: missing '+', '.join(missing))
            # Provenance is the captured training source, never silently replaced by current source.
            for relative, expected in read(copied/'source_manifest.json',{}).items():
                path=safe_child(copied/'source_snapshot',relative)
                if not path.is_file() or digest(path)!=expected:
                    raise ValueError(f'Run source snapshot mismatch: {path}')
            manifest=read(copied/'asset_manifest.json',{})
            assets=dict(manifest.get('meshes',{}))
            if manifest.get('urdf') and manifest.get('urdf_sha256'):
                assets[manifest['urdf']]=manifest['urdf_sha256']
            for relative, expected in assets.items():
                key=f'{expected}/{Path(relative).name}'
                if key not in asset_hashes:
                    source=safe_child(project,relative)
                    if not source.is_file() or digest(source)!=expected:
                        warnings.append(f'{name}: original asset missing/changed: {relative}')
                        continue
                    stable_copy(source,temporary/'assets'/key,expected)
                    asset_hashes[key]=expected
            appearance=read(copied/'play_visuals.json',{})
            if appearance.get('sky_local_path') and appearance.get('sky',{}).get('sha256'):
                sky=Path(appearance['sky_local_path']).resolve()
                expected=appearance['sky']['sha256']; key=f'{expected}/{sky.name}'
                if sky.is_relative_to(project) and sky.is_file() and digest(sky)==expected:
                    if key not in asset_hashes:
                        stable_copy(sky,temporary/'assets'/key,expected); asset_hashes[key]=expected
                else:
                    warnings.append(f'{name}: original playback sky missing/changed/outside project')
            consistency=read(copied/'source_consistency.json',{})
            if consistency and consistency.get('status')!='passed': warnings.append(f'{name}: runtime source consistency failed')
            checkpoint_paths=sorted(p for p in copied.rglob('*.pt') if 'source_snapshot' not in p.parts)
            source_cp=request.get('checkpoint')
            expected=meta.get('source_sha256') or meta.get('checkpoint_sha256')
            if source_cp and expected:
                src=Path(source_cp)
                # Any input model is kept independently, including resume-source optimizer state.
                target=temporary/'input_checkpoints'/f'{expected}.pt'
                if not target.exists():
                    if src.is_file(): stable_copy(src,target,expected)
                    else: warnings.append(f'{name}: input checkpoint unavailable: {source_cp}')
            entry={'directory':str(copied.relative_to(temporary)), 'source_run':str(run),
                   'mode':request.get('mode'), 'run_status':meta.get('status','unknown'),
                   'payload_kg':request.get('payload_kg'), 'result_status':result.get('status'),
                   'checkpoints':[str(p.relative_to(temporary)) for p in checkpoint_paths],
                   'evaluation_success_rate':result.get('success_rate'), 'missing_required':missing,
                   'protocol':result.get('protocol','fixed_payload_screen'),
                   'payload_events':[str(p.relative_to(temporary)) for p in copied.glob('*payload_events.jsonl')],
                   'videos':[str(p.relative_to(temporary)) for p in copied.glob('*.mp4')],
                   'raw_traces':[str(p.relative_to(temporary)) for p in copied.glob('*_trace.npz')],
                   'physics_traces':[str(p.relative_to(temporary)) for p in copied.glob('*_physics.npz')]}
            inventory.append(entry)
        # Environment manifests are tiny; dependency wheels / simulator installs are not return data.
        for relative in ('offline/wheels.json','offline/requirements.lock','offline/isaaclab_source_manifest.json','offline/bundle_manifest.json'):
            src=project/relative
            if src.is_file(): stable_copy(src,temporary/'environment'/Path(relative).name)
        plots=[]
        try:
            from review_plots import make_plots
            plots=make_plots(temporary)
        except Exception as error:
            warnings.append(f'Overview plots unavailable; raw logs are preserved: {error}')
        report={'schema_version':1,'created_at':now(),'route':str(route),'host':socket.gethostname(),
                'session_status':read(Path(session)/'session.json',{}).get('status') if session else None,
                'runs':inventory,'warnings':warnings,'overview_plots':plots,'asset_sha256':asset_hashes,
                'collection_complete_without_warnings':not warnings,'all_saved_checkpoints_included':True,'behavior_acceptance':'not_evaluated',
                'notes':['Execution completion is not behavior acceptance. Inspect the raw trajectories and failures.',
                         'No training-time 1kHz full-population dump; post-training evaluation has optional 1kHz traces.',
                         'Asset cache, Python caches, wheels and Isaac Sim installation are excluded.']}
        write(temporary/'review.json',report)
        summary=['# 离线训练审查资料','',f'路线：{route}；收集时间：{report["created_at"]}',
                 '', '**把这个文件夹整体用U盘带回即可，不需要另外找模型、日志或视频。**', '',
                 '- `review.json`：运行索引、完成状态、缺失资料和待审查项。',
                 '- `*_overview.png`：训练概览曲线，原始CSV/JSONL/TensorBoard一并保留。',
                 '- `runs/`：全部已保存模型、TensorBoard、CSV/JSONL、实际配置与源码快照、原始报错。',
                 '- `*_trace.npz` / `*_trace.csv`：50Hz轨迹，保留自动重置前的终止帧。',
                 '- `*_physics.npz`：启用时的1kHz状态/接触/实际与计算力矩，关节顺序见trace.json。',
                 '- `*.mp4` / `*_preview.png`：回放录像和预览，视频对应时间见video_frames.csv。',
                 '- `*payload_events.jsonl` / `*_payload.json`：键盘变载事件、实际质量/质心/惯量核验；交互回放不计入固定载荷成功率。',
                 '- `assets/`、`input_checkpoints/`：按SHA保存的原始网格/URDF及输入模型。',
                 '- `environment/`：环境/硬件信息及软件、交付版本清单。',
                 '', '校验：在任意装有Python 3的电脑执行 `python3 verify_review.py .`。',
                 '模型应由本路线匹配的源码/环境回放；不要把R6模型直接用于其他路线。',
                 '完成、筛查成功率和视频都不等于重载技能、热负载或实机验收。', '', '## 当前记录']
        for item in inventory:
            summary.append(f'- {item["mode"]}: {item["run_status"]}; {item["directory"]}')
        summary += ['', '## 缺失或失败信息']+[f'- {w}' for w in warnings] if warnings else ['', '未发现收集时的缺失文件；行为结论需另行审查。']
        (temporary/'README.md').write_text('\n'.join(summary)+'\n')
        stable_copy(Path(__file__).with_name('verify_review.py'),temporary/'verify_review.py')
        hashes={str(p.relative_to(temporary)):digest(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
        write(temporary/'manifest.json',{'schema_version':1,'created_at':now(),'sha256':hashes,
              'bytes':sum((temporary/p).stat().st_size for p in hashes)})
        verify(temporary)
        temporary.rename(final)
    except BaseException:
        # A .collecting directory is deliberately never advertised as a complete review folder.
        raise
    return final
