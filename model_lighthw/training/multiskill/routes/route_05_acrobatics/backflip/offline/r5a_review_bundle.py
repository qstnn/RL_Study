"""R5A 离线回收：模型、运行来源、物理轨迹、视频和失败诊断放在同一目录。"""
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[7]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def inside(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Path outside root: '+str(name))
    return path


def copy_file(src,dst,expected=None):
    if src.is_symlink(): raise ValueError('Symlink in review inputs: '+str(src))
    before=digest(src)
    if expected and before!=expected: raise ValueError('Source provenance mismatch: '+str(src))
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    if digest(dst)!=before or digest(src)!=before: raise RuntimeError('File changed during collection: '+str(src))


def copy_tree(src,dst,exclude=()):
    for path in sorted(src.rglob('*')):
        rel=path.relative_to(src)
        if any(part in {'__pycache__','.git','asset_cache',*exclude} for part in rel.parts): continue
        if path.is_symlink(): raise ValueError('Symlink in review inputs: '+str(path))
        if path.is_file() and path.suffix!='.pyc': copy_file(path,dst/rel)


def training_plot(run,out):
    source=run/'training_curve.csv'
    if not source.is_file(): return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with source.open() as f: rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for ax,key,title in zip(axes,('mean_reward','mean_episode_length'),('Mean episode reward','Mean episode length (policy steps)')):
        points=[(int(r['completed_updates']),float(r[key])) for r in rows if r.get(key)]
        if points: ax.plot(*zip(*points))
        ax.set_ylabel(title); ax.grid(alpha=.3)
    axes[-1].set_xlabel('Completed PPO updates')
    fig.tight_layout(); fig.savefig(out,dpi=130); plt.close(fig)


def collect(runs,route,session=None):
    if route!='05a': raise ValueError('R5A collector accepts route 05a only')
    created=datetime.now(ZoneInfo('Asia/Shanghai'))
    folder=ROOT/'usb_review/route_05a'/created.strftime('%Y-%m-%d_%H-%M-%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    summaries=[]; cases=[]; assets={}; stages=set()
    try:
        for index,path in enumerate(runs):
            run=Path(path).resolve(strict=True)
            if not run.is_relative_to(ROOT): raise ValueError('Run must be inside this project: '+str(run))
            worker=read(run/'worker_status.json',{})
            pid=worker.get('pid')
            if pid and worker.get('stage')!='completed':
                process=Path('/proc')/str(pid)/'cmdline'
                try: command=process.read_bytes().replace(b'\0',b' ').decode(errors='replace')
                except (FileNotFoundError,PermissionError): command=''
                if str(run) in command and 'worker.py' in command:
                    raise RuntimeError('Training/replay worker is still active: '+str(run))
            label=f'{index:02d}_{run.name}'
            target=folder/'runs'/label
            copy_tree(run,target)
            meta=read(run/'run.json',{})
            stage=read(run/'contract.json',{}).get('stage',meta.get('stage','unknown'))
            stages.add(stage)
            result=read(run/'result.json',{})
            role=read(run/'review_role.json',{}).get('role','unspecified')
            summary=dict(folder=str(target.relative_to(folder)),original_path=str(run),role=role,
                mode=meta.get('mode'),status=meta.get('status','missing_run_metadata'),
                error=meta.get('error'),worker_error=read(run/'worker_error.json'),
                runtime_check=read(run/'runtime_check.json'),
                completed_updates=result.get('completed_updates'),checkpoint=result.get('checkpoint'))
            summaries.append(summary)
            for case in result.get('cases',[]): cases.append(dict(run=label,mode=meta.get('mode'),role=role,**case))
            manifest=read(run/'asset_manifest.json',{})
            if manifest:
                run_assets={manifest['urdf']:manifest['urdf_sha256'],**manifest['meshes']}
                for name,sha in run_assets.items():
                    if name in assets and assets[name]!=sha: raise ValueError('Runs use conflicting assets: '+name)
                    assets[name]=sha
            if meta.get('mode')=='train': training_plot(target,target/'training_curve.png')
        if session:
            session=Path(session).resolve(strict=True)
            # Runs already copied once; keep station logs and environment diagnostics separately.
            target=folder/'session'; target.mkdir()
            for src in sorted(session.iterdir()):
                if src.is_file() and src.name not in {'.station.lock','latest_review.json'}:
                    copy_file(src,target/src.name)
            if (session/'environment').is_dir(): copy_tree(session/'environment',target/'environment')
            if (session/'migration').is_dir(): copy_tree(session/'migration',target/'migration')
        for name,sha in assets.items(): copy_file(inside(ROOT,name),folder/'provenance'/name,sha)
        # Include the actual vendored framework and version/lock files, without copying installed environments/wheels.
        framework=ROOT/'offline/isaaclab_source_manifest.json'
        if framework.is_file():
            record=read(framework)
            entries=record.get('sha256',record.get('files',{}))
            if not isinstance(entries,dict): raise ValueError('Unknown framework manifest schema')
            for name,sha in entries.items():
                if isinstance(sha,dict): sha=sha['sha256']
                copy_file(inside(ROOT,name),folder/'provenance'/name,sha)
        for path in sorted((ROOT/'offline').iterdir()):
            if path.is_file() and path.suffix in {'.json','.txt','.md'}:
                copy_file(path,folder/'provenance/offline'/path.name)
        copy_file(Path(__file__).resolve(),folder/'provenance/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/offline/r5a_review_bundle.py')
        evaluation=[c for c in cases if c['mode']=='evaluate']
        groups={role:[c for c in evaluation if c['role']==role] for role in sorted({c['role'] for c in evaluation})}
        evaluation_scope='final' if 'final' in groups else 'all_available'
        if 'final' in groups: evaluation=groups['final']
        playback=[c for c in cases if c['mode']=='play']
        write(folder/'summary.json',dict(schema_version=1,route='05a',stage=next(iter(stages)) if len(stages)==1 else 'mixed',stages=sorted(stages),
            created_at=created.isoformat(),runs=summaries,cases=cases,
            evaluation_scope=evaluation_scope,
            evaluation_groups={role:dict(cases=len(rows),successes=sum(bool(c.get('success')) for c in rows)) for role,rows in groups.items()},
            evaluation_successes=sum(bool(c.get('success')) for c in evaluation),evaluation_cases=len(evaluation),
            playback_successes=sum(bool(c.get('success')) for c in playback),playback_cases=len(playback),
            collector_sha256=digest(Path(__file__).resolve()),
            formal_behavior_acceptance='requires_review',full_backflip_implemented=False,
            hardware_acceptance='not_evaluated',excluded=['regenerable asset_cache','Python bytecode','installed environments','wheel binaries']))
        (folder/'README.txt').write_text(
            'R5A J1 训练回收包：请把整个文件夹用 U 盘拷回。\n'
            '先看 summary.json，再看 session/ 和 runs/。训练完成不等于行为通过。\n'
            '包含全部模型、训练曲线、来源快照、1 kHz 物理轨迹、逐例判定、回放 MP4（若已成功生成）、错误日志。\n'
            'provenance/ 保存模型资产、框架和依赖清单。完整后空翻及真实硬件尚未验收。\n'
            '缺少视频或评估时，先检查失败日志；不删除失败样本。\n'
            '校验：在 R5A 项目根目录运行 bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh verify --folder 此文件夹路径\n')
        hashes={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        write(folder/'review_manifest.json',dict(schema='r5a_usb_review_v1',created_at=created.isoformat(),sha256=hashes))
        verify(folder)
        (ROOT/'usb_review/route_05a/LATEST.txt').write_text(str(folder)+'\n')
        return folder
    except Exception as error:
        write(folder/'collection_error.json',dict(error=f'{type(error).__name__}: {error}',status='incomplete'))
        raise


def verify(folder):
    folder=Path(folder).resolve(strict=True)
    manifest=read(folder/'review_manifest.json')
    if not manifest or manifest.get('schema')!='r5a_usb_review_v1': raise ValueError('Not a complete R5A review folder')
    failed=[]
    for name,expected in manifest['sha256'].items():
        path=inside(folder,name)
        if not path.is_file() or digest(path)!=expected: failed.append(name)
    if failed: raise ValueError('Review checksum mismatch: '+', '.join(failed))
    return dict(status='passed',files=len(manifest['sha256']),folder=str(folder),behavior_acceptance='not_evaluated')
