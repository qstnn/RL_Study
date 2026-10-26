"""R4a先验证教师、再模仿、再自主PPO；任何阶段均不自动晋级或自动冻结。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common_guided import HERE, REPO, now, read, write, snapshot, require_sources, require_teacher
from geometry import audit
from teacher import candidates


def launch(run, script):
    """父进程核对持久化结果，防止Isaac关闭掩盖worker异常退出。"""
    process = None
    try:
        with (run/'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable,'-u',str(HERE/script),'--run',str(run)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            code = process.wait()
        if code or (run/'worker_error.json').exists() or not (run/'result.json').exists():
            raise RuntimeError(f'Worker incomplete; inspect {run}/console.log and worker_error.json')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','teacher-search','teacher-replay','collect','pretrain','evaluate','train'))
    p.add_argument('--output',type=Path)
    p.add_argument('--search',type=Path)
    p.add_argument('--candidate-id',type=int)
    p.add_argument('--family',choices=('capture','return'),default='capture')
    p.add_argument('--layout',choices=('original','pair'),default='original')
    p.add_argument('--reference',type=Path)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--actor',type=Path)
    p.add_argument('--validation',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44,45])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--updates',type=int,default=200)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--epochs',type=int,default=200)
    return p


def main():
    p=parser();a=p.parse_args()
    if a.mode=='check':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(HERE/'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not a.seeds or len(set(a.seeds))!=len(a.seeds): p.error('seeds must be nonempty and unique')
    if not 2<=a.num_envs<=4096 or not 1<=a.updates<=200 or a.save_every<=0 or a.epochs<=0:
        p.error('Use 2..4096 envs, 1..200 updates and positive save interval/epochs')
    learning=a.mode in ('collect','pretrain','evaluate','train')
    rows=None; selected=None
    if a.mode=='teacher-search': rows=candidates(a.family)
    elif a.mode=='teacher-replay':
        if a.search is None or a.candidate_id is None: p.error('teacher-replay requires --search and --candidate-id')
        a.search=a.search.resolve(strict=True);require_sources(a.search)
        result=read(a.search/'result.json')
        if result['kind']!='teacher_search': p.error('Expected a teacher-search result')
        originals=read(a.search/'candidates.json')
        ids=[i for i,r in enumerate(originals) if r['id']==a.candidate_id]
        if len(ids)!=1: p.error('Candidate not in recorded search')
        if a.layout=='original': rows=originals;selected=ids
        else: rows=[originals[ids[0]],originals[ids[0]]];selected=[0,1]
        if a.reference:
            a.reference=a.reference.resolve(strict=True);require_sources(a.reference)
    else:
        from learning import preflight
        preflight(a)
    run=(a.output or HERE/'runs'/f'{now().replace(":","-")}_{a.mode}').resolve()
    run.mkdir(parents=True,exist_ok=False)
    request={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    request.update(selected=selected,headless=True)
    snapshot(run,learning=learning)
    write(run/'request.json',request)
    write(run/'geometry.json',audit())
    if rows is not None: write(run/'candidates.json',rows)
    write(run/'session.json',dict(started_at=now(),status='running',mode=a.mode))
    print(f'[R4A GUIDED] {a.mode}: {run}',flush=True)
    try:
        launch(run,'learning.py' if learning else 'physics_worker.py')
        require_sources(run,learning=learning)
        write(run/'session.json',dict(started_at=read(run/'session.json')['started_at'],finished_at=now(),status='completed',mode=a.mode))
        print(f'[R4A GUIDED COMPLETED] {run}',flush=True)
    except BaseException as exc:
        write(run/'error.json',dict(error=str(exc),at=now()))
        raise
    return 0


if __name__=='__main__': raise SystemExit(main())
