"""短段接管、实际教师段数据、有限BC与纯学生验证；不启动正式PPO。"""
import argparse
from pathlib import Path
import subprocess
import sys
import torch
from common_segments import HERE,ACTOR,load_actor,snapshot,require,read,write,sha,now
from programs import PROGRAMS,validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    p=sub.add_parser('prepare');p.add_argument('--collect',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--device',default='cpu');p.add_argument('--initial',type=Path,default=ACTOR)
    for command in ['collect','evaluate']:
        p=sub.add_parser(command);p.add_argument('--actor',type=Path,default=ACTOR);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--hold-seconds',type=float,default=10.,choices=[10.,30.,60.,300.])
        p.add_argument('--seeds',type=int,nargs='+',default=[43] if command=='collect' else [43,44,45])
        if command=='collect':p.add_argument('--program',choices=list(PROGRAMS),nargs='+',default=['early','late'])
    p=sub.add_parser('worker');p.add_argument('run',type=Path)
    p=sub.add_parser('verify');p.add_argument('run',type=Path)
    a=parser.parse_args();torch.set_num_threads(4)
    if a.command=='check':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(HERE/'tests'))
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if a.command=='prepare':
        from learning_segments import prepare
        prepare(a.output.resolve(),[p.resolve() for p in a.collect]);return
    if a.command=='train':
        assert 1<=a.steps<=12000
        from learning_segments import train
        train(a.output.resolve(),a.dataset.resolve(),a.steps,a.device,a.initial.resolve());return
    if a.command=='worker':
        from worker_segments import worker
        worker(a.run.resolve());return
    if a.command=='verify':
        from verify_segments import verify
        print(verify(a.run.resolve()));return
    assert a.seeds and len(set(a.seeds))==len(a.seeds)
    if a.command=='collect':
        assert len(set(a.program))==len(a.program) and a.hold_seconds==10.
        for name in a.program:validate(PROGRAMS[name])
        programs=[dict(id=name,windows=PROGRAMS[name]) for name in a.program]
    else:programs=[dict(id='autonomous',windows=None)]
    actor=a.actor.resolve();load_actor(actor)
    run=a.output.resolve();run.mkdir(parents=True,exist_ok=False);snapshot(run)
    write(run/'request.json',dict(actor=str(actor),actor_sha256=sha(actor),hold_seconds=a.hold_seconds,
        seeds=a.seeds,programs=programs,return_after_s=None,autonomous=a.command=='evaluate'))
    started=now();write(run/'session.json',dict(status='running',started_at=started))
    process=None
    try:
        with (run/'console.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',str(run)],stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        if code or not (run/'verification.json').exists():raise RuntimeError(f'Worker failed ({code}); inspect {run}/console.log')
        require(run);write(run/'session.json',dict(status='completed',started_at=started,finished_at=now()))
    except BaseException as exc:
        write(run/'session.json',dict(status='failed',started_at=started,finished_at=now(),error=str(exc)));raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
