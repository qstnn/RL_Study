"""独立学生数据、有限BC、自主/明确接管验证；不启动PPO。"""
import argparse
from pathlib import Path
import subprocess
import sys
import torch
from common_student import HERE, snapshot, require, write, read, sha, now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--input',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--device',default='cpu');p.add_argument('--initial',type=Path)
    p=sub.add_parser('evaluate');p.add_argument('--actor',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hold-seconds',type=float,default=10.,choices=[10.,30.,60.,300.]);p.add_argument('--seeds',type=int,nargs='+',default=[43,44,45])
    p.add_argument('--switch',type=float,help='明确的教师接管诊断；指定后绝不算自主')
    p=sub.add_parser('worker');p.add_argument('run',type=Path)
    p=sub.add_parser('verify');p.add_argument('run',type=Path)
    a=parser.parse_args();torch.set_num_threads(4)
    if a.command=='prepare':
        from learning_student import prepare
        prepare(a.output.resolve(),[p.resolve() for p in a.input]);return
    if a.command=='train':
        assert 1<=a.steps<=12000
        from learning_student import train
        train(a.output.resolve(),a.dataset.resolve(),a.steps,a.device,a.initial.resolve() if a.initial else None);return
    if a.command=='worker':
        from worker_student import worker
        worker(a.run.resolve());return
    if a.command=='verify':
        from verify_student import verify
        print(verify(a.run.resolve()));return
    assert len(set(a.seeds))==len(a.seeds) and a.seeds
    if a.switch is not None:
        assert 1. <= a.switch <= 6.
    from common_student import load_actor
    actor=a.actor.resolve();load_actor(actor)
    run=a.output.resolve();run.mkdir(parents=True,exist_ok=False);snapshot(run)
    write(run/'request.json',dict(actor=str(actor),actor_sha256=sha(actor),hold_seconds=a.hold_seconds,
        seeds=a.seeds,switch_s=a.switch,return_after_s=None,autonomous=a.switch is None))
    started=now();write(run/'session.json',dict(status='running',started_at=started))
    process=None
    try:
        with (run/'console.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',str(run)],stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        if code or not (run/'verification.json').exists():
            raise RuntimeError(f'Worker failed ({code}); inspect {run}/console.log')
        require(run)
        write(run/'session.json',dict(status='completed',started_at=started,finished_at=now()))
    except BaseException as exc:
        write(run/'session.json',dict(status='failed',started_at=started,finished_at=now(),error=str(exc)))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill();process.wait()


if __name__=='__main__':
    main()
