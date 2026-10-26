"""捕获记忆单因素：同数据有限BC、纯学生回放与准入检查。"""
from pathlib import Path
import argparse
import subprocess
import sys
import torch
from common_memory import HERE,read,write,now,sha,snapshot,require,load_actor


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('check');s.add_parser('prepare')
    q=s.add_parser('train');q.add_argument('--mode',choices=['zero','exposed'],required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--device',default='cpu')
    q=s.add_parser('evaluate');q.add_argument('--actor',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--seeds',type=int,nargs='+',default=[43,44,45]);q.add_argument('--hold-seconds',type=float,choices=[10.,30.,60.,300.],default=10.)
    for name in ['worker','verify']:
        q=s.add_parser(name);q.add_argument('run',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    if a.command=='check':
        import unittest
        tests=unittest.defaultTestLoader.discover(str(HERE/'tests'))
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1)
    if a.command=='prepare':
        from data_memory import prepare
        prepare();return
    if a.command=='train':
        from learning_memory import train
        train(a.output.resolve(),a.mode,a.device);return
    if a.command=='worker':
        from worker_memory import worker
        worker(a.run.resolve());return
    if a.command=='verify':
        from verify_memory import verify
        print(verify(a.run.resolve()));return
    assert a.seeds and len(set(a.seeds))==len(a.seeds)
    actor=a.actor.resolve();_,value=load_actor(actor)
    run=a.output.resolve();run.mkdir(parents=True,exist_ok=False);snapshot(run)
    write(run/'request.json',dict(actor=str(actor),actor_sha256=sha(actor),memory_mode=value['memory_mode'],
        hold_seconds=a.hold_seconds,seeds=a.seeds,programs=[dict(id='autonomous',windows=None)],return_after_s=None,autonomous=True))
    started=now();write(run/'session.json',dict(status='running',started_at=started));process=None
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
