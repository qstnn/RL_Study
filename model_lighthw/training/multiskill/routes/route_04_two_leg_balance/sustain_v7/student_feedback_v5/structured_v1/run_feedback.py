"""有限结构化BC与教师全关测试；正式PPO仍需独立准入。"""
from pathlib import Path
import argparse
import subprocess
import sys
import torch
from common_feedback import read,write,now,sha,snapshot,require,load_actor


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('fit');q.add_argument('--variant',choices=['full','no_velocity'],default='full');q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('evaluate');q.add_argument('--actor',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--seeds',type=int,nargs='+',default=[43]);q.add_argument('--hold-seconds',type=float,choices=[10.,30.,60.,300.],default=10.)
    for name in ('worker','verify'):
        q=s.add_parser(name);q.add_argument('run',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    if a.command=='fit':
        from fit_feedback import fit
        fit(a.output.resolve(),a.variant);return
    if a.command=='worker':
        from worker_feedback import worker
        worker(a.run.resolve());return
    if a.command=='verify':
        from verify_feedback import verify
        print(verify(a.run.resolve()));return
    assert a.seeds and len(set(a.seeds))==len(a.seeds)
    actor=a.actor.resolve();_,value=load_actor(actor);run=a.output.resolve();run.mkdir(parents=True,exist_ok=False);snapshot(run)
    write(run/'request.json',dict(actor=str(actor),actor_sha256=sha(actor),variant=value['variant'],hold_seconds=a.hold_seconds,seeds=a.seeds,
        programs=[dict(id='autonomous',windows=None)],return_after_s=None,autonomous=True))
    started=now();write(run/'session.json',dict(status='running',started_at=started));process=None
    try:
        with (run/'console.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',str(run)],stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        if code or not (run/'verification.json').exists():raise RuntimeError(f'Worker failed ({code}); see {run}/worker_error.json')
        require(run);write(run/'session.json',dict(status='completed',started_at=started,finished_at=now()))
    except BaseException as exc:
        write(run/'session.json',dict(status='failed',started_at=started,finished_at=now(),error=str(exc)));raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
