"""独立持续站立诊断：有限混合对照或已明确选定候选的长时复核，不启动训练。"""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback
import torch
from common_sustain import HERE, ACTOR, ACTOR_SHA, read, write, sha, now, snapshot, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'runs/2026-10-22_sustain_blend_6')
    p.add_argument('--hold-seconds', type=float, default=10., choices=[10., 30., 60., 300.])
    p.add_argument('--blend', type=float, choices=[.5, .65, .8])
    p.add_argument('--return-after', type=float, help='显式测试命令在捕获后指定秒数发出；不改变环境自动行为')
    p.add_argument('--worker', type=Path)
    p.add_argument('--verify', type=Path)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.check:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if a.verify:
        from verify_sustain import verify
        print(verify(a.verify.resolve()))
        return
    if a.worker:
        from worker_sustain import worker
        worker(a.worker.resolve())
        return
    if a.hold_seconds > 10. and a.blend is None:
        p.error('Long observation requires one explicitly selected blend')
    if a.return_after is not None and (a.blend is None or not 2. <= a.return_after <= a.hold_seconds):
        p.error('Return check requires one blend and 2 <= return-after <= hold-seconds')
    run = a.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    write(run / 'request.json', dict(actor=str(ACTOR), actor_sha256=ACTOR_SHA,
        blends=[a.blend] if a.blend is not None else [.5, .65, .8], hold_seconds=a.hold_seconds,
        switch_s=3.2, seed=43, return_after_s=a.return_after, autonomous=False, ppo_updates=0))
    started = now()
    write(run / 'session.json', dict(status='running', started_at=started))
    print('[R4 SUSTAIN DIAGNOSTIC]', run, flush=True)
    process = None
    try:
        with (run / 'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(run)], stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code or not (run / 'result.json').exists():
            raise RuntimeError(f'Worker failed ({code}); inspect {run}/console.log')
        require(run)
        write(run / 'session.json', dict(status='completed', started_at=started, finished_at=now()))
    except BaseException as exc:
        write(run / 'session.json', dict(status='failed', started_at=started, finished_at=now(), error=str(exc)))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
