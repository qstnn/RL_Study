"""显式v6恢复对照、采集、纠偏和自主审查；没有自动评估或PPO入口。"""
from pathlib import Path
import argparse
import subprocess
import sys
import torch
from common_smooth import (HERE, REPO, LEGACY_ACTOR, LEGACY_BASELINE, TEACHER,
    read, write, sha, now, require, snapshot, input_actor, teacher_evidence)
from learning_smooth import evidence_check, dataset_check
from geometry import audit


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('replay', 'evaluate', 'collect', 'pretrain', 'check'))
    p.add_argument('--output', type=Path)
    p.add_argument('--actor', type=Path, default=LEGACY_ACTOR)
    p.add_argument('--baseline', type=Path, default=LEGACY_BASELINE)
    p.add_argument('--diagnosis', type=Path)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--switch-times', nargs='+', type=float, default=[3.6, 4.])
    p.add_argument('--seeds', nargs='+', type=int, default=[43, 44, 45])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    return p


def preflight(a):
    if not 1 <= a.epochs <= 200:
        raise ValueError('One bounded correction: at most 200 epochs')
    if not a.seeds or len(a.seeds) > 3 or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('One to three unique seeds')
    if a.mode in ('replay', 'evaluate'):
        input_actor(a.actor)
    if a.mode == 'replay':
        if a.actor.resolve() != LEGACY_ACTOR.resolve() or a.baseline.resolve() != LEGACY_BASELINE.resolve():
            raise ValueError('This probe explicitly transfers the third Actor and its recorded baseline')
        if (not a.switch_times or len(a.switch_times) > 4 or len(set(a.switch_times)) != len(a.switch_times)
                or any(not 3. <= x <= 5. or abs(x / .02 - round(x / .02)) > 1e-5 for x in a.switch_times)):
            raise ValueError('At most four unique control-aligned switches in [3, 5] seconds')
        if 2 * len(a.seeds) * len(a.switch_times) > 12:
            raise ValueError('At most 12 teacher cases per explicit invocation')
        teacher_evidence(TEACHER)
        result = read(a.baseline / 'result.json')
        assert result['actor_sha256'] == sha(a.actor) and result['autonomous']
        for seed in a.seeds:
            name = f'trace_seed{seed}.npz'
            assert sha(a.baseline / name) == result['trace_hashes'][name]
    if a.mode == 'collect':
        if not a.diagnosis or not evidence_check(a.diagnosis)['full_successes']:
            raise ValueError('Collection needs actual full-protocol success')
    if a.mode == 'pretrain':
        if not a.dataset:
            raise ValueError('Pretrain requires an explicit verified dataset')
        dataset_check(a.dataset)


def main():
    torch.set_num_threads(4)
    a = parser().parse_args()
    if a.mode == 'check':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    preflight(a)
    run = (a.output or HERE / 'runs' / f'{now().replace(":", "-")}_{a.mode}').resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    request = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if a.mode == 'replay':
        request['baseline_result_sha256'] = sha(a.baseline / 'result.json')
    write(run / 'request.json', request)
    write(run / 'geometry.json', audit())
    started = now()
    write(run / 'session.json', dict(status='running', started_at=started, mode=a.mode))
    print('[R4A SMOOTH]', a.mode, run, flush=True)
    process = None
    try:
        with (run / 'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(HERE / 'worker_smooth.py'), '--run', str(run)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code or (run / 'worker_error.json').exists() or not (run / 'result.json').exists():
            raise RuntimeError(f'Worker did not complete: {run}/console.log')
        require(run)
        write(run / 'session.json', dict(status='completed', started_at=started, finished_at=now(), mode=a.mode))
        print('[R4A SMOOTH COMPLETED]', run, flush=True)
    except BaseException as exc:
        write(run / 'error.json', dict(error=str(exc), at=now()))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
