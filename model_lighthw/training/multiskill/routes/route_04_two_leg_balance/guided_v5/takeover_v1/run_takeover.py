"""显式教师接管诊断→成功后缀采集→第三轮纠偏；无自动晋级或PPO入口。"""
import argparse
from pathlib import Path
import subprocess
import sys
import torch
from common_takeover import (HERE, GUIDED, REPO, read, write, now, sha, snapshot, require,
    original_actor, teacher_evidence, correction_evidence, original_dataset, diagnosis_check, dataset_check, load_actor)
from geometry import audit


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('diagnose', 'collect', 'pretrain', 'evaluate', 'check'))
    p.add_argument('--output', type=Path)
    p.add_argument('--actor', type=Path)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--diagnosis', type=Path)
    p.add_argument('--switch-times', type=float, nargs='+', default=[3.6, 4., 4.4, 4.8])
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--seeds', type=int, nargs='+', default=[43, 44, 45])
    p.add_argument('--epochs', type=int, default=400)
    return p


def preflight(a):
    if not 1 <= a.epochs <= 400:
        raise ValueError('This correction budget is at most 400 epochs')
    if a.mode == 'diagnose':
        if not all((a.actor, a.teacher, a.baseline)):
            raise ValueError('diagnose needs --actor, --teacher and --baseline')
        if (not a.switch_times or len(a.switch_times) > 4 or len(set(a.switch_times)) != len(a.switch_times)
                or any(not 3. <= x <= 5. or abs(x / .02 - round(x / .02)) > 1e-5 for x in a.switch_times)):
            raise ValueError('At most four unique 20ms-aligned switches between 3 and 5 seconds')
        original_actor(a.actor)
        teacher_evidence(a.teacher)
        evidence = correction_evidence(a.baseline)
        if evidence['actor_sha256'] != sha(a.actor):
            raise ValueError('Baseline belongs to another actor')
        if not (a.baseline / f'trace_seed{a.seed}.npz').exists():
            raise ValueError('Requested seed has no autonomous prefix reference')
    elif a.mode == 'collect':
        if not a.dataset or not a.diagnosis:
            raise ValueError('collect needs --dataset parent and --diagnosis')
        original_dataset(a.dataset)
        if diagnosis_check(a.diagnosis)['full_successes'] == 0:
            raise ValueError('No successful takeover: cannot collect third correction')
    elif a.mode == 'pretrain':
        if not a.dataset:
            raise ValueError('pretrain needs --dataset')
        dataset_check(a.dataset)
    elif a.mode == 'evaluate':
        if not a.actor or not a.seeds or len(set(a.seeds)) != len(a.seeds):
            raise ValueError('evaluate needs --actor and unique seeds')
        load_actor(a.actor)


def main():
    torch.set_num_threads(4)
    p = parser()
    a = p.parse_args()
    if a.mode == 'check':
        import unittest
        tests = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
    preflight(a)
    run = (a.output or HERE / 'runs' / f'{now().replace(":", "-")}_{a.mode}').resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    request = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    write(run / 'request.json', request)
    write(run / 'geometry.json', audit())
    write(run / 'session.json', dict(started_at=now(), mode=a.mode, status='running'))
    print(f'[R4A TAKEOVER] {a.mode}: {run}', flush=True)
    process = None
    try:
        with (run / 'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(HERE / 'worker_takeover.py'), '--run', str(run)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code or (run / 'worker_error.json').exists() or not (run / 'result.json').exists():
            raise RuntimeError(f'Incomplete worker; inspect {run}/console.log')
        require(run)
        write(run / 'session.json', dict(started_at=read(run / 'session.json')['started_at'], finished_at=now(),
            mode=a.mode, status='completed'))
        print(f'[R4A TAKEOVER COMPLETED] {run}', flush=True)
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
