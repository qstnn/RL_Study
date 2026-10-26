"""R4b vertical front standing: verified teacher -> IL -> autonomous review -> PPO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from front_vertical_il_v1.common import HERE, read, write, now, sha, snapshot, require


def subprocess_stage(run, *args):
    command = [sys.executable, '-u', str(HERE/'run.py'), *map(str, args)]
    log = run/('console_'+str(args[0])+'.log')
    print('[R4b stage]', ' '.join(map(str, args)), flush=True)
    with log.open('a') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); stream.write(line); stream.flush()
            if process.wait(): raise RuntimeError(f'Stage failed; inspect {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check','smoke','prepare-il','evaluate','ppo-smoke','train-ppo','_worker','_ppo'])
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--stage', default='smoke')
    p.add_argument('--hold-seconds', type=int, choices=[10,60,300], default=10)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--bc-steps', type=int, default=3000)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--resume', type=Path)
    p.add_argument('--time-limit', type=float, default=360.)
    a = p.parse_args()
    if min(a.bc_steps, a.updates) <= 0 or (a.max_steps is not None and a.max_steps <= 0): p.error('budgets must be positive')
    import torch
    torch.set_num_threads(4)
    if a.command == 'check':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE/'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): raise SystemExit(1)
        from front_vertical_il_v1.settings import contract
        print(json.dumps(contract(), ensure_ascii=False, indent=2)); return
    if a.command == '_worker':
        from front_vertical_il_v1.worker import worker
        worker(a.run, a.stage, a.hold_seconds, a.seed, a.max_steps); return
    if a.command == '_ppo':
        from front_vertical_il_v1.train_ppo import worker
        worker(a.run, a.output, a.updates, a.resume, a.time_limit); return
    if a.command in ('prepare-il', 'smoke'):
        run = (a.output or HERE/'runs'/(now().replace(':','-')+'_'+a.command)).resolve()
        run.mkdir(parents=True, exist_ok=False); snapshot(run)
        write(run/'geometry.json', read(HERE/'geometry.json'))
        write(run/'request.json', dict(command=a.command, bc_steps=a.bc_steps, target_pitch_deg=90.,
            observation_noise=False, randomization=False, formal_ppo=False))
    else:
        if a.run is None: p.error('--run is required for a completed R4b IL directory')
        run = a.run.resolve(); require(run)
    print(f'[R4b RUN] {run}', flush=True)
    session = run/(a.command+'_session.json')
    write(session, dict(status='running', started_at=now()))
    try:
        if a.command == 'smoke':
            subprocess_stage(run, '_worker', '--run', run, '--stage', 'smoke', '--max-steps', 2)
        elif a.command == 'prepare-il':
            from front_vertical_il_v1.verify import verify
            from front_vertical_il_v1.teacher import candidates
            subprocess_stage(run, '_worker', '--run', run, '--stage', 'search')
            report = verify(run, 'search')
            good = [r for r in candidates() if sum(c['observation_pass'] for c in report['cases'] if c['candidate']==r['id']) == 2]
            if not good:
                write(run/'result.json', dict(status='needs_teacher_correction', teacher_passes=report['passes'],
                    bc_updates=0, ppo_updates=0, ppo_admission=False))
                print('[R4b STOP] No two-slot 90-degree teacher success. Diagnostics saved; no IL/PPO started.', flush=True)
            else:
                # Select among genuine passes, preferring lower observed residual speed.
                selected = min(good, key=lambda r: max(c['max_stable_speed_m_s'] for c in report['cases'] if c['candidate']==r['id']))
                write(run/'selected_teacher.json', selected)
                subprocess_stage(run, '_worker', '--run', run, '--stage', 'collect', '--hold-seconds', 60)
                verified = verify(run, 'collect')
                if verified['passes'] != 2:
                    write(run/'result.json', dict(status='teacher_60s_failed', bc_updates=0, ppo_updates=0))
                    print('[R4b STOP] Teacher 60s failed; no IL/PPO started.', flush=True)
                else:
                    from front_vertical_il_v1.learning import fit
                    fit(run, a.bc_steps)
                    write(run/'result.json', dict(status='il_completed_awaiting_user_review', bc_updates=a.bc_steps,
                        ppo_updates=0, ppo_admission=False, checkpoint=str(run/'student.pt')))
        elif a.command == 'evaluate':
            from front_vertical_il_v1.verify import verify
            evidence = []
            for hold, seed in [(10,43),(10,44),(10,45),(60,43)]:
                stage = f'student_{hold}s_seed{seed}'
                subprocess_stage(run, '_worker', '--run', run, '--stage', stage, '--hold-seconds', hold, '--seed', seed)
                v = verify(run, stage)
                path = run/stage/'verification.json'
                evidence.append(dict(path=str(path), sha256=sha(path)))
                if v['passes'] != 2: break
            passed = len(evidence)==4 and all(read(v['path'])['passes']==2 for v in evidence)
            write(run/'admission.json', dict(autonomous_admission=passed, actor_sha256=sha(run/'student.pt'),
                evidence=evidence, ppo_runtime_verified=False, formal_ppo_admission=False))
        elif a.command in ('ppo-smoke', 'train-ppo'):
            from front_vertical_il_v1.learning import require_admission
            require_admission(run)
            if a.command == 'ppo-smoke':
                first, resumed = run/'ppo_smoke_initial', run/'ppo_smoke_resumed'
                subprocess_stage(run, '_ppo', '--run', run, '--output', first, '--updates',32,'--time-limit',20)
                initial = read(first/'result.json')
                subprocess_stage(run, '_ppo', '--run', run, '--output', resumed, '--updates',1,
                                 '--time-limit',20,'--resume',initial['checkpoint'])
                final = read(resumed/'result.json')
                passed = (initial['completed_updates']==32 and final['completed_updates']==33
                    and final['optimizer_steps']==[66] and initial['bootstrap_events']>=16
                    and final['physical_terminations']==0 and final['driver_reset_calls']==0)
                write(run/'ppo_smoke.json', dict(ppo_runtime_verified=passed, checkpoint=final['checkpoint'],
                    checkpoint_sha256=final['checkpoint_sha256'], actor_sha256=sha(run/'student.pt'),
                    initial_result_sha256=sha(first/'result.json'), resumed_result_sha256=sha(resumed/'result.json')))
                if not passed: raise ValueError('PPO smoke did not pass; formal training remains closed')
            else:
                smoke = read(run/'ppo_smoke.json')
                if not smoke['ppo_runtime_verified'] or smoke['actor_sha256'] != sha(run/'student.pt'):
                    raise ValueError('Matching GPU PPO update/save/new-process-resume proof required')
                if sha(smoke['checkpoint']) != smoke['checkpoint_sha256']: raise ValueError('PPO source changed')
                out = (a.output or run/('ppo_'+now().replace(':','-'))).resolve()
                subprocess_stage(run, '_ppo', '--run', run, '--output', out, '--updates', a.updates,
                                 '--resume', smoke['checkpoint'])
        write(session, dict(status='completed', finished_at=now()))
    except BaseException as exc:
        write(session, dict(status='failed_or_interrupted', finished_at=now(), error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
