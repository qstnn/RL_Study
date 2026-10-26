"""第四轮Actor实际状态接管诊断；固定strong教师和六例预算，不训练。"""
from pathlib import Path
import argparse
import subprocess
import sys
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SMOOTH = HERE.parents[1]
sys.path.insert(0, str(SMOOTH))
from common_smooth import (GUIDED, REPO, read, write, sha, now, require, snapshot,
    load_actor, actor_network, arrays, successful, prefix_check, teacher_evidence, feedback_candidate)
from verify_smooth import verify
from worker_smooth import setup
from worker_takeover import rollout
from learning import corrected_trace_arrays
from geometry import audit as geometry_audit

ACTOR = SMOOTH / 'runs/2026-10-22_bc_corrected_04/actor.pt'
ACTOR_SHA = 'e986f82bdd9c4832084e4a9a000f2f05274bf914c7e869a4bc94dc8b6af88f65'
BASELINE = SMOOTH / 'runs/2026-10-22_bc_corrected_04_autonomous'
TEACHER = GUIDED / 'runs/2026-10-22_teacher13_pair_final'


def probe_sources():
    return {str((HERE / name).relative_to(REPO)): sha(HERE / name)
            for name in ('probe.py', 'probe.sh')}


def baseline_check():
    require(BASELINE)
    load_actor(ACTOR)
    result = read(BASELINE / 'result.json')
    assert result['kind'] == 'smooth_autonomous_evaluation' and result['autonomous']
    assert result['actor_sha256'] == sha(ACTOR) == ACTOR_SHA
    assert result['cases'] == read(BASELINE / 'evaluation.json')['cases']
    verify(BASELINE)
    for name, digest in result['trace_hashes'].items():
        assert sha(BASELINE / name) == digest
    return result


def validate_switches(switches):
    if (not switches or len(switches) > 3 or len(set(switches)) != len(switches)
            or any(x not in (2.8, 3.2, 3.6) for x in switches)):
        raise ValueError('Only a nonempty subset of the three planned switches: 2.8, 3.2, 3.6 seconds')


def check_geometry(reference, current):
    """复核同源静力几何的浮点舍入；执行时仍逐字复用基线，避免引入目标变化。"""
    if isinstance(reference, dict):
        assert isinstance(current, dict) and set(reference) == set(current)
        for key in reference:
            check_geometry(reference[key], current[key])
    elif isinstance(reference, list):
        assert isinstance(current, list) and len(reference) == len(current)
        for a, b in zip(reference, current):
            check_geometry(a, b)
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        assert np.isfinite(reference) and np.isfinite(current)
        np.testing.assert_allclose(reference, current, rtol=1e-12, atol=1e-10)
    else:
        assert reference == current


def probe_contract():
    return dict(version='r4a_fourth_actor_takeover_diagnosis_v1', actor_sha256=ACTOR_SHA,
                switches=[2.8, 3.2, 3.6], max_cases=6, seed=43, slots=2, feedback='strong',
                physics_reset_at_takeover=False, teacher_shadow_from_reset=True,
                training_enabled=False, ppo_enabled=False, automatic_collection=False)


def validate_request(request):
    validate_switches(request['switch_times'])
    if (Path(request['actor']) != ACTOR or Path(request['baseline']) != BASELINE
            or Path(request['teacher']) != TEACHER or request['seed'] != 43
            or request['feedback'] != 'strong' or request['actor_sha256'] != ACTOR_SHA
            or request['ppo_updates'] != 0):
        raise ValueError('Only the explicitly pinned fourth Actor, baseline and strong teacher are allowed')
    assert sha(ACTOR) == ACTOR_SHA
    assert request['baseline_result_sha256'] == sha(BASELINE / 'result.json')
    assert request['teacher_result_sha256'] == sha(TEACHER / 'result.json')


def full_prefix_check(data, baseline, switch):
    """全量前缀含实际关节与高频滤波状态；另核对切换瞬间的Actor观测和输出。"""
    original = prefix_check(data, baseline, switch)
    steps = round(switch / .02)
    keys = sorted(set(data) & set(baseline) - {'policy_teacher_action'})
    for key in keys:
        n = steps if key.startswith('policy_') else steps * 20
        np.testing.assert_array_equal(data[key][:n], baseline[key][:n], err_msg=f'Prefix {key}')
    for key in ('policy_observation', 'policy_actor_action'):
        np.testing.assert_array_equal(data[key][steps], baseline[key][steps], err_msg=f'Switch state {key}')
    assert data['active'][:steps * 20].all()
    return dict(original_max_error=max(original.values()), exact_array_count=len(keys),
                exact_switch_observation=True, exact_switch_actor_output=True)


def check_probe_sources(run):
    require(run)
    assert read(run / 'probe_source_manifest.json') == probe_sources()
    assert all(sha(run / 'source_snapshot' / name) == digest for name, digest in probe_sources().items())
    assert read(run / 'probe_contract.json') == probe_contract()
    assert read(run / 'geometry.json') == read(BASELINE / 'geometry.json')


def verify_probe(run):
    check_probe_sources(run)
    baseline_check()
    request, result = read(run / 'request.json'), read(run / 'result.json')
    validate_request(request)
    assert request['actor_sha256'] == result['actor_sha256'] == sha(ACTOR)
    assert request['baseline_result_sha256'] == sha(BASELINE / 'result.json')
    assert result['teacher_result_sha256'] == sha(TEACHER / 'result.json')
    assert result['kind'] == 'fourth_actor_takeover_probe' and not result['autonomous']
    assert result['cases'] == read(run / 'evaluation.json')['cases']
    candidate = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
    assert result['candidate'] == candidate
    assert len(result['cases']) == 2 * len(request['switch_times'])
    assert {c['switch_s'] for c in result['cases']} == set(request['switch_times'])
    assert all(c['seed'] == 43 for c in result['cases'])
    assert {c['trace'] for c in result['cases']} == set(result['trace_hashes'])
    assert len({(c['switch_s'], c['trace_env']) for c in result['cases']}) == len(result['cases'])
    model = actor_network().eval()
    model.load_state_dict(load_actor(ACTOR)['actor_state_dict'])
    baseline = arrays(BASELINE / 'trace_seed43.npz')
    checks = []
    for name, digest in result['trace_hashes'].items():
        assert sha(run / name) == digest
        cases = [c for c in result['cases'] if c['trace'] == name]
        assert {c['trace_env'] for c in cases} == {0, 1}
        switch = cases[0]['switch_s']
        d = arrays(run / name)
        prefix = full_prefix_check(d, baseline, switch)
        active = d['policy_active'].astype(bool)
        np.testing.assert_array_equal(active, d['active'][::20])
        with torch.no_grad():
            output = model(torch.from_numpy(d['policy_observation']).float()).numpy()
        actor_error = float(abs(output[active] - d['policy_actor_action'][active]).max())
        assert actor_error < 2e-6
        labels = corrected_trace_arrays(d, candidate, read(run / 'geometry.json'))
        teacher_error = float(abs(labels['actions'] - d['policy_teacher_action'][active]).max())
        assert teacher_error < 2e-5
        np.testing.assert_array_equal(d['policy_takeover'][active], (d['policy_time'] >= switch - 1e-6)[active])
        expected = np.where(d['policy_takeover'][..., None], d['policy_teacher_action'], d['policy_actor_action'])
        np.testing.assert_array_equal(expected[active], d['policy_action'][active])
        expected = expected.clip(-1, 1)
        expected[:10] = 0
        np.testing.assert_array_equal(expected[active], d['actions'][::20][active])
        checks.append(dict(trace=name, prefix=prefix, actor_max_error=actor_error,
                           teacher_max_error=teacher_error, executed_action_error=0.))
    return dict(verification=verify(run), traces=checks, full_successes=sum(successful(c) for c in result['cases']),
                teacher_success_is_not_ppo_admission=True, autonomous_ppo_admission=False,
                training_runs=0, collected_frames=0, ppo_updates=0)


def worker(run):
    app = env = None
    try:
        check_probe_sources(run)
        request = read(run / 'request.json')
        validate_request(request)
        candidate = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
        saved = load_actor(ACTOR)
        app, env = setup(run)
        model = actor_network().to(env.device).eval()
        model.load_state_dict(saved['actor_state_dict'])
        cases = []
        for switch in request['switch_times']:
            cases.extend(rollout(env, app, run, model, 43, switch, candidate, BASELINE))
        report = dict(cases=cases, ppo_updates=0, autonomous=False, behavior_accepted=False,
                      full_successes=sum(successful(c) for c in cases))
        write(run / 'evaluation.json', report)
        report.update(kind='fourth_actor_takeover_probe', actor_sha256=sha(ACTOR), candidate=candidate,
                      teacher_result_sha256=sha(TEACHER / 'result.json'),
                      trace_hashes={p.name: sha(p) for p in run.glob('trace_seed*.npz')})
        write(run / 'result.json', report)
        write(run / 'probe_verification.json', verify_probe(run))
    except BaseException as exc:
        write(run / 'worker_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


def main():
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', type=Path)
    p.add_argument('--verify', type=Path)
    p.add_argument('--check', action='store_true', help='CPU budget, request and prefix isolation checks')
    p.add_argument('--baseline-check', action='store_true', help='Independent baseline symptom check; failed behavior exits 1')
    p.add_argument('--switch-times', type=float, nargs='+', default=[2.8, 3.2, 3.6])
    a = p.parse_args()
    if a.check:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if a.baseline_check:
        verify(BASELINE)
        cases = read(BASELINE / 'result.json')['cases']
        accepted = all(successful(c) for c in cases)
        print('GREEN' if accepted else 'RED', 'autonomous full success', sum(successful(c) for c in cases), '/', len(cases))
        raise SystemExit(0 if accepted else 1)
    if a.verify:
        print(verify_probe(a.verify.resolve()))
        return
    if a.worker:
        worker(a.worker.resolve())
        return
    validate_switches(a.switch_times)
    baseline_check()
    teacher_evidence(TEACHER)
    run = (a.output or HERE / 'runs/2026-10-22_fourth_actor_takeover_6_verified').resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    import shutil
    for name in probe_sources():
        target = run / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    write(run / 'probe_source_manifest.json', probe_sources())
    write(run / 'probe_contract.json', probe_contract())
    actual_geometry = geometry_audit()
    baseline_geometry = read(BASELINE / 'geometry.json')
    check_geometry(baseline_geometry, actual_geometry)
    write(run / 'geometry_recheck.json', actual_geometry)
    shutil.copy2(BASELINE / 'geometry.json', run / 'geometry.json')
    write(run / 'request.json', dict(actor=str(ACTOR), actor_sha256=sha(ACTOR), baseline=str(BASELINE),
        baseline_result_sha256=sha(BASELINE / 'result.json'), teacher=str(TEACHER),
        teacher_result_sha256=sha(TEACHER / 'result.json'),
        feedback='strong', switch_times=a.switch_times, seed=43, ppo_updates=0))
    started = now()
    write(run / 'session.json', dict(status='running', started_at=started))
    print('[R4A FOURTH TAKEOVER]', run, a.switch_times, flush=True)
    process = None
    try:
        with (run / 'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(run)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code or (run / 'worker_error.json').exists() or not (run / 'probe_verification.json').exists():
            raise RuntimeError(f'Probe did not finish: inspect {run}/console.log')
        check_probe_sources(run)
        write(run / 'session.json', dict(status='completed', started_at=started, finished_at=now()))
        print(read(run / 'probe_verification.json'), flush=True)
    except BaseException as exc:
        write(run / 'error.json', dict(error=str(exc), at=now()))
        write(run / 'session.json', dict(status='failed', started_at=started, finished_at=now()))
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
