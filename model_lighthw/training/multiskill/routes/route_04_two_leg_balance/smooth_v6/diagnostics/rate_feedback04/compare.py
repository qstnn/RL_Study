"""第四轮Actor固定3.2秒接管，仅比较俯仰角速度反馈；六例上限，不训练。"""
from pathlib import Path
import argparse
import importlib.util
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'takeover04'
spec = importlib.util.spec_from_file_location('r4_fourth_takeover_frozen', OLD / 'probe.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
from common_smooth import (REPO, read, write, sha, now, require, snapshot, load_actor,
    actor_network, arrays, successful, teacher_evidence, feedback_candidate)
from verify_smooth import verify
from worker_smooth import setup
from worker_takeover import rollout
from learning import corrected_trace_arrays
from geometry import audit as geometry_audit

ACTOR, ACTOR_SHA, BASELINE, TEACHER = prior.ACTOR, prior.ACTOR_SHA, prior.BASELINE, prior.TEACHER
CONTROL = OLD / 'runs/2026-10-22_fourth_actor_takeover_6_verified'
CONTROL_TRACE = 'trace_seed43_switch3200.npz'
GAINS = [.20, .25, .30]
SWITCH = 3.2
DEFAULT_RUN = HERE / 'runs/2026-10-22_rate_feedback_6'


def comparison_sources():
    return {**prior.probe_sources(), **{str((HERE / n).relative_to(REPO)): sha(HERE / n)
            for n in ('compare.py', 'compare.sh')}}


def comparison_contract():
    return dict(version='r4a_fourth_actor_pitch_rate_comparison_v1', actor_sha256=ACTOR_SHA,
                switch_s=SWITCH, gains=GAINS, max_cases=6, seed=43, slots=2,
                changed_candidate_fields=['pitch_rate_gain'], base_feedback='strong',
                physics_reset_at_takeover=False, teacher_shadow_from_reset=True,
                training_enabled=False, ppo_enabled=False, automatic_collection=False)


def validate_gains(gains):
    # 不允许临时扩展搜索范围；完整三组对照以固定次序运行。
    if gains != GAINS:
        raise ValueError('Exactly the planned pitch-rate gains [0.20, 0.25, 0.30] are required')


def candidate_for_gain(base, gain):
    if gain not in GAINS:
        raise ValueError('Pitch-rate gain outside the three-case budget')
    assert base['pitch_rate_gain'] == .20 and base['velocity_gain'] == 2. and base['drift_gain'] == 0.
    return {**base, 'pitch_rate_gain': gain}


def request_payload():
    return dict(actor=str(ACTOR), actor_sha256=sha(ACTOR), baseline=str(BASELINE),
        baseline_result_sha256=sha(BASELINE / 'result.json'), teacher=str(TEACHER),
        teacher_result_sha256=sha(TEACHER / 'result.json'), control=str(CONTROL),
        control_result_sha256=sha(CONTROL / 'result.json'), control_trace_sha256=sha(CONTROL / CONTROL_TRACE),
        base_feedback='strong', switch_s=SWITCH, gains=GAINS, seed=43, ppo_updates=0)


def validate_request(request):
    validate_gains(request['gains'])
    if request != request_payload() or request['actor_sha256'] != ACTOR_SHA:
        raise ValueError('Only pinned fourth Actor, control trace, fixed 3.2 s and three gains are allowed')


def control_check():
    prior.check_probe_sources(CONTROL)
    assert read(CONTROL / 'session.json')['status'] == 'completed'
    result = read(CONTROL / 'result.json')
    assert result['actor_sha256'] == ACTOR_SHA and result['kind'] == 'fourth_actor_takeover_probe'
    assert result['candidate'] == feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
    assert result['trace_hashes'][CONTROL_TRACE] == sha(CONTROL / CONTROL_TRACE)


def check_comparison_sources(run):
    require(run)
    manifest = comparison_sources()
    assert read(run / 'comparison_source_manifest.json') == manifest
    assert all(sha(run / 'source_snapshot' / n) == h for n, h in manifest.items())
    assert read(run / 'comparison_contract.json') == comparison_contract()
    assert read(run / 'geometry.json') == read(BASELINE / 'geometry.json')


def exact_control_check(data, control):
    assert set(data) == set(control)
    for key in sorted(data):
        np.testing.assert_array_equal(data[key], control[key], err_msg=f'0.20 replay: {key}')
    return dict(exact_array_count=len(data), maximum_error=0.)


def verify_comparison(run):
    check_comparison_sources(run)
    prior.baseline_check()
    control_check()
    request, result = read(run / 'request.json'), read(run / 'result.json')
    validate_request(request)
    assert result['kind'] == 'fourth_actor_pitch_rate_comparison' and not result['autonomous']
    assert result['actor_sha256'] == ACTOR_SHA and result['ppo_updates'] == 0
    assert result['cases'] == read(run / 'evaluation.json')['cases']
    base = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
    assert result['base_candidate'] == base
    assert result['teacher_result_sha256'] == request['teacher_result_sha256']
    cases = result['cases']
    assert len(cases) == 6 and len(result['trace_hashes']) == 3
    assert {(c['pitch_rate_gain'], c['trace_env']) for c in cases} == {(g, s) for g in GAINS for s in (0, 1)}
    assert all(c['seed'] == 43 and c['switch_s'] == SWITCH for c in cases)
    assert {c['trace'] for c in cases} == set(result['trace_hashes'])
    model = actor_network().eval()
    model.load_state_dict(load_actor(ACTOR)['actor_state_dict'])
    baseline = arrays(BASELINE / 'trace_seed43.npz')
    checks = []
    for name, digest in result['trace_hashes'].items():
        assert sha(run / name) == digest
        group = [c for c in cases if c['trace'] == name]
        assert {c['trace_env'] for c in group} == {0, 1}
        gain = group[0]['pitch_rate_gain']
        assert all(c['pitch_rate_gain'] == gain for c in group)
        candidate = candidate_for_gain(base, gain)
        d = arrays(run / name)
        prefix = prior.full_prefix_check(d, baseline, SWITCH)
        active = d['policy_active'].astype(bool)
        np.testing.assert_array_equal(active, d['active'][::20])
        with torch.no_grad():
            output = model(torch.from_numpy(d['policy_observation']).float()).numpy()
        actor_error = float(abs(output[active] - d['policy_actor_action'][active]).max())
        assert actor_error < 2e-6
        labels = corrected_trace_arrays(d, candidate, read(run / 'geometry.json'))
        teacher_error = float(abs(labels['actions'] - d['policy_teacher_action'][active]).max())
        assert teacher_error < 2e-5
        np.testing.assert_array_equal(d['policy_takeover'][active], (d['policy_time'] >= SWITCH - 1e-6)[active])
        expected = np.where(d['policy_takeover'][..., None], d['policy_teacher_action'], d['policy_actor_action'])
        np.testing.assert_array_equal(expected[active], d['policy_action'][active])
        expected = expected.clip(-1, 1)
        expected[:10] = 0
        np.testing.assert_array_equal(expected[active], d['actions'][::20][active])
        control = exact_control_check(d, arrays(CONTROL / CONTROL_TRACE)) if gain == .20 else None
        checks.append(dict(trace=name, pitch_rate_gain=gain, prefix=prefix, control_reproduction=control,
            actor_max_error=actor_error, teacher_max_error=teacher_error, executed_action_error=0.))
    return dict(verification=verify(run), traces=checks, full_successes=sum(successful(c) for c in cases),
        teacher_success_is_not_ppo_admission=True, autonomous_ppo_admission=False,
        training_runs=0, collected_frames=0, ppo_updates=0)


def worker(run):
    app = env = None
    try:
        check_comparison_sources(run)
        request = read(run / 'request.json')
        validate_request(request)
        control_check()
        base = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate'])
        saved = load_actor(ACTOR)
        app, env = setup(run)
        model = actor_network().to(env.device).eval()
        model.load_state_dict(saved['actor_state_dict'])
        cases = []
        for gain in request['gains']:
            print('[PITCH RATE GAIN]', gain, 'switch', SWITCH, flush=True)
            group = rollout(env, app, run, model, 43, SWITCH, candidate_for_gain(base, gain), BASELINE)
            names = {c['trace'] for c in group}
            assert len(names) == 1
            original = run / names.pop()
            target = run / f'trace_seed43_rate{round(gain * 100):03d}.npz'
            assert not target.exists()
            original.rename(target)
            for case in group:
                case.update(trace=target.name, pitch_rate_gain=gain)
            cases.extend(group)
        report = dict(cases=cases, ppo_updates=0, autonomous=False, behavior_accepted=False,
                      full_successes=sum(successful(c) for c in cases))
        write(run / 'evaluation.json', report)
        report.update(kind='fourth_actor_pitch_rate_comparison', actor_sha256=sha(ACTOR), base_candidate=base,
            teacher_result_sha256=sha(TEACHER / 'result.json'),
            trace_hashes={p.name: sha(p) for p in run.glob('trace_seed*.npz')})
        write(run / 'result.json', report)
        write(run / 'comparison_verification.json', verify_comparison(run))
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
    p.add_argument('--output', type=Path, default=DEFAULT_RUN)
    p.add_argument('--worker', type=Path)
    p.add_argument('--verify', type=Path)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if a.verify:
        print(verify_comparison(a.verify.resolve()))
        return
    if a.worker:
        worker(a.worker.resolve())
        return
    prior.baseline_check()
    control_check()
    run = a.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot(run)
    for name in comparison_sources():
        target = run / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    write(run / 'comparison_source_manifest.json', comparison_sources())
    write(run / 'comparison_contract.json', comparison_contract())
    geometry = geometry_audit()
    prior.check_geometry(read(BASELINE / 'geometry.json'), geometry)
    write(run / 'geometry_recheck.json', geometry)
    shutil.copy2(BASELINE / 'geometry.json', run / 'geometry.json')
    write(run / 'request.json', request_payload())
    started = now()
    write(run / 'session.json', dict(status='running', started_at=started))
    print('[R4A PITCH RATE COMPARISON]', run, GAINS, flush=True)
    process = None
    try:
        with (run / 'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(run)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code or (run / 'worker_error.json').exists() or not (run / 'comparison_verification.json').exists():
            raise RuntimeError(f'Comparison did not finish: inspect {run}/console.log')
        check_comparison_sources(run)
        write(run / 'session.json', dict(status='completed', started_at=started, finished_at=now()))
        print(read(run / 'comparison_verification.json'), flush=True)
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
