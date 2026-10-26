"""连续物理状态的影子教师接管；独立阶段运行，不自动训练或评估。"""
import argparse
from pathlib import Path
import traceback
import numpy as np
import torch
from common_takeover import (read, write, sha, now, snapshot, require, contract, sources,
    original_actor, load_actor, actor_network, teacher_evidence, arrays, prefix_check,
    diagnosis_check, assemble, dataset_check, successful, verify, feedback_candidate)


def rollout(env, app, run, actor, seed, switch_s=None, candidate=None, baseline=None):
    """全程运行原Actor/影子教师；到时才切执行源，不reset状态、历史或教师触发器。"""
    from teacher import Teacher
    b = env.unwrapped
    env.seed(seed)
    frames, controls, ended = [], [], {}
    with torch.inference_mode():
        obs, _ = env.reset()
        b.mode[:] = 1
        b.history_reset[:] = True
        obs, _ = env.get_observations()
        b.last_reports = []
        active = torch.ones(2, dtype=torch.bool, device=b.device)
        teacher = Teacher([candidate, candidate], read(run / 'geometry.json'), b.device) if candidate else None

        def capture(base, tick):
            row = {k: v.detach().clone() for k, v in base.current_metrics.items()}
            t = base.physics_steps * base.physics_dt
            row.update(t=t.clone(), task_t=base.state.clock(t).clone(), mode=base.mode.clone(), active=active.clone(),
                origin=base.state.origin.clone(), target_gravity=base.command()[1].clone(),
                start_time=base.state.start_time.clone(), actions=base.actions.clone(),
                leg_targets=base.joint_targets.clone(), torque=base.robot.data.applied_torque[:, base.joint_ids].clone())
            frames.append(row)

        b.physics_observer = capture
        try:
            for step in range(b.max_episode_length):
                if not app.is_running():
                    raise RuntimeError('Simulation closed before terminal reports')
                t = b.state.clock(b.physics_steps * b.physics_dt)
                mu = actor(obs)
                label = teacher(b, t) if teacher else torch.zeros_like(mu)
                takeover = (t >= switch_s - 1e-6) if teacher else torch.zeros_like(active)
                action = torch.where(takeover[:, None], label, mu)
                controls.append({k: v.cpu().numpy().copy() for k, v in dict(observation=obs, action=action,
                    actor_action=mu, teacher_action=label, takeover=takeover, active=active, time=t).items()})
                obs, _, done, _ = env.step(action)
                for c in b.last_reports:
                    i = c['env_id']
                    if bool(active[i]) and bool(done[i]):
                        ended[i] = dict(c)
                active &= ~done.bool()
                if not active.any():
                    break
        finally:
            b.physics_observer = None
    if len(ended) != 2:
        raise ValueError('Missing terminal reports')
    data = {k: torch.stack([r[k] for r in frames]).cpu().numpy() for k in frames[0]}
    data.update({f'policy_{k}': np.stack([r[k] for r in controls]) for k in controls[0]})
    suffix = f'_switch{round(switch_s * 1000)}' if switch_s is not None else ''
    name = f'trace_seed{seed}{suffix}.npz'
    np.savez_compressed(run / name, **data)
    if baseline is not None:
        prefix_check(data, arrays(baseline / f'trace_seed{seed}.npz'), switch_s)
    cases = [dict(ended[i], trace_env=i, trace=name, seed=seed, switch_s=switch_s) for i in (0, 1)]
    print(f'[TAKEOVER] seed={seed} switch={switch_s} hold={[c["best_hold"] for c in cases]} success={[c["success"] for c in cases]}', flush=True)
    return cases


def gpu(run, request):
    from physics_worker import setup
    app = env = None
    try:
        app, env = setup(run, 2, 42)
        diagnostic = request['mode'] == 'diagnose'
        saved = original_actor(request['actor']) if diagnostic else load_actor(request['actor'])
        actor = actor_network().to(env.device).eval()
        actor.load_state_dict(saved['actor_state_dict'])
        cases = []
        if diagnostic:
            teacher = teacher_evidence(request['teacher'])
            candidate = feedback_candidate(request['feedback'], teacher['candidate'])
            for switch in request['switch_times']:
                cases.extend(rollout(env, app, run, actor, request['seed'], switch,
                    candidate, Path(request['baseline'])))
        else:
            for seed in request['seeds']:
                cases.extend(rollout(env, app, run, actor, seed))
        report = dict(cases=cases, ppo_updates=0, behavior_accepted=False,
            autonomous=not diagnostic, full_successes=sum(successful(c) for c in cases))
        write(run / 'evaluation.json', report)
        report.update(kind='teacher_takeover_diagnosis' if diagnostic else 'takeover_actor_evaluation',
            candidate=candidate if diagnostic else None,
            feedback=request['feedback'] if diagnostic else None,
            actor_sha256=sha(request['actor']), teacher_result_sha256=sha(Path(request['teacher']) / 'result.json') if diagnostic else None,
            trace_hashes={p.name: sha(p) for p in run.glob('trace_seed*.npz')}, verification=verify(run))
        write(run / 'result.json', report)
        if diagnostic:
            write(run / 'takeover_verification.json', diagnosis_check(run))
    except BaseException as exc:
        # Isaac关闭可能掩盖Python异常，必须先落盘，再关闭仿真。
        write(run / 'worker_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


def collect(run, request):
    data, additions, audit = assemble(request['dataset'], request['diagnosis'])
    np.savez_compressed(run / 'dataset.npz', **data)
    write(run / 'result.json', dict(kind='verified_takeover_dataset', at=now(),
        parent_dataset=request['dataset'], parent_dataset_sha256=sha(Path(request['dataset']) / 'dataset.npz'),
        diagnosis=request['diagnosis'], diagnosis_result_sha256=sha(Path(request['diagnosis']) / 'result.json'),
        dataset_sha256=sha(run / 'dataset.npz'), frames=len(data['phase']),
        episodes=int(data['episode'].max()) + 1, sample_kind_counts=np.bincount(data['sample_kind'], minlength=3).tolist(),
        phase_counts=np.bincount(data['phase'], minlength=5).tolist(), additions=additions, diagnosis_audit=audit,
        ppo_updates=0, note='Only actual teacher suffixes of full-success recoveries are added; actor prefixes and failures excluded'))
    dataset_check(run)


def pretrain(run, request):
    from learning import fit_actor
    data, meta = dataset_check(request['dataset'])
    model, history = fit_actor(data, request['epochs'], request['seed'])
    path = run / 'actor.pt'
    torch.save(dict(kind='bc_takeover_actor', workflow=contract(), sources=sources(),
        actor_state_dict=model.state_dict(), dataset=request['dataset'], dataset_sha256=meta['dataset_sha256'],
        epochs=request['epochs'], seed=request['seed'], critic_transferred=False, optimizer_transferred=False,
        actor_initialized_from_scratch=True, behavior_accepted=False), path)
    write(run / 'result.json', dict(kind='takeover_actor_imitation', actor=str(path), actor_sha256=sha(path),
        epochs=request['epochs'], history=history, ppo_updates=0, autonomous_behavior_tested=False,
        automatic_evaluations=0, review_status='awaiting_user_request', behavior_accepted=False))
    load_actor(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    torch.set_num_threads(4)
    try:
        require(run)
        request = read(run / 'request.json')
        if request['mode'] in ('diagnose', 'evaluate'):
            gpu(run, request)
        elif request['mode'] == 'collect':
            collect(run, request)
        else:
            pretrain(run, request)
        require(run)
    except BaseException as exc:
        write(run / 'worker_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise


if __name__ == '__main__':
    main()
