"""v7学生闭环；自主路径完全不构造教师，不使用旧Actor或旧观测。"""
import traceback
import numpy as np
import torch
from common_student import read, write, now, sha, require, parent, load_actor


@torch.inference_mode()
def rollout(env, app, run, actor, request, seed):
    b = env.unwrapped
    env.seed(seed)
    obs, _ = env.reset()
    b.last_reports = []
    active = torch.ones(2, dtype=torch.bool, device=b.device)
    teacher = None
    if request['switch_s'] is not None:
        from teacher_sustain import SustainTeacher
        teacher = SustainTeacher([parent.candidate(.8)]*2, read(run/'geometry.json'), b.device)
    frames, controls, ended, terminals = [], [], {}, []

    def capture(base, tick):
        t = base.physics_steps * base.physics_dt
        row = {k: v.detach().clone() for k,v in base.current_metrics.items()}
        row.update(t=t.clone(), task_t=base.state.clock(t).clone(), mode=base.mode.clone(), active=active.clone(),
            origin=base.state.origin.clone(), target_gravity=base.command()[1].clone(), phase=base.command()[0].clone(),
            start_time=base.state.start_time.clone(), actions=base.actions.clone(), leg_targets=base.joint_targets.clone(),
            torque=base.robot.data.applied_torque[:, base.joint_ids].clone(),
            joint_velocity=base.robot.data.joint_vel[:, base.joint_ids].clone())
        for key in ('capture_time','first_break_time','break_reason','return_time','stand_request',
                    'post_capture_s','held','interrupted','observation_complete','failed','boundary'):
            row[key] = getattr(base.state, key).clone()
        frames.append(row)

    b.physics_observer = capture
    try:
        for step in range(b.max_episode_length):
            assert app.is_running(), 'Simulation closed before reports'
            mu = actor(obs)
            task = b.state.clock(b.physics_steps*b.physics_dt)
            label = teacher(b, task) if teacher is not None else torch.zeros_like(mu)
            takeover = task >= request['switch_s']-1e-6 if teacher is not None else torch.zeros_like(active)
            action = torch.where(takeover[:,None], label, mu)
            control = {k: v.cpu().numpy().copy() for k,v in dict(observation=obs, actor_action=mu,
                teacher_action=label, action=action, time=task, takeover=takeover, active=active,
                physics_time=b.physics_steps*b.physics_dt, return_time=b.state.return_time,
                hold_origin=b.state.hold_origin).items()}
            obs, reward, done, extras = env.step(action)
            control['reward'] = reward.cpu().numpy().copy()
            control['reward_terms'] = torch.stack(list(b.reward_terms.values()),-1).cpu().numpy().copy()
            controls.append(control)
            for c in b.last_reports:
                i = c['env_id']
                if active[i] and done[i]:
                    assert extras['terminal_observation_valid'][i]
                    assert torch.isfinite(extras['terminal_critic_observation'][i]).all()
                    ended[i] = dict(c)
                    terminals.append(dict(slot=i, timeout=bool(extras['time_outs'][i]),
                        failed=bool(extras['physical_terminated'][i]),
                        terminal_held=float(extras['terminal_critic_observation'][i,85]),
                        reset_held=float(extras['observations']['critic'][i,85])))
            active &= ~done.bool()
            if not active.any():
                break
    finally:
        b.physics_observer = None
    assert len(ended) == 2
    data = {k: torch.stack([r[k] for r in frames]).cpu().numpy() for k in frames[0]}
    data.update({f'policy_{k}': np.stack([r[k] for r in controls]) for k in controls[0]})
    name = f'trace_seed{seed}.npz'
    np.savez_compressed(run/name, **data)
    cases = [dict(ended[i], trace_env=i, trace=name, seed=seed, blend_s=.8 if teacher else None,
        teacher=teacher is not None, switch_s=request['switch_s']) for i in (0,1)]
    write(run/f'terminal_seed{seed}.json', terminals)
    write(run/'reward_names.json', list(b.reward_terms))
    print('[STUDENT]', seed, [(c['duration_s'],c['best_hold'],c['post_capture_s'],c['reason'],c['success']) for c in cases], flush=True)
    return cases


def worker(run):
    app = env = None
    try:
        require(run)
        request = read(run/'request.json')
        assert sha(request['actor']) == request['actor_sha256']
        actor, _ = load_actor(request['actor'])
        from worker_sustain import setup
        app, env = setup(run, request)
        # setup共享物理读回；显式纠正其仅适用于旧诊断的观测注释。
        runtime = read(run/'sustain_runtime.json')
        runtime.update(old_actor_uses_separate_legacy_adapter=False, actor_uses_v7_observation=True,
            teacher_constructed=request['switch_s'] is not None)
        write(run/'sustain_runtime.json', runtime)
        actor = actor.to(env.device).eval()
        cases = []
        for seed in request['seeds']:
            cases.extend(rollout(env, app, run, actor, request, seed))
            write(run/'evaluation.json', dict(cases=cases))
        write(run/'result.json', dict(cases=cases, autonomous=request['switch_s'] is None,
            actor_sha256=sha(request['actor']), observation_passes=sum(c['success'] for c in cases),
            ppo_admission=False, ppo_updates=0, trace_hashes={p.name:sha(p) for p in run.glob('trace_seed*.npz')}))
        from verify_student import verify
        write(run/'verification.json', verify(run))
    except BaseException as exc:
        write(run/'worker_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()
