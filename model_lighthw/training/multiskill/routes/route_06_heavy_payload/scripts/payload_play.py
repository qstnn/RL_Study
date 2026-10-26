"""键盘变载回放；质量阶跃及每次重置独立记录，不作为固定载荷筛查。"""
import csv
import math
import time

from launch import URDF, spec, write_json

PROTOCOL = 'interactive_payload_change_v1'


def play_variable_payload(env, runner, args, run, app):
    import torch
    from payload_runtime import RuntimePayload
    from keyboard_payload import KeyboardPayload

    base = env.unwrapped
    runner.load(args['checkpoint'], load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    runtime = RuntimePayload(base, URDF, spec.CONTRACT)
    base.r6_payload_runtime = runtime
    control = KeyboardPayload(base, run, runtime)
    audit = base.r6_review
    reports = []
    budget = math.ceil(args['play_seconds']/base.step_dt)
    total_steps, episode = 0, 0
    pending_mass_verification = False
    exit_reason = 'duration_reached'
    env.seed(args['seed'])
    print('[R6 PAYLOAD] Manual variable-load play; fixed-payload success rate is not computed.', flush=True)
    try:
        while total_steps < budget and app.is_running() and not control.stop_requested:
            label = f"seed{args['seed']}_episode{episode}"
            rows, reason = [], None
            start_mass = runtime.mass_kg
            with torch.inference_mode():
                obs, _ = env.reset()
                # 普通reset保留运行时质量；再次核验后才启动新轨迹。
                runtime.verify_current()
                initial = base.scene['robot'].data.root_state_w[0].cpu().tolist()
                control.begin_episode(label)
                if hasattr(control, 'set_status'):
                    control.set_status('Running | R: restart current load | F10: finish')
                audit.begin(run, label, record_video=True, physics_trace=True)
                step = 0
                try:
                    while total_steps < budget:
                        wall_start = time.perf_counter()
                        if not app.is_running():
                            reason = 'app_closed'; break
                        changed = control.before_step(step, base.step_dt)
                        pending_mass_verification = pending_mass_verification or changed
                        if control.stop_requested:
                            reason = 'user_finished'; break
                        if control.reset_requested:
                            reason = 'user_reset'; break
                        if base.sim.is_stopped():
                            reason = 'timeline_stopped'; break
                        if not base.sim.is_playing():
                            base.sim.render()
                            time.sleep(.01)
                            continue
                        if changed:
                            obs = base.obs_buf['policy']
                        action = policy(obs)
                        if not torch.isfinite(action).all():
                            reason = 'non_finite_action'; break
                        obs, reward, done, extras = env.step(action)
                        step += 1
                        total_steps += 1
                        if pending_mass_verification:
                            # 暂停期间的变载也必须等下一真实物理步后回读。
                            # 回读发生在真实物理步之后，避免只检查setter输入缓冲。
                            verified = runtime.verify_current()
                            with (run/'payload_after_step.jsonl').open('a') as stream:
                                import json
                                stream.write(json.dumps(dict(label=label, step=step,
                                    t=step*base.step_dt, verification=verified), allow_nan=False)+'\n')
                            pending_mass_verification = False
                        frame = audit.frames[-1]
                        row = dict(frame['metrics'], t=step*base.step_dt,
                                   payload_kg=float(frame['payload_kg']),
                                   loaded_mass_kg=float(frame['loaded_mass_kg']),
                                   terminated=int(frame['terminated']), timeout=int(frame['timeout']))
                        rows.append(row)
                        control.update_display(frame)
                        if done.any():
                            reason = 'timeout' if bool(frame['timeout']) else 'terminated'
                            break
                        if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                            reason = 'non_finite_rollout'; break
                        # 给操作员真实的按键时间；慢于实时的渲染不做补帧跳步。
                        time.sleep(max(0., base.step_dt-(time.perf_counter()-wall_start)))
                    if reason is None:
                        reason = 'duration_reached'
                except BaseException:
                    reason = 'worker_exception'
                    raise
                finally:
                    audit.finish(reason)
                    control.finish_episode(reason)
            with (run/f'{label}.csv').open('w', newline='') as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            report = dict(seed=args['seed'], episode=episode, protocol=PROTOCOL,
                reason=reason, samples=len(rows), duration_s=len(rows)*base.step_dt,
                initial_payload_kg=start_mass, final_payload_kg=runtime.mass_kg,
                payload_kg=None, payload_range_kg=([min(r['payload_kg'] for r in rows),
                    max(r['payload_kg'] for r in rows)] if rows else [start_mass, runtime.mass_kg]),
                initial_root_state=initial, raw_trace=label+'_trace.npz',
                terminal_evidence=label+'_trace.json', success=None,
                final_acceptance='manual_variable_payload_not_standard_screen')
            write_json(run/f'{label}.json', report)
            reports.append(report)
            episode += 1
            exit_reason = reason
            if reason in ('app_closed', 'user_finished', 'timeline_stopped', 'duration_reached'):
                break
            if reason == 'user_reset':
                continue
            if reason.startswith('non_finite'):
                break
            # 自动reset之前的失败帧已保存；等待用户R开始下一段，不把重置伪装成恢复。
            print('[R6 PAYLOAD] Trial ended: '+reason+'. Change load if needed, then R to restart; F10 to finish.', flush=True)
            if hasattr(control, 'set_status'):
                control.set_status('Trial ended: '+reason+' | R: restart | F10: finish')
            while app.is_running() and not control.stop_requested and not control.reset_requested:
                base.sim.render()
                with torch.inference_mode():
                    changed = control.before_step(step, base.step_dt)
                pending_mass_verification = pending_mass_verification or changed
                if base.sim.is_stopped():
                    exit_reason = 'timeline_stopped'; break
                time.sleep(.02)
            if base.sim.is_stopped():
                break
    finally:
        control.close()
    if control.stop_requested:
        exit_reason = 'user_finished'
    elif not app.is_running():
        exit_reason = 'app_closed'
    result = dict(mode='play', status='completed', protocol=PROTOCOL, cases=reports,
        success_rate=None, payload_kg=None, initial_payload_kg=args['payload_kg'],
        final_payload_kg=runtime.mass_kg, requested_duration_s=args['play_seconds'],
        recorded_duration_s=total_steps*base.step_dt, exit_reason=exit_reason,
        ppo_updates=0, diagnostic_schema_version=3,
        unstepped_mass_change_at_exit=pending_mass_verification,
        final_acceptance='manual_variable_payload_not_standard_screen',
        mass_change_model='instantaneous rigid-body parameter change at fixed payload center; no loading collision or momentum-transfer model')
    write_json(run/'evaluation.json', result)
    return result
