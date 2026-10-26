"""真实GPU接管及持续保持证据；没有Actor训练、自动数据晋级或PPO。"""
from pathlib import Path
import importlib.util
import traceback
import numpy as np
import torch
from common_sustain import (HERE, ROUTE, ACTOR, ACTOR_SHA, read, write, now, sha,
    require, check_legacy, load_actor, actor_network, candidate)


def setup(run, request):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from two_leg_balance_sustain_v7.env import SustainBalanceCfg, SustainBalanceEnv
    cfg = SustainBalanceCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 2, 42, 'cuda:0'
    cfg.observation_hold_s = request['hold_seconds']
    cfg.episode_length_s = request['hold_seconds'] + 15.
    cfg.stop_on_observation = request['return_after_s'] is None
    cfg.target_pitches = tuple(c['pitch_rad'] for c in read(run / 'geometry.json')['candidates'])
    cfg.scene.robot.spawn.asset_path = str(run / 'assets/urdf/lighthw.urdf')
    cfg.scene.robot.spawn.usd_dir = str(run / 'asset_cache')
    cfg.scene.robot.spawn.usd_file_name = 'sustain.usd'
    class Wrapper(RslRlVecEnvWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
    env = Wrapper(SustainBalanceEnv(cfg=cfg), clip_actions=1.)
    spec = importlib.util.spec_from_file_location('r4_original_runtime', ROUTE / 'split_v4/scripts/worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime(env, run)
    b = env.unwrapped
    assert not cfg.is_finite_horizon
    assert (b.scene.env_origins[:, :2].abs() < 190.).all()
    write(run / 'sustain_runtime.json', dict(time_limit_s=cfg.episode_length_s,
        observation_s=cfg.observation_hold_s, stop_on_observation=cfg.stop_on_observation,
        scene_origins=b.scene.env_origins.cpu().tolist(), collision_filter='inherited original scene.filter_collisions',
        world_boundary_abs_xy_m=198., no_automatic_return=True, no_actor_episode_clock=True,
        old_actor_uses_separate_legacy_adapter=True, ppo_admission=False))
    env.reset()
    return app, env


@torch.inference_mode()
def rollout(env, app, run, actor, request, blend):
    from teacher_sustain import SustainTeacher
    from observations import LegacyPrefixObservation
    b = env.unwrapped
    frames, controls, ended = [], [], {}
    env.seed(request['seed'])
    obs, _ = env.reset()
    b.last_reports = []
    legacy = LegacyPrefixObservation()
    teacher = SustainTeacher([candidate(blend)] * 2, read(run / 'geometry.json'), b.device)
    active = torch.ones(2, dtype=torch.bool, device=b.device)
    terminal_checks = []

    def capture(base, tick):
        t = base.physics_steps * base.physics_dt
        row = {k: v.detach().clone() for k, v in base.current_metrics.items()}
        row.update(t=t.clone(), task_t=base.state.clock(t).clone(), mode=base.mode.clone(), active=active.clone(),
            origin=base.state.origin.clone(), target_gravity=base.command()[1].clone(), phase=base.command()[0].clone(),
            start_time=base.state.start_time.clone(), actions=base.actions.clone(),
            leg_targets=base.joint_targets.clone(), torque=base.robot.data.applied_torque[:, base.joint_ids].clone(),
            joint_velocity=base.robot.data.joint_vel[:, base.joint_ids].clone())
        for key in ('capture_time', 'first_break_time', 'break_reason', 'return_time', 'stand_request',
                    'post_capture_s', 'held', 'interrupted', 'observation_complete', 'failed', 'boundary'):
            row[key] = getattr(base.state, key).clone()
        frames.append(row)

    b.physics_observer = capture
    try:
        with torch.inference_mode():
            for step in range(b.max_episode_length):
                if not app.is_running():
                    raise RuntimeError('Simulation closed before reports')
                if request['return_after_s'] is not None:
                    ids = torch.nonzero(active & b.state.stand_request &
                        (b.state.post_capture_s >= request['return_after_s'] - 1e-8)).flatten()
                    if ids.numel():
                        b.request_return(ids)
                        obs, _ = env.get_observations()
                old_obs = legacy(b)
                mu = actor(old_obs)
                task_t = b.state.clock(b.physics_steps * b.physics_dt)
                label = teacher(b, task_t)
                takeover = task_t >= request['switch_s'] - 1e-6
                action = torch.where(takeover[:, None], label, mu)
                control = {k: v.cpu().numpy().copy() for k, v in dict(
                    observation=obs, legacy_observation=old_obs, actor_action=mu, teacher_action=label,
                    action=action, time=task_t, takeover=takeover, active=active,
                    physics_time=b.physics_steps * b.physics_dt, return_time=b.state.return_time,
                    hold_origin=b.state.hold_origin).items()}
                obs, reward, done, extras = env.step(action)
                control['reward'] = reward.cpu().numpy().copy()
                control['reward_terms'] = torch.stack(list(b.reward_terms.values()), -1).cpu().numpy().copy()
                controls.append(control)
                for c in b.last_reports:
                    i = c['env_id']
                    if bool(active[i]) and bool(done[i]):
                        assert extras['terminal_observation_valid'][i]
                        assert torch.isfinite(extras['terminal_critic_observation'][i]).all()
                        assert bool(extras['time_outs'][i]) == bool(c['timeout'])
                        assert bool(extras['physical_terminated'][i]) == bool(c['failed'])
                        terminal_checks.append(dict(slot=i, terminal_stand_request=float(extras['terminal_critic_observation'][i, 60]),
                            terminal_held=float(extras['terminal_critic_observation'][i, 85]),
                            reset_held=float(extras['observations']['critic'][i, 85]),
                            timeout=bool(c['timeout']), failed=bool(c['failed'])))
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
    name = f'trace_seed43_blend{round(blend * 1000):03d}.npz'
    np.savez_compressed(run / name, **data)
    cases = [dict(ended[i], trace_env=i, trace=name, seed=43, switch_s=3.2, blend_s=blend,
                  teacher=True, observation_target_s=request['hold_seconds']) for i in (0, 1)]
    write(run / f'terminal_blend{round(blend * 1000):03d}.json', terminal_checks)
    write(run / 'reward_names.json', list(b.reward_terms))
    print('[SUSTAIN]', blend, [(c['capture_time'], c['post_capture_s'], c['first_break_time'], c['reason'], c['success']) for c in cases], flush=True)
    return cases


def worker(run):
    app = env = None
    try:
        require(run)
        check_legacy()
        request = read(run / 'request.json')
        assert request['actor_sha256'] == ACTOR_SHA and Path(request['actor']) == ACTOR
        app, env = setup(run, request)
        actor = actor_network().to(env.device).eval()
        actor.load_state_dict(load_actor(ACTOR)['actor_state_dict'])
        cases = []
        for blend in request['blends']:
            cases.extend(rollout(env, app, run, actor, request, blend))
            write(run / 'evaluation.json', dict(cases=cases, autonomous=False, ppo_updates=0))
        result = dict(cases=cases, kind='continuous_standing_teacher_takeover', autonomous=False,
            actor_sha256=sha(ACTOR), observation_passes=sum(c['success'] for c in cases),
            ppo_admission=False, ppo_updates=0, new_bc_epochs=0, collected_training_frames=0,
            trace_hashes={p.name: sha(p) for p in run.glob('trace_seed*.npz')})
        write(run / 'result.json', result)
        from verify_sustain import verify
        write(run / 'verification.json', verify(run))
    except BaseException as exc:
        write(run / 'worker_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()
