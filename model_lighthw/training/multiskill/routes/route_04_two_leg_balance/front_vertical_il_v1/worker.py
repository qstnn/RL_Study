"""Native GPU collection, teacher-off evaluation, and separately requested PPO."""
from pathlib import Path
import json
import traceback
import numpy as np
import torch
from .common import read, write, require, now, sha
from .model import Actor, Features, pitch_rate
from .teacher import Teacher, candidates
from .trace import BufferedPhysicsTrace


def setup(run, n, hold=10, training=False, time_limit=None):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from .env import SustainBalanceCfg, SustainBalanceEnv
    from .runtime import runtime
    cfg = SustainBalanceCfg()
    cfg.scene.num_envs = n; cfg.seed = 42; cfg.sim.device = 'cuda:0'
    cfg.observation_hold_s = hold; cfg.stop_on_observation = not training
    cfg.episode_length_s = time_limit if time_limit is not None else (360. if training else hold+15.)
    cfg.scene.robot.spawn.asset_path = str(run/'assets/urdf/lighthw.urdf')
    cfg.scene.robot.spawn.usd_dir = str(run/'asset_cache'); cfg.scene.robot.spawn.usd_file_name = 'front_vertical.usd'
    env = RslRlVecEnvWrapper(SustainBalanceEnv(cfg=cfg), clip_actions=1.)
    runtime(env, run)
    assert (env.unwrapped.mode == 0).all() and cfg.target_pitches[0] == np.pi/2
    return app, env


@torch.inference_mode()
def rollout(env, app, folder, geometry, rows, hold, seed, actor=None, max_steps=None):
    folder.mkdir(parents=True, exist_ok=False)
    b = env.unwrapped; n = env.num_envs
    env.seed(seed); raw, _ = env.reset(); b.last_reports = []
    features = Features(n, b.device, [r['trigger_deg'] for r in rows], [r['blend_s'] for r in rows])
    obs = features(raw, torch.ones(n, dtype=torch.bool, device=b.device))
    teacher = Teacher(rows, geometry, b.device) if actor is None else None
    active = torch.ones(n, dtype=torch.bool, device=b.device)
    trace = BufferedPhysicsTrace(capacity=2000); controls = []; ended = {}
    def capture(physical, tick):
        t = physical.physics_steps*physical.physics_dt
        row = dict(physical.current_metrics)
        row.update(t=t, task_t=physical.state.clock(t), mode=physical.mode, active=active,
                   origin=physical.state.origin, target_gravity=physical.command()[1], phase=physical.command()[0],
                   actions=physical.actions, torque=physical.robot.data.applied_torque[:, physical.joint_ids])
        for key in ('capture_time', 'first_break_time', 'return_time', 'stand_request', 'post_capture_s'):
            row[key] = getattr(physical.state, key)
        trace.append(row)
    b.physics_observer = capture
    try:
        for step in range(min(b.max_episode_length, max_steps or b.max_episode_length)):
            if not app.is_running(): raise RuntimeError('Simulator closed')
            action = teacher(b, obs) if teacher is not None else actor(obs)
            controls.append(dict(observation=obs.cpu().numpy().copy(), raw=raw.cpu().numpy().copy(),
                                 action=action.cpu().numpy().copy(), active=active.cpu().numpy().copy()))
            raw, reward, done, _ = env.step(action)
            for row in b.last_reports:
                slot = row['env_id']
                if bool(active[slot]) and bool(done[slot]):
                    ended[slot] = dict(row, trace_env=slot, trace='trace.npz', blend_s=rows[slot]['blend_s'],
                                       candidate=rows[slot]['id'])
            active &= ~done.bool()
            if (step+1) % 50 == 0 or not active.any():
                status = dict(simulation_s=(step+1)*.02, active=int(active.sum()),
                    captured=int(b.state.held.sum()), max_hold_s=float(b.state.post_capture_s.max()))
                write(folder/'progress.json', status)
                print('[R4b]', folder.name, json.dumps(status), flush=True)
            if not active.any(): break
            obs = features(raw, done.bool())
    finally:
        b.physics_observer = None
    data = trace.arrays()
    data.update({'policy_'+k: np.stack([r[k] for r in controls]) for k in controls[0]})
    np.savez_compressed(folder/'trace.npz', **data)
    result = dict(cases=list(ended.values()), teacher_off=teacher is None, hold_seconds=hold,
                  seed=seed, trace_sha256=sha(folder/'trace.npz'), complete=len(ended)==n,
                  actor_dim=325, new_ppo_updates=0)
    write(folder/'result.json', result)
    return result


def worker(run, stage, hold=10, seed=43, steps=None):
    run = Path(run); request = require(run); geometry = read(run/'geometry.json')
    app = env = None
    try:
        if stage == 'search':
            rows = [r for r in candidates() for _ in range(2)]
        else:
            selected = read(run/'selected_teacher.json') if (run/'selected_teacher.json').exists() else candidates()[0]
            rows = [selected]*2
        app, env = setup(run, len(rows), hold)
        if stage.startswith('student'):
            from .learning import load_actor
            actor, _ = load_actor(run/'student.pt'); actor = actor.to(env.device).eval()
        else:
            actor = None
        return rollout(env, app, run/stage, geometry, rows, hold, seed, actor, steps)
    except BaseException as exc:
        write(run/f'{stage}_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise
    finally:
        if env is not None: env.close()
        if app is not None: app.close()
