"""独立R4a教师搜索/重放：真实固定初态、原动作物理、零PPO。"""
import argparse
import math
from pathlib import Path
import traceback
import numpy as np
from common_guided import BASE, read, write, now, sha, require_sources
from teacher import Teacher, pitch_and_rate


def setup(run, count, seed=42):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from two_leg_balance_split_v4.env import TwoLegBalanceCfg, TwoLegBalanceEnv
    from worker import runtime
    cfg = TwoLegBalanceCfg()
    cfg.scene.num_envs = count
    cfg.seed = seed
    cfg.sim.device = 'cuda:0'
    cfg.mode = 'rear'
    cfg.reward_profile = 'soft_transfer_v2'
    cfg.target_pitches = tuple(x['pitch_rad'] for x in read(run / 'geometry.json')['candidates'])
    cfg.scene.robot.spawn.asset_path = str(run / 'assets/urdf/lighthw.urdf')
    cfg.scene.robot.spawn.usd_dir = str(run / 'asset_cache')
    cfg.scene.robot.spawn.usd_file_name = 'r4.usd'
    class Wrapper(RslRlVecEnvWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
    env = Wrapper(TwoLegBalanceEnv(cfg=cfg), clip_actions=1.)
    runtime(env, run)
    env.reset()
    return app, env


def search(env, run, rows, horizon_s=7.2):
    import torch
    from two_leg_balance_split_v4.core import desired_gravity, masks
    b = env.unwrapped
    teacher = Teacher(rows, read(run / 'geometry.json'), b.device)
    n = len(rows)
    active = torch.ones(n, dtype=torch.bool, device=b.device)
    first = torch.full((n,), 99., device=b.device)
    strict_ticks = torch.zeros(n, device=b.device)
    pair_ticks = strict_ticks.clone()
    peak_torque = strict_ticks.clone()
    min_attitude = first.clone()
    reports = {}
    target = desired_gravity(torch.full((n,), teacher.target_pitch, device=b.device))
    def observer(base, tick):
        t = base.state.clock(base.physics_steps * base.physics_dt)
        hold = (t >= 4.) & (t < 7.) & active & ~base.state.failed
        m = base.current_metrics
        v = masks(m, base.mode, target)
        strict = hold & (base.state.hold_time > 0)
        first.copy_(torch.where(strict, torch.minimum(first, t), first))
        strict_ticks.add_(strict)
        pair_ticks.add_(hold & v['pair'])
        min_attitude.copy_(torch.where(hold, torch.minimum(min_attitude, torch.rad2deg(v['attitude'])), min_attitude))
        peak_torque.copy_(torch.where(active, torch.maximum(peak_torque, m['torque_peak']), peak_torque))
    b.physics_observer = observer
    try:
        with torch.inference_mode():
            for step in range(math.ceil(horizon_s/b.step_dt)):
                if not active.any(): break
                action = teacher(b, b.state.clock(b.physics_steps * b.physics_dt))
                action[~active] = 0.
                _, _, dones, _ = env.step(action)
                for row in b.last_reports:
                    i = row['env_id']
                    if active[i] and dones[i]: reports[i] = dict(row)
                active &= ~dones.bool()
                if step % 50 == 0: print(f'[GUIDED SEARCH] step={step} alive={int(active.sum())}', flush=True)
    finally:
        b.physics_observer = None
    output = []
    for i, row in enumerate(rows):
        report = reports.get(i)
        if report is None:
            report = {k:float(getattr(b.state, k)[i]) for k in ('best_hold','best_pair','start_time')}
            report.update({k:bool(getattr(b.state, k)[i]) for k in ('entered','held','failed','success')})
            report['reason'] = int(b.state.reason[i])
        output.append(dict(candidate=row, episode=report, first_strict_task_s=float(first[i]) if first[i] < 99 else None,
            strict_total_s=float(strict_ticks[i])*.001, hold_pair_total_s=float(pair_ticks[i])*.001,
            min_hold_attitude_deg=float(min_attitude[i]), peak_applied_torque_Nm=float(peak_torque[i]),
            trigger_task_s=float(teacher.trigger_t[i])))
    output.sort(key=lambda r:(-int(r['episode']['success']),-r['episode']['best_hold'], -r['strict_total_s'], -r['hold_pair_total_s']))
    result = dict(kind='teacher_search', generated_at=now(), ppo_updates=0, count=n, horizon_s=horizon_s,
        best_hold_s=max(r['episode']['best_hold'] for r in output), held=sum(r['episode']['held'] for r in output),
        successes=sum(r['episode']['success'] for r in output),
        dataset_eligible=False, full_protocol_tested=len(reports)==len(rows), rows=output)
    write(run / 'result.json', result)
    print('[GUIDED SEARCH RESULT]', result['best_hold_s'], result['held'], [r['candidate']['id'] for r in output[:8]], flush=True)


def replay(env, run, rows, selected, request):
    import torch
    from verify_trace import verify
    b = env.unwrapped
    slots = torch.tensor(selected, device=b.device)
    pick = lambda x: x.index_select(0, slots)
    cases = []
    for seed in request['seeds']:
        env.seed(seed)
        with torch.inference_mode():
            obs, _ = env.reset()
        b.last_reports = []
        teacher = Teacher(rows, read(run / 'geometry.json'), b.device)
        active = torch.ones(len(rows), dtype=torch.bool, device=b.device)
        frames, controls, ended = [], [], {}
        def observer(base, tick):
            m = {k:pick(v).detach().clone() for k,v in base.current_metrics.items()}
            t = base.physics_steps * base.physics_dt
            m.update(t=pick(t).clone(), task_t=pick(base.state.clock(t)).clone(), mode=pick(base.mode).clone(),
                active=pick(active).clone(), origin=pick(base.state.origin).clone(),
                target_gravity=pick(base.command()[1]).clone(), start_time=pick(base.state.start_time).clone(),
                actions=pick(base.actions).clone(), leg_targets=pick(base.joint_targets).clone(),
                joint_pos=pick(base.robot.data.joint_pos[:,base.joint_ids]).clone(),
                joint_vel=pick(base.robot.data.joint_vel[:,base.joint_ids]).clone(),
                torque=pick(base.robot.data.applied_torque[:,base.joint_ids]).clone())
            frames.append(m)
        b.physics_observer = observer
        try:
            with torch.inference_mode():
                for step in range(b.max_episode_length):
                    action = teacher(b, b.state.clock(b.physics_steps*b.physics_dt))
                    action[~active] = 0.
                    controls.append(dict(observation=pick(obs).cpu().numpy().copy(), action=pick(action).cpu().numpy().copy(),
                                         active=pick(active).cpu().numpy().copy()))
                    obs, _, dones, _ = env.step(action)
                    for report in b.last_reports:
                        i = report['env_id']
                        if i in selected and active[i] and dones[i]: ended[i] = dict(report)
                    active &= ~dones.bool()
                    if not pick(active).any(): break
        finally:
            b.physics_observer = None
        if len(ended) != len(selected): raise ValueError('Teacher replay did not reach terminal reports')
        arrays = {k:torch.stack([r[k] for r in frames]).cpu().numpy() for k in frames[0]}
        arrays.update({f'policy_{k}':np.stack([r[k] for r in controls]) for k in controls[0]})
        path = run / f'trace_seed{seed}.npz'
        np.savez_compressed(path, **arrays)
        for j,i in enumerate(selected):
            cases.append(dict(ended[i], trace_env=j, trace=path.name, seed=seed))
        print(f'[GUIDED REPLAY] seed={seed} hold={[ended[i]["best_hold"] for i in selected]}', flush=True)
    write(run/'evaluation.json', dict(cases=cases, ppo_updates=0, behavior_accepted=False, controller='privileged teacher'))
    verification = verify(run)
    all_held = all(c['held'] and c['best_hold'] >= 1.9999 for c in cases)
    all_success = all(c['success'] and c['timeout'] for c in cases)
    reference_ok = False
    if request.get('reference'):
        ref = read(Path(request['reference'])/'result.json')
        reference_ok = (ref['kind']=='teacher_replay' and ref['layout']=='original' and ref['all_success'] and
                        ref['candidate']==rows[selected[0]] and ref['all_held'] and len(ref['cases'])>=3)
    write(run/'result.json', dict(kind='teacher_replay', generated_at=now(), ppo_updates=0,
        candidate=rows[selected[0]], layout=request['layout'], count=len(rows), recorded_slots=selected,
        cases=cases, all_held=all_held, all_success=all_success, verification=verification,
        dataset_eligible=bool(all_held and all_success and request['layout']=='pair' and len(cases)>=6 and reference_ok),
        reference=request.get('reference'), reference_sha256=sha(Path(request['reference'])/'result.json') if request.get('reference') else None,
        trace_hashes={p.name:sha(p) for p in run.glob('trace_seed*.npz')}, behavior_accepted=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=__import__('pathlib').Path, required=True)
    run = p.parse_args().run.resolve()
    request = read(run/'request.json'); rows = read(run/'candidates.json')
    app = env = None
    try:
        require_sources(run)
        app, env = setup(run,len(rows),request.get('seed',42))
        if request['mode']=='teacher-search': search(env,run,rows,14. if request.get('family')=='return' else 7.2)
        else: replay(env,run,rows,request['selected'],request)
        require_sources(run)
    except BaseException as exc:
        write(run/'worker_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None: env.close()
        if app is not None: app.close()


if __name__=='__main__': main()
