"""v6显式阶段worker：固定原物理、1kHz实际振动、独立来源和失败落盘。"""
from pathlib import Path
import argparse
import importlib.util
import traceback
import torch
from common_smooth import (REPO, HERE, read, write, sha, now, require, input_actor,
    actor_network, TEACHER, teacher_evidence, feedback_candidate, successful)
from worker_takeover import rollout


def setup(run):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from two_leg_balance_smooth_v6.env import SmoothBalanceCfg, SmoothBalanceEnv
    cfg = SmoothBalanceCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 2, 42, 'cuda:0'
    cfg.target_pitches = tuple(c['pitch_rad'] for c in read(run / 'geometry.json')['candidates'])
    cfg.scene.robot.spawn.asset_path = str(run / 'assets/urdf/lighthw.urdf')
    cfg.scene.robot.spawn.usd_dir = str(run / 'asset_cache')
    cfg.scene.robot.spawn.usd_file_name = 'smooth.usd'
    class Wrapper(RslRlVecEnvWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space.spaces['critic'])
    env = Wrapper(SmoothBalanceEnv(cfg=cfg), clip_actions=1.)
    spec = importlib.util.spec_from_file_location('original_runtime_checks', HERE.parent / 'split_v4/scripts/worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime(env, run)
    env.reset()
    return app, env


def gpu(run, request):
    app = env = None
    try:
        saved = input_actor(request['actor'])
        app, env = setup(run)
        model = actor_network().to(env.device).eval()
        model.load_state_dict(saved['actor_state_dict'])
        teacher = request['mode'] == 'replay'
        candidate = feedback_candidate('strong', teacher_evidence(TEACHER)['candidate']) if teacher else None
        cases = []
        for seed in request['seeds']:
            for switch in request['switch_times'] if teacher else [None]:
                cases.extend(rollout(env, app, run, model, seed, switch, candidate,
                    Path(request['baseline']) if teacher else None))
        result = dict(cases=cases, autonomous=not teacher, behavior_accepted=False, ppo_updates=0,
            full_successes=sum(successful(c) for c in cases))
        write(run / 'evaluation.json', result)
        result.update(kind='smooth_teacher_replay' if teacher else 'smooth_autonomous_evaluation',
            actor_sha256=sha(request['actor']), candidate=candidate,
            teacher_result_sha256=sha(TEACHER / 'result.json') if teacher else None,
            trace_hashes={p.name: sha(p) for p in run.glob('trace_seed*.npz')})
        write(run / 'result.json', result)
        from learning_smooth import evidence_check
        write(run / 'evidence_verification.json', evidence_check(run))
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
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    try:
        require(run)
        request = read(run / 'request.json')
        if request['mode'] in ('evaluate', 'replay'):
            gpu(run, request)
        else:
            from learning_smooth import collect, pretrain
            (collect if request['mode'] == 'collect' else pretrain)(run, request)
        require(run)
    except BaseException as exc:
        write(run / 'worker_error.json', dict(error=str(exc), traceback=traceback.format_exc(), at=now()))
        raise


if __name__ == '__main__':
    main()
