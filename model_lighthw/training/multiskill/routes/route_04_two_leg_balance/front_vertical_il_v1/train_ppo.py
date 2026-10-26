"""Explicit, gated continuous PPO; no periodic or end-of-training frozen evaluation."""
from pathlib import Path
import json
import torch
from .common import require, read, write, sha
from .learning import require_admission
from .settings import OBSERVATION, PPO_CONFIG
from .model import Features
from .ppo_driver import ContinuingDriver, DriverConfig


def worker(run, output, updates, resume=None, time_limit=360.):
    from .worker import setup
    run, output = Path(run), Path(output)
    require(run); actor, saved = require_admission(run)
    actor.prefix.requires_grad_(False)
    app = env = None
    output.mkdir(parents=True, exist_ok=False)
    try:
        app, env = setup(run, 16, training=True, time_limit=time_limit)
        actor = actor.to(env.device)
        profile = saved['teacher']
        driver = ContinuingDriver(env, actor,
            Features(16, env.device, profile['trigger_deg'], profile['blend_s']), DriverConfig(**PPO_CONFIG),
            observation_contract=OBSERVATION, policy_contract='R4b_IL_'+sha(run/'student.pt'), device=env.device)
        if resume is not None: driver.load_checkpoint(resume)
        start = driver.completed_updates
        def record(episodes):
            with (output/'episodes.jsonl').open('a') as stream:
                for row in episodes: stream.write(json.dumps(row)+'\n')
        env.unwrapped.episode_sink = record
        prefix = {k: v.detach().clone() for k, v in actor.prefix.state_dict().items()}
        for index in range(updates):
            row = driver.run_iteration()
            row['local_updates'] = index+1
            with (output/'updates.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            write(output/'progress.json', row)
            print(f'[R4b PPO] {index+1}/{updates} global={driver.completed_updates} losses={row["losses"]}', flush=True)
            if (index+1) % 100 == 0 or index+1 == updates:
                final = output/f'model_{driver.completed_updates:06d}.pt'; driver.save_checkpoint(final)
        if any(not torch.equal(v, actor.prefix.state_dict()[k]) for k, v in prefix.items()):
            raise ValueError('Frozen IL rise changed')
        optimizer_steps = sorted({int(s['step']) for s in driver.alg.optimizer.state.values() if 'step' in s})
        result = dict(start_updates=start, new_updates=updates, completed_updates=driver.completed_updates,
            checkpoint=str(final.resolve()), checkpoint_sha256=sha(final), optimizer_steps=optimizer_steps,
            bootstrap_events=driver.bootstrap_events, physical_terminations=driver.physical_terminations,
            driver_reset_calls=driver.driver_reset_calls, prefix_unchanged=True,
            source_manifest=read(run/'sources.json'), actor_sha256=sha(run/'student.pt'),
            review_status='awaiting_user_request', behavior_accepted=False)
        write(output/'result.json', result)
    finally:
        if env is not None: env.close()
        if app is not None: app.close()
