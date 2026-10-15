"""R6仿真进程；仅由launch.py创建完整输入和独立输出目录后启动。"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback
from launch import REPO, SOURCE, spec, now, sha256, write_json
import metrics


def measured(base):
    import torch
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.catalog import WHEEL_BODIES
    data, sensor = base.scene['robot'].data, base.scene['contact_forces']
    ids = sensor.find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
    other = [i for i, name in enumerate(sensor.body_names) if name not in WHEEL_BODIES]
    ratio = data.applied_torque.abs()/data.joint_effort_limits.clamp_min(1e-6)
    force = sensor.data.net_forces_w
    result = {'tilt_deg': torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1., 1.))*180./torch.pi,
              'speed': data.root_lin_vel_b[:, :2].norm(dim=-1), 'yaw_rate': data.root_ang_vel_b[:, 2].abs(),
              'height_error': (data.root_pos_w[:, 2]-.416).abs(), 'height': data.root_pos_w[:, 2],
              'four_wheel': (force[:, ids, 2] > 5.).all(-1).float(),
              'body_contact': (force[:, other].norm(dim=-1) > 1.).any(-1).float(),
              'torque_saturation': (ratio >= .99).float().mean(-1),
              'torque_peak': data.applied_torque.abs().max(-1).values,
              'x': data.root_pos_w[:, 0], 'y': data.root_pos_w[:, 1]}
    for i, name in enumerate(WHEEL_BODIES):
        result['force_z_'+name] = force[:, ids[i], 2]
    for i, name in enumerate(base.scene['robot'].joint_names):
        result['torque_'+name] = data.applied_torque[:, i]
    return result


def check_runtime(env, payload, run):
    import torch
    from isaaclab.utils.math import matrix_from_quat
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.catalog import POLICY_TERMS, LEG_NAMES, WHEEL_NAMES
    base, robot = env.unwrapped, env.unwrapped.scene['robot']
    dimensions = [env.num_obs, env.num_privileged_obs, env.num_actions]
    if dimensions != [58, 237, 16]:
        raise ValueError(f'R6 actual dimensions: {dimensions}, expected 58/237/16')
    if list(base.observation_manager.active_terms['policy']) != list(POLICY_TERMS)+['payload']:
        raise ValueError('R6 actor observation order mismatch')
    action_order = []
    for key in ('joint_pos', 'joint_vel'):
        action_order.extend(robot.joint_names[i] for i in base.action_manager.get_term(key)._joint_ids)
    if action_order != list(LEG_NAMES+WHEEL_NAMES):
        raise ValueError('R6 action order mismatch')
    obs, extras = env.get_observations()
    if not torch.isfinite(obs).all() or not torch.isfinite(extras['observations']['critic']).all():
        raise ValueError('Non-finite initial observation')
    expected_obs = torch.full_like(obs[:, -1], payload['payload_kg']/65.)
    torch.testing.assert_close(obs[:, -1], expected_obs)
    torch.testing.assert_close(extras['observations']['critic'][:, -1], expected_obs)
    actual_mass = robot.root_physx_view.get_masses().cpu()
    torch.testing.assert_close(actual_mass.sum(-1), torch.full_like(actual_mass[:, 0], payload['loaded_robot_mass_kg']), atol=.002, rtol=1e-4)
    body_id = robot.find_bodies('Base_link')[0][0]
    torch.testing.assert_close(robot.data.body_com_pos_b[0, body_id].cpu(), torch.tensor(payload['base_com_m'], dtype=actual_mass.dtype), atol=2e-4, rtol=1e-3)
    rotation = matrix_from_quat(robot.data.body_com_quat_b[0, body_id]).cpu()
    inertia_com = robot.root_physx_view.get_inertias()[0, body_id].cpu().reshape(3, 3)
    inertia_body = rotation @ inertia_com @ rotation.T
    torch.testing.assert_close(inertia_body, torch.tensor(payload['base_inertia_kg_m2'], dtype=inertia_body.dtype), atol=2e-4, rtol=.005)
    if abs(base.step_dt-.02)>1e-9 or abs(base.cfg.sim.dt-.001)>1e-9:
        raise ValueError('R6 control/physics rate mismatch')
    write_json(run/'runtime_check.json', {'checked_at': now(), 'dimensions': dimensions, 'action_order': action_order,
        'loaded_mass_kg': float(actual_mass[0].sum()), 'payload_kg': payload['payload_kg'],
        'mass_com_inertia_verified': True, 'behavior_acceptance': 'not_evaluated'})


def evaluate(env, runner, args, run, app):
    import torch
    runner.load(args['checkpoint'], load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    reports = []
    seeds = args['eval_seeds'] if args['mode'] == 'evaluate' else [args['seed']]
    episodes = args['eval_episodes'] if args['mode'] == 'evaluate' else 1
    for seed in seeds:
        env.seed(seed)
        for episode in range(episodes):
            with torch.inference_mode():
                obs, _ = env.reset()
                initial = env.unwrapped.scene['robot'].data.root_state_w[0].cpu().tolist()
                rows, reason = [], None
                for step in range(1000):
                    if not app.is_running():
                        reason = 'app_closed'; break
                    action = policy(obs)
                    if not torch.isfinite(action).all():
                        reason = 'non_finite_action'; break
                    obs, reward, done, extras = env.step(action)
                    if done.any():
                        reason = 'timeout' if env.unwrapped.reset_time_outs.any() else 'terminated'
                        break  # 自动重置后的出生帧不计入持载成功样本。
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                        reason = 'non_finite_rollout'; break
                    row = {k: float(v[0]) for k, v in measured(env.unwrapped).items()}
                    row.update(t=(step+1)*.02, payload_kg=args['payload_kg'])
                    rows.append(row)
            label = f'seed{seed}_episode{episode}'
            with (run/f'{label}.csv').open('w', newline='') as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            report = metrics.summarize(rows, reason, spec.SCREEN)
            report.update(seed=seed, episode=episode, payload_kg=args['payload_kg'], initial_root_state=initial)
            write_json(run/f'{label}.json', report); reports.append(report)
    result = {'mode': args['mode'], 'status': 'completed', 'cases': reports,
              'success_rate': sum(r['success'] for r in reports)/len(reports), 'payload_kg': args['payload_kg'],
              'thresholds': spec.SCREEN, 'ppo_updates': 0, 'final_acceptance': 'not_evaluated'}
    write_json(run/'evaluation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    stage = 'reading_inputs'

    def mark_stage(value):
        nonlocal stage
        stage = value
        write_json(run/'worker_status.json', {'stage': stage, 'updated_at': now(), 'pid': os.getpid()})

    app, env, runner = None, None, None
    try:
        mark_stage('reading_inputs')
        args = json.loads((run/'request.json').read_text())
        payload = json.loads((run/'payload.json').read_text())
        sys.path.insert(0, str(REPO/'source/rl_training'))
        mark_stage('app_launcher_import')
        from isaaclab.app import AppLauncher
        mark_stage('app_start')
        app = AppLauncher(headless=args['headless'], device=args['device']).app
        mark_stage('task_imports')
        import gymnasium as gym
        import torch
        import rl_training.tasks
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        class LoggedEnv(RslRlVecEnvWrapper):
            def __init__(self, *pos, **kw):
                self.r6_samples = []
                super().__init__(*pos, **kw)

            def step(self, action):
                if not torch.isfinite(action).all():
                    raise RuntimeError('Non-finite policy action')
                output = super().step(action)
                obs, reward, done, extras = output
                if not torch.isfinite(obs).all() or not torch.isfinite(reward).all() or not torch.isfinite(extras['observations']['critic']).all():
                    raise RuntimeError('Non-finite R6 rollout')
                if args['mode'] == 'train':
                    valid = ~done.bool()
                    row = {'terminated': int(self.unwrapped.reset_terminated.sum()), 'timeout': int(self.unwrapped.reset_time_outs.sum())}
                    if valid.any():
                        row.update({k: float(v[valid].mean()) for k, v in measured(self.unwrapped).items() if k in ('tilt_deg','speed','height','four_wheel','torque_peak','torque_saturation')})
                    self.r6_samples.append(row)
                return output

        from training import make_runner_class, resume_runner
        PayloadRunner = make_runner_class(run, args['payload_kg'])

        mark_stage('config_load')
        spec.register_task()
        cfg = load_cfg_from_registry(spec.TASK, 'env_cfg_entry_point')
        agent = load_cfg_from_registry(spec.TASK, 'rsl_rl_cfg_entry_point')
        cfg.scene.num_envs, cfg.payload_mass_kg = args['num_envs'], args['payload_kg']
        cfg.seed = agent.seed = args['seed']
        cfg.sim.device = agent.device = args['device']
        cfg.scene.robot.spawn.asset_path = str(run/'input/robot_payload.urdf')
        cfg.scene.robot.spawn.usd_dir = str(run/'asset_cache')
        cfg.scene.robot.spawn.usd_file_name = 'lighthw_r6.usd'
        agent.max_iterations = args['max_iterations']
        if args['mode'] in ('play','evaluate'):
            cfg.observations.policy.enable_corruption = False
            cfg.episode_length_s = 21.0  # 20秒评估窗之前不触发自动超时重置。
        dump_yaml(str(run/'params/env.yaml'), cfg)
        dump_yaml(str(run/'params/agent.yaml'), agent)
        mark_stage('environment_creation')
        env = LoggedEnv(gym.make(spec.TASK, cfg=cfg), clip_actions=agent.clip_actions)
        mark_stage('runtime_validation')
        check_runtime(env, payload, run)
        mark_stage(args['mode'])
        if args['mode'] == 'smoke':
            done_count = 0
            with torch.inference_mode():
                for step in range(100):
                    _, _, done, _ = env.step(torch.zeros((env.num_envs, 16), device=env.device))
                    done_count += int(done.sum())
            result = {'mode': 'smoke', 'status': 'completed', 'steps': 100, 'done_count': done_count, 'ppo_updates': 0}
        else:
            saved = spec.checkpoint(args['checkpoint']) if args['checkpoint'] else None
            if saved and args['mode'] == 'train':
                agent.algorithm.learning_rate = saved['infos']['learning_rate']
            runner = PayloadRunner(env, agent.to_dict(), log_dir=str(run/'training') if args['mode']=='train' else None, device=args['device'])
            if args['mode'] == 'train':
                if saved:
                    resume_runner(runner, args['checkpoint'], saved)
                start = runner.current_learning_iteration
                runner.learn(num_learning_iterations=args['max_iterations'], init_at_random_ep_len=False)
                final = run/'training'/f'model_{runner.current_learning_iteration}.pt'
                spec.checkpoint(final)
                completed = runner.current_learning_iteration-start+1
                result = {'mode': 'train', 'status': 'completed', 'checkpoint': str(final), 'checkpoint_sha256': sha256(final),
                          'start_iteration': start, 'last_iteration': runner.current_learning_iteration,
                          'completed_updates': completed, 'requested_updates': args['max_iterations'],
                          'payload_kg': args['payload_kg'], 'behavior_acceptance': 'not_evaluated'}
                write_json(run/'training_result.json', result)
            else:
                result = evaluate(env, runner, args, run, app)
        write_json(run/'result.json', result)
        mark_stage('completed')
    except BaseException as error:
        # 必须在app.close之前保存；Kit关闭过程可能直接结束进程并吞掉外层traceback。
        write_json(run/'worker_error.json', {'failed_at': now(), 'stage': stage,
            'exception_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
        print(f'[R6 WORKER ERROR] {stage}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
