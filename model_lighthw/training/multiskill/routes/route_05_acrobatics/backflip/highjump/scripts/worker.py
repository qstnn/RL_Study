"""R5A J1 仿真子进程；异常必须在 Kit 退出前写入原始诊断。"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from launch import REPO,ROUTE,SOURCE,spec,now,sha256,write_json,checkpoint,implementation_hashes


def verify_sources(run,phase):
    expected=json.loads((run/'source_manifest.json').read_text())
    changed=[name for name,sha in expected.items() if not (REPO/name).is_file() or sha256(REPO/name)!=sha]
    write_json(run/'source_consistency.json',dict(phase=phase,files=len(expected),changed=changed,status='failed' if changed else 'passed'))
    if changed: raise RuntimeError('Source changed while running: '+', '.join(changed))


def runtime_check(env,model,run):
    import torch
    import isaaclab.sim as sim
    from pxr import Usd
    base=env.unwrapped; robot=base.robot
    dims=[env.num_obs,env.num_privileged_obs,env.num_actions]
    import inspect
    write_json(run/'interface_check.json',dict(environment_type=type(base).__name__,
        environment_file=inspect.getfile(type(base)),wrapper_file=inspect.getfile(type(env)),
        observation_spaces=list(base.single_observation_space.spaces),cfg_state_space=base.cfg.state_space,
        dimensions=dims))
    if dims!=[400,109,16]: raise ValueError(f'R5A actual dimensions {dims}')
    order=[robot.joint_names[i] for i in base.leg_ids+base.wheel_ids]
    if order!=list(spec.LEG_JOINTS+spec.WHEELS): raise ValueError('Action mapping differs')
    torch.testing.assert_close(base.mass,torch.full_like(base.mass,model['unloaded_robot_mass_kg']),atol=.002,rtol=1e-4)
    expected_effort=torch.tensor([24. if 'wheel' in n else 48. for n in robot.joint_names],device=base.device)
    expected_velocity=torch.tensor([37.7 if 'wheel' in n else 12.57 for n in robot.joint_names],device=base.device)
    torch.testing.assert_close(robot.data.joint_effort_limits[0],expected_effort)
    torch.testing.assert_close(robot.data.joint_vel_limits[0],expected_velocity)
    base_id=robot.find_bodies('Base_link')[0][0]
    torch.testing.assert_close(robot.data.body_com_pos_b[0,base_id].cpu(),torch.tensor(model['base_com_m']),atol=2e-4,rtol=1e-3)
    actual=robot.root_physx_view.get_inertias()[0,base_id].cpu().reshape(3,3)
    torch.testing.assert_close(actual,torch.tensor(model['base_inertia_kg_m2']),atol=2e-4,rtol=.005)
    obs,extras=env.get_observations()
    if not torch.isfinite(obs).all() or not torch.isfinite(extras['observations']['critic']).all(): raise ValueError('Initial observation is non-finite')
    stage=sim.get_current_stage(); collision=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot')):
        attribute=prim.GetAttribute('physxArticulation:enabledSelfCollisions')
        if attribute and attribute.HasAuthoredValueOpinion(): collision.append((str(prim.GetPath()),bool(attribute.Get())))
    if not collision or not all(value for _,value in collision): raise ValueError('Actual self collisions not enabled')
    write_json(run/'runtime_check.json',dict(checked_at=now(),dimensions=dims,action_order=order,
        total_mass_kg=float(base.mass[0]),payload_kg=0.,actual_effort_limits=robot.data.joint_effort_limits[0].cpu().tolist(),
        actual_velocity_limits=robot.data.joint_vel_limits[0].cpu().tolist(),self_collision_attributes=collision,
        physics_dt=base.physics_dt,policy_dt=base.step_dt,base_inertia=actual.tolist(),behavior_acceptance='not_evaluated'))


def make_runner(run):
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner
    class Runner(OnPolicyRunner):
        def __init__(self,*args,**kwargs):
            from copy import deepcopy
            super().__init__(*args,**kwargs)
            self.completed_updates=0
            policy=self.alg.policy
            actor_params=list(policy.actor.parameters())+[policy.log_std]
            critic_params=list(policy.critic.parameters())
            self.alg.optimizer=torch.optim.Adam([
                {'params':actor_params,'lr':spec.CONTRACT['actor_learning_rate']},
                {'params':critic_params,'lr':spec.CONTRACT['critic_learning_rate']}])
            original_update=self.alg.update
            def guarded_update():
                warm=self.completed_updates<spec.CONTRACT['critic_warmup_updates']
                for param in policy.actor.parameters():param.requires_grad_(not warm)
                policy.log_std.requires_grad_(False)
                self.alg.optimizer.param_groups[0]['lr']=spec.CONTRACT['actor_learning_rate']
                self.alg.optimizer.param_groups[1]['lr']=spec.CONTRACT['critic_learning_rate']
                before=deepcopy(policy.state_dict());adam_before=deepcopy(self.alg.optimizer.state_dict())
                losses=original_update()
                with torch.no_grad():
                    observations=self.alg.storage.observations.flatten(0,1)
                    old_mu=self.alg.storage.mu.flatten(0,1);old_sigma=self.alg.storage.sigma.flatten(0,1)
                    policy.update_distribution(observations)
                    mu=policy.action_mean;sigma=policy.action_std
                    kl=(torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1).mean()
                    finite=all(torch.isfinite(x).all() for x in policy.state_dict().values())
                    accepted=bool(finite and torch.isfinite(kl) and kl<=.01)
                if not accepted:
                    policy.load_state_dict(before);self.alg.optimizer.load_state_dict(adam_before)
                if warm:
                    assert all(torch.equal(before[k],v) for k,v in policy.state_dict().items() if k.startswith('actor.') or k=='log_std')
                self.completed_updates+=1
                record=dict(completed_updates=self.completed_updates,critic_only=warm,accepted=accepted,
                    rollout_KL=float(kl) if torch.isfinite(kl) else None,actor_lr=spec.CONTRACT['actor_learning_rate'],
                    critic_lr=spec.CONTRACT['critic_learning_rate'],fixed_noise_std=spec.CONTRACT['exploration_std'])
                with (run/'update_audit.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                losses.update(rollout_KL=float(kl) if torch.isfinite(kl) else 0.,accepted_update=float(accepted))
                return losses
            self.alg.update=guarded_update

        def save(self,path,infos=None):
            infos=dict(r5a_contract=spec.CONTRACT,implementation_sha256=implementation_hashes(),
                completed_updates=int(self.current_learning_iteration)+1,learning_rate=float(self.alg.learning_rate),
                critic_warmup_updates=spec.CONTRACT['critic_warmup_updates'],fixed_noise_std=spec.CONTRACT['exploration_std'],
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),physics_state_saved=False,
                lineage=json.loads((run/'lineage.json').read_text()) if (run/'lineage.json').is_file() else None)
            super().save(path,infos)
        def log(self,locs,*args,**kwargs):
            super().log(locs,*args,**kwargs)
            rew=locs.get('rewbuffer',[]); lengths=locs.get('lenbuffer',[])
            row=dict(completed_updates=int(locs['it'])+1,mean_reward=sum(rew)/len(rew) if rew else None,
                     mean_episode_length=sum(lengths)/len(lengths) if lengths else None,learning_rate=self.alg.learning_rate,
                     collection_seconds=locs.get('collection_time'),learning_seconds=locs.get('learn_time'))
            path=run/'training_curve.csv'; new=not path.exists()
            with path.open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if new: writer.writeheader()
                writer.writerow(row)
    return Runner


def evaluate(env,runner,args,run,app):
    import torch
    from capture import Capture
    base=env.unwrapped
    runner.load(args['checkpoint'],load_optimizer=False); policy=runner.get_inference_policy(device=env.device)
    seeds=args['eval_seeds'] if args['mode']=='evaluate' else [args['seed']]
    count=args['eval_episodes'] if args['mode']=='evaluate' else args['play_episodes']
    reports=[]
    for seed in seeds:
        env.seed(seed)
        for episode in range(count):
            label=f'seed{seed}_episode{episode}'
            steps=0; reason='interrupted'; capture=None
            with torch.inference_mode():
                obs,_=env.reset()
                capture=Capture(base,run,label,video=args['record_video'])
                try:
                    for step in range(base.max_episode_length):
                        if not app.is_running(): raise RuntimeError('Playback window closed before review finished')
                        started=time.monotonic()
                        obs,reward,done,extras=env.step(policy(obs)); steps+=1
                        if done.any():
                            reason='terminated' if bool(base.reset_terminated[0]) else 'timeout'; break
                        if not torch.isfinite(obs).all() or not torch.isfinite(reward).all(): raise RuntimeError('Non-finite replay')
                        if not args['headless']: time.sleep(max(0.,base.step_dt-(time.monotonic()-started)))
                finally:
                    report=capture.finish(reason,steps)
            report.update(seed=seed,episode=episode); write_json(run/(label+'.json'),report); reports.append(report)
            print(f'[R5A REVIEW] {label}: success={report["success"]}, standing COM rise={report["first_flight_peak_rise"]:.3f} m, wheel clearance={report["wheel_clearance_peak"]:.3f} m, free flight={report["free_flight_time"]:.3f} s',flush=True)
    result=dict(mode=args['mode'],status='completed',cases=reports,success_rate=sum(r['success'] for r in reports)/len(reports),
                ppo_updates=0,screen=spec.SCREEN,full_backflip_acceptance='not_evaluated',hardware_acceptance='not_evaluated')
    write_json(run/'evaluation.json',result); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run.resolve(); app=None; env=None; runner=None; stage='inputs'
    def mark(value):
        nonlocal stage
        stage=value; write_json(run/'worker_status.json',dict(stage=stage,updated_at=now(),pid=os.getpid()))
    try:
        args=json.loads((run/'request.json').read_text()); model=json.loads((run/'model.json').read_text())
        verify_sources(run,'before_app'); mark('app_start')
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=args['headless'],device=args['device'],enable_cameras=args['record_video']).app
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        class JumpVecEnvWrapper(RslRlVecEnvWrapper):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                # Gym Dict.__contains__ checks a sample, not a key. The bundled
                # wrapper uses that test and misses DirectRLEnv critic spaces.
                spaces=self.unwrapped.single_observation_space.spaces
                self.num_privileged_obs=gym.spaces.flatdim(spaces['critic'])
                actor,extras=self.get_observations()
                critic=extras['observations']['critic']
                if actor.shape!=(self.num_envs,self.num_obs) or critic.shape!=(self.num_envs,self.num_privileged_obs):
                    raise ValueError('R5A observation tensors disagree with Gym spaces')
                repeated,_=self.get_observations()
                torch.testing.assert_close(actor,repeated,rtol=0,atol=0)
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.env import JumpLandingEnvCfg
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.runner_cfg import JumpRunnerCfg
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.appearance import sky_path
        cfg=JumpLandingEnvCfg(); agent=JumpRunnerCfg()
        cfg.reference_profile=args["reference_profile"];cfg.probe_sweep=args["probe_sweep"]
        cfg.scene.num_envs=args['num_envs']; cfg.seed=agent.seed=args['seed']
        cfg.sim.device=agent.device=args['device']; agent.max_iterations=args['max_iterations']
        cfg.scene.robot.spawn.asset_path=str(run/'input/robot.urdf')
        cfg.scene.robot.spawn.usd_dir=str(run/'asset_cache'); cfg.scene.robot.spawn.usd_file_name='r5a.usd'
        if args['mode']=='play': cfg.scene.sky_light.spawn.texture_file=sky_path()
        saved=checkpoint(args['checkpoint']) if args['checkpoint'] else None
        if saved:
            info=saved['infos']
            write_json(run/'lineage.json',info.get('migration') or info.get('lineage') or dict(source_checkpoint=args['checkpoint'],source_sha256=sha256(args['checkpoint'])))
        if saved and args['mode']=='train': agent.algorithm.learning_rate=saved['infos']['learning_rate']
        dump_yaml(str(run/'params/env.yaml'),cfg); dump_yaml(str(run/'params/agent.yaml'),agent)
        mark('environment_creation'); spec.register()
        env=JumpVecEnvWrapper(gym.make(spec.TASK,cfg=cfg,render_mode='rgb_array' if args['record_video'] else None),clip_actions=1.)
        base=env.unwrapped
        def episodes(rows):
            with (run/'episodes.jsonl').open('a') as f:
                for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
        base.episode_sink=episodes
        mark('runtime_check'); runtime_check(env,model,run); mark(args['mode'])
        if args['mode']=='smoke':
            from capture import Capture
            trace=Capture(base,run,'reference_probe')
            steps=0
            with torch.inference_mode():
                try:
                    for _ in range(base.max_episode_length):
                        _,reward,done,_=env.step(torch.zeros(env.num_envs,16,device=env.device)); steps+=1
                        if bool(done[0]): break
                finally:
                    report=trace.finish('terminated' if bool(base.reset_terminated[0]) else 'timeout',steps)
            result=dict(mode='smoke',status='completed',steps=steps,ppo_updates=0,reference_probe=report,
                        completed_episodes=base.last_reports,behavior_acceptance='not_evaluated')
        else:
            runner=make_runner(run)(env,agent.to_dict(),log_dir=str(run/'training') if args['mode']=='train' else None,device=args['device'])
            runner.git_status_repos=[]
            if args['mode']=='train':
                start=0
                if saved:
                    warm=saved['infos'].get('initialization')=='actor_warm_start'
                    info=runner.load(args['checkpoint'],load_optimizer=not warm)
                    start=info['completed_updates']; runner.current_learning_iteration=start;runner.completed_updates=start
                    runner.alg.learning_rate=info['learning_rate']
                    if not warm:
                        torch.set_rng_state(info['torch_rng'].cpu()); torch.cuda.set_rng_state_all(info['cuda_rng'])
                        random.setstate(info['python_rng']); np.random.set_state(info['numpy_rng'])
                    write_json(run/'optimizer_initialization.json',dict(actor_warm_start=warm,completed_updates=start,
                        optimizer_states=len(runner.alg.optimizer.state),physics_state_restored=False))
                runner.learn(num_learning_iterations=args['max_iterations'],init_at_random_ep_len=False)
                final=run/'training'/f'model_{runner.current_learning_iteration}.pt'; checked=checkpoint(final)
                completed=checked['infos']['completed_updates']-start
                result=dict(mode='train',status='completed',checkpoint=str(final),checkpoint_sha256=sha256(final),
                    start_completed_updates=start,total_completed_updates=start+completed,completed_updates=completed,
                    requested_updates=args['max_iterations'],behavior_acceptance='not_evaluated',physics_state_resumed=False)
                write_json(run/'training_result.json',result)
            else: result=evaluate(env,runner,args,run,app)
        verify_sources(run,'after_execution'); write_json(run/'result.json',result); mark('completed')
    except BaseException as error:
        write_json(run/'worker_error.json',dict(stage=stage,error=str(error),exception_type=type(error).__name__,traceback=traceback.format_exc(),failed_at=now()))
        print('[R5A WORKER ERROR]',stage,type(error).__name__,error,file=sys.stderr,flush=True); traceback.print_exc(); raise
    finally:
        if runner is not None and runner.writer is not None: runner.writer.close()
        if env is not None: env.close()
        if app is not None: app.close()


if __name__=='__main__': main()
