"""冻结同一个V1 Actor，在新工况下比较48/24与96/48 Nm；不做PPO更新。"""
import argparse
import csv
from pathlib import Path
import sys
import traceback

from launch import ROOT, REPO, SOURCE, SPEC, CART, PROTOCOL, BASELINE, now, sha256, write_json, snapshot, module_from_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=BASELINE)
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve(strict=True)
    SPEC.initialization_checkpoint(checkpoint)
    stamp=now().replace(':','-').replace('T','_').split('+')[0]
    output=ROOT/'validation'/('motor_caps_'+stamp)
    output.mkdir(parents=True,exist_ok=False)
    cases=PROTOCOL.cases()+[dict(slope_deg=0.,payload_kg=0.,towed_kg=0.,speed=.3,goal_distance_m=1.)]
    profiles=[dict(name='original_48_24',leg=48.,wheel=24.),dict(name='expanded_96_48',leg=96.,wheel=48.)]
    jobs=[dict(case=case,profile=profile) for profile in profiles for case in cases]
    digest=sha256(checkpoint)
    write_json(output/'request.json',dict(checkpoint=str(checkpoint),sha256=digest,seed=43,ppo_updates=0,
        ground_friction=2.3,rolling_coefficient=.02,hitch_height_m=.10,jobs=jobs,
        scope='Frozen V1 policy on V2 tasks. Diagnostic only; changing caps is not policy retraining.'))
    snapshot(output)
    print('CAP_COMPARISON',output,flush=True)
    sys.path.insert(0,str(REPO/'source/rl_training'))
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    env=None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from rsl_rl.runners import OnPolicyRunner
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v2 import core,mdp,torque
        verify_runtime=module_from_file("_motor_cap_worker",ROOT/"scripts/worker.py").verify_runtime
        SPEC.register_task()
        cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point')
        agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
        cfg.scene.num_envs=len(jobs);cfg.seed=agent.seed=43
        cfg.sim.device=agent.device='cuda:0'
        cfg.episode_length_s=61.;cfg.fixed_speed_mps=0.
        cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_dir=str(output/'cache/robot');cfg.scene.robot.spawn.usd_file_name='lighthw.usd'
        CART.write_cart_urdf(output/'cart.urdf')
        cfg.scene.trailer.spawn.asset_path=str(output/'cart.urdf')
        cfg.scene.trailer.spawn.usd_dir=str(output/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
        dump_yaml(str(output/'env_template.yaml'),cfg);dump_yaml(str(output/'agent.yaml'),agent)
        env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
        base=env.unwrapped;robot=base.scene['robot']
        base.r1_slope_rad[:]=torch.tensor([j['case']['slope_deg']*torch.pi/180 for j in jobs],device=env.device)
        base.r1_tangent[:],base.r1_normal[:]=core.plane_basis(base.r1_slope_rad)
        base.r1_payload_kg[:]=torch.tensor([j['case']['payload_kg'] for j in jobs],device=env.device)
        base.r1_towed_kg[:]=torch.tensor([j['case']['towed_kg'] for j in jobs],device=env.device)
        lanes=torch.tensor([SPEC.ANGLES.index(j['case']['slope_deg']) for j in jobs],device=env.device)
        base.scene.terrain.terrain_types[:]=lanes
        base.scene.terrain.env_origins[:]=base.scene.terrain.terrain_origins[0,lanes]
        base.cfg.fixed_speed_mps=torch.tensor([j['case']['speed'] for j in jobs],device=env.device)
        mdp.set_payload_properties(base)
        with torch.inference_mode():obs,_=env.reset()
        verify_runtime(env,output)
        # 改的是真正参与PD裁剪的逐环境执行器限值，不改PhysX的1e9保护上限。
        for name,key in [('joint','leg'),('wheel','wheel')]:
            actuator=robot.actuators[name]
            actuator.effort_limit[:]=torch.tensor([j['profile'][key] for j in jobs],device=env.device)[:,None]
        base.r1_motor_limits=torque.actuator_limits(robot,validate=True)[:,base.r1_joint_ids]
        limits=base.r1_motor_limits.cpu().numpy()
        for i,job in enumerate(jobs):
            np.testing.assert_array_equal(limits[i],np.array([job['profile']['leg']]*12+[job['profile']['wheel']]*4))
        write_json(output/'actual_caps.json',dict(joint_names=SPEC.CONTRACT['action_order'],limits_Nm=limits.tolist(),
                   note='runtime_check was collected before per-environment overrides; this file records actual comparison limits'))
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=env.device)
        runner.load(str(checkpoint),load_optimizer=False)
        policy=runner.get_inference_policy(device=env.device)
        alive=torch.ones(len(jobs),dtype=torch.bool,device=env.device)
        rows=[[] for _ in jobs];frames=[[] for _ in jobs];reasons=[None for _ in jobs]
        with torch.inference_mode():
            for step in range(PROTOCOL.MAX_DISTANCE_STEPS):
                action=policy(obs)
                assert bool(torch.isfinite(action).all())
                action[~alive]=0.
                obs,_,done,_=env.step(action)
                names=list(base.r1_last_metrics)
                values=torch.stack(list(base.r1_last_metrics.values()),dim=-1).cpu().tolist()
                samples={k:getattr(base.r1_torque_recorder,k).cpu().numpy() for k in ('applied','requested','velocity')}
                ended=done.bool().cpu().tolist()
                for i,recording in enumerate(alive.cpu().tolist()):
                    if not recording:continue
                    row=dict(t=step*.02,**dict(zip(names,values[i])))
                    rows[i].append(row)
                    frames[i].append({k:v[i].copy() for k,v in samples.items()})
                    if ended[i]:reasons[i]='terminated';alive[i]=False
                    elif PROTOCOL.finished(rows[i],jobs[i]['case']):alive[i]=False
                    elif len(rows[i])>=PROTOCOL.max_steps(jobs[i]['case']):
                        reasons[i]='evaluation_budget_exhausted';alive[i]=False
                if step%100==0:print('CAP_PROGRESS',step,'active',int(alive.sum()),flush=True)
                if not bool(alive.any()):break
        reports=[]
        for i,job in enumerate(jobs):
            trace={key:np.concatenate([f[key] for f in frames[i]]) for key in frames[i][0]}
            assert len(trace['applied'])==len(rows[i])*20
            assert np.all(abs(trace['applied'])<=limits[i]+1.e-4)
            report=PROTOCOL.summarize(rows[i],job['case'],reasons[i])
            report.update(profile=job['profile'],case_index=i)
            motor=torque.summarize_trace(trace,SPEC.CONTRACT['action_order'],limits[i],report['success'])
            report['motor_report']=f'case{i:02d}_motor.json'
            with (output/f'case{i:02d}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[i][0]));writer.writeheader();writer.writerows(rows[i])
            np.savez_compressed(output/f'case{i:02d}_motor_trace.npz',**trace,limits=limits[i],
                                joint_names=np.asarray(SPEC.CONTRACT['action_order']),physics_dt=.001)
            write_json(output/f'case{i:02d}.json',report);write_json(output/report['motor_report'],motor)
            reports.append(report)
        assert sha256(checkpoint)==digest
        write_json(output/'comparison.json',dict(finished_at=now(),checkpoint=str(checkpoint),sha256=digest,
            ppo_updates=0,cases=reports,passed_cases=sum(r['success'] for r in reports),
            note='One deterministic seed; no new task-trained policy or minimum motor rating established.'))
        print('CAP_FINISHED',output,flush=True)
    except BaseException as exc:
        write_json(output/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        traceback.print_exc()
        if env is None:
            import os
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':
    main()
