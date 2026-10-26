"""R1冻结校准：短测只定位原因，完整协议才作原门槛验收。PPO0。"""
import argparse,json,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'transfer_review'))
from review import (REPO,V3,SOURCES,GATE,gate,cpu_check,FrozenRunner,CART,SPEC,
                    now,write_json,snapshot,asset_manifest,sha256,module_from_file)
from launch import PROTOCOL

VARIANTS={
 'baseline':{},
 'wheelcap':dict(wheel_cap=3000.),
 'wheelcap_d1':dict(wheel_cap=3000.,wheel_damping=1.),
}

def physical_record(env):
    import isaaclab.sim as sim_utils
    from pxr import Usd,UsdPhysics,PhysxSchema
    base=env.unwrapped;robot=base.scene['robot'];stage=sim_utils.get_current_stage();bodies=[]
    for root in base.scene.env_prim_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root+'/Robot')):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(dict(path=str(prim.GetPath()),name=prim.GetName(),angular_cap_deg_s=PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxAngularVelocityAttr().Get()))
    if len(bodies)!=17*base.num_envs:raise RuntimeError('Incomplete physical body audit')
    import torch
    zero=base.r1_payload_kg.cpu()==0
    mass_error=float((robot.root_physx_view.get_masses()[zero]-base.r1_unloaded_mass[zero]).abs().max())
    com_error=float((robot.root_physx_view.get_coms()[zero,:,:3]-base.r1_unloaded_com[zero,:,:3]).abs().max())
    inertia_error=float((robot.root_physx_view.get_inertias()[zero]-base.r1_unloaded_inertia[zero]).abs().max())
    if max(mass_error,com_error,inertia_error)>1e-5:raise RuntimeError('Zero-payload physical properties changed')
    return dict(zero_payload_equivalence=dict(mass_max_error=mass_error,com_position_max_error=com_error,inertia_link_frame_max_error=inertia_error),bodies=bodies,mass=robot.root_physx_view.get_masses().cpu().tolist(),
                com=robot.root_physx_view.get_coms().cpu().tolist(),inertia=robot.root_physx_view.get_inertias().cpu().tolist(),
                material=robot.root_physx_view.get_material_properties().cpu().tolist(),
                wheel_damping=robot.actuators['wheel'].damping.cpu().tolist(),leg_effort=robot.actuators['joint'].effort_limit.cpu().tolist(),
                wheel_effort=robot.actuators['wheel'].effort_limit.cpu().tolist())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--variant',choices=tuple(VARIANTS),required=True)
    parser.add_argument('--candidate',choices=tuple(SOURCES),action='append');parser.add_argument('--full',action='store_true')
    parser.add_argument('--seconds',type=float,default=8.)
    args=parser.parse_args();names=args.candidate or ['route53'];treatment=VARIANTS[args.variant]
    if not 3.<args.seconds<=30.:parser.error('Short probe duration must be >3 and <=30 seconds')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checks=cpu_check();write_json(out/'cpu_checks.json',checks)
    write_json(out/'plan.json',dict(created_at=now(),variant=args.variant,treatment=treatment,candidates=names,full_protocol=args.full,
      short_seconds=None if args.full else args.seconds,gate=GATE,seed=43,ppo_updates=0,teacher_velocity='privileged diagnostic only',
      source_checkpoints={k:dict(v,checkpoint=str(v['checkpoint'])) for k,v in SOURCES.items()},hypotheses=['wheel body cap','contact parameters','torque cap/command profile','zero payload physical equivalence']))
    write_json(out/'asset_manifest.json',asset_manifest());snapshot(out);CART.write_cart_urdf(out/'assets/cart.urdf')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app;env=None;original_budget=PROTOCOL.max_steps
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        worker=module_from_file('_r1_calibration_worker',V3/'scripts/worker.py')
        if worker.protocol is not PROTOCOL:raise RuntimeError('Wrong protocol module')
        SPEC.register_task();cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
        cfg.scene.num_envs=16;cfg.seed=agent.seed=43;cfg.episode_length_s=65.;cfg.fixed_speed_mps=.3
        cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,15.,0.;cfg.observations.policy.enable_corruption=False
        if 'wheel_cap' in treatment:
            from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.wheel_rigid_cap_env_cfg import spawn_lighthw_wheel_rigid_cap
            cfg.scene.robot.spawn.func=spawn_lighthw_wheel_rigid_cap
        cfg.scene.robot.actuators['wheel'].damping=treatment.get('wheel_damping',3.)
        cfg.ground_friction=treatment.get('friction',2.3)
        cfg.leg_torque_limit=treatment.get('leg_limit',96.);cfg.wheel_torque_limit=treatment.get('wheel_limit',48.)
        cfg.command_profile=treatment.get('command_profile','slew')
        cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
        cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
        dump_yaml(str(out/'params/env.yaml'),cfg)
        env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
        if not args.full:PROTOCOL.max_steps=lambda case:min(original_budget(case),round(args.seconds/PROTOCOL.DT))
        summary=[]
        for name in names:
            folder=out/name;folder.mkdir();runner=FrozenRunner(name,env)
            request=dict(mode='evaluate',eval_distance=1.,eval_seeds=[43],seed=43,checkpoint=str(SOURCES[name]['checkpoint']),command_profile=cfg.command_profile,headless=True)
            write_json(folder/'request.json',dict(request,short_probe=not args.full,variant=args.variant))
            # 物理轨迹之外增加逐策略帧的轮绝对角速度，区分关节rad/s与刚体deg/s。
            base=env.unwrapped;robot=base.scene['robot'];wheel_ids=robot.find_bodies(['FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link'],preserve_order=True)[0]
            absolute=[];original_policy=runner.get_inference_policy
            def traced_policy(device=None):
                infer=original_policy(device)
                def step(obs):
                    if len(absolute)%250==0:print('CALIBRATION_PROGRESS',name,args.variant,'control_step',len(absolute),flush=True)
                    absolute.append(robot.data.body_ang_vel_w[:,wheel_ids].detach().cpu().numpy().copy())
                    return infer(obs)
                return step
            runner.get_inference_policy=traced_policy
            print('CALIBRATION_START',name,args.variant,flush=True)
            with torch.inference_mode():worker.evaluate(env,runner,request,folder,app)
            import numpy as np
            e=json.loads((folder/'evaluation.json').read_text());runner.save_inputs(folder,e['cases'])
            np.savez_compressed(folder/'wheel_body_angular_velocity.npz',angular_velocity_world_rad_s=np.stack(absolute),valid_case_lengths=np.asarray([c['samples'] for c in e['cases']]))
            physical=physical_record(env);write_json(folder/'physical.json',physical)
            # V3旧报告把阻尼写成常数3；隔离对照必须报告实际执行器值并保留旧原件。
            runtime=json.loads((folder/'runtime_check.json').read_text())
            damping_values=sorted(set(np.asarray(physical['wheel_damping']).ravel().tolist()))
            if len(damping_values)!=1:raise RuntimeError('Unexpected heterogeneous wheel damping')
            if runtime['wheel_damping']!=damping_values[0]:
                write_json(folder/'runtime_check_v3_original.json',runtime)
                runtime.update(wheel_damping=damping_values[0],wheel_damping_source='actual actuator tensor in physical.json')
                write_json(folder/'runtime_check.json',runtime)
            for body in physical['bodies']:
                expected=treatment.get('wheel_cap',1000.) if body['name'].endswith('_wheel_link') else 1000.
                if body['angular_cap_deg_s']!=expected:raise RuntimeError('Angular cap mismatch '+str(body))
            if args.full:decision=gate(e)
            else:decision=dict(training_ready=False,short_probe_only=True,early_failures=[dict(index=i,duration_s=c['observed']['duration_s'],causes=c['observed']['termination_causes']) for i,c in enumerate(e['cases']) if c['failure_reason']!='budget_exhausted' and not c['complete']],note='Reaching the short budget is not full task acceptance.')
            write_json(folder/'decision.json',decision)
            row=dict(candidate=name,variant=args.variant,complete=e['complete_count'],pass_count=e['pass_count'],**decision);summary.append(row)
            write_json(out/'summary.json',dict(status='running',candidates=summary,ppo_updates=0));print('CALIBRATION_RESULT',json.dumps(row,ensure_ascii=False),flush=True)
        for source in SOURCES.values():
            if sha256(source['checkpoint'])!=source['sha256']:raise RuntimeError('Source checkpoint changed')
        write_json(out/'summary.json',dict(status='completed',finished_at=now(),candidates=summary,ppo_updates=0))
    except BaseException as exc:
        write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
        import os
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
    finally:
        PROTOCOL.max_steps=original_budget
        if env is not None:env.close()
        app.close()
if __name__=='__main__':main()
