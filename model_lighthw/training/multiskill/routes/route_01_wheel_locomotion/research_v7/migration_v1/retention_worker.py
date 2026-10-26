"""独立GPU旧能力单case回归入口；不训练、不调用教师、不改变原12项协议。"""
import argparse
import csv
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import REPO, V7, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.student import load_student, environment_reconstruction_contract
from migration_v1.student_worker import serial, write_json, dependency_sources
from migration_v1.runtime_checks import check_action_mapping
from migration_v1 import retention_protocol as protocol


def prepare(checkpoint, case_index, output,*,diagnostic_physics_override=None):
    """严格加载来源先于AppLauncher/建输出；非拖载不要求重拖教师时钟相等。"""
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Use a fresh retention output directory')
    case = protocol.select_case(case_index)
    policy,manifest = load_student(checkpoint,device='cpu',inference_mode='row')
    profile = environment_reconstruction_contract(manifest)['physics_profile']
    actual_profile,diagnostic = protocol.execution_profile(profile,diagnostic_physics_override)
    sources = {str(path):sha256(path) for path in set(dependency_sources())|
        {protocol.PROTOCOL_SOURCE,Path(protocol.__file__).resolve(),Path(__file__).resolve()}}
    return dict(policy=policy,manifest=manifest,profile=profile,case=case,case_index=case_index,
        output=output,checkpoint=str(Path(checkpoint).resolve()),sources=sources,
        execution_profile=actual_profile,diagnostic_physics_override=diagnostic)


def run_simulation(prepared):
    from isaaclab.app import AppLauncher
    out = prepared['output']; out.mkdir(parents=True,exist_ok=False)
    for name,digest in prepared['sources'].items():
        source = Path(name)
        relative = source.relative_to(REPO) if source.is_relative_to(REPO) else Path('_dependencies')/source.relative_to(REPO.parent)
        destination = out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,destination)
        if sha256(destination) != digest or sha256(source) != digest:
            raise EvidenceError('Retention dependency changed before application start')
    write_json(out/'retention_source_manifest.json',prepared['sources'])
    app = AppLauncher(headless=True,device='cuda:0').app
    env = None; traces = []; rows = []; motor = []
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.sim import get_current_stage
        from pxr import PhysxSchema, UsdPhysics
        sys.path.insert(0,str(V7/'scripts'))
        import research_common as common
        from rounded_cart import replace_wheels
        from migration_v1.student_recording import recorded_wheel_action_type, attach_recorders
        case,source_profile = prepared['case'],prepared['profile']
        diagnostic = prepared.get('diagnostic_physics_override')
        variant = None if diagnostic is None else diagnostic['variant']
        profile,verified_diagnostic = protocol.execution_profile(source_profile,variant)
        if prepared.get('execution_profile',source_profile) != profile or diagnostic != verified_diagnostic:
            raise EvidenceError('Prepared diagnostic execution profile was changed')
        artifact_kind = 'autonomous_retention_execution' if diagnostic is None else 'diagnostic_retention_physics_probe'
        cfg,agent = common.configure(out,1,43,*profile['robot_solver'],noise=False)
        cfg.sim.device = 'cuda:0'; protocol.configure_case(cfg,case,profile)
        for material in (cfg.scene.terrain.physics_material,cfg.sim.physics_material):
            material.static_friction = material.dynamic_friction = profile['ground_friction']
            material.friction_combine_mode = 'min'
        write_json(out/'rounded_cart_geometry.json',replace_wheels(out/'assets/cart.urdf'))
        cfg.actions.joint_vel.class_type = recorded_wheel_action_type()
        # 原UnifiedVelocityCommand：3s、1/.3加速度、原静止/行进/减速/2+5；没有重拖准备子类。
        env = RslRlVecEnvWrapper(gym.make(common.SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
        base = env.unwrapped
        base.r1_robot_hitch[:,2] += profile['robot_hitch_offset_m']
        policy = prepared['policy']; policy.actor.to('cuda:0').eval().assert_invariants()
        env.seed(43); observation,_ = env.reset()
        evaluation = common.launch.module_from_file('_r1_retention_runtime_verifier',common.V6/'scripts/worker.py')
        runtime = serial(evaluation.verify_runtime(env,out))
        robot,cart = base.scene['robot'],base.scene['trailer']
        term = base.command_manager.get_term('base_velocity')
        initial = serial(dict(robot_root_state=robot.data.root_state_w[0],cart_root_state=cart.data.root_state_w[0],
            joint_position_action_order=robot.data.joint_pos[0,base.r1_joint_ids],
            joint_velocity_action_order=robot.data.joint_vel[0,base.r1_joint_ids],
            cart_joint_velocity=cart.data.joint_vel[0],env_origin=base.scene.env_origins[0],
            episode_step=int(base.episode_length_buf[0]),robot_tensor_device=str(robot.data.root_state_w.device),
            simulation_device=str(cfg.sim.device),command_vx=float(term.command[0,0]),
            requested_vx=float(term.requested_vx[0]),goal_distance_m=cfg.eval_distance_m))
        physical = serial(dict(body_names=robot.body_names,masses_kg=robot.root_physx_view.get_masses()[0],
            coms=robot.root_physx_view.get_coms()[0],inertias=robot.root_physx_view.get_inertias()[0],
            unloaded_masses_kg=base.r1_unloaded_mass[0],unloaded_inertias=base.r1_unloaded_inertia[0]))
        write_json(out/'initial_state_runtime.json',initial); write_json(out/'payload_physical_readback.json',physical)
        solver,terrain = {},{}
        for prim in get_current_stage().Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or (str(prim.GetPath()).startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
                solver[str(prim.GetPath())] = {str(attr.GetName()):attr.Get() for attr in prim.GetAttributes() if 'Iteration' in str(attr.GetName())}
            if str(prim.GetPath()).startswith(cfg.scene.terrain.prim_path+'/') and prim.HasAPI(UsdPhysics.MaterialAPI):
                material = UsdPhysics.MaterialAPI(prim); px = PhysxSchema.PhysxMaterialAPI(prim)
                terrain[str(prim.GetPath())] = dict(static=material.GetStaticFrictionAttr().Get(),
                    dynamic=material.GetDynamicFrictionAttr().Get(),combine=px.GetFrictionCombineModeAttr().Get())
        materials = dict(assets={name:base.scene[name].root_physx_view.get_material_properties()[:,:,:2].cpu().tolist()
            for name in ('robot','trailer')},terrain=terrain)
        write_json(out/'solver_runtime_usd.json',solver); write_json(out/'friction_runtime.json',materials)
        verified = protocol.check_runtime(runtime,initial,solver,materials,
            dict(robot=cfg.scene.robot.spawn.asset_path,cart=out/'assets/cart.urdf'),profile,case,physical,
            diagnostic_physics_override=variant,source_profile=source_profile)
        write_json(out/'retention_runtime_review.json',verified)
        write_json(out/'request.json',dict(artifact_kind=artifact_kind,checkpoint=prepared['checkpoint'],
            checkpoint_sha256=sha256(prepared['checkpoint']),source_student_manifest=prepared['manifest'],
            case_index=prepared['case_index'],cases=[case],seed=43,device='cuda:0',noise=False,
            initial_state_mode='ordinary',physical_trailer_connected=False,nominal_towed_kg=0.,
            command_profile='slew',unloaded_acceleration_mps2=1.,loaded_acceleration_mps2=.3,
            goal_distance_m=1.,budget_s=protocol.original.max_steps(case)*.02,
            source_protocol=str(protocol.PROTOCOL_SOURCE),source_protocol_sha256=sha256(protocol.PROTOCOL_SOURCE),
            actor_inference='row',actor_unique_action_source=True,teacher_calls=0,action_override=False,
            source_towing_physics_profile=source_profile,diagnostic_physics_override=diagnostic,
            execution_physics=dict(profile,loaded_acceleration_mps2=.3,unloaded_acceleration_mps2=1.),
            command_protocol_note='Original non-towing protocol; source towing acceleration/preparation is not inherited.',
            ppo_updates=0,formal_ppo_authorized=False))
        recorder = attach_recorders(base,evaluation,common,out,write_json)
        write_json(out/'physics_recorder.json',dict(sample_hz=1000,body_names=robot.body_names,
            body_masses=recorder.masses,native_friction=recorder.contact_meta,
            actuator_torque_note='implicit PD requested/applied estimates, not measured final solver drive torque'))
        mapping = {}; latest_raw = {}; process = base.action_manager.process_action
        def record_native_action(action):
            process(action)
            legs = base.action_manager.get_term('joint_pos'); wheels = base.action_manager.get_term('joint_vel')
            cpu = lambda value:value.detach().cpu().numpy().copy()
            mapping.clear(); mapping.update(actor_raw=cpu(latest_raw['value']),manager_raw=cpu(action),
                leg_target=cpu(legs.processed_actions),wheel_target=cpu(wheels.processed_actions),
                default_legs=cpu(robot.data.default_joint_pos[:,legs._joint_ids]),leg_clip=cpu(legs._clip))
            mapping['check'] = check_action_mapping(**mapping,wrapper_clip=env.clip_actions)
        base.action_manager.process_action = record_native_action
        reason = None
        with torch.inference_mode():
            for step in range(protocol.original.max_steps(case)):
                if not app.is_running():
                    reason = 'application_closed'; break
                elapsed = float(base.episode_length_buf[0])*base.step_dt
                phase = int(base.r1_eval_phase[0]); command = float(term.command[0,0])
                protocol.check_pre_action(observation.cpu().numpy(),elapsed,phase,command,case)
                raw = policy(observation); latest_raw['value'] = raw.clone()
                frame = dict(observation=observation[0].cpu().numpy().copy(),actor_action=raw[0].cpu().numpy().copy(),
                    pre_action_time_s=elapsed,phase=phase,command_vx=command)
                observation,_,done,_ = env.step(raw); recorder.capture()
                frame.update(executed_action=mapping['manager_raw'][0],leg_target=mapping['leg_target'][0],
                    wheel_target=mapping['wheel_target'][0],**mapping['check'])
                traces.append(frame)
                values = evaluation.cpu_measurements(base.r1_last_metrics)
                rows.append(dict({key:float(value[0]) for key,value in values.items()},t=(step+1)*.02))
                motor.append({name:getattr(base.r1_torque_recorder,name)[0].cpu().numpy().copy()
                    for name in ('requested','applied','velocity')})
                if bool(done[0]):
                    reason = 'terminated'; break
                if protocol.original.finished(rows,case):
                    break
                if (step+1)%100 == 0:
                    print('R1_RETENTION_STEP',prepared['case_index'],step+1,'phase',rows[-1]['eval_phase'],flush=True)
            else:
                reason = 'budget_exhausted'
        if not rows:
            raise EvidenceError('No actual retention transition was recorded')
        prefix = out/'seed43_case00'
        with prefix.with_suffix('.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(str(prefix)+'_policy_trace.npz',**{key:np.stack([frame[key] for frame in traces]) for key in traces[0]})
        physics = {key:np.concatenate([frame[0] for frame in values]) for key,values in recorder.history.items()}
        np.savez_compressed(str(prefix)+'_physics_trace.npz',**physics)
        motors = {key:np.concatenate([frame[key] for frame in motor]) for key in motor[0]}
        limits = base.r1_motor_limits[0].cpu().numpy()
        np.savez_compressed(str(prefix)+'_motor_trace.npz',**motors,limits=limits,
            joint_names=np.asarray(common.SPEC.CONTRACT['action_order']),physics_dt=.001)
        if any(sha256(name) != digest for name,digest in prepared['sources'].items()):
            raise EvidenceError('Retention dependency changed during execution')
        if policy.teacher_calls != 0 or policy.actor_calls != len(traces):
            raise EvidenceError('Retention executed Actor-call count differs from actions')
        if (len(motors['applied']) != len(rows)*20 or not all(np.isfinite(value).all() for value in motors.values())
                or np.any(np.abs(motors['applied']) > limits+1.e-5)):
            raise EvidenceError('Retention actual 1kHz motor evidence/limits invalid')
        no_tow = protocol.check_no_towing(physics,rows)
        review = protocol.original.summarize(rows,case,reason)
        review.update(artifact_kind=artifact_kind,case_index=prepared['case_index'],
            diagnostic_physics_override=diagnostic,source_towing_physics_profile=source_profile,
            execution_physics=dict(profile,loaded_acceleration_mps2=.3,unloaded_acceleration_mps2=1.),
            autonomous_retention_accepted=False,
            runtime_review=verified,no_towing_review=no_tow,student_execution=policy.execution_record(),
            actor_unique_action_source=True,teacher_calls=policy.teacher_calls,action_override=False,
            source_protocol_sha256=sha256(protocol.PROTOCOL_SOURCE),
            action_mapping_review=dict(control_frames=len(traces),actor_calls=policy.actor_calls,
                **{key:max(frame[key] for frame in traces) for key in
                    ('actor_to_manager_max_error','leg_mapping_max_error','wheel_mapping_max_error')}),
            motor_review=dict(sample_hz=1000,actual_samples=len(motors['applied']),
                peak_abs_applied_leg_Nm=float(np.abs(motors['applied'][:,:12]).max()),
                peak_abs_applied_wheel_Nm=float(np.abs(motors['applied'][:,12:]).max())),
            unified_actor_accepted=False,formal_ppo_authorized=False)
        write_json(prefix.with_suffix('.json'),review); write_json(out/'retention_behavior_review.json',review)
        print('R1_RETENTION_COMPLETE',prepared['case_index'],review['success'],review['failed_checks'],flush=True)
        return review
    except BaseException as error:
        write_json(out/'retention_worker_error.json',dict(error=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),behavior_accepted=False,formal_ppo_authorized=False))
        if traces:
            np.savez_compressed(out/'aborted_retention_policy_trace.npz',**{key:np.stack([frame[key] for frame in traces]) for key in traces[0]})
        if 'recorder' in locals() and any(recorder.history.values()):
            np.savez_compressed(out/'aborted_retention_physics_trace.npz',**{key:np.concatenate([frame[0] for frame in values]) for key,values in recorder.history.items() if values})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--case-index',type=int,choices=range(12),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diagnostic-physics-override',choices=protocol.DIAGNOSTIC_PHYSICS_VARIANTS,default=None)
    args = parser.parse_args(argv)
    try:
        result = run_simulation(prepare(args.checkpoint,args.case_index,args.output,
            diagnostic_physics_override=args.diagnostic_physics_override))
    except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
        parser.exit(2,'Retention execution rejected: '+str(error)+'\n')
    if not result['success']:
        parser.exit(2,'Actual retention behavior failed the original protocol.\n')


if __name__ == '__main__':
    main()
