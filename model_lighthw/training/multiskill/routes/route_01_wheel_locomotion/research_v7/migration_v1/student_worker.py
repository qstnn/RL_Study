"""独立学生普通初态GPU单向回放；严格来源门先于AppLauncher和输出目录。

此入口尚未执行实际学生仿真。显式固定准备时钟（默认3s）是待自主验证候选；没有教师接管，
不训练，不能凭接口/文件存在授权PPO。
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration_v1.contract import V7, REPO, SOURCE, sha256
from migration_v1.dataset import EvidenceError
from migration_v1.student import load_student
from migration_v1.runtime_checks import (select_case, check_runtime, check_action_mapping,
    preparation_step_offset, check_source_preparation, check_pre_action_preparation)

OPEN_LOOP_DIAGNOSTIC_VERSION = 'r1_open_loop_teacher_sequence_v1'


def serial(value):
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def dependency_sources():
    """冻结具体执行类与记录器；不调用会复制全部路线的旧snapshot函数。"""
    old = V7.parent/'unified_v5/results/2026-10-19_coordinated_towing'
    v6 = V7.parent/'unified_v6/scripts'
    # Explicit student dependencies: unrelated PPO/environment development must
    # not invalidate an independent student execution that never imports it.
    files = {Path(__file__).parent/name for name in ('__init__.py', 'contract.py', 'dataset.py',
        'policy.py', 'bc.py', 'preparation_sampling.py', 'student.py', 'runtime_checks.py',
        'student_recording.py', 'student_worker.py')}
    files |= set((SOURCE/'wheel_unified_v6').glob('*.py'))
    files |= {SOURCE/name for name in ('catalog.py','wheel_env_cfg.py','rewards.py')}
    files |= {v6/name for name in ('launch.py','worker.py','protocol.py','quality.py')}
    files |= {V7/'scripts'/name for name in ('research_common.py','settings.py','rounded_cart.py',
        'contact_state.py','momentum_probe.py','real_cart_audit.py','rolling_load_audit.py','controlled_load.py')}
    files |= {old/'coordinated_control.py', old/'pose_ik.py',
        REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'}
    files |= {REPO/'model_lighthw/training/multiskill/common.py',
        REPO/'source/rl_training/rl_training/assets/deeprobotics.py',
        REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py'}
    files |= set((REPO/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_lighthw').glob('*.py'))
    isaac = REPO.parent/'Isaaclab/source'
    files |= {isaac/name for name in (
        'isaaclab/isaaclab/envs/mdp/actions/joint_actions.py',
        'isaaclab/isaaclab/envs/mdp/commands/velocity_command.py',
        'isaaclab/isaaclab/managers/action_manager.py',
        'isaaclab/isaaclab/managers/command_manager.py',
        'isaaclab/isaaclab/managers/observation_manager.py',
        'isaaclab/isaaclab/envs/manager_based_env.py',
        'isaaclab/isaaclab/envs/manager_based_rl_env.py',
        'isaaclab/isaaclab/assets/articulation/articulation.py',
        'isaaclab/isaaclab/assets/articulation/articulation_data.py',
        'isaaclab/isaaclab/scene/interactive_scene.py',
        'isaaclab/isaaclab/utils/buffers/circular_buffer.py',
        'isaaclab/isaaclab/actuators/actuator_pd.py',
        'isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py')}
    urdf = REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
    files |= {(urdf.parent/mesh.get('filename')).resolve(strict=True) for mesh in ET.parse(urdf).getroot().findall('.//mesh')}
    return sorted(files)


def prepare(checkpoint, slope_deg, output, preparation_seconds=3.):
    """无App/Isaac导入；失败时没有输出目录。"""
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError('Use a fresh autonomous student output directory')
    policy, manifest = load_student(checkpoint, device='cpu', inference_mode='row')
    case, profile = select_case(manifest, slope_deg)
    preparation = check_source_preparation(manifest, preparation_seconds, case_slope=case['slope_deg'])
    sources = {str(path): sha256(path) for path in dependency_sources()}
    return dict(policy=policy, manifest=manifest, case=case, profile=profile, output=output,
        checkpoint=str(Path(checkpoint).resolve()), sources=sources, preparation=preparation)


def fixed_preparation_command_type(preparation_seconds):
    """App后调用的原命令子类；只有从reset起算的固定时钟，无teacher/状态门。"""
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6.commands import UnifiedVelocityCommand
    class FixedPreparationCommand(UnifiedVelocityCommand):
        def _update_command(self):
            self.cycle_start_step[:] = preparation_step_offset(preparation_seconds, self._env.step_dt)
            super()._update_command()
    return FixedPreparationCommand


def run_simulation(prepared):
    # Default student entry remains actor-only. The separate diagnostic entry
    # verifies an original recorded sequence without impersonating an Actor.
    diagnostic = prepared.get('execution_kind') == OPEN_LOOP_DIAGNOSTIC_VERSION
    if prepared.get('execution_kind') is not None and not diagnostic:
        raise EvidenceError('Unknown shared physical execution source')
    if diagnostic:
        from migration_v1.open_loop_teacher_probe import verify_prepared, verify_source_files
        verify_prepared(prepared)
    # The sole AppLauncher import follows the selected entry's strict source gate.
    from isaaclab.app import AppLauncher
    out = prepared['output']; out.mkdir(parents=True, exist_ok=False)
    for name, digest in prepared['sources'].items():
        source = Path(name)
        relative = source.relative_to(REPO) if source.is_relative_to(REPO) else Path('_dependencies')/source.relative_to(REPO.parent)
        destination = out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        if sha256(destination) != digest:
            raise EvidenceError('Student execution dependency changed before application start')
    write_json(out/('open_loop_source_manifest.json' if diagnostic else 'student_source_manifest.json'), prepared['sources'])
    app = AppLauncher(headless=True, device='cuda:0').app
    env = None
    traces, rows, motor = [], [], []
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.sim import get_current_stage
        from pxr import PhysxSchema, UsdPhysics
        sys.path.insert(0, str(V7/'scripts'))
        import research_common as common
        from rounded_cart import replace_wheels
        from migration_v1.student_recording import recorded_wheel_action_type, attach_recorders
        case, profile = prepared['case'], prepared['profile']
        cfg, agent = common.configure(out, 1, 43, 16, 4, noise=False)
        cfg.sim.device = 'cuda:0'; cfg.fixed_slope_deg = case['slope_deg']
        cfg.fixed_speed_mps = case['speed']; cfg.fixed_payload_kg = 0.; cfg.fixed_towed_kg = 40.
        cfg.eval_distance_m = case['goal_distance_m']
        cfg.episode_length_s = prepared['trial_duration_s']+5. if diagnostic else 45.
        cfg.ground_friction = profile['ground_friction']; cfg.rolling_coefficient = profile['rolling_coefficient']
        cfg.hitch_height_m = profile['hitch_height_m']; cfg.loaded_acceleration = profile['loaded_acceleration_mps2']
        for material in (cfg.scene.terrain.physics_material, cfg.sim.physics_material):
            material.static_friction = material.dynamic_friction = profile['ground_friction']
            material.friction_combine_mode = 'min'
        geometry = replace_wheels(out/'assets/cart.urdf')
        write_json(out/'rounded_cart_geometry.json', geometry)
        cfg.actions.joint_vel.class_type = recorded_wheel_action_type()
        preparation_seconds = prepared['preparation']['preparation_seconds']
        cfg.commands.base_velocity.class_type = fixed_preparation_command_type(preparation_seconds)
        # 原V6状态机只增加固定时钟偏移；不安装EntryVelocityCommand或任何teacher ready。
        env = RslRlVecEnvWrapper(gym.make(common.SPEC.TASK, cfg=cfg), clip_actions=agent.clip_actions)
        base = env.unwrapped
        base.r1_robot_hitch[:,2] += profile['robot_hitch_offset_m']
        if diagnostic:
            sequence = prepared['sequence']
        else:
            policy = prepared['policy']; policy.actor.to('cuda:0').eval().assert_invariants()
        env.seed(43)
        observation, _ = env.reset()
        evaluation = common.launch.module_from_file('_r1_student_runtime_verifier', common.V6/'scripts/worker.py')
        # 原验证器返回tuple维度，落盘数据合同是JSON list；先做同样的无损规范化。
        runtime = serial(evaluation.verify_runtime(env, out))
        r, c = base.scene['robot'], base.scene['trailer']
        initial = dict(robot_root_state=r.data.root_state_w[0], cart_root_state=c.data.root_state_w[0],
            joint_position_action_order=r.data.joint_pos[0,base.r1_joint_ids],
            joint_velocity_action_order=r.data.joint_vel[0,base.r1_joint_ids],
            cart_joint_velocity=c.data.joint_vel[0], env_origin=base.scene.env_origins[0],
            episode_step=int(base.episode_length_buf[0]), cart_mass_kg=float(base.r1_cart_masses[0].sum()),
            robot_tensor_device=str(r.data.root_state_w.device), simulation_device=str(cfg.sim.device),
            command_vx=float(base.command_manager.get_command('base_velocity')[0,0]))
        initial = serial(initial); write_json(out/'initial_state_runtime.json', initial)
        solver, terrain = {}, {}
        for prim in get_current_stage().Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or (str(prim.GetPath()).startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
                solver[str(prim.GetPath())] = {str(x.GetName()): x.Get() for x in prim.GetAttributes() if 'Iteration' in str(x.GetName())}
            if str(prim.GetPath()).startswith(cfg.scene.terrain.prim_path+'/') and prim.HasAPI(UsdPhysics.MaterialAPI):
                material = UsdPhysics.MaterialAPI(prim); px = PhysxSchema.PhysxMaterialAPI(prim)
                terrain[str(prim.GetPath())] = dict(static=material.GetStaticFrictionAttr().Get(),
                    dynamic=material.GetDynamicFrictionAttr().Get(), combine=px.GetFrictionCombineModeAttr().Get())
        materials = dict(assets={name: base.scene[name].root_physx_view.get_material_properties()[:,:,:2].cpu().tolist()
                                for name in ('robot','trailer')}, terrain=terrain)
        write_json(out/'solver_runtime_usd.json', solver); write_json(out/'friction_runtime.json', materials)
        verified = check_runtime(runtime, initial, solver, materials,
            dict(robot=cfg.scene.robot.spawn.asset_path, cart=out/'assets/cart.urdf'), profile, case)
        write_json(out/'student_runtime_review.json', verified)
        request = dict(cases=[case], slope_deg=case['slope_deg'], physical_cart_mass_kg=40., physical_trailer_connected=True,
            trial_duration_s=prepared['trial_duration_s'] if diagnostic else 40.,
            initial_state_mode='ordinary', initial_condition='native ordinary zero-speed reset',
            initial_spring_energy_J=None, rounded_cart_wheels=True, records_drawbar_pre_step=True,
            fixed_preparation_s=preparation_seconds, fixed_preparation_is_candidate=True,
            preparation_protocol=prepared['preparation'], move_start_s=preparation_seconds, seed=43,
            position_iterations=16, velocity_iterations=4, cart_solver_iterations=[16,4],
            noise=False, ppo_updates=0, route_acceptance=False, formal_ppo_authorized=False,
            base_network_contract=common.SPEC.CONTRACT, execution_physics=dict(profile, device='cuda:0'))
        if diagnostic:
            request.update(prepared['diagnostic_metadata'],
                source_teacher_manifest=prepared['source_teacher_manifest'], source_episode=prepared['source_episode'],
                source_sequence_sha256=prepared['source_sequence_sha256'], source_frames=prepared['control_frames'])
        else:
            request.update(artifact_kind='autonomous_student_execution', checkpoint=prepared['checkpoint'],
                checkpoint_sha256=sha256(prepared['checkpoint']), source_student_manifest=prepared['manifest'],
                actor_inference='row', actor_unique_action_source=True, teacher_calls=0, action_override=False)
        write_json(out/'request.json', request)
        recorder = attach_recorders(base, evaluation, common, out, write_json)
        write_json(out/'physics_recorder.json', dict(sample_hz=1000, body_names=r.body_names,
            body_masses=recorder.masses, native_friction=recorder.contact_meta,
            actuator_torque_note='implicit PD requested/applied estimates, not measured final solver drive torque'))
        mapping = {}; latest_raw = {}
        process = base.action_manager.process_action
        def record_native_action(action):
            process(action)
            legs = base.action_manager.get_term('joint_pos'); wheels = base.action_manager.get_term('joint_vel')
            cpu = lambda value: value.detach().cpu().numpy().copy()
            mapping.clear(); mapping.update(actor_raw=cpu(latest_raw['value']), manager_raw=cpu(action),
                leg_target=cpu(legs.processed_actions), wheel_target=cpu(wheels.processed_actions),
                default_legs=cpu(r.data.default_joint_pos[:,legs._joint_ids]), leg_clip=cpu(legs._clip))
            mapping['check'] = check_action_mapping(**mapping, wrapper_clip=env.clip_actions)
        base.action_manager.process_action = record_native_action
        terminated = False; actual_pre_action_move_start = None
        frame_limit = prepared['control_frames'] if diagnostic else 2000
        raw_error_key = 'sequence_to_manager_max_error' if diagnostic else 'actor_to_manager_max_error'
        raw_action_key = 'recorded_teacher_action' if diagnostic else 'actor_action'
        with torch.inference_mode():
            for step in range(frame_limit):
                if not app.is_running():
                    raise RuntimeError('Application closed before declared student window')
                elapsed = float(base.episode_length_buf[0])*base.step_dt
                phase = int(base.r1_eval_phase[0]); command = float(base.command_manager.get_command('base_velocity')[0,0])
                check_pre_action_preparation(elapsed, phase, command, preparation_seconds)
                if phase >= 1 and actual_pre_action_move_start is None:
                    actual_pre_action_move_start = elapsed
                raw = sequence.next_action(step, elapsed, observation.device) if diagnostic else policy(observation)
                latest_raw['value'] = raw.clone()
                frame = dict(observation=observation[0].cpu().numpy().copy(),
                    pre_action_time_s=elapsed,
                    phase=base.r1_eval_phase[0].cpu().numpy().copy(),
                    command_vx=base.command_manager.get_command('base_velocity')[0,0].cpu().numpy().copy())
                frame[raw_action_key] = raw[0].cpu().numpy().copy()
                if diagnostic:
                    frame.update(source_frame_index=step, source_time_s=float(sequence.source_time_s[step]))
                observation, _, done, _ = env.step(raw)
                recorder.capture()
                frame.update(executed_action=mapping['manager_raw'][0], leg_target=mapping['leg_target'][0],
                    wheel_target=mapping['wheel_target'][0],
                    leg_mapping_max_error=mapping['check']['leg_mapping_max_error'],
                    wheel_mapping_max_error=mapping['check']['wheel_mapping_max_error'])
                frame[raw_error_key] = mapping['check']['actor_to_manager_max_error']
                traces.append(frame)
                values = evaluation.cpu_measurements(base.r1_last_metrics)
                rows.append(dict({key: float(value[0]) for key,value in values.items()}, t=(step+1)*.02))
                motor.append({name: getattr(base.r1_torque_recorder,name)[0].cpu().numpy().copy()
                              for name in ('requested','applied','velocity')})
                if bool(done[0]):
                    terminated = True; break
                if (step+1)%100 == 0:
                    if diagnostic:
                        print('R1_OPEN_LOOP_STEP',step+1,'phase',rows[-1]['eval_phase'],'actor_calls',0,flush=True)
                    else:
                        print('R1_STUDENT_STEP',step+1,'phase',rows[-1]['eval_phase'],'teacher_calls',policy.teacher_calls,flush=True)
        prefix = out/'seed43_case00'
        with prefix.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(str(prefix)+'_policy_trace.npz', **{key: np.stack([frame[key] for frame in traces]) for key in traces[0]})
        np.savez_compressed(str(prefix)+'_physics_trace.npz', **{key: np.concatenate([frame[0] for frame in values]) for key,values in recorder.history.items()})
        np.savez_compressed(str(prefix)+'_motor_trace.npz', **{key: np.concatenate([frame[key] for frame in motor]) for key in motor[0]},
            limits=base.r1_motor_limits[0].cpu().numpy(), joint_names=np.asarray(common.SPEC.CONTRACT['action_order']), physics_dt=.001)
        write_json(prefix.with_suffix('.json'), dict(samples=len(rows), complete=len(rows)==frame_limit,
            terminated=terminated, artifact_kind=request['artifact_kind'], success=False))
        if any(sha256(name) != digest for name,digest in prepared['sources'].items()):
            raise EvidenceError('Student runtime dependency changed during execution')
        if diagnostic:
            verify_source_files(prepared)
            if sequence.calls != len(traces):
                raise EvidenceError('Recorded sequence consumption differs from executed control frames')
            replayed = np.stack([frame[raw_action_key] for frame in traces])
            original = sequence.actions[:len(traces)]
            if replayed.dtype != original.dtype or replayed.tobytes() != original.tobytes():
                raise EvidenceError('Executed replay raw action bytes differ from the original sequence')
        elif policy.teacher_calls != 0 or policy.actor_calls != len(traces):
            raise EvidenceError('Student Actor call accounting differs from executed action trace')
        from real_cart_audit import analyze
        review = analyze(out, write_report=False)
        review.update(preparation_protocol=dict(prepared['preparation'],
                actual_pre_action_move_start_s=actual_pre_action_move_start,
                student_clock_runtime_verified=actual_pre_action_move_start is not None),
            runtime_profile_verified=verified['runtime_profile_verified'],
            ordinary_reset_verified=verified['ordinary_reset_verified'],
            action_mapping_review=dict(control_frames=len(traces), actor_calls=0 if diagnostic else policy.actor_calls,
                **{key: max(frame[key] for frame in traces) for key in
                   (raw_error_key,'leg_mapping_max_error','wheel_mapping_max_error')}),
            unified_actor_accepted=False, formal_ppo_authorized=False)
        if diagnostic:
            review.update(prepared['diagnostic_metadata'], open_loop_execution=sequence.execution_record(),
                open_loop_complete_protocol_pass=bool(review['move_brake_hold_pass']),
                source_sequence_sha256=prepared['source_sequence_sha256'], sequence_bytes_preserved=True)
            write_json(out/'open_loop_behavior_review.json', review)
            print('R1_OPEN_LOOP_COMPLETE',review['open_loop_complete_protocol_pass'],'actor_calls',0,flush=True)
        else:
            review.update(artifact_kind='autonomous_student_execution', student_execution=policy.execution_record(),
                actor_unique_action_source=True, teacher_calls=policy.teacher_calls, action_override=False,
                autonomous_anchor_pass=bool(review['move_brake_hold_pass']))
            write_json(out/'student_behavior_review.json', review)
            print('R1_STUDENT_COMPLETE',review['autonomous_anchor_pass'],'teacher_calls',policy.teacher_calls,flush=True)
        return review
    except BaseException as error:
        label = 'open_loop' if diagnostic else 'student'
        write_json(out/(label+'_worker_error.json'), dict(error=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), student_behavior_accepted=False, formal_ppo_authorized=False))
        if traces:
            np.savez_compressed(out/('aborted_'+label+'_policy_trace.npz'), **{key: np.stack([frame[key] for frame in traces]) for key in traces[0]})
        if 'recorder' in locals() and any(recorder.history.values()):
            np.savez_compressed(out/('aborted_'+label+'_physics_trace.npz'), **{key: np.concatenate([frame[0] for frame in values]) for key,values in recorder.history.items() if values})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--slope-deg', type=float, choices=(25., -20.), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preparation-seconds', type=float, default=3.,
        help='Fixed clock from ordinary reset; all teachers for selected slope must match this actual start (default: 3)')
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.checkpoint, args.slope_deg, args.output, args.preparation_seconds)
        result = run_simulation(prepared)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        parser.exit(2, 'Autonomous student execution rejected: '+str(error)+'\n')
    if not result['autonomous_anchor_pass']:
        parser.exit(2, 'Autonomous student did not pass the complete physical protocol.\n')


if __name__ == '__main__':
    main()
