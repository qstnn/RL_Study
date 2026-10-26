"""原牵引点施加沿坡向下的已知负载；拖车停放远处，无PPO。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

import worker
from research_common import ROOT, launch
from controlled_load import LoadSchedule, analyze_run


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--load-levels',type=float,nargs='+',default=[0,50,100,150,173,190,220])
    parser.add_argument('--load-start',type=float,default=5.)
    parser.add_argument('--load-ramp',type=float,default=2.)
    parser.add_argument('--load-hold',type=float,default=5.)
    parser.add_argument('--duration-budget',type=float,choices=(60.,180.),default=60.,
                        help='Explicit longer episode for a continuous staircase; no reset between levels')
    parser.add_argument('--hip-servo-damping',type=float,default=3.,
                        help='Research hip velocity damping at 1kHz; retains original 80Nm/rad gain and 96Nm torque limit')
    parser.add_argument('--hip-servo-stiffness',type=float,default=80.,
                        help='Research hip position gain at 1kHz; original torque limit is unchanged')
    parser.add_argument('--sagittal-servo-stiffness',type=float,default=80.,
                        help='Research thigh/calf stiffness; actual gain is also used to invert force feedforward')
    parser.add_argument('--command-speed',type=float,default=.2,
                        help='Uphill speed command in m/s; diagnostic tracking is scored against this value')
    parser.add_argument('--wheel-controller',choices=('actor','load_allocated','common_feedforward'),default='actor',
                        help='Research-only wheel controller; load allocation uses privileged measured contact forces')
    parser.add_argument('--wheel-control-start',type=float,default=3.,
                        help='Start the one-second allocated-wheel handoff, preserving the Actor prefix')
    parser.add_argument('--front-force-share',type=float,default=None,
                        help='Explicit fixed front axle fraction for common_feedforward; no measured left/right load feedback')
    parser.add_argument('--wheel-heading-gain',type=float,default=0.,
                        help='Explicit lane yaw feedback in wheel targets, with interval-mean yaw rate damping')
    parser.add_argument('--balance-options',type=Path,help='Research-only captured leg reference and bounded attitude feedback JSON')
    parser.add_argument('--force-feedforward-file',type=Path,help='Precomputed research load-to-leg-target calibration, with source evidence')
    parser.add_argument('--balance-angular-rate-sampling',choices=('last','interval_mean'),default='last',
                        help='Use the prior control interval world angular velocity mean, expressed in the current body frame')
    parser.add_argument('--conditioning-towed-kg',type=float,choices=(0.,40.),default=40.,
                        help='Frozen policy task context; the disconnected cart never supplies this physical load')
    selected,remaining=parser.parse_known_args()
    if not np.isfinite(selected.command_speed) or not .1<=selected.command_speed<=.6:
        parser.error('Research speed command must be within [0.1,0.6] m/s')
    if not np.isfinite(selected.wheel_control_start) or not 3<=selected.wheel_control_start<=16:
        parser.error('Wheel control start must be within [3,16] seconds')
    if selected.front_force_share is not None and (selected.wheel_controller!='common_feedforward' or not .1<=selected.front_force_share<=.5):
        parser.error('Fixed front force share requires common_feedforward and [0.1,0.5]')
    if not np.isfinite(selected.wheel_heading_gain) or not 0<=selected.wheel_heading_gain<=8 or (selected.wheel_heading_gain and selected.wheel_controller=='actor'):
        parser.error('Wheel heading gain requires explicit wheel control and [0,8]')
    if not np.isfinite(selected.hip_servo_damping) or not 3<=selected.hip_servo_damping<=20:
        parser.error('Hip servo damping must be within [3,20] Nm/(rad/s)')
    if not np.isfinite(selected.hip_servo_stiffness) or not 80<=selected.hip_servo_stiffness<=320:
        parser.error('Hip servo stiffness must be within [80,320] Nm/rad')
    if not np.isfinite(selected.sagittal_servo_stiffness) or not 80<=selected.sagittal_servo_stiffness<=320:
        parser.error('Sagittal servo stiffness must be within [80,320] Nm/rad')
    if selected.hip_servo_stiffness!=80 and selected.balance_options:
        options=json.loads(selected.balance_options.read_text())
        if options.get('leg_force_gain',0) and options.get('hip_force_scale',1):
            parser.error('Modified hip stiffness requires hip force feedforward disabled')
    if selected.wheel_controller!='actor':
        if any(x in remaining for x in ('--pose-file','--freeze-wheel-baseline')):
            parser.error('Allocated wheel comparison must not combine WBC or a frozen wheel baseline')
        if '--wheel-boost-gain' in remaining and selected.wheel_control_start<16:
            parser.error('Actor wheel boost is only allowed as the prefix of a delayed (16s) allocation handoff')
    schedule=LoadSchedule(tuple(selected.load_levels),selected.load_start,selected.load_ramp,selected.load_hold,selected.duration_budget)
    balance=None
    calibration=json.loads(selected.force_feedforward_file.read_text()) if selected.force_feedforward_file else None
    if calibration and not selected.balance_options:parser.error('Force calibration requires balance options')
    if selected.balance_options:
        from partial_support_control import BalanceOptions, PartialSupportControl
        balance=PartialSupportControl(BalanceOptions(**json.loads(selected.balance_options.read_text())),calibration,
            np.tile([selected.hip_servo_stiffness,selected.sagittal_servo_stiffness,selected.sagittal_servo_stiffness],4))
        first_loaded=next((stage for stage in schedule.stages() if stage['level_N']>0),None)
        if first_loaded is None or balance.options.capture_end_s+balance.options.blend_s>first_loaded['ramp_start_s']:
            parser.error('Balance capture and blend must finish before first nonzero ramp')
        if any(x in remaining for x in ('--pose-file','--forward-pose-bias','--pitch-pose-bias','--calf-load-bias','--front-down-mm')):
            parser.error('Balance comparison must not combine another leg controller')
    if '--seconds' in remaining:parser.error('Duration comes from the load schedule, not --seconds')
    if '--output' not in remaining:parser.error('--output is required')
    out=Path(remaining[remaining.index('--output')+1]).resolve()
    if '--cases' in remaining:parser.error('This diagnostic owns its one +25 degree case')
    # Prepare only the case file beside the output; worker still creates a fresh run directory.
    out.parent.mkdir(parents=True,exist_ok=True)
    case_file=out.parent/(out.name+'_case.json')
    if out.exists() or case_file.exists():parser.error('Use a new output path')
    launch.write_json(case_file,[dict(slope_deg=25.,payload_kg=0.,towed_kg=selected.conditioning_towed_kg,speed=selected.command_speed,goal_distance_m=100.)])
    original_configure=worker.configure
    original_write=launch.write_json
    original_module=launch.module_from_file
    holder={}

    def load_tensor(env,physics=False):
        import torch
        elapsed=env.episode_length_buf.detach().cpu().numpy()*.02
        if physics:
            elapsed+=((env._sim_step_counter-env.r1_load_reset_counter-1)%env.cfg.decimation)*env.physics_dt
        return torch.as_tensor(schedule.value(elapsed),device=env.device,dtype=torch.float32),elapsed

    def configure(*args,**kwargs):
        cfg,agent=original_configure(*args,**kwargs)
        cfg.eval_distance_m=100.;cfg.episode_length_s=max(65.,schedule.duration_s+5.)
        if selected.hip_servo_damping!=3.:
            cfg.scene.robot.actuators['joint'].damping={'.*_hip_joint':selected.hip_servo_damping,
                                                      '.*_thigh_joint':3.,'.*_calf_joint':3.}
        if selected.hip_servo_stiffness!=80. or selected.sagittal_servo_stiffness!=80.:
            cfg.scene.robot.actuators['joint'].stiffness={'.*_hip_joint':selected.hip_servo_stiffness,
                                                        '.*_thigh_joint':selected.sagittal_servo_stiffness,
                                                        '.*_calf_joint':selected.sagittal_servo_stiffness}
        cfg.fixed_towed_kg=selected.conditioning_towed_kg
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v6 import cart,cart_asset,commands
        import torch
        from isaaclab.utils.math import quat_apply
        if selected.conditioning_towed_kg==0.:
            original_command=commands.UnifiedVelocityCommand._update_command
            def update_with_same_speed_protocol(command):
                # Only classify the command as loaded while computing its ramp/phase.
                # Restore real nominal task context before observations or physics.
                env=command._env;context=env.r1_towed_kg.clone()
                try:
                    env.r1_towed_kg.fill_(40.)
                    original_command(command)
                finally:env.r1_towed_kg.copy_(context)
            commands.UnifiedVelocityCommand._update_command=update_with_same_speed_protocol
        reset=cfg.events.reset_base.func
        def reset_without_cart(env,env_ids):
            reset(env,env_ids)
            ids=torch.as_tensor(env_ids,device=env.device)
            trailer=env.scene['trailer'];state=trailer.data.default_root_state[ids].clone()
            state[:,:3]=env.scene.env_origins[ids]+env.r1_normal[ids]*2.
            state[:,1]+=100.;state[:,3:7]=env.scene['robot'].data.root_quat_w[ids];state[:,7:]=0.
            trailer.write_root_pose_to_sim(state[:,:7],env_ids=ids)
            trailer.write_root_velocity_to_sim(state[:,7:],env_ids=ids)
            env.r1_load_reset_counter=env._sim_step_counter
            holder['env']=env
        cfg.events.reset_base.func=reset_without_cart

        def known_load(robot_point,cart_point,robot_velocity,cart_velocity,active,**kwargs):
            env=holder['env'];magnitude,_=load_tensor(env)
            return magnitude[:,None]*env.r1_tangent,torch.ones_like(magnitude),magnitude

        def apply_load(env):
            holder['env']=env
            robot,trailer=env.scene['robot'],env.scene['trailer']
            magnitude,elapsed=load_tensor(env,physics=True)
            point=quat_apply(robot.data.root_quat_w,env.r1_robot_hitch)+robot.data.root_pos_w
            arm=point-robot.data.body_com_pos_w[:,env.r1_base_id]
            force=-magnitude[:,None]*env.r1_tangent
            torque=torch.cross(arm,force,dim=-1)
            robot.set_external_force_and_torque(force[:,None,:],torque[:,None,:],body_ids=[env.r1_base_id],is_global=True)
            # Gravity is disabled by the original trailer asset. No force connects it to the robot.
            zero=torch.zeros((env.num_envs,trailer.num_bodies,3),device=env.device)
            trailer.set_external_force_and_torque(zero,zero,is_global=True)
            env.r1_drawbar_length.fill_(1.);env.r1_drawbar_force[:]=magnitude
            env.r1_drawbar_peak=torch.maximum(env.r1_drawbar_peak,magnitude)
            env.r1_rolling_torque.zero_()
            env.r1_controlled_load=dict(time_s=torch.as_tensor(elapsed,device=env.device,dtype=torch.float64),
                force_w_N=force.clone(),torque_w_Nm=torque.clone(),arm_w_m=arm.clone(),
                buffer_force_w_N=robot._external_force_b[:,env.r1_base_id].clone(),
                buffer_torque_w_Nm=robot._external_torque_b[:,env.r1_base_id].clone(),
                cart_force_norm_N=trailer._external_force_b.abs().sum((1,2)),
                cart_torque_norm_Nm=trailer._external_torque_b.abs().sum((1,2)),
                cart_distance_m=(trailer.data.root_pos_w-robot.data.root_pos_w).norm(dim=-1))
        cart.apply_cart_forces=apply_load
        cart_asset.drawbar_force=known_load
        return cfg,agent

    def module_from_file(name,path):
        module=original_module(name,path)
        if name=='_r1_v7_physical_recorder':
            parent=module.PhysicsRecorder
            class LoadRecorder(parent):
                def __init__(self,base,evaluation_worker):
                    super().__init__(base,evaluation_worker)
                    holder['recorder']=self
                    import torch
                    for key,shape in dict(time_s=(),force_w_N=(3,),torque_w_Nm=(3,),arm_w_m=(3,),
                                          buffer_force_w_N=(3,),buffer_torque_w_Nm=(3,),
                                          cart_force_norm_N=(),cart_torque_norm_Nm=(),cart_distance_m=()).items():
                        self.buffers['controlled_load_'+key]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device,dtype=torch.float64)
                        self.history['controlled_load_'+key]=[]
                    previous=base.r1_torque_recorder.sample
                    def sample_applied():
                        index=base.r1_torque_recorder.cursor;previous()
                        if base.r1_torque_recorder.cursor>index:
                            for key,value in base.r1_controlled_load.items():self.buffers['controlled_load_'+key][:,index]=value
                            self.buffers['applied_drawbar_force_w'][:,index]=base.r1_controlled_load['force_w_N']
                    base.r1_torque_recorder.sample=sample_applied
                    if '--pose-file' not in remaining:
                        from contact_state import record_contact_geometry
                        record_contact_geometry(self,base)
            module.PhysicsRecorder=LoadRecorder
        return module

    def write_json(path,content):
        if Path(path).name=='request.json':
            import os,torch
            content=dict(content,formal_behavior_evaluation=False,controlled_load=asdict(schedule),
                physical_trailer_connected=False,conditioning_towed_kg=selected.conditioning_towed_kg,
                wheel_controller=selected.wheel_controller,
                wheel_control_start_s=selected.wheel_control_start,
                front_force_share=selected.front_force_share,
                wheel_heading_gain=selected.wheel_heading_gain,
                hip_servo_damping_Nms=selected.hip_servo_damping,
                hip_servo_stiffness_Nm_rad=selected.hip_servo_stiffness,
                sagittal_servo_stiffness_Nm_rad=selected.sagittal_servo_stiffness,
                balance_options=asdict(balance.options) if balance else None,
                force_feedforward_calibration=calibration,
                force_feedforward_source_sha256=launch.sha256(selected.force_feedforward_file) if calibration else None,
                balance_angular_rate_sampling=selected.balance_angular_rate_sampling,
                cpu_threads=dict(torch=torch.get_num_threads(),**{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
                protocol='known_backward_load_staircase',duration_s=schedule.duration_s,command_speed_mps=selected.command_speed,
                note='Nominal towing context is an explicit controller input only. Both contexts use the same slow command ramp and continuous-move protocol. Physical trailer is parked with zero force. Known load is applied at the original hitch; no uphill force or root restraint.')
            if selected.wheel_controller!='actor':content['controller']='frozen_actor_legs_'+selected.wheel_controller+'_wheels'
            if balance:content['controller']='captured_leg_reference_with_bounded_attitude_feedback'
        original_write(path,content)
        if Path(path).name=='evaluation.json' and (Path(path).parent/'seed43_case00_physics_trace.npz').exists():
            report=analyze_run(Path(path).parent,schedule)
            print('CONTROLLED_LOAD_RESULT',json.dumps(dict(duration_s=report['duration_s'],
                protocol='legacy_four_contact_tracking',
                load_audit_passed=report['load_audit_passed'],highest_passed_plateau_N=report['highest_passed_plateau_N']),ensure_ascii=False),flush=True)
            from rolling_load_audit import analyze_rolling_run
            rolling=analyze_rolling_run(Path(path).parent,schedule)
            print('ROLLING_LOAD_RESULT',json.dumps(dict(protocol=rolling['protocol']['version'],
                highest_passed_load_N=rolling['highest_passed_load_N'],physical_trailer_connected=False,
                route_acceptance=False),ensure_ascii=False),flush=True)

    def summarize(rows,case,reason):
        return dict(case=case,samples=len(rows),complete=False,success=False,failure_reason=reason,
            protocol='controlled_load_diagnostic',observed=dict(duration_s=len(rows)*.02),
            route_acceptance=False,failed_checks=['not_a_route_acceptance_test'])

    def wheel_transform(base,action):
        import torch
        from wheel_load_control import allocated_targets,heading_velocity_offset
        elapsed=float(base.episode_length_buf[0])*base.step_dt
        trace=dict(allocated_force_N=torch.zeros((1,4),device=base.device),
                   allocated_normal_N=torch.zeros((1,4),device=base.device),
                   wheel_heading_offset_rad_s=torch.zeros((1,4),device=base.device),
                   allocated_active=torch.zeros(1,device=base.device))
        if elapsed<selected.wheel_control_start:return action,trace
        rec=holder['recorder'];normal=base.r1_normal[0].detach().cpu().numpy()
        loads=(rec.buffers['native_ground_normal_w'][0].detach().cpu().numpy()@normal).mean(0)
        previous=holder.get('filtered_normals',loads)
        filtered=previous+(.02/.12)*(loads-previous);holder['filtered_normals']=filtered
        velocity=float((rec.buffers['whole_com_velocity_w'][0,-1]*base.r1_tangent[0]).sum())
        command=float(base.command_manager.get_command('base_velocity')[0,0])
        load=float(schedule.value(elapsed))
        try:
            value=allocated_targets(command,velocity,load,filtered,
                float(rec.buffers['whole_mass_kg'][0,-1]),float(base.r1_slope_rad[0]),base.cfg.ground_friction,
                uniform=selected.wheel_controller=='common_feedforward',front_share=selected.front_force_share)
        except ValueError:
            if selected.wheel_controller=='common_feedforward' and 'previous_wheel_target' in holder:
                action=action.clone();action[0,12:]=torch.as_tensor(holder['previous_wheel_target']/5.,device=base.device,dtype=action.dtype)
            return action,trace
        original=action[0,12:].detach().cpu().numpy()*5.
        if selected.wheel_heading_gain:
            from scipy.spatial.transform import Rotation
            quat=rec.buffers['body_quat_w'][0,-1].detach().cpu().numpy()
            R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            tangent=base.r1_tangent[0].detach().cpu().numpy();side=np.cross(normal,tangent)
            heading=np.arctan2(R[:,0]@side,R[:,0]@tangent)
            yaw_rate=float((rec.buffers['body_angular_velocity_w'][0,:,base.r1_base_id].mean(0)*base.r1_normal[0]).sum())
            y=(rec.buffers['wheel_pos_w'][0,-1]-rec.buffers['whole_com_w'][0,-1]).detach().cpu().numpy()@side
            offset=heading_velocity_offset(heading,yaw_rate,y,selected.wheel_heading_gain)
            value['target_rad_s']=np.clip(value['target_rad_s']+offset,-50.,50.)
            trace['wheel_heading_offset_rad_s']=torch.as_tensor(offset[None],device=base.device)
        u=np.clip(elapsed-selected.wheel_control_start,0.,1.);blend=u*u*(3.-2.*u)
        wanted=original*(1-blend)+value['target_rad_s']*blend
        previous=holder.get('previous_wheel_target',original)
        target=previous+np.clip(wanted-previous,-.1,.1)
        holder['previous_wheel_target']=target
        action=action.clone();action[0,12:]=torch.as_tensor(target/5.,device=base.device,dtype=action.dtype)
        trace.update(allocated_force_N=torch.as_tensor(value['force_N'][None],device=base.device),
                     allocated_normal_N=torch.as_tensor(filtered[None],device=base.device),
                     allocated_active=torch.ones(1,device=base.device))
        return action,trace

    worker.configure=configure;launch.module_from_file=module_from_file;launch.write_json=write_json
    def coordinated_transform(base,action):
        import torch
        trace={}
        if selected.wheel_controller!='actor':action,trace=wheel_transform(base,action)
        if balance:
            rec=holder['recorder'];buffers=rec.buffers
            force=buffers['native_ground_normal_w'][0]+buffers['friction_force_w'][0]
            # 原生合力矩以整机COM为参考，转换到轮心后与轮心Jacobian成对使用。
            moment=buffers['native_contact_moment_com_w'][0]+buffers['native_friction_moment_com_w'][0]
            moment=moment-torch.cross(buffers['wheel_pos_w'][0]-buffers['whole_com_w'][0,:,None,:],force,dim=-1)
            omega=buffers['body_omega_b'][0,-1]
            if selected.balance_angular_rate_sampling=='interval_mean' and buffers['body_quat_w'][0,-1].norm()>1.e-8:
                from isaaclab.utils.math import quat_apply_inverse
                # 先在固定世界坐标中平均，再转回当前机身；避免1kHz接触脉冲
                # 的单点角速度在50Hz姿态D项中混叠为大幅关节目标摆动。
                mean_world=buffers['body_angular_velocity_w'][0,:,base.r1_base_id].mean(0)
                omega=quat_apply_inverse(buffers['body_quat_w'][0,-1],mean_world)
            leg,extra=balance.step(float(base.episode_length_buf[0])*base.step_dt,
                action[0,:12].detach().cpu().numpy(),buffers['joint_pos'][0,-1].detach().cpu().numpy(),
                buffers['body_quat_w'][0,-1].detach().cpu().numpy(),base.r1_normal[0].detach().cpu().numpy(),
                omega.detach().cpu().numpy(),
                force.mean(0).detach().cpu().numpy(),moment.mean(0).detach().cpu().numpy(),
                buffers['joint_velocity'][0,-1].detach().cpu().numpy(),
                known_load_N=float(schedule.value(float(base.episode_length_buf[0])*base.step_dt)))
            action=action.clone();action[0,:12]=torch.as_tensor(leg,device=base.device,dtype=action.dtype)
            trace.update({key:torch.as_tensor(np.asarray(value)[None],device=base.device) for key,value in extra.items()})
        return action,trace
    if selected.wheel_controller!='actor' or balance:worker.action_transform=coordinated_transform
    worker.protocol_v7.max_steps=lambda case:round(schedule.duration_s/.02)
    worker.protocol_v7.finished=lambda rows,case:False
    worker.protocol_v7.summarize=summarize
    original_write(out.parent/(out.name+'_load_plan.json'),dict(schedule=asdict(schedule),stages=schedule.stages(),command_speed_mps=selected.command_speed,wheel_controller=selected.wheel_controller,
        criteria=dict(mean_speed_min_mps=.1,speed_mae_max_mps=.05,four_contact_fraction_min=.95,
                      tilt_max_deg=10.5,abs_heading_max_deg=5.,native_slip_p95_max_mps=.05),
        controller_handoff_s=1.5 if '--pose-file' in remaining else None,ppo_updates=0))
    sys.argv=[sys.argv[0],*remaining,'--cases',str(case_file)]
    worker.main()


if __name__=='__main__':main()
