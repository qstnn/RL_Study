"""单次有界物理对照/教师试验；无PPO，完整保存原16维接口和1kHz物理证据。"""
import argparse,json,sys,traceback
from pathlib import Path
from research_common import *
import protocol_v7
from settings import GROUND_FRICTION
from inference import infer_actor

# Optional isolated research transform; production and ordinary V7 runs leave it unset.
action_transform=None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=BASELINE);p.add_argument('--seed',type=int,default=43)
    p.add_argument('--position-iterations',type=int,default=4);p.add_argument('--velocity-iterations',type=int,default=1)
    p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    p.add_argument('--hitch-height',type=float,default=.07)
    p.add_argument('--robot-hitch-offset','--robot-hitch-rise',dest='robot_hitch_offset',type=float,default=0.,help='Research-only signed robot-end hitch height offset from --hitch-height; cart end unchanged')
    p.add_argument('--ground-friction',type=float,default=GROUND_FRICTION,help='Explicit physical friction; loaded_cart_worker live ordinary control uses the same value; legacy pose-file WBC remains mu=3')
    p.add_argument('--base-com-shift',type=float,default=0.,help='Counterfactual base-link COM shift uphill in metres, with unchanged mass and COM inertia; not the original robot')
    p.add_argument('--loaded-acceleration',type=float,default=.3)
    p.add_argument('--seconds',type=float);p.add_argument('--pose-file',type=Path);p.add_argument('--pose-index',type=int,default=0)
    p.add_argument('--actor-inference',choices=('row','batch'),default='row')
    p.add_argument('--actor-static-command',action='store_true',help='Teacher probe: mask the policy command-history channels while preserving the environment command')
    p.add_argument('--wheel-boost-gain',type=float,default=0.,help='Teacher probe: bounded extra wheel action per m/s of lane speed error after movement begins')
    p.add_argument('--wheel-boost-cap',type=float,default=.20)
    p.add_argument('--wheel-boost-weights',type=float,nargs=4,default=[1.,1.,1.,1.],metavar=('FR','FL','RR','RL'))
    p.add_argument('--freeze-wheel-baseline',action='store_true',help='Teacher probe: hold the 2-3s mean static Actor wheel targets after movement begins')
    p.add_argument('--forward-pose-bias',type=float,default=0.,help='Teacher probe: fraction of V7 x+0.04m IK leg-target difference, smoothly applied at 2-3s')
    p.add_argument('--pitch-pose-bias',type=float,default=0.,help='Teacher probe: fraction of V7 -3deg IK leg-target difference')
    p.add_argument('--calf-load-bias',type=float,default=0.,help='Teacher probe: common positive calf raw-action offset; lowers front wheel centers and raises rear wheel centers near the reference pose')
    p.add_argument('--front-down-mm',type=float,default=0.,help='Teacher probe: signed symmetric front-wheel vertical target in mm, with tangent movement cancelled by local IK Jacobian')
    p.add_argument('--cart-contact-trace',action='store_true',help='Record separate native ground contact/friction view for all cart bodies; diagnostic only')
    p.add_argument('--pose-bias-start',type=float,default=2.)
    p.add_argument('--pose-bias-duration',type=float,default=1.)
    p.add_argument('--stop-on-first-failure',action='store_true',help='Reject candidate at first teacher failure; other cases are aborted, never accepted')
    p.add_argument('--teacher-options',type=Path)
    p.add_argument('--noise',action='store_true');a=p.parse_args()
    teacher_options=json.loads(a.teacher_options.read_text()) if a.teacher_options else {}
    if a.seconds is not None and not 0<a.seconds<=15.:p.error('Short diagnostics must be <=15s')
    if not 0<a.hitch_height<.4 or not 0<a.loaded_acceleration<=.6:
        p.error('Hitch height must be (0, 0.4)m and loaded acceleration (0, 0.6]m/s2')
    if not -.03<=a.robot_hitch_offset<=.2 or not .01<=a.hitch_height+a.robot_hitch_offset<.4:
        p.error('Robot-end hitch offset must be [-0.03, 0.2]m and leave its height in [0.01, 0.4)m')
    if not GROUND_FRICTION<=a.ground_friction<=5. or (a.pose_file and a.ground_friction!=GROUND_FRICTION):
        p.error('Ground friction must be within [V7 base,5], and WBC pose tests require the V7 base value')
    if not 0<=a.base_com_shift<=.12 or (a.pose_file and a.base_com_shift):
        p.error('Base COM shift must be [0,0.12]m; WBC pose tests require the original robot mass model')
    if not 0<=a.wheel_boost_gain<=3. or not 0<a.wheel_boost_cap<=1.:
        p.error('Wheel boost gain must be [0,3] and cap (0,1] raw action')
    if any(not 0<=weight<=1. for weight in a.wheel_boost_weights):
        p.error('Wheel boost weights must be within [0,1]')
    if not 0<=a.forward_pose_bias<=1. or not 0<=a.pitch_pose_bias<=1. or not 0<=a.calf_load_bias<=.5 or not -2.<=a.front_down_mm<=2. or not 0<=a.pose_bias_start<=3. or not .1<=a.pose_bias_duration<=5.:
        p.error('Pose bias fractions must be [0,1], calf bias [0,.5], front vertical target [-2,2]mm, start [0,3]s and duration [0.1,5]s')
    pose_bias_source=Path(__file__).resolve().parents[1]/'results/2026-10-20_mu3_integration_shortcheck/poses.json'
    pose_bias_data=json.loads(pose_bias_source.read_text()) if a.forward_pose_bias or a.pitch_pose_bias else None
    forward_bias_raw=None;pitch_bias_raw=None
    if pose_bias_data is not None:
        selected={p['id']:p for p in pose_bias_data['selected']}
        q0=selected['h0.38_p0_x0.00']['q'][:12]
        q1=selected['h0.38_p0_x0.04']['q'][:12]
        qp=selected['h0.38_p-3_x0.00']['q'][:12]
        forward_bias_raw=[(end-start)/scale for start,end,scale in zip(q0,q1,[.08,.14,.14]*4)]
        pitch_bias_raw=[(end-start)/scale for start,end,scale in zip(q0,qp,[.08,.14,.14]*4)]
    cases=json.loads(a.cases.read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checkpoint=a.checkpoint.resolve(strict=True);saved=SPEC.initialization_checkpoint(checkpoint)
    pose=json.loads(a.pose_file.read_text())['selected'][a.pose_index] if a.pose_file else None
    launch.write_json(out/'request.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=launch.sha256(checkpoint),cases=cases,
        seed=a.seed,noise=a.noise,actor_inference=a.actor_inference,actor_static_command=a.actor_static_command,wheel_boost_gain=a.wheel_boost_gain,wheel_boost_cap=a.wheel_boost_cap,wheel_boost_weights=a.wheel_boost_weights,freeze_wheel_baseline=a.freeze_wheel_baseline,forward_pose_bias=a.forward_pose_bias,forward_pose_bias_raw=forward_bias_raw,pitch_pose_bias=a.pitch_pose_bias,pitch_pose_bias_raw=pitch_bias_raw,calf_load_bias=a.calf_load_bias,front_down_mm=a.front_down_mm,front_down_raw_per_mm=[0.,-.0252,.0449,0.,-.0251,.0449]+[0.]*6,cart_contact_trace=a.cart_contact_trace,pose_bias_start_s=a.pose_bias_start,pose_bias_duration_s=a.pose_bias_duration,pose_bias_source=str(pose_bias_source) if pose_bias_data else None,pose_bias_source_sha256=launch.sha256(pose_bias_source) if pose_bias_data else None,ground_friction=a.ground_friction,position_iterations=a.position_iterations,velocity_iterations=a.velocity_iterations,
        controller='dynamic_wbc_candidate' if pose else 'frozen_baseline',teacher_contact_estimator='geometry_and_20ms_load' if pose else None,pose=pose,seconds=a.seconds,formal_behavior_evaluation=a.seconds is None,
        teacher_options=teacher_options,stop_on_first_failure=a.stop_on_first_failure,ppo_updates=0,source_contract=saved['infos']['r1_unified_contract'],base_network_contract=SPEC.CONTRACT,execution_physics=dict(device=a.device,hitch_height_m=a.hitch_height,robot_hitch_offset_m=a.robot_hitch_offset,robot_hitch_nominal_height_m=a.hitch_height+a.robot_hitch_offset,cart_hitch_nominal_height_m=a.hitch_height,loaded_acceleration_mps2=a.loaded_acceleration,ground_friction=a.ground_friction,friction_combine="min",rolling_coefficient=.02,base_com_shift_m=a.base_com_shift,robot_mass_variant=a.base_com_shift>0)))
    snapshot(out)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=a.device).app;env=None
    try:
        import gymnasium as gym,numpy as np,torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        cfg,agent=configure(out,len(cases),a.seed,a.position_iterations,a.velocity_iterations,a.noise)
        cfg.sim.device=a.device;cfg.hitch_height_m=a.hitch_height;cfg.loaded_acceleration=a.loaded_acceleration;cfg.ground_friction=a.ground_friction
        for material in (cfg.scene.terrain.physics_material,cfg.sim.physics_material):
            material.static_friction=material.dynamic_friction=a.ground_friction
        if a.base_com_shift:
            import xml.etree.ElementTree as ET
            source_urdf=Path(cfg.scene.robot.spawn.asset_path).resolve(strict=True)
            variant=out/'assets/lighthw_base_com_shift.urdf'
            tree=ET.parse(source_urdf);root=tree.getroot()
            origin=root.find("./link[@name='Base_link']/inertial/origin")
            if origin is None:raise RuntimeError('Base-link inertial origin missing')
            xyz=[float(x) for x in origin.get('xyz').split()];original_xyz=xyz.copy();xyz[0]+=a.base_com_shift
            origin.set('xyz',' '.join(str(x) for x in xyz))
            for mesh in root.findall('.//mesh'):
                mesh.set('filename',str((source_urdf.parent/mesh.get('filename')).resolve(strict=True)))
            tree.write(variant,encoding='unicode',xml_declaration=True)
            cfg.scene.robot.spawn.asset_path=str(variant)
            base_mass = float(root.find("./link[@name='Base_link']/inertial/mass").get('value'))
            whole_mass = sum(float(m.get('value')) for m in root.findall('./link/inertial/mass'))
            launch.write_json(out/'mass_variant.json',dict(original_urdf=str(source_urdf),original_sha256=launch.sha256(source_urdf),variant_urdf=str(variant),variant_sha256=launch.sha256(variant),link='Base_link',base_mass_kg=base_mass,original_inertial_xyz_m=original_xyz,variant_inertial_xyz_m=xyz,whole_robot_mass_kg=whole_mass,approx_whole_com_shift_m=base_mass/whole_mass*a.base_com_shift,interpretation='Hypothetical internal mass relocation; original robot and hardware not validated'))
        dump_yaml(str(out/'params/env.yaml'),cfg);dump_yaml(str(out/'params/agent.yaml'),agent)
        env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions);base=env.unwrapped
        if a.robot_hitch_offset:
            # Counterfactual geometry only: move the robot end while the passive cart end stays fixed.
            # The real spring/collision/contact solver remains active; no extra drive or root force is added.
            base.r1_robot_hitch[:,2]+=a.robot_hitch_offset
        launch.write_json(out/'hitch_geometry_runtime.json',dict(
            robot_local_xyz_m=base.r1_robot_hitch[0].detach().cpu().tolist(),
            cart_local_xyz_m=base.r1_cart_hitch[0].detach().cpu().tolist(),
            robot_end_nominal_height_m=a.hitch_height+a.robot_hitch_offset,
            cart_end_nominal_height_m=a.hitch_height,
            robot_end_offset_m=a.robot_hitch_offset,
            spring_rest_length_m=1.,
            interpretation='Research-only asymmetric hitch; passive 40kg trailer and original robot mass/motor limits'))
        evaluation_worker=launch.module_from_file('_r1_v7_eval_worker',V6/'scripts/worker.py')
        def serial(value):
            if isinstance(value,(np.ndarray,torch.Tensor)):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [serial(v) for v in value]
            return value
        evaluation_worker.write_json=lambda path,value:launch.write_json(path,serial(value))
        evaluation_worker.protocol=protocol_v7
        protocol_v7.cases=lambda distance:cases
        if a.seconds is not None:protocol_v7.max_steps=lambda case:round(a.seconds/.02)
        oldmodule=launch.module_from_file('_r1_v7_physical_recorder',OLD/'coordinated_control.py')
        physical=oldmodule.PhysicsRecorder(base,evaluation_worker)
        # 记录被动拖车的轮速与显式滚阻，区分真实拖行、轮锁止和弹簧伸长。
        cart=base.scene['trailer']
        cart_wheels=cart.find_bodies([f'cart_wheel_{i}' for i in range(4)],preserve_order=True)[0]
        cart_joints=cart.find_joints([f'cart_wheel_{i}_joint' for i in range(4)],preserve_order=True)[0]
        cart_shapes=dict(cart_pos_w=(3,),cart_velocity_w=(3,),cart_wheel_center_velocity_w=(4,3),
            cart_wheel_angular_velocity_w=(4,3),cart_joint_velocity=(4,),cart_rolling_torque_Nm=(4,))
        for key,shape in cart_shapes.items():
            physical.buffers[key]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device);physical.history[key]=[]
        sample=physical.sample
        def physical_sample():
            index=physical.cursor;sample()
            physical.buffers['cart_pos_w'][:,index]=cart.data.root_pos_w
            physical.buffers['cart_velocity_w'][:,index]=cart.data.root_com_lin_vel_w
            physical.buffers['cart_wheel_center_velocity_w'][:,index]=cart.data.body_lin_vel_w[:,cart_wheels]
            physical.buffers['cart_wheel_angular_velocity_w'][:,index]=cart.data.body_ang_vel_w[:,cart_wheels]
            physical.buffers['cart_joint_velocity'][:,index]=cart.data.joint_vel[:,cart_joints]
            physical.buffers['cart_rolling_torque_Nm'][:,index]=base.r1_rolling_torque
        physical.sample=physical_sample
        if a.cart_contact_trace:
            cart_sensor=base.scene['trailer_contacts']
            all_paths=list(cart_sensor.body_physx_view.prim_paths)
            names=[f'cart_wheel_{i}' for i in range(4)]+['cart_base']
            requested=[next(path for path in all_paths if f'/env_{env_i}/' in path and path.endswith('/'+name))
                for env_i in range(base.num_envs) for name in names]
            if len(set(requested))!=len(requested):raise RuntimeError('Duplicate cart native contact sensor path')
            ground=physical.contact_meta['ground_filters']
            cart_view=cart_sensor._physics_sim_view.create_rigid_contact_view(requested,
                filter_patterns=[ground]*len(requested),max_contact_data_count=len(requested)*128)
            actual=list(cart_view.sensor_paths)
            if len(actual)!=len(requested) or set(actual)!=set(requested):
                raise RuntimeError('Cart native contact view sensor order mismatch')
            order=torch.tensor([actual.index(path) for path in requested],device=base.device)
            offsets=torch.arange(128,device=base.device)
            for name,shape in dict(cart_ground_friction_w=(5,3),cart_ground_normal_w=(5,3),
                cart_ground_point_count=(5,),cart_ground_min_separation_m=(5,)).items():
                physical.buffers[name]=torch.zeros((base.num_envs,base.cfg.decimation,*shape),device=base.device)
                physical.history[name]=[]
            launch.write_json(out/'cart_contact_view.json',dict(requested_paths=requested,actual_paths=actual,
                ground_filters=ground,capacity=cart_view.max_contact_data_count,body_order=names))
            previous_sample=physical.sample
            def sample_with_cart_contact():
                index=physical.cursor;previous_sample()
                friction,_,counts,starts=cart_view.get_friction_data(dt=base.physics_dt)
                counts=counts.long();starts=starts.long()
                if (counts>len(offsets)).any():raise RuntimeError('Cart friction contact patches exceed buffer')
                valid=offsets[None,None,:]<counts[:,:,None]
                ix=(starts[:,:,None]+offsets[None,None,:]).clamp(0,len(friction)-1)
                tangential=torch.where(valid[:,:,:,None],friction[ix],0.).sum((1,2))
                normal=cart_view.get_contact_force_matrix(dt=base.physics_dt).sum(1)
                force,points,normals,separation,point_counts,point_starts=cart_view.get_contact_data(dt=base.physics_dt)
                point_counts=point_counts.long();point_starts=point_starts.long()
                if (point_counts>len(offsets)).any():raise RuntimeError('Cart native contact points exceed buffer')
                point_valid=offsets[None,None,:]<point_counts[:,:,None]
                point_ix=(point_starts[:,:,None]+offsets[None,None,:]).clamp(0,len(separation)-1)
                min_sep=torch.where(point_valid,separation[point_ix,0],1000.).flatten(1).min(1).values
                for key,value in dict(cart_ground_friction_w=tangential,cart_ground_normal_w=normal,
                    cart_ground_point_count=point_counts.sum(1),cart_ground_min_separation_m=min_sep).items():
                    physical.buffers[key][:,index]=value[order].reshape(base.num_envs,5,*value.shape[1:])
            physical.sample=sample_with_cart_contact
        from controller import Teacher
        if pose:
            from contact_state import record_contact_geometry
            record_contact_geometry(physical,base)
        teacher=Teacher(base,pose,physical,teacher_options) if pose else None
        if teacher:
            protocol_v7.finished=protocol_v7.track_completion(protocol_v7.finished,cases,teacher.closed)
        from pxr import PhysxSchema,UsdPhysics
        from isaaclab.sim import get_current_stage
        solver={}
        for prim in get_current_stage().Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or (str(prim.GetPath()).startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
                solver[str(prim.GetPath())]={str(x.GetName()):x.Get() for x in prim.GetAttributes() if 'Iteration' in str(x.GetName())}
        launch.write_json(out/'solver_runtime_usd.json',solver)
        bodies={}
        for prim in get_current_stage().Traverse():
            if str(prim.GetPath()).startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api=PhysxSchema.PhysxRigidBodyAPI(prim)
                bodies[str(prim.GetPath())]=dict(max_angular_velocity_deg_s=api.GetMaxAngularVelocityAttr().Get(),
                    enable_gyroscopic_forces=api.GetEnableGyroscopicForcesAttr().Get())
        launch.write_json(out/'body_settings_runtime.json',bodies)
        original_verify=evaluation_worker.verify_runtime
        def verify_with_friction(env,run):
            original_verify(env,run)
            materials={}
            for name in ('robot','trailer'):
                values=base.scene[name].root_physx_view.get_material_properties()[:,:,:2]
                if not torch.allclose(values,torch.full_like(values,a.ground_friction),atol=1.e-6,rtol=0.):
                    raise RuntimeError('Actual '+name+' friction differs from V7 setting')
                materials[name]=dict(static_unique=values[:,:,0].unique().tolist(),dynamic_unique=values[:,:,1].unique().tolist())
            terrain={}
            for prim in get_current_stage().Traverse():
                if str(prim.GetPath()).startswith(cfg.scene.terrain.prim_path+'/') and prim.HasAPI(UsdPhysics.MaterialAPI):
                    api=UsdPhysics.MaterialAPI(prim);physx=PhysxSchema.PhysxMaterialAPI(prim)
                    values=dict(static=api.GetStaticFrictionAttr().Get(),dynamic=api.GetDynamicFrictionAttr().Get(),combine=physx.GetFrictionCombineModeAttr().Get())
                    terrain[str(prim.GetPath())]=values
                    if values['static']!=a.ground_friction or values['dynamic']!=a.ground_friction or values['combine']!='min':
                        raise RuntimeError('Actual terrain friction/combine differs from V7 setting: '+str(values))
            if not terrain:raise RuntimeError('No authored terrain friction material found')
            launch.write_json(run/'friction_runtime.json',dict(expected=a.ground_friction,assets=materials,terrain=terrain,passed=True))
        evaluation_worker.verify_runtime=verify_with_friction

        actor=torch.nn.Sequential(torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16)).to(env.device).eval().requires_grad_(False)
        actor.load_state_dict({k[6:]:v for k,v in saved['model_state_dict'].items() if k.startswith('actor.')})
        traces={};diagnostics=[];counter=0;aborted=set()
        wheel_baseline_sum=torch.zeros((base.num_envs,4),device=base.device)
        wheel_baseline_count=torch.zeros((base.num_envs,1),device=base.device)
        leg_bias=(torch.tensor(forward_bias_raw,device=base.device)*a.forward_pose_bias+torch.tensor(pitch_bias_raw,device=base.device)*a.pitch_pose_bias) if pose_bias_data is not None else (torch.zeros(12,device=base.device) if a.calf_load_bias or a.front_down_mm else None)
        if leg_bias is not None:
            leg_bias[[2,5,8,11]]+=a.calf_load_bias
            leg_bias+=a.front_down_mm*torch.tensor([0.,-.0252,.0449,0.,-.0251,.0449]+[0.]*6,device=base.device)
        class Runner:
            def load(self,path,load_optimizer=False):
                if load_optimizer or Path(path).resolve()!=checkpoint:raise ValueError('Unexpected checkpoint')
            def get_inference_policy(self,device=None):
                def infer(obs):
                    nonlocal counter
                    policy_obs=obs.clone() if a.actor_static_command else obs
                    if a.actor_static_command:
                        policy_obs[:,30:45]=0.
                    original=infer_actor(actor,policy_obs,a.actor_inference);action=teacher.apply(original,counter) if teacher else original.clone()
                    elapsed=base.episode_length_buf*base.step_dt
                    if a.freeze_wheel_baseline:
                        gathering=(elapsed>=2.)&(elapsed<3.)
                        wheel_baseline_sum[gathering]+=original[gathering,12:]
                        wheel_baseline_count[gathering]+=1
                        held=(elapsed>=3.)&(wheel_baseline_count[:,0]>0)
                        action[held,12:]=wheel_baseline_sum[held]/wheel_baseline_count[held]
                    if leg_bias is not None:
                        fraction=((elapsed-a.pose_bias_start)/a.pose_bias_duration).clamp(0.,1.)
                        smooth=fraction*fraction*(3.-2.*fraction)
                        action[:,:12]+=smooth[:,None]*leg_bias[None,:]
                    if a.wheel_boost_gain:
                        commanded=base.command_manager.get_command('base_velocity')[:,0]
                        lane_speed=(base.scene['robot'].data.root_com_lin_vel_w*base.r1_tangent).sum(-1)
                        boost=(a.wheel_boost_gain*(commanded-lane_speed)).clamp(-a.wheel_boost_cap,a.wheel_boost_cap)
                        enabled=torch.tensor([case['speed']>0 for case in cases],device=base.device)&(elapsed>=3.)
                        weights=torch.as_tensor(a.wheel_boost_weights,device=base.device,dtype=action.dtype)
                        action[enabled,12:]+=boost[enabled,None]*weights
                    transform_trace={}
                    if action_transform is not None:
                        action,transform_trace=action_transform(base,action)
                    values=dict(observation=obs,executed_action=action,actor_action=original,phase=base.r1_eval_phase,
                        command_vx=base.command_manager.get_command('base_velocity')[:,0])
                    values.update(transform_trace)
                    for key,value in evaluation_worker.cpu_measurements(values).items():traces.setdefault(key,[]).append(value.copy())
                    if teacher:
                        diagnostics.append({str(i):serial(v) for i,v in teacher.last.items()})
                    counter+=1
                    if counter%100==0:
                        failed={i:dict(reason=v['reason'],time_s=v['time_s']) for i,v in teacher.failures.items()} if teacher else {}
                        print('R1_RESEARCH_STEP',counter,'progress_m',base.r1_eval_progress.detach().cpu().tolist(),'failures',failed,flush=True)
                    return action
                return infer
        step=env.step;cart_origins=torch.zeros((base.num_envs,3),device=base.device);anchored=torch.zeros(base.num_envs,dtype=torch.bool,device=base.device)
        def recorded_step(action):
            result=step(action);physical.capture()
            phase=base.r1_last_metrics['eval_phase'];start=(phase>=1)&~anchored
            current=physical.buffers['cart_pos_w'][:,-1]
            cart_origins[start]=current[start];anchored[start]=True
            progress=((current-cart_origins)*base.r1_tangent).sum(-1)
            base.r1_last_metrics['cart_progress_m']=torch.where(anchored,progress,0.)
            if teacher and teacher.failures:
                failed=torch.tensor(list(teacher.failures),device=base.device,dtype=torch.long)
                result[2][failed]=1  # 此候选结束并保留失败原因；不将回退Actor记为成功教师。
                if a.stop_on_first_failure:
                    aborted.update(set(range(base.num_envs))-set(teacher.failures)-teacher.closed)
                    result[2][:]=1
            if teacher:
                teacher.closed.update(result[2].nonzero().flatten().cpu().tolist())
            return result
        env.step=recorded_step
        args=dict(mode='evaluate',eval_distance=1.,eval_seeds=[a.seed],seed=a.seed,checkpoint=str(checkpoint),command_profile='slew',headless=True)
        with torch.inference_mode():evaluation_worker.evaluate(env,Runner(),args,out,app)
        report=json.loads((out/'evaluation.json').read_text())
        traces={k:np.stack(v) for k,v in traces.items()}
        physics={k:np.stack(v) for k,v in physical.history.items()}
        for i,case in enumerate(report['cases']):
            n=case['samples'];prefix=f'seed{a.seed}_case{i:02d}'
            np.savez_compressed(out/(prefix+'_policy_trace.npz'),**{k:v[:n,i] for k,v in traces.items()})
            np.savez_compressed(out/(prefix+'_physics_trace.npz'),**{k:v[:n,i].reshape(n*base.cfg.decimation,*v.shape[3:]) for k,v in physics.items()})
            if teacher and i in teacher.failures:
                case['success']=False;case['teacher_failure']=serial(teacher.failures[i]);case.setdefault('failed_checks',[]).append('teacher_controller')
            if i in aborted:
                case['success']=False;case['complete']=False;case['candidate_aborted']=True
                case.setdefault('failed_checks',[]).append('candidate_aborted')
            if a.seconds is not None:case['success']=False;case['short_diagnostic']=True
            launch.write_json(out/(prefix+'.json'),case)
        report.update(pass_count=sum(r['success'] for r in report['cases']),success_rate=sum(r['success'] for r in report['cases'])/len(cases),
            aborted_cases=sorted(aborted),short_diagnostic=a.seconds is not None,teacher=bool(teacher),teacher_failures=serial(teacher.failures) if teacher else {})
        launch.write_json(out/'evaluation.json',report)
        launch.write_json(out/'dynamics_crosscheck.json',serial(teacher.audit) if teacher else {})
        launch.write_json(out/'qp_trace.json',diagnostics)
        launch.write_json(out/'physics_recorder.json',dict(sample_hz=1000,body_names=base.scene['robot'].body_names,
            body_masses=physical.masses.cpu().tolist(),native_friction=physical.contact_meta,normal_sensor_only=True,
            generalized_velocity='base_COM_world_v, base_world_omega, action_order_joint_qd',
            native_mass_excludes_armature=True,teacher_control_hz=50,physics_hz=1000))
        print('R1_RESEARCH_COMPLETE',report['pass_count'],'/',len(cases),'short',bool(a.seconds),flush=True)
    except BaseException as exc:
        launch.write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        # 控制保护异常仍失败退出；只保存诊断流，绝不生成evaluation或成功标签。
        # 与完整档案使用不同文件名，避免将部分运行交给迁移数据读取器。
        try:
            diagnostic=dict(complete=False,success=False,teacher_labels_allowed=False,
                reason='aborted streams for failure diagnosis only',files=[],
                alignment_note='A policy item is an inference request; only items with a completed matching physics step were executed. An env.step exception may leave one extra policy item.')
            if 'traces' in locals() and isinstance(traces,dict):
                arrays={k:np.stack(v) for k,v in traces.items() if isinstance(v,list) and v}
                if arrays:
                    np.savez_compressed(out/'aborted_policy_trace.npz',**arrays)
                    diagnostic['files'].append('aborted_policy_trace.npz')
                    diagnostic['policy_steps']={k:len(v) for k,v in arrays.items()}
            if 'physical' in locals():
                arrays={k:np.stack(v) for k,v in physical.history.items() if v}
                if arrays:
                    np.savez_compressed(out/'aborted_physics_trace.npz',**arrays)
                    diagnostic['files'].append('aborted_physics_trace.npz')
                    diagnostic['physics_steps']={k:len(v) for k,v in arrays.items()}
            launch.write_json(out/'aborted_trace_manifest.json',diagnostic)
        except BaseException as diagnostic_exc:
            launch.write_json(out/'aborted_trace_error.json',dict(type=type(diagnostic_exc).__name__,message=str(diagnostic_exc)))
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        import os
        os._exit(1)
    finally:
        if env is not None:env.close()
        app.close()

if __name__=='__main__':main()
