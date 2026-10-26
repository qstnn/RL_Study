"""诊断专用：静力补偿、牵引高度、PD 和初始预拉伸的配对对照；不作为部署控制器。"""
from pathlib import Path
import argparse,csv,json,sys,traceback

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/user/RL_lab/rl_training')
V2=ROOT.parent/'unified_v3'
sys.path.insert(0,str(V2/'scripts'));sys.path.insert(0,str(REPO/'source/rl_training'))
from launch import SPEC,CART,module_from_file,now,write_json,sha256

parser=argparse.ArgumentParser();parser.add_argument('--duration',type=float,default=8.)
parser.add_argument('--output',type=Path,required=True);parser.add_argument('--static_teacher',action='store_true');args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
cases=[dict(controller='encoder_integral',kp=kp,kd=kd,hitch=.10,slope_deg=25.,payload_kg=0.,towed_kg=40.,preload=False,
            friction=2.3,ground_spawn=True,leg_ki=li,wheel_ki=wi)
       for kp,kd in ((80.,3.),(160.,6.)) for li in (0.,100.,300.) for wi in (0.,10.,30.)]
models={}
args.static_teacher=False
write_json(out/'request.json',dict(created_at=now(),cases=cases,duration_s=args.duration,seed=43,ppo_updates=0,
    checkpoint_sha256={k:sha256(p) for k,p in models.items()},caps_Nm=[96,48],friction=2.3,rolling=.02,
    note='Default-pose PD is a capability probe, not a trained policy or static-position guarantee.'))
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app;env=None
try:
    import numpy as np
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.math import quat_apply
    from rsl_rl.runners import OnPolicyRunner
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v3 import mdp,core,torque
    worker=module_from_file('_capability_v2_worker',V2/'scripts/worker.py')
    original=mdp.measurements
    def measured(base,command=None):
        result=original(base,command);robot=base.scene['robot'];cart=base.scene['trailer']
        for name,asset,offset in (('robot_hitch_m',robot,base.r1_robot_hitch),('cart_hitch_m',cart,base.r1_cart_hitch)):
            point=quat_apply(asset.data.root_quat_w,offset)+asset.data.root_pos_w
            result[name]=((point-base.scene.env_origins)*base.r1_normal).sum(-1)
        ids=base.r1_joint_ids[:12]
        ids_w=base.r1_joint_ids[12:]
        for j,idx in enumerate(ids):result['q'+str(j)]=robot.data.joint_pos[:,idx]
        for j,idx in enumerate(ids_w):result['w'+str(j)]=robot.data.joint_vel[:,idx]
        contact=base.scene['contact_forces'].data.net_forces_w[:,base.r1_wheel_sensor_ids]
        norms=(contact*base.r1_normal[:,None,:]).sum(-1)
        along=(contact*base.r1_tangent[:,None,:]).sum(-1)
        for j in range(4):
            result['normal'+str(j)]=norms[:,j];result['traction'+str(j)]=along[:,j]
        result['leg_error_max_rad']=(robot.data.joint_pos_target[:,ids]-robot.data.joint_pos[:,ids]).abs().amax(-1)
        return result
    mdp.measurements=measured
    SPEC.register_task();cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=len(cases);cfg.seed=agent.seed=43;cfg.fixed_speed_mps=0.;cfg.episode_length_s=args.duration+5
    cfg.observations.policy.enable_corruption=False
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.wheel_rigid_cap_env_cfg import spawn_lighthw_wheel_rigid_cap
    cfg.scene.robot.spawn.func=spawn_lighthw_wheel_rigid_cap
    cfg.scene.robot.actuators['wheel'].damping=1.
    cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
    CART.write_cart_urdf(out/'cart.urdf');cfg.scene.trailer.spawn.asset_path=str(out/'cart.urdf')
    cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
    env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions);base=env.unwrapped
    device=env.device;robot=base.scene['robot']
    base.r1_slope_rad[:]=torch.tensor([c['slope_deg']*torch.pi/180 for c in cases],device=device)
    base.r1_tangent[:],base.r1_normal[:]=core.plane_basis(base.r1_slope_rad)
    for key in ('payload_kg','towed_kg'):getattr(base,'r1_'+key)[:]=torch.tensor([c[key] for c in cases],device=device)
    lanes=torch.tensor([SPEC.ANGLES.index(c['slope_deg']) for c in cases],device=device)
    base.scene.terrain.terrain_types[:]=lanes;base.scene.terrain.env_origins[:]=base.scene.terrain.terrain_origins[0,lanes]
    h=torch.tensor([c['hitch'] for c in cases],device=device)
    base.r1_robot_hitch[:,2]=h-.43;base.r1_cart_hitch[:,2]=h-.30
    actuator=robot.actuators['joint']
    actuator.stiffness[:]=torch.tensor([c['kp'] for c in cases],device=device)[:,None]
    actuator.damping[:]=torch.tensor([c['kd'] for c in cases],device=device)[:,None]
    mdp.set_payload_properties(base)
    with torch.inference_mode():obs,_=env.reset()
    teacher=None
    if args.static_teacher:
        sys.path.insert(0,str(V2/'diagnostics'))
        from static_support import nominal_support
        static=[nominal_support(REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf',c['slope_deg'],c['hitch']) for c in cases]
        write_json(out/'static_support.json',static)
        teacher=torch.tensor([c['torques_Nm'] for c in static],device=device)
        weights=torch.tensor([c['wheel_normals_N'] for c in static],device=device).clamp_min(0)
        weights/=weights.sum(-1,keepdim=True)
        extension=torch.tensor([s['preload_extension_m'] if c['preload'] else 0. for s,c in zip(static,cases)],device=device)
        cart=base.scene['trailer'];pose=cart.data.root_state_w[:,:7].clone();pose[:,:3]-=base.r1_tangent*extension[:,None]
        with torch.inference_mode():cart.write_root_pose_to_sim(pose)
        origin=robot.data.root_pos_w.clone()
    if True:
        state=robot.data.root_state_w.clone()
        for i,c in enumerate(cases):
            if c['ground_spawn']:state[i,:3]-=base.r1_normal[i]*(.43-.416223)
        with torch.inference_mode():robot.write_root_state_to_sim(state)
        for asset in (robot,base.scene['trailer']):
            materials=asset.root_physx_view.get_material_properties()
            materials[:,:,:2]=torch.tensor([c['friction'] for c in cases])[:,None,None]
            asset.root_physx_view.set_material_properties(materials,torch.arange(len(cases)))
        # Min combine requires matching ground material; in this diagnostic use max only
        # on the ground so each per-environment tire coefficient determines friction.
        import isaaclab.sim as sim_utils
        from pxr import Usd,UsdPhysics,PhysxSchema
        for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath('/World/ground')):
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                PhysxSchema.PhysxMaterialAPI(prim).CreateFrictionCombineModeAttr().Set('min')
        wid=robot.find_bodies(['FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link'],preserve_order=True)[0]
        np.savez_compressed(out/'initial_dynamics.npz',
           jacobian=robot.root_physx_view.get_jacobians().cpu().numpy(),gravity=robot.root_physx_view.get_gravity_compensation_forces().cpu().numpy(),
           joint_ids=np.array(base.r1_joint_ids),wheel_ids=np.array(wid),body_names=np.array(robot.body_names),
           body_positions=robot.data.body_pos_w.cpu().numpy(),root_state=robot.data.root_state_w.cpu().numpy())
    import types
    for name,key in (('joint','leg_ki'),('wheel','wheel_ki')):
        act=robot.actuators[name];act.integral=torch.zeros_like(act.stiffness)
        act.gain=torch.tensor([c[key] for c in cases],device=device)[:,None]
        original_compute=act.compute;original_reset=act.reset
        def wrapped(self,control_action,joint_pos,joint_vel,original_compute=original_compute,is_leg=name=='joint'):
            error=(control_action.joint_positions-joint_pos) if is_leg else (control_action.joint_velocities-joint_vel)
            delta=self.gain*error*.001
            result=original_compute(control_action,joint_pos,joint_vel)
            proposed=(self.integral+delta).clamp(-self.effort_limit,self.effort_limit)
            demanded=self.computed_effort+proposed
            saturated=(demanded.abs()>self.effort_limit)&(delta*demanded>0)
            self.integral=torch.where(saturated,self.integral,proposed)
            self.computed_effort=self.computed_effort+self.integral
            self.applied_effort=self._clip_effort(self.computed_effort)
            result.joint_efforts=self.applied_effort
            return result
        def reset(self,env_ids,original_reset=original_reset):
            original_reset(env_ids);self.integral[env_ids]=0.
        act.compute=types.MethodType(wrapped,act);act.reset=types.MethodType(reset,act)
    policies={}
    for mode,path in models.items():
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=device)
        runner.load(str(path),load_optimizer=False);policies[mode]=runner.get_inference_policy(device=device)
    selected={mode:torch.tensor([c['controller']==mode for c in cases],device=device) for mode in models}
    active=torch.ones(len(cases),dtype=torch.bool,device=device);rows=[[] for _ in cases];frames=[[] for _ in cases]
    with torch.inference_mode():
        for step in range(round(args.duration/.02)):
            action=torch.zeros(len(cases),16,device=device)
            for mode,policy in policies.items():
                if selected[mode].any():action[selected[mode]]=policy(obs[selected[mode]])
            if teacher is not None:
                scale=torch.tensor([.08,.14,.14]*4,device=device)
                kp=torch.tensor([c['kp'] for c in cases],device=device)
                action[:,:12]=teacher[:,:12]/kp[:,None]/scale
                velocity=(robot.data.root_lin_vel_w*base.r1_tangent).sum(-1)
                displacement=((robot.data.root_pos_w-origin)*base.r1_tangent).sum(-1)
                correction=(-80.*velocity-20.*displacement).clamp(-30.,30.)
                action[:,12:]=(teacher[:,12:]+correction[:,None]*weights*.0875)/1./5.
            assert torch.isfinite(action).all();action[~active]=0.
            obs,_,done,_=env.step(action)
            keys=list(base.r1_last_metrics);values=torch.stack(list(base.r1_last_metrics.values()),-1).cpu().tolist()
            trace={k:getattr(base.r1_torque_recorder,k).cpu().numpy() for k in ('applied','requested','velocity')}
            for i,alive in enumerate(active.cpu().tolist()):
                if not alive:continue
                rows[i].append(dict(t=(step+1)*.02,**dict(zip(keys,values[i]))))
                frames[i].append({k:v[i].copy() for k,v in trace.items()})
                if done[i]:active[i]=False
            if step%100==0:print('CAPABILITY',step,'alive',int(active.sum()),flush=True)
            if not active.any():break
    reports=[]
    for i,(case,data) in enumerate(zip(cases,rows)):
        last=data[-1];hold=[r for r in data if r['t']>args.duration-3.]
        mean=lambda key:float(np.mean([r[key] for r in hold])) if hold else None
        trace={k:np.concatenate([f[k] for f in frames[i]]) for k in frames[i][0]};limits=base.r1_motor_limits[i].cpu().numpy()
        assert np.all(abs(trace['applied'])<=limits+1e-4)
        survived=len(data)==round(args.duration/.02) and not any(r['terminated'] for r in data)
        report=dict(index=i,case=case,duration_s=len(data)*.02,survived=survived,
            terminal_causes=[k for k in last if k.startswith('termination_') and last[k]],
            min_robot_hitch_m=min(r['robot_hitch_m'] for r in data),final_height=last['height'],peak_tilt_deg=max(r['tilt_deg'] for r in data),
            final_hitch_m=last['robot_hitch_m'],hold_speed_mps=mean('speed'),hold_four_wheel=mean('four_wheel'),
            peak_leg_error_rad=max(r['leg_error_max_rad'] for r in data),leg_peak_Nm=float(abs(trace['applied'][:,:12]).max()),
            wheel_peak_Nm=float(abs(trace['applied'][:,12:]).max()),clipped_fraction=float((abs(trace['requested'])>limits+1e-5).mean()),
            x_displacement_m=last['x']-data[0]['x'],y_displacement_m=last['y']-data[0]['y'])
        with (out/f'case{i:02d}.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        np.savez_compressed(out/f'case{i:02d}_motor.npz',**trace,limits=limits)
        reports.append(report);print('CAPABILITY_CASE',json.dumps(report),flush=True)
    write_json(out/'result.json',dict(finished_at=now(),ppo_updates=0,cases=reports,
        note='Standing capability probe only; survival, holding and motor-rating acceptance are separate.'))
except BaseException as exc:
    write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    traceback.print_exc();raise
finally:
    if env is not None:env.close()
    app.close()
