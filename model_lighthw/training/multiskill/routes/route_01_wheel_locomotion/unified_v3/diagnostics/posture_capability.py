"""固定支撑点的前短后长姿态探针；无PPO，无重力补偿，首次终止截断。"""
from pathlib import Path
import argparse,csv,json,sys,traceback

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/user/RL_lab/rl_training')
V2=ROOT.parent/'unified_v2'
sys.path.insert(0,str(V2/'scripts'));sys.path.insert(0,str(REPO/'source/rl_training'))
from launch import SPEC,CART,module_from_file,now,write_json,sha256

parser=argparse.ArgumentParser();parser.add_argument('--duration',type=float,default=8.)
parser.add_argument('--output',type=Path,required=True);parser.add_argument('--static_teacher',action='store_true');args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
cases=[dict(controller='posture_pose',kp=80.,kd=3.,hitch=.10,slope_deg=s,payload_kg=p,towed_kg=t,
            bias_deg=b*(1 if s>0 else -1))
       for s in (25.,-25.) for p,t in ((0.,0.),(15.,0.),(0.,40.)) for b in (0.,5.,10.)]
models={mode:V2/f'runs/2026-10-16_10-19-29.659784/training/model_{i}.pt' for mode,i in (('model3900',3900),('model4200',4200))}
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
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v2 import mdp,core,torque
    worker=module_from_file('_capability_v2_worker',V2/'scripts/worker.py')
    original=mdp.measurements
    def measured(base,command=None):
        result=original(base,command);robot=base.scene['robot'];cart=base.scene['trailer']
        for name,asset,offset in (('robot_hitch_m',robot,base.r1_robot_hitch),('cart_hitch_m',cart,base.r1_cart_hitch)):
            point=quat_apply(asset.data.root_quat_w,offset)+asset.data.root_pos_w
            result[name]=((point-base.scene.env_origins)*base.r1_normal).sum(-1)
        ids=base.r1_joint_ids[:12]
        result['leg_error_max_rad']=(robot.data.joint_pos_target[:,ids]-robot.data.joint_pos[:,ids]).abs().amax(-1)
        return result
    mdp.measurements=measured
    SPEC.register_task();cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=len(cases);cfg.seed=agent.seed=43;cfg.fixed_speed_mps=0.;cfg.episode_length_s=args.duration+5
    cfg.observations.policy.enable_corruption=False
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
    from posture_ik import solve
    poses=[solve(REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf',c['bias_deg']) for c in cases]
    write_json(out/'posture_ik.json',poses)
    if any(p['max_foot_error_m']>.001 for p in poses):raise RuntimeError('Pose is not reachable within 1mm')
    q=torch.tensor([p['q'] for p in poses],device=device,dtype=torch.float32)
    joints=robot.data.joint_pos.clone();joints[:,base.r1_joint_ids[:12]]=q
    root_state=robot.data.root_state_w.clone()
    angle=base.r1_slope_rad-torch.tensor([c['bias_deg'] for c in cases],device=device)*torch.pi/180
    root_state[:,3]=torch.cos(angle/2);root_state[:,4]=0.;root_state[:,5]=-torch.sin(angle/2);root_state[:,6]=0.
    with torch.inference_mode():
        robot.write_joint_state_to_sim(joints,torch.zeros_like(joints))
        robot.write_root_state_to_sim(root_state)
        base.scene.write_data_to_sim();base.scene.update(base.physics_dt)
    nominal=torch.tensor([p['default'] for p in poses],device=device)
    fixed_actions=(q-nominal)/torch.tensor([.08,.14,.14]*4,device=device)
    teacher=None
    if args.static_teacher:
        from static_support import nominal_support
        static=[nominal_support(REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf',c['slope_deg'],c['hitch']) for c in cases]
        write_json(out/'static_support.json',static)
        teacher=torch.tensor([c['torques_Nm'] for c in static],device=device)
        weights=torch.tensor([c['wheel_normals_N'] for c in static],device=device).clamp_min(0)
        weights/=weights.sum(-1,keepdim=True)
        extension=torch.tensor([c['preload_extension_m'] for c in static],device=device)
        cart=base.scene['trailer'];pose=cart.data.root_state_w[:,:7].clone();pose[:,:3]-=base.r1_tangent*extension[:,None]
        with torch.inference_mode():cart.write_root_pose_to_sim(pose)
        origin=robot.data.root_pos_w.clone()
    runtime=worker.verify_runtime(env,out)
    runtime['per_environment_leg_kp']=actuator.stiffness[:,0].cpu().tolist();runtime['per_environment_leg_kd']=actuator.damping[:,0].cpu().tolist()
    runtime['per_environment_hitch_initial_m']=h.cpu().tolist();write_json(out/'runtime_check.json',runtime)
    policies={}
    for mode,path in []:
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=device)
        runner.load(str(path),load_optimizer=False);policies[mode]=runner.get_inference_policy(device=device)
    selected={mode:torch.tensor([c['controller']==mode for c in cases],device=device) for mode in policies}
    active=torch.ones(len(cases),dtype=torch.bool,device=device);rows=[[] for _ in cases];frames=[[] for _ in cases]
    with torch.inference_mode():
        for step in range(round(args.duration/.02)):
            action=torch.zeros(len(cases),16,device=device)
            action[:,:12]=fixed_actions
            for mode,policy in policies.items():
                if selected[mode].any():action[selected[mode]]=policy(obs[selected[mode]])
            if teacher is not None:
                scale=torch.tensor([.08,.14,.14]*4,device=device)
                kp=torch.tensor([c['kp'] for c in cases],device=device)
                action[:,:12]=teacher[:,:12]/kp[:,None]/scale
                velocity=(robot.data.root_lin_vel_w*base.r1_tangent).sum(-1)
                displacement=((robot.data.root_pos_w-origin)*base.r1_tangent).sum(-1)
                correction=(-80.*velocity-20.*displacement).clamp(-30.,30.)
                action[:,12:]=(teacher[:,12:]+correction[:,None]*weights*.0875)/3./5.
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
