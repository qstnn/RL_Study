"""Isaac Sim统一R1 worker；实际训练、冻结回放和输出互相隔离。"""
import argparse
from copy import deepcopy
import csv
import json
import random
import shutil
from pathlib import Path
import sys
import time
from launch import ROOT,REPO,SOURCE,SPEC,now,sha256,validate_checkpoint,write_json
from launch import PROTOCOL as protocol
import quality


def verify_runtime(env,run):
    import torch
    from isaaclab.utils.math import matrix_from_quat
    base=env.unwrapped
    robot=base.scene["robot"]
    actual=(env.num_obs,env.num_privileged_obs,env.num_actions)
    if actual!=(300,268,16):
        raise RuntimeError(f"Actual observation/action dimensions: {actual}")
    if base.observation_manager.active_terms["policy"]!=list(SPEC.POLICY_TERMS):
        raise RuntimeError("Policy term order does not match warm-start history mapping")
    action_names=[]
    for key in ("joint_pos","joint_vel"):
        term=base.action_manager.get_term(key)
        action_names.extend(robot.joint_names[i] for i in term._joint_ids)
    if action_names!=SPEC.CONTRACT["action_order"]:
        raise RuntimeError("Unified action order mismatch")
    obs,extra=env.get_observations()
    if not torch.isfinite(obs).all() or not torch.isfinite(extra["observations"]["critic"]).all():
        raise RuntimeError("Non-finite initial observations")
    expected_context=torch.stack((base.r1_payload_kg/15.,base.r1_towed_kg/40.),-1)
    if not torch.allclose(obs[:,-2:],expected_context,atol=1e-7,rtol=0.):
        raise RuntimeError("Nominal job context is missing from the newest policy frame")
    if not torch.allclose(robot.actuators["wheel"].damping,torch.full_like(robot.actuators["wheel"].damping,SPEC.CONTRACT["wheel_damping"])):
        raise RuntimeError("Wheel velocity gain differs from the V8 contract")
    # Critic射线实际命中平面，与每环境所声明的坡度交叉核对。
    hits=base.scene["height_scanner"].data.ray_hits_w
    if not torch.isfinite(hits).all():
        raise RuntimeError("Terrain ray misses in the initial observation")
    expected_z=base.scene.env_origins[:,None,2]+(hits[:,:,0]-base.scene.env_origins[:,None,0])*base.r1_slope_rad.tan()[:,None]
    ray_error=float((hits[:,:,2]-expected_z).abs().max())
    if ray_error>.005:
        raise RuntimeError(f"Raycast differs from configured slope by {ray_error}m")
    masses=robot.root_physx_view.get_masses().sum(-1)
    expected=base.r1_unloaded_mass.sum(-1)+base.r1_payload_kg.cpu()
    if not torch.allclose(masses,expected,atol=1e-4):
        raise RuntimeError("Actual robot mass differs from requested payload")
    cart=base.scene["trailer"]
    cart_mass=cart.root_physx_view.get_masses().sum(-1)
    active=base.r1_towed_kg.cpu()>0
    if not torch.allclose(cart_mass[active],base.r1_towed_kg.cpu()[active],atol=1e-4):
        raise RuntimeError("Actual trailer mass differs from request")
    if torch.count_nonzero(cart.data.default_joint_stiffness) or torch.count_nonzero(cart.data.default_joint_damping):
        raise RuntimeError("Trailer wheels unexpectedly have a drive")
    material=robot.root_physx_view.get_material_properties()
    if not torch.allclose(material[:,:,:2],torch.full_like(material[:,:,:2],base.cfg.ground_friction)):
        raise RuntimeError("Actual robot friction differs from the explicit assumption")
    report={"checked_at":now(),"dimensions":actual,"action_order":action_names,
            "policy_terms":base.observation_manager.active_terms["policy"],
            "critic_terms":base.observation_manager.active_terms["critic"],"policy_history_frames":5,
            "num_envs":base.num_envs,"physics_dt":base.physics_dt,"policy_dt":base.step_dt,
            "slopes_deg":sorted(set(round(float(v),3) for v in (base.r1_slope_rad*180/3.141592653589793).cpu())),
            "payloads_kg":sorted(set(base.r1_payload_kg.cpu().tolist())),
            "towed_masses_kg":sorted(set(base.r1_towed_kg.cpu().tolist())),
            "physical_robot_masses_kg":sorted(set(round(float(v),5) for v in masses)),
            "physical_active_cart_masses_kg":sorted(set(round(float(v),5) for v in cart_mass[active])),
            "disabled_cart_note":"0kg tow environments park a disconnected gravity-disabled cart away from lanes",
            "terrain_ray_max_error_m":ray_error,"ground_friction":base.cfg.ground_friction,
            "rolling_coefficient":base.cfg.rolling_coefficient,"wheel_damping":float(robot.actuators["wheel"].damping[0,0]),"passive_cart_drive_stiffness":0.,
            "payload_examples":base.r1_payload_audit,"physical_com_inertia_verified":base.r1_inertia_verified,"unloaded_inertia_urdf_verified":base.r1_unloaded_inertia_urdf_verified,"nonbase_inertia_unchanged":base.r1_nonbase_inertia_unchanged,"behavior_acceptance":"not_evaluated"}
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8 import torque,mdp
    software=torque.actuator_limits(robot,validate=True)
    ordered=software[:,base.r1_joint_ids]
    if not torch.allclose(ordered[:,:12],torch.full_like(ordered[:,:12],base.cfg.leg_torque_limit)):
        raise RuntimeError("Actual leg actuator cap differs from request")
    if not torch.allclose(ordered[:,12:],torch.full_like(ordered[:,12:],base.cfg.wheel_torque_limit)):
        raise RuntimeError("Actual wheel actuator cap differs from request")
    if bool(((base.r1_payload_kg>0)&(base.r1_towed_kg>0)).any()):
        raise RuntimeError("Payload and towing were incorrectly combined")
    forbidden=torch.isclose(base.r1_slope_rad,torch.full_like(base.r1_slope_rad,-25.*torch.pi/180.),atol=1e-6,rtol=0.)&(base.r1_towed_kg>0)
    if bool(forbidden.any()):raise RuntimeError("The revised task forbids trailer towing on -25 degree downhill")
    report["minus25_towing_count"]=int(forbidden.sum())
    report.update(payload_towing_mutually_exclusive=True,software_effort_limits_Nm=ordered[0].cpu().tolist(),
                  solver_effort_limits_Nm=robot.data.joint_effort_limits[0,base.r1_joint_ids].cpu().tolist(),
                  motor_sample_hz=1000,hitch_height_initial_m=base.cfg.hitch_height_m,
                  robot_hitch_local_m=base.r1_robot_hitch[0].cpu().tolist(),cart_hitch_local_m=base.r1_cart_hitch[0].cpu().tolist(),
                  loaded_command_mps=[base.cfg.loaded_speed_min,base.cfg.loaded_speed_max],unloaded_only_3mps_requirement=True)
    contexts=torch.stack((base.r1_slope_rad*180/torch.pi,base.r1_payload_kg,base.r1_towed_kg),dim=-1)
    report["unique_condition_count"]=int(torch.unique(contexts,dim=0).shape[0])
    unloaded=(base.r1_payload_kg==0)&(base.r1_towed_kg==0)
    report["class_counts"]={"unloaded":int(unloaded.sum()),"payload_only":int((base.r1_payload_kg>0).sum()),"towing_only":int((base.r1_towed_kg>0).sum())}
    if base.cfg.fixed_speed_mps is None:
        if report["unique_condition_count"]!=64:
            raise RuntimeError("Training must contain all64 permitted conditions from reset")
        term=base.command_manager.get_term("base_velocity")
        actual_command=term.command[:,0]
        if bool((actual_command!=0).any()):raise RuntimeError("Reset must start with an applied zero command")
        speed=term.requested_vx
        loaded=~unloaded
        standing=base.command_manager.get_term("base_velocity").is_standing_env
        if bool((speed[standing]!=0).any()):
            raise RuntimeError("Standing reset must expose a zero command from the first observation")
        if bool((speed[loaded]>base.cfg.loaded_speed_max+1.e-6).any()) or bool((speed[loaded]<0).any()):
            raise RuntimeError("Loaded command violates the low-speed training reference")
        if bool(((speed>=3.)&loaded).any()):
            raise RuntimeError("High speed was incorrectly assigned to loaded or towing cases")
        report["initial_command_check"]={"applied_zero_count":int((actual_command==0).sum()),"loaded_max_mps":float(speed[loaded].max()) if loaded.any() else None,
            "unloaded_high_speed_count":int(((speed>=3.)&unloaded).sum()),
            "loaded_high_speed_count":int(((speed>=3.)&loaded).sum()),"zero_count":int((speed==0).sum()),"standing_count":int(standing.sum())}
    report.update(command_profile=base.cfg.command_profile,unloaded_acceleration_mps2=base.cfg.unloaded_acceleration,
                  loaded_acceleration_mps2=base.cfg.loaded_acceleration,hitch_margin_m=base.cfg.hitch_margin_m,
                  leg_kp=robot.actuators["joint"].stiffness[0].cpu().tolist(),leg_kd=robot.actuators["joint"].damping[0].cpu().tolist())
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8.wheel_caps import audit_runtime_caps
    report["rigid_body_angular_caps"]=audit_runtime_caps(base)
    wheel_term=base.action_manager.get_term("joint_vel")
    expected_clip=torch.tensor(SPEC.CONTRACT["wheel_clip"],device=env.device)
    if not torch.allclose(wheel_term._clip,expected_clip.expand_as(wheel_term._clip)):
        raise RuntimeError("Runtime wheel target clip differs from the V8 contract")
    joint_limits=robot.data.joint_vel_limits[:,wheel_term._joint_ids]
    if not torch.allclose(joint_limits,torch.full_like(joint_limits,SPEC.CONTRACT["wheel_joint_velocity_limit_rad_s"])):
        raise RuntimeError("Runtime wheel joint speed limit differs from the V8 contract")
    solver_velocity=robot.root_physx_view.get_dof_max_velocities()[:,wheel_term._joint_ids]
    actuator_velocity=robot.actuators["wheel"].velocity_limit
    for label,values in (("solver",solver_velocity),("actuator",actuator_velocity)):
        if not torch.allclose(values,torch.full_like(values,SPEC.CONTRACT["wheel_joint_velocity_limit_rad_s"])):
            raise RuntimeError("Actual wheel "+label+" speed limit differs from the V8 contract")
    noise_terms={}
    for name in ("base_ang_vel","projected_gravity","joint_pos","joint_vel"):
        term_cfg=getattr(base.cfg.observations.policy,name)
        noise_terms[name]={"n_min":term_cfg.noise.n_min,"n_max":term_cfg.noise.n_max,"scale":term_cfg.scale}
    command_scale=torch.as_tensor(base.cfg.observations.policy.velocity_commands.scale or 1.,device=env.device)
    if not torch.all(command_scale==1.):raise RuntimeError("V8 noise schedule requires unscaled command observation")
    expected_vx=base.command_manager.get_command("base_velocity")[:,0]*(command_scale if command_scale.ndim==0 else command_scale[0])
    if not torch.allclose(obs[:,42],expected_vx,atol=1.e-6):raise RuntimeError("V8 conditional-noise command column mismatch")
    report["conditional_exploration_columns"]={"current_vx":42,"current_towed_context":299,"noise_true_state_inputs":False}
    report.update(wheel_target_clip_rad_s=wheel_term._clip[0].cpu().tolist(),
                  wheel_joint_velocity_limits_rad_s=joint_limits[0].cpu().tolist(),
                  wheel_solver_velocity_limits_rad_s=solver_velocity[0].cpu().tolist(),
                  wheel_actuator_velocity_limits_rad_s=actuator_velocity[0].cpu().tolist(),
                  episode_length_s=base.cfg.episode_length_s,observation_noise_scale=base.cfg.observation_noise_scale,
                  policy_enable_corruption=base.cfg.observations.policy.enable_corruption,noise_terms=noise_terms)
    report["condition_counts"]=[dict(slope_deg=round(float(row[0])),payload_kg=float(row[1]),towed_kg=float(row[2]),count=int((contexts==row).all(-1).sum())) for row in torch.unique(contexts,dim=0)]
    report["solver_iterations"]=[base.cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,base.cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count]
    write_json(run/"runtime_check.json",report)
    return report


def install_reference_cache(runner,checkpoint,saved):
    """读取初始化明确绑定且 SHA 匹配的成功标签缓存，禁止隐式全样本模仿。"""
    import numpy as np
    import torch
    initialization=(saved.get("infos") or {}).get("initialization")
    record=initialization.get("reference_cache") if isinstance(initialization,dict) else None
    if record is None:
        if runner.alg.reference_loss_coef>0.:
            raise ValueError("Enabled reference loss requires an initialization-bound success cache")
        return None
    if not isinstance(record,dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError("Reference cache metadata requires path and sha256")
    path=Path(record["path"])
    if not path.is_absolute():path=Path(checkpoint).resolve().parent/path
    path=path.resolve(strict=True)
    if sha256(path)!=record["sha256"]:
        raise ValueError("Initialization reference cache SHA256 mismatch")
    with np.load(path,allow_pickle=False) as values:
        required={"observation","action","successful"}
        if not required.issubset(values.files):
            raise ValueError("Reference cache must contain observation/action/successful arrays")
        accepted=values["successful"]
        if accepted.dtype!=np.bool_ or accepted.ndim!=1:
            raise ValueError("Reference successful must be a one-dimensional boolean array")
        cache_kind=str(values["cache_kind"].item()) if "cache_kind" in values.files and values["cache_kind"].shape==() else None
        if (getattr(runner.alg,"reference_max_group_mse",0.)>0.
                and cache_kind!="deployed_actor_success_reference"):
            raise ValueError("Grouped reference guard requires verified Actor-success data; teacher labels are offline-only")
        grouped={"group_id","case_key","stage"}.issubset(values.files)
        if not grouped or cache_kind!="deployed_actor_success_reference":
            raise ValueError("V8 requires explicitly validated case/stage Actor reference data")
        if getattr(runner.alg,"reference_max_group_mse",0.)>0. and (not grouped or "stage" not in values.files):
            raise ValueError("Grouped reference guard requires explicit group_id, case_key and stage arrays")
        group_ids=values["group_id"] if grouped else None
        case_keys=[];case_stages={}
        if grouped:
            keys=values["case_key"]
            if keys.shape!=values["successful"].shape or keys.dtype.kind!="U":
                raise ValueError("Reference case_key must be one Unicode key per sample")
            allowed={quality.case_key(case) for case in protocol.cases()}
            if group_ids.shape!=accepted.shape or group_ids.dtype.kind not in "iu" or (group_ids<0).any():
                raise ValueError("Reference group ids must be one nonnegative integer per sample")
            case_keys=sorted(set(keys[accepted].tolist()))
            if not set(case_keys).issubset(allowed):
                raise ValueError("Reference contains a case outside the declared task")
            stages=values["stage"] if "stage" in values.files else np.full(len(keys),"legacy")
            allowed_stages={"wait","move","brake","stop_settle","stop_hold","initial_settle","static_mid","static_hold"}
            if (stages.shape!=accepted.shape or stages.dtype.kind!="U"
                    or (runner.alg.reference_max_group_mse>0. and not set(stages[accepted]).issubset(allowed_stages))):
                raise ValueError("Reference stages must explicitly identify startup, movement and steady hold")
            case_stages={key:sorted(set(stages[accepted&(keys==key)].tolist())) for key in case_keys}
            for group in np.unique(group_ids[accepted]):
                members=accepted&(group_ids==group)
                if len(set(keys[members].tolist()))!=1 or len(set(stages[members].tolist()))!=1:
                    raise ValueError("A reference group must not mix different cases or stages")
        runner.alg.set_reference_data(values["observation"],values["action"],values["successful"],group_ids=group_ids)
        total=len(values["successful"])
    if sha256(path)!=record["sha256"]:
        raise ValueError("Reference cache changed while loading")
    observations=runner.alg.reference_observations
    count=0 if observations is None else len(observations)
    if runner.alg.reference_loss_coef>0. and count==0:
        raise ValueError("Enabled reference loss has no accepted successful samples")
    normalized_square_sum=0.;max_error=0.
    with torch.inference_mode():
        for start in range(0,count,8192):
            predicted=runner.alg.policy.act_inference(observations[start:start+8192])
            error=predicted-runner.alg.reference_actions[start:start+8192]
            normalized_square_sum+=float((error/runner.alg.reference_action_scale).square().sum())
            max_error=max(max_error,float(error.abs().max()))
    diagnostics=runner.alg._reference_diagnostics()
    if (runner.alg.reference_max_group_mse>0.
            and (not np.isfinite(diagnostics["reference_group_mse_max"])
                 or diagnostics["reference_group_mse_max"]>runner.alg.reference_max_group_mse)):
        raise ValueError("Initialization already exceeds the per-group reference bound; validate a matching Actor/cache")
    return dict(path=str(path),sha256=record["sha256"],total_samples=total,successful_samples=count,
                rejected_samples=total-count,cache_kind=cache_kind,loss_coef=runner.alg.reference_loss_coef,
                action_scale=runner.alg.reference_action_scale.detach().cpu().tolist(),
                initial_normalized_mse=normalized_square_sum/max(1,count*16),
                initial_raw_max_error=max_error,case_keys=case_keys,case_stages=case_stages,
                group_count=len(runner.alg._reference_members),
                sampling="uniform_case_phase_group_then_uniform_frame" if grouped else "legacy_single_group",
                max_group_mse=runner.alg.reference_max_group_mse,
                initial_group_diagnostics=diagnostics)


def export_actor(runner,obs,run):
    import torch
    actor=deepcopy(runner.alg.policy.actor).cpu().eval()
    module=torch.jit.script(actor)
    directory=run/"exported";directory.mkdir(exist_ok=True)
    path=directory/"actor.pt";module.save(str(path))
    with torch.inference_mode():
        expected=runner.alg.policy.act_inference(obs).cpu()
        actual=module(obs.cpu())
        if not torch.allclose(expected,actual,atol=1e-5,rtol=1e-5):
            raise RuntimeError("Exported actor differs from actual inference policy")
    write_json(directory/"contract.json",SPEC.CONTRACT)
    write_json(directory/"verification.json",{"actor_input":300,"actor_output":16,"samples":len(obs),
               "max_abs_error":float((expected-actual).abs().max()),"sha256":sha256(path),"history_required":True})


def cpu_measurements(values):
    """按形状/类型/设备合并只读回传，保持逐字段数值、dtype及输出顺序。

    冻结评估原先每20ms对数百字段分别执行GPU同步；共享GPU时尤其昂贵。
    只做stack/copy，不归一化、不降采样、不改动物理或策略状态。
    """
    import torch
    groups={}
    for name,value in values.items():
        groups.setdefault((tuple(value.shape),value.dtype,value.device),[]).append((name,value))
    result={}
    for group in groups.values():
        packed=torch.stack([value.detach() for _,value in group]).cpu().numpy()
        for index,(name,_) in enumerate(group):result[name]=packed[index]
    return {name:result[name] for name in values}


def evaluate(env,runner,args,run,app):
    import numpy as np
    import torch
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8 import core,mdp,torque
    base=env.unwrapped
    runner.load(args["checkpoint"],load_optimizer=False)
    policy=runner.get_inference_policy(device=env.device)
    cases=protocol.cases(args["eval_distance"],args.get("suite","boundary")) if args["mode"]=="evaluate" else [{"slope_deg":args["slope_deg"],"payload_kg":args["payload_kg"],"towed_kg":args["towed_kg"],"speed":args["speed"],"goal_distance_m":args["eval_distance"]}]
    if len(cases)!=env.num_envs:raise RuntimeError("Evaluation must allocate one isolated environment per case")
    seeds=args["eval_seeds"] if args["mode"]=="evaluate" else [args["seed"]]
    reports=[]
    base.r1_slope_rad[:]=torch.tensor([c["slope_deg"]*torch.pi/180 for c in cases],device=env.device)
    base.r1_tangent[:],base.r1_normal[:]=core.plane_basis(base.r1_slope_rad)
    for key in ("payload_kg","towed_kg"):
        getattr(base,"r1_"+key)[:]=torch.tensor([c[key] for c in cases],device=env.device)
    base.cfg.fixed_speed_mps=torch.tensor([c["speed"] for c in cases],device=env.device)
    lanes=torch.tensor([SPEC.ANGLES.index(c["slope_deg"]) for c in cases],device=env.device)
    base.scene.terrain.terrain_types[:]=lanes
    base.scene.terrain.env_origins[:]=base.scene.terrain.terrain_origins[0,lanes]
    mdp.set_payload_properties(base)
    for seed in seeds:
        env.seed(seed)
        rows=[[] for _ in cases];frames=[[] for _ in cases];reasons=[None]*len(cases)
        policy_frames=[[] for _ in cases]
        active=torch.ones(len(cases),dtype=torch.bool,device=env.device)
        with torch.inference_mode():
            obs,_=env.reset()
            # 批量验收在创建环境后重排了坡度与质量，必须复查最终实际物理上下文。
            if seed==seeds[0]:verify_runtime(env,run)
            initial=base.scene["robot"].data.root_state_w.cpu().tolist()
            if seed==seeds[0] and not args["headless"]:
                # 只在启动时定位一次；随后由用户在Isaac Sim窗口自由移动或手动启用跟随。
                p=base.scene["robot"].data.root_pos_w[0].cpu().numpy()
                base.sim.set_camera_view(p+[5.,-6.,3.5],p+[-.7,0.,0.])
            for step in range(max(protocol.max_steps(c) for c in cases)):
                start=time.monotonic()
                if not app.is_running():
                    for i in active.nonzero().flatten().tolist():reasons[i]="application_closed"
                    break
                action=(torch.cat([policy(obs[i:i+1]) for i in range(len(obs))]) if args.get("actor_inference","row")=="row" else policy(obs))
                policy_record=cpu_measurements({"observation":obs,"action":action,"phase":base.r1_eval_phase})
                if not torch.isfinite(action).all():raise RuntimeError("Non-finite frozen Actor action")
                action[~active]=0.
                obs,_,done,_=env.step(action)
                measurements=cpu_measurements(base.r1_last_metrics)
                recorder=base.r1_torque_recorder
                motor_frames=cpu_measurements({name:getattr(recorder,name) for name in ('applied','requested','velocity')})
                done_cpu=done.cpu().numpy()
                for i in active.nonzero().flatten().tolist():
                    row={k:float(v[i]) for k,v in measurements.items()};row["t"]=(step+1)*protocol.DT
                    policy_frames[i].append({key:value[i].copy() for key,value in policy_record.items()})
                    rows[i].append(row);frames[i].append({name:values[i].copy() for name,values in motor_frames.items()})
                    if bool(done_cpu[i]):reasons[i]="terminated";active[i]=False
                    elif protocol.finished(rows[i],cases[i]):active[i]=False
                    elif step+1>=protocol.max_steps(cases[i]):reasons[i]="budget_exhausted";active[i]=False
                if (step+1)%250==0:
                    print(f"[V8 EVAL] step={step+1} active={int(active.sum())}",flush=True)
                if not active.any():break
                if not args["headless"]:
                    time.sleep(max(0.,protocol.DT-(time.monotonic()-start)))
        for i,case in enumerate(cases):
            prefix=f"seed{seed}_case{i:02d}"
            with (run/(prefix+".csv")).open("w",newline="") as stream:
                if rows[i]:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[i][0]));writer.writeheader();writer.writerows(rows[i])
            if policy_frames[i]:
                np.savez_compressed(run/(prefix+"_policy_trace.npz"),**{key:np.stack([r[key] for r in policy_frames[i]]) for key in policy_frames[i][0]})
            report=protocol.summarize(rows[i],case,reasons[i])
            if frames[i]:
                trace={k:np.concatenate([frame[k] for frame in frames[i]]) for k in frames[i][0]}
                if len(trace["applied"])!=len(rows[i])*20:raise RuntimeError("Incomplete 1kHz motor trace")
                limits=base.r1_motor_limits[i].cpu().numpy()
                np.savez_compressed(run/(prefix+"_motor_trace.npz"),**trace,limits=limits,
                                    joint_names=np.asarray(SPEC.CONTRACT["action_order"]),physics_dt=.001)
                write_json(run/(prefix+"_motor.json"),torque.summarize_trace(trace,SPEC.CONTRACT["action_order"],limits,report["success"]))
                report["motor_report"]=prefix+"_motor.json"
            report.update(seed=seed,initial_root_state=initial[i],command_profile=args["command_profile"])
            write_json(run/(prefix+".json"),report);reports.append(report)
        base.drain_metrics();base.r1_torque_window.drain()
        if "application_closed" in reasons:break
    write_json(run/"evaluation.json",{"checkpoint":args["checkpoint"],"sha256":sha256(args["checkpoint"]),
        "ppo_updates":0,"cases":reports,"all_cases_complete":all(r["complete"] for r in reports),
        "complete_count":sum(r["complete"] for r in reports),"pass_count":sum(r["success"] for r in reports),
        "success_rate":sum(r["success"] for r in reports)/len(reports),"final_hardware_acceptance":"not_evaluated",
        "command_profile":args["command_profile"],"screening_only":True,
        "note":"Fixed initial-state screen; repeated seeds are not independent randomized robustness trials. Step profile is stress only."})


def periodic_evaluation(runner,env,args,run,iteration,*,baseline=False):
    """冻结验收文件保持不变；质量状态进入后续训练/最佳 checkpoint。"""
    import subprocess
    import torch
    completed_updates=runner.alg.completed_updates
    if iteration+1!=completed_updates:
        raise RuntimeError("Evaluation iteration differs from the actual optimizer update counter")
    directory=run/"periodic_evaluations"
    checkpoint=directory/"checkpoints"/f"model_{iteration}.pt"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():
        raise RuntimeError("Refusing to replace a previously evaluated checkpoint")
    runner.save(str(checkpoint))
    before=sha256(checkpoint)
    roots=env.unwrapped.scene["robot"].data.root_state_w.clone()
    label=f"update_{completed_updates:04d}"
    output=directory/label
    cmd=[sys.executable,str(ROOT/"scripts/evaluate.py"),"--headless","--checkpoint",str(checkpoint),
         "--eval_seeds","43","--output_dir",str(output)]
    for key in ("device","ground_friction","rolling_coefficient","leg_torque_limit","wheel_torque_limit",
                "hitch_height","unloaded_acceleration","loaded_acceleration","eval_distance","critic_warmup_updates"):
        cmd.extend(["--"+key,str(args.get(key,50))])
    print(f"[V8] {'Baseline before further updates' if baseline else 'Pause training'}: "
          f"{completed_updates} completed updates; frozen {len(protocol.cases(args['eval_distance']))}-case screen -> {output}",flush=True)
    with (directory/(label+"_launcher.log")).open("w") as stream:
        child=subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    if child.returncode or not (output/"evaluation.json").is_file() or (output/"worker_error.json").exists():
        raise RuntimeError(f"Frozen evaluation failed to execute; inspect {output}")
    if sha256(checkpoint)!=before or not torch.equal(roots,env.unwrapped.scene["robot"].data.root_state_w):
        raise RuntimeError("Frozen evaluation changed the parent checkpoint or physical state")
    result=json.loads((output/"evaluation.json").read_text())
    expected_keys={quality.case_key(case) for case in protocol.cases(args["eval_distance"])}
    actual_keys={quality.case_key(row["case"]) for row in result["cases"]}
    if (result["sha256"]!=before or len(result["cases"])!=len(expected_keys) or actual_keys!=expected_keys):
        raise RuntimeError("Frozen evaluation checkpoint or case coverage mismatch")
    quality_directory=run/"quality"
    previous=runner.quality_state
    initialization_record=getattr(runner,"initialization_record",None)
    required=(initialization_record.get("required_initialization_cases")
              if isinstance(initialization_record,dict) else None)
    if baseline and completed_updates==0 and previous is None and required is not None:
        if not isinstance(required,list) or any(not isinstance(key,str) for key in required):
            raise ValueError("required_initialization_cases must be a list of quality.case_key strings")
        passing={quality.case_key(row["case"]) for row in result["cases"] if row["success"]}
        lost=sorted(set(required)-passing)
        write_json(run/"initialization_baseline_check.json",dict(
            completed_updates=0,checkpoint=str(checkpoint),checkpoint_sha256=before,
            evaluation=str(output/"evaluation.json"),required_cases=required,
            required_count=len(set(required)),failed_required_cases=lost,passed=not lost))
        if lost:
            # 尚未建立quality保护集合，也尚未开始Critic预热或Actor PPO；
            # 不能将准备阶段已证明的能力静默降级成当前较弱的初始通过集。
            raise RuntimeError("Initialization baseline regression: required cases did not pass: "+repr(lost))
    if baseline and runner.alg.reference_loss_coef>0.:
        reference=(initialization_record or {}).get("reference_cache") or {}
        covered=set(reference.get("case_keys",[]))
        passing={quality.case_key(row["case"]) for row in result["cases"] if row["success"]}
        protected=set(previous["protected"]) if previous is not None else passing
        missing=sorted(protected-covered)
        write_json(run/"reference_coverage_check.json",dict(protected_cases=sorted(protected),
                   covered_cases=sorted(covered),missing_cases=missing,passed=not missing))
        if missing:
            raise RuntimeError("Protected baseline cases missing from success reference: "+repr(missing))
        declared={quality.case_key(case):case for case in protocol.cases(args["eval_distance"])}
        missing_stages={}
        for key in protected:
            required_stages=({"initial_settle","static_mid","static_hold"} if declared[key]["speed"]==0.
                             else {"wait","move","brake","stop_settle","stop_hold"})
            absent=required_stages-set(reference.get("case_stages",{}).get(key,[]))
            if absent:missing_stages[key]=sorted(absent)
        write_json(run/"reference_stage_coverage_check.json",dict(completed_updates=completed_updates,
                   passed=not missing_stages,missing_stages=missing_stages))
        if missing_stages:
            raise RuntimeError("Protected cases have missing reference stages: "+repr(missing_stages))
    # 精确恢复前的复核不能把同一已评估更新重复计成第二次策略退化。
    repeated=baseline and previous is not None and previous.get("last",{}).get("completed_updates")==completed_updates
    if repeated:
        passing={quality.case_key(row["case"]) for row in result["cases"] if row["success"]}
        regressions=sorted(set(previous["protected"])-passing)
        verdict=dict(previous["last"],promoted=False,repeat_preflight_only=True,
                     pass_count=result["pass_count"],complete_count=result["complete_count"],
                     regressions=regressions,eligible=not regressions)
    else:
        verdict=quality.review(result,checkpoint,quality_directory,completed_updates,patience=2)
        runner.quality_state=json.loads((quality_directory/"quality_state.json").read_text())
        if verdict["promoted"]:
            # 网络/Adam 与冻结源相同，额外保存刚取得的质量状态；冻结源 SHA 不变。
            runner.save(str(quality_directory/"best_eligible.pt"))
    record=dict(completed_updates=completed_updates,checkpoint=str(checkpoint),sha256=before,
                evaluation=str(output/"evaluation.json"),pass_count=result["pass_count"],
                complete_count=result["complete_count"],case_count=len(result["cases"]),ppo_updates_in_evaluation=0,
                parent_physical_state_unchanged=True,baseline=baseline,quality=verdict)
    if verdict.get("best_checkpoint") and Path(verdict["best_checkpoint"]).is_file():
        record["best_checkpoint_sha256"]=sha256(verdict["best_checkpoint"])
    with (directory/"index.jsonl").open("a") as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(f"[V8] Frozen screen: {record['pass_count']}/{record['case_count']} pass; "
          f"regressions={len(verdict['regressions'])}, eligible={verdict['eligible']}, stop={verdict['stop']}",flush=True)
    if verdict["stop"]:
        raise quality.QualityRegression("Two consecutive evaluated updates lost protected cases: "+
                                        repr(verdict["regressions"]))
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    run=parser.parse_args().run.resolve()
    args=json.loads((run/"request.json").read_text())
    sys.path.insert(0,str(REPO/"source/rl_training"))
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=args["headless"],device=args["device"]).app
    env=runner=None
    try:
        import gymnasium as gym
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner
        import rsl_rl.runners.on_policy_runner as runner_module
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8.ppo import V8PPO
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8 import spec
        spec.register_task()
        cfg=load_cfg_from_registry(spec.TASK,"env_cfg_entry_point")
        agent=load_cfg_from_registry(spec.TASK,"rsl_rl_cfg_entry_point")
        if args["mode"]=="train" and args.get("fresh") and agent.algorithm.reference_loss_coef>0.:
            raise ValueError("Fresh training has no verified success cache; use a cache-bound initialization checkpoint")
        cfg.scene.num_envs=args["num_envs"]
        cfg.seed=agent.seed=args["seed"]
        cfg.sim.device=agent.device=args["device"]
        cfg.ground_friction=args["ground_friction"];cfg.rolling_coefficient=args["rolling_coefficient"]
        cfg.leg_torque_limit=args["leg_torque_limit"];cfg.wheel_torque_limit=args["wheel_torque_limit"]
        cfg.scene.robot.actuators["joint"].effort_limit=cfg.leg_torque_limit
        cfg.scene.robot.actuators["wheel"].effort_limit=cfg.wheel_torque_limit
        cfg.hitch_height_m=args["hitch_height"];cfg.eval_distance_m=args["eval_distance"]
        cfg.command_profile=args["command_profile"]
        cfg.unloaded_acceleration=args["unloaded_acceleration"];cfg.loaded_acceleration=args["loaded_acceleration"]
        cfg.scene.robot.spawn.usd_dir=str(run/"asset_cache/robot")
        cfg.scene.robot.spawn.usd_file_name="lighthw.usd"
        cfg.scene.trailer.spawn.asset_path=str(run/"assets/cart.urdf")
        cfg.scene.trailer.spawn.usd_dir=str(run/"asset_cache/trailer")
        cfg.scene.trailer.spawn.usd_file_name="trailer.usd"
        agent.max_iterations=args["max_iterations"]
        if args.get("critic_warmup_updates") is not None:
            if args["mode"]=="train" and not args.get("validation") and args["critic_warmup_updates"]!=agent.algorithm.critic_warmup_updates:
                raise ValueError("Changing Critic warmup length is restricted to validation")
            if args["critic_warmup_updates"]<0:
                raise ValueError("Critic warmup length must be nonnegative")
            agent.algorithm.critic_warmup_updates=args["critic_warmup_updates"]
        if args.get("skip_baseline_evaluation") and not args.get("validation"):
            raise ValueError("Formal training must establish a frozen behavior baseline")
        if args["mode"] in ("play","evaluate"):
            # 默认不注册机器人跟随视角；需要跟随时由用户在窗口中设置。
            cfg.viewer.origin_type="world"
            cfg.episode_length_s=65.
            from configuration import evaluation_template
            evaluation_template(cfg,args)
            cfg.observations.policy.enable_corruption=args.get("observation_noise",False)
        cfg.scene.terrain.physics_material.static_friction=cfg.scene.terrain.physics_material.dynamic_friction=cfg.ground_friction
        cfg.sim.physics_material.static_friction=cfg.sim.physics_material.dynamic_friction=cfg.ground_friction
        cfg.scene.terrain.fixed_slope_deg=cfg.fixed_slope_deg
        cfg.scene.terrain.fixed_payload_kg=cfg.fixed_payload_kg
        cfg.scene.terrain.fixed_towed_kg=cfg.fixed_towed_kg
        dump_yaml(str(run/"params/env.yaml"),cfg);dump_yaml(str(run/"params/agent.yaml"),agent)
        env=RslRlVecEnvWrapper(gym.make(spec.TASK,cfg=cfg),clip_actions=agent.clip_actions)
        with torch.inference_mode():env.reset()
        verify_runtime(env,run)

        class UnifiedRunner(OnPolicyRunner):
            quality_state=None
            initialization_record=None

            def save(self,path,infos=None):
                import numpy as np
                spec.validate_policy_input_mask(self.alg.policy,self.alg.optimizer)
                completed=self.alg.completed_updates
                metadata=dict(infos or {},r1_unified_contract=spec.CONTRACT,
                              environment_signature=args["environment_signature"],completed_updates=completed,
                              v8_optimizer_state=self.alg.training_state_dict(),
                              quality_state=deepcopy(self.quality_state),initialization=deepcopy(self.initialization_record),
                              rng_state=dict(torch_cpu=torch.get_rng_state(),
                                             torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                             numpy=np.random.get_state(),python=random.getstate()))
                Path(path).parent.mkdir(parents=True,exist_ok=True)
                previous=self.current_learning_iteration
                # RSL 新 runner 的 current_iteration=0；真正零更新的 checkpoint 必须为 iter=-1。
                self.current_learning_iteration=completed-1
                if not hasattr(self,"logger_type"):
                    self.logger_type=self.cfg.get("logger","tensorboard").lower()
                try:
                    return super().save(path,infos=metadata)
                finally:
                    self.current_learning_iteration=previous

            def load(self,path,load_optimizer=True):
                import numpy as np
                infos=super().load(path,load_optimizer=load_optimizer)
                if load_optimizer:
                    if not infos or "v8_optimizer_state" not in infos:
                        raise ValueError("Exact V8 continuation requires the saved optimizer training state")
                    self.alg.load_training_state_dict(infos["v8_optimizer_state"])
                    if self.alg.completed_updates!=self.current_learning_iteration+1 or infos["completed_updates"]!=self.alg.completed_updates:
                        raise ValueError("Checkpoint iteration/algorithm update counters disagree")
                    self.quality_state=deepcopy(infos.get("quality_state"))
                    self.initialization_record=deepcopy(infos.get("initialization"))
                    if "rng_state" not in infos:
                        raise ValueError("Exact V8 continuation requires saved RNG states")
                    rng=infos["rng_state"]
                    torch.set_rng_state(rng["torch_cpu"].cpu())
                    if rng["torch_cuda"]:
                        torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda"]])
                    np.random.set_state(rng["numpy"]);random.setstate(rng["python"])
                return infos

            def log(self,locs,width=80,pad=35):
                super().log(locs,width,pad)
                rows=env.unwrapped.drain_metrics()
                output=run/"training/condition_metrics.csv"
                exists=output.exists()
                with output.open("a",newline="") as stream:
                    for row in rows:
                        row=dict(iteration=locs["it"],**row)
                        writer=csv.DictWriter(stream,fieldnames=list(row))
                        if not exists:writer.writeheader();exists=True
                        writer.writerow(row)
                        if row["samples"] and self.writer:
                            for key in ("vx","velocity_error","four_wheel","terminated","drawbar_force_N"):
                                self.writer.add_scalar(f"R1/{row['group']}/{key}",row[key],locs["it"])
                motor_rows=env.unwrapped.r1_torque_window.drain()
                output=run/"training/motor_metrics.csv"
                exists=output.exists()
                with output.open("a",newline="") as stream:
                    for row in motor_rows:
                        row=dict(iteration=locs["it"],**row)
                        writer=csv.DictWriter(stream,fieldnames=list(row))
                        if not exists:writer.writeheader();exists=True
                        writer.writerow(row)

                output=run/"training/condition_optimizer_metrics.csv"
                exists=output.exists()
                with output.open("a",newline="") as stream:
                    for row in self.alg.condition_diagnostics:
                        row=dict(iteration=locs["it"],**row)
                        writer=csv.DictWriter(stream,fieldnames=list(row))
                        if not exists:writer.writeheader();exists=True
                        writer.writerow(row)
                optimizer_metrics={"iteration":locs["it"],"completed_updates":self.alg.completed_updates,**locs["loss_dict"]}
                output=run/"training/optimizer_metrics.csv"
                exists=output.exists()
                with output.open("a",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(optimizer_metrics))
                    if not exists:writer.writeheader()
                    writer.writerow(optimizer_metrics)
                if args["eval_interval"] and self.alg.completed_updates%args["eval_interval"]==0:
                    periodic_evaluation(self,env,args,run,locs["it"])

        if args["mode"]=="smoke":
            terminations=0
            with torch.inference_mode():
                for _ in range(100):
                    obs,reward,done,extra=env.step(torch.zeros(env.num_envs,16,device=env.device))
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():raise RuntimeError("Non-finite smoke")
                    terminations+=int(done.sum())
            write_json(run/"smoke.json",{"steps":100,"ppo_updates":0,"terminations":terminations,
                "motor_metrics":env.unwrapped.r1_torque_window.drain(),"condition_metrics":env.unwrapped.drain_metrics(),"behavior_acceptance":"not_evaluated",
                "note":"Zero-action control is not a trained loaded-slope policy; finite/reset-chain check only"})
            return
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_direct_v8.policy import MotionExplorationActorCritic
        native_ppo=runner_module.PPO
        native_actor=runner_module.ActorCritic
        try:
            runner_module.PPO=V8PPO
            runner_module.ActorCritic=MotionExplorationActorCritic
            runner=UnifiedRunner(env,agent.to_dict(),log_dir=str(run/"training") if args["mode"]=="train" else None,device=args["device"])
        finally:
            runner_module.PPO=native_ppo
            runner_module.ActorCritic=native_actor
        if not isinstance(runner.alg,V8PPO):
            raise RuntimeError("The V8 worker did not instantiate its controlled PPO implementation")
        spec.install_policy_input_mask(runner.alg.policy)
        if args["mode"]=="train":
            initialization={"mode":"fresh","optimizer_restored":False,"start_update":0}
            if args["resume"]:
                saved=spec.checkpoint(args["resume"])
                runner.load(args["resume"],load_optimizer=True)
                runner.current_learning_iteration=runner.alg.completed_updates
                def exact(left,right):
                    if isinstance(left,torch.Tensor):return torch.equal(left.cpu(),right.cpu())
                    if isinstance(left,dict):return left.keys()==right.keys() and all(exact(left[k],right[k]) for k in left)
                    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
                    return left==right
                model_match=exact(runner.alg.policy.state_dict(),saved["model_state_dict"])
                optimizer_match=exact(runner.alg.optimizer.state_dict(),saved["optimizer_state_dict"])
                algorithm_match=exact(runner.alg.training_state_dict(),saved["infos"]["v8_optimizer_state"])
                if not model_match or not optimizer_match or not algorithm_match:
                    raise RuntimeError("Resume did not restore exact network/Adam/algorithm state")
                initialization.update(mode="resume",optimizer_restored=True,start_update=runner.current_learning_iteration,
                    model_restored_exact=model_match,optimizer_restored_exact=optimizer_match,
                    algorithm_restored_exact=algorithm_match,rng_restored=True,
                    restored_learning_rates={group["name"]:group["lr"] for group in runner.alg.optimizer.param_groups},
                    source_adam_steps=sorted({int(v["step"]) for v in saved["optimizer_state_dict"]["state"].values()}),
                    simulation_state_restored=False)
                previous_initialization=saved["infos"].get("initialization")
                if isinstance(previous_initialization,dict) and previous_initialization.get("reference_cache"):
                    initialization["reference_cache"]=deepcopy(previous_initialization["reference_cache"])
                if isinstance(previous_initialization,dict) and "required_initialization_cases" in previous_initialization:
                    initialization["required_initialization_cases"]=deepcopy(previous_initialization["required_initialization_cases"])
            elif args["initialize_from"]:
                saved=spec.initialization_checkpoint(args["initialize_from"])
                initial_noise=runner.alg.policy.log_std.clone()
                runner.alg.policy.load_state_dict(spec.transfer_policy(saved,runner.alg.policy.state_dict()),strict=True)
                source_actor=deepcopy(runner.alg.policy.actor)
                source_actor.load_state_dict({key.removeprefix("actor."):value for key,value in saved["model_state_dict"].items() if key.startswith("actor.")})
                source_critic=deepcopy(runner.alg.policy.critic)
                source_critic_dim=saved["model_state_dict"]["critic.0.weight"].shape[1]
                source_critic[0]=torch.nn.Linear(source_critic_dim,source_critic[0].out_features,device=env.device)
                source_critic.load_state_dict({key.removeprefix("critic."):value for key,value in saved["model_state_dict"].items() if key.startswith("critic.")})
                with torch.inference_mode():
                    probe,extra=env.get_observations()
                    difference=(source_actor(probe)-runner.alg.policy.act_inference(probe)).abs().max()
                    critic_obs=extra["observations"]["critic"]
                    critic_difference=(source_critic(critic_obs[:,:source_critic_dim])-runner.alg.policy.evaluate(critic_obs)).abs().max()
                    if difference>1e-5 or critic_difference>1e-5:
                        raise RuntimeError("GPU Actor/Critic transfer changed the source functions")
                if source_critic_dim<268 and torch.count_nonzero(runner.alg.policy.critic[0].weight[:,source_critic_dim:]):
                    raise RuntimeError("V4 Critic task-state extension must start with zero weights")
                if not torch.equal(initial_noise,runner.alg.policy.log_std) or runner.alg.optimizer.state:
                    raise RuntimeError("Network transfer must preserve calibrated exploration noise and empty Adam")
                initialization.update(mode="actor_critic_transfer_new_optimizer",source_iteration=saved.get("iter",-1),
                    critic_reset=False,critic_source_input_dim=source_critic_dim,critic_destination_input_dim=268,
                    critic_new_columns_zero=source_critic_dim<268,optimizer_restored=False,
                    source_version=saved["infos"]["r1_unified_contract"]["version"],
                    gpu_actor_function_max_abs_error=float(difference),gpu_critic_function_max_abs_error=float(critic_difference),
                    noise_reset=True,initial_noise_std=runner.alg.policy.log_std.exp().detach().cpu().tolist(),adam_initial_state_count=0)
                reference=install_reference_cache(runner,args["initialize_from"],saved)
                if reference is not None:initialization["reference_cache"]=reference
                source_initialization=saved["infos"].get("initialization")
                if isinstance(source_initialization,dict) and "required_initialization_cases" in source_initialization:
                    required=source_initialization["required_initialization_cases"]
                    if not isinstance(required,list) or any(not isinstance(key,str) for key in required):
                        raise ValueError("required_initialization_cases must be a list of quality.case_key strings")
                    initialization["required_initialization_cases"]=deepcopy(required)
            spec.validate_policy_input_mask(runner.alg.policy,runner.alg.optimizer)
            if runner.alg.reference_loss_coef>0. and runner.alg.reference_observations is None:
                raise ValueError("Reference-preserving training requires verified successful samples before warmup")
            initialization.update(permanent_ignored_input_columns=[205,285],
                                  critic_warmup_updates=runner.alg.critic_warmup_updates,
                                  algorithm_class=type(runner.alg).__name__)
            runner.initialization_record=deepcopy(initialization)
            write_json(run/"initialization.json",initialization)
            start_updates=runner.alg.completed_updates
            if runner.quality_state is not None:
                quality_directory=run/"quality";quality_directory.mkdir(exist_ok=True)
                old_best=Path(runner.quality_state["best_checkpoint"])
                if not old_best.is_file():
                    raise ValueError("The resumed protected best checkpoint is missing: "+str(old_best))
                local_best=quality_directory/"best_eligible.pt"
                shutil.copyfile(old_best,local_best)
                runner.quality_state["best_checkpoint"]=str(local_best.resolve())
                if runner.quality_state.get("last"):
                    runner.quality_state["last"]["best_checkpoint"]=str(local_best.resolve())
                write_json(quality_directory/"quality_state.json",runner.quality_state)
            stopped=False;stop_reason=None
            try:
                if not args.get("skip_baseline_evaluation",False):
                    periodic_evaluation(runner,env,args,run,start_updates-1,baseline=True)
                env.unwrapped.drain_metrics();env.unwrapped.r1_torque_window.drain()
                runner.learn(num_learning_iterations=args["max_iterations"],init_at_random_ep_len=False)
                if args["eval_interval"] and (runner.quality_state is None or
                        runner.quality_state.get("last",{}).get("completed_updates")!=runner.alg.completed_updates):
                    periodic_evaluation(runner,env,args,run,runner.alg.completed_updates-1)
            except quality.QualityRegression as exc:
                stopped=True;stop_reason=str(exc)
                print("[V8] Training stopped on confirmed behavior regression: "+stop_reason,flush=True)
            completed=runner.alg.completed_updates
            final=run/"training"/f"model_{completed-1}.pt"
            runner.save(str(final))
            saved=spec.checkpoint(final)
            steps=sorted({int(state["step"]) for state in saved["optimizer_state_dict"]["state"].values()})
            quality_state=runner.quality_state
            last_quality=quality_state.get("last",{}) if quality_state else {}
            current_eligible=(last_quality.get("completed_updates")==completed and last_quality.get("eligible",False))
            # 预算结束不等于当前策略合格。只导出本次真实验收合格的当前策略；
            # 一次退化尚未触发停止时也不把它放进默认 exported/actor.pt。
            if not stopped and current_eligible:
                with torch.inference_mode():
                    obs,_=env.get_observations();export_actor(runner,obs,run)
            write_json(run/"training_result.json",{"status":"stopped_on_regression" if stopped else "completed",
                "checkpoint":str(final),"sha256":sha256(final),"stop_reason":stop_reason,
                "updates_this_run":completed-start_updates,"requested_updates":args["max_iterations"],
                "completed_updates":completed,"critic_only_updates_this_run":
                    min(completed,runner.alg.critic_warmup_updates)-min(start_updates,runner.alg.critic_warmup_updates),
                "actor_update_rounds_this_run":max(0,completed-runner.alg.critic_warmup_updates)-max(0,start_updates-runner.alg.critic_warmup_updates),
                "adam_steps":steps,"single_actor":True,"current_checkpoint_eligible":current_eligible,
                "actor_exported":not stopped and current_eligible,
                "export_note":"current checkpoint passed protected-case review" if not stopped and current_eligible else
                    "current actor not exported; use best_eligible_checkpoint after checking its evaluation scope",
                "best_eligible_checkpoint":quality_state.get("best_checkpoint") if quality_state else None,
                "quality_state":quality_state,"behavior_acceptance":"see_frozen_evaluations" if quality_state else "not_evaluated"})
        else:
            evaluate(env,runner,args,run,app)
    except BaseException as exc:
        import traceback
        write_json(run/"worker_error.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        traceback.print_exc()
        if env is None:
            # 部分构造失败时Kit清理可能等待未初始化对象；错误已落盘，终止独立worker释放GPU。
            import os
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        raise
    finally:
        if runner is not None and runner.writer is not None:runner.writer.close()
        if env is not None:env.close()
        app.close()


if __name__=="__main__":
    main()
