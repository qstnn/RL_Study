"""Isaac Sim统一R1 worker；实际训练、冻结回放和输出互相隔离。"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import time
from launch import ROOT,REPO,SOURCE,SPEC,now,sha256,validate_checkpoint,write_json
from launch import PROTOCOL as protocol


def verify_runtime(env,run):
    import torch
    from isaaclab.utils.math import matrix_from_quat
    base=env.unwrapped
    robot=base.scene["robot"]
    actual=(env.num_obs,env.num_privileged_obs,env.num_actions)
    if actual!=(300,242,16):
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
        raise RuntimeError("Wheel velocity gain differs from the V4 contract")
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
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v4 import torque,mdp
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
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_lighthw.wheel_rigid_cap_env_cfg import audit_runtime_caps
    report["rigid_body_angular_caps"]=audit_runtime_caps(base)
    write_json(run/"runtime_check.json",report)
    return report


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


def evaluate(env,runner,args,run,app):
    import numpy as np
    import torch
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v4 import core,mdp,torque
    base=env.unwrapped
    runner.load(args["checkpoint"],load_optimizer=False)
    policy=runner.get_inference_policy(device=env.device)
    cases=protocol.cases(args["eval_distance"]) if args["mode"]=="evaluate" else [{"slope_deg":args["slope_deg"],"payload_kg":args["payload_kg"],"towed_kg":args["towed_kg"],"speed":args["speed"],"goal_distance_m":args["eval_distance"]}]
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
                action=policy(obs)
                if not torch.isfinite(action).all():raise RuntimeError("Non-finite frozen Actor action")
                action[~active]=0.
                obs,_,done,_=env.step(action)
                measurements={k:v.cpu().numpy() for k,v in base.r1_last_metrics.items()}
                for i in active.nonzero().flatten().tolist():
                    row={k:float(v[i]) for k,v in measurements.items()};row["t"]=(step+1)*protocol.DT
                    rows[i].append(row);frames[i].append(base.r1_torque_recorder.numpy_trace(i))
                    if bool(done[i]):reasons[i]="terminated";active[i]=False
                    elif protocol.finished(rows[i],cases[i]):active[i]=False
                    elif step+1>=protocol.max_steps(cases[i]):reasons[i]="budget_exhausted";active[i]=False
                if not active.any():break
                if not args["headless"]:
                    time.sleep(max(0.,protocol.DT-(time.monotonic()-start)))
        for i,case in enumerate(cases):
            prefix=f"seed{seed}_case{i:02d}"
            with (run/(prefix+".csv")).open("w",newline="") as stream:
                if rows[i]:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[i][0]));writer.writeheader();writer.writerows(rows[i])
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


def periodic_evaluation(runner,env,args,run,iteration):
    """主训练暂停；独立进程冻结同一Actor评估当前许可工况，不更新训练网络/优化器/仿真状态。"""
    import subprocess
    import torch
    checkpoint=run/f"training/model_{iteration}.pt"
    runner.save(str(checkpoint))
    before=sha256(checkpoint)
    roots=env.unwrapped.scene["robot"].data.root_state_w.clone()
    output=run/f"periodic_evaluations/update_{iteration+1:04d}"
    output.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,str(ROOT/"scripts/evaluate.py"),"--headless","--checkpoint",str(checkpoint),
         "--eval_seeds","43","--output_dir",str(output)]
    for key in ("device","ground_friction","rolling_coefficient","leg_torque_limit","wheel_torque_limit",
                "hitch_height","unloaded_acceleration","loaded_acceleration","eval_distance"):
        cmd.extend(["--"+key,str(args[key])])
    print(f"[V4] Pause PPO after {iteration+1} updates; frozen {len(protocol.cases(args['eval_distance']))}-case screen -> {output}",flush=True)
    with (output.parent/f"update_{iteration+1:04d}_launcher.log").open("w") as stream:
        completed=subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode or not (output/"evaluation.json").is_file() or (output/"worker_error.json").exists():
        raise RuntimeError(f"Periodic evaluation failed to execute; inspect {output}")
    if sha256(checkpoint)!=before or not torch.equal(roots,env.unwrapped.scene["robot"].data.root_state_w):
        raise RuntimeError("Frozen evaluation changed the parent checkpoint or physical state")
    result=json.loads((output/"evaluation.json").read_text())
    record=dict(completed_updates=iteration+1,checkpoint=str(checkpoint),sha256=before,
                evaluation=str(output/"evaluation.json"),pass_count=result["pass_count"],
                complete_count=result["complete_count"],case_count=len(result["cases"]),ppo_updates_in_evaluation=0,
                parent_physical_state_unchanged=True)
    with (output.parent/"index.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
    print(f"[V4] Frozen screen: {record['pass_count']}/{record['case_count']} pass, {record['complete_count']} complete; resume PPO",flush=True)


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
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v4 import spec
        spec.register_task()
        cfg=load_cfg_from_registry(spec.TASK,"env_cfg_entry_point")
        agent=load_cfg_from_registry(spec.TASK,"rsl_rl_cfg_entry_point")
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
        if args["mode"] in ("play","evaluate"):
            # 默认不注册机器人跟随视角；需要跟随时由用户在窗口中设置。
            cfg.viewer.origin_type="world"
            cfg.episode_length_s=65.
            cfg.fixed_slope_deg=args["slope_deg"];cfg.fixed_payload_kg=args["payload_kg"]
            cfg.fixed_towed_kg=args["towed_kg"];cfg.fixed_speed_mps=args["speed"]
            cfg.observations.policy.enable_corruption=False
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
            def save(self,path,infos=None):
                spec.validate_policy_input_mask(self.alg.policy,self.alg.optimizer)
                metadata=dict(infos or {},r1_unified_contract=spec.CONTRACT,environment_signature=args["environment_signature"],
                              completed_updates=self.current_learning_iteration+1)
                return super().save(path,infos=metadata)

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

                if args["eval_interval"] and (locs["it"]+1)%args["eval_interval"]==0:
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
        runner=UnifiedRunner(env,agent.to_dict(),log_dir=str(run/"training") if args["mode"]=="train" else None,device=args["device"])
        spec.install_policy_input_mask(runner.alg.policy)
        if args["mode"]=="train":
            initialization={"mode":"fresh","optimizer_restored":False,"start_update":0}
            if args["resume"]:
                saved=spec.checkpoint(args["resume"])
                runner.load(args["resume"],load_optimizer=True)
                runner.current_learning_iteration=int(saved["iter"])+1
                runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]["lr"]
                def exact(left,right):
                    if isinstance(left,torch.Tensor):return torch.equal(left.cpu(),right.cpu())
                    if isinstance(left,dict):return left.keys()==right.keys() and all(exact(left[k],right[k]) for k in left)
                    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
                    return left==right
                model_match=exact(runner.alg.policy.state_dict(),saved["model_state_dict"])
                optimizer_match=exact(runner.alg.optimizer.state_dict(),saved["optimizer_state_dict"])
                if not model_match or not optimizer_match:raise RuntimeError("Resume did not restore exact network/Adam state")
                initialization.update(mode="resume",optimizer_restored=True,start_update=runner.current_learning_iteration,
                    model_restored_exact=model_match,optimizer_restored_exact=optimizer_match,
                    restored_learning_rate=runner.alg.learning_rate,
                    source_adam_steps=sorted({int(v["step"]) for v in saved["optimizer_state_dict"]["state"].values()}))
            elif args["initialize_from"]:
                saved=spec.initialization_checkpoint(args["initialize_from"])
                initial_critic=deepcopy(runner.alg.policy.critic.state_dict())
                initial_noise=runner.alg.policy.log_std.clone()
                runner.alg.policy.load_state_dict(spec.transfer_policy(saved,runner.alg.policy.state_dict()),strict=True)
                initialization.update(mode="actor_transfer_new_critic_optimizer",source_iteration=saved["iter"],
                                      critic_reset=True,optimizer_restored=False,source_version=saved["infos"]["r1_unified_contract"]["version"])
                source_actor=deepcopy(runner.alg.policy.actor)
                source_actor.load_state_dict({key.removeprefix("actor."):value for key,value in saved["model_state_dict"].items() if key.startswith("actor.")})
                source_actor=source_actor.to(env.device)
                with torch.inference_mode():
                    probe,_=env.get_observations()
                    difference=(source_actor(probe)-runner.alg.policy.act_inference(probe)).abs().max()
                    if difference>1e-5:raise RuntimeError("GPU initialization actor function differs")
                if not all(torch.equal(v,runner.alg.policy.critic.state_dict()[k]) for k,v in initial_critic.items()):
                    raise RuntimeError("New task must retain its fresh critic initialization")
                if not torch.equal(initial_noise,runner.alg.policy.log_std) or runner.alg.optimizer.state:
                    raise RuntimeError("Actor transfer must preserve fresh exploration noise and empty Adam")
                initialization.update(gpu_actor_function_max_abs_error=float(difference),noise_reset=True,
                    initial_noise_std=runner.alg.policy.log_std.exp().detach().cpu().tolist(),adam_initial_state_count=0)
            spec.validate_policy_input_mask(runner.alg.policy,runner.alg.optimizer)
            initialization["permanent_ignored_input_columns"]=[205,285]
            write_json(run/"initialization.json",initialization)
            env.unwrapped.drain_metrics()
            env.unwrapped.r1_torque_window.drain()
            runner.learn(num_learning_iterations=args["max_iterations"],init_at_random_ep_len=False)
            files=sorted((run/"training").glob("model_*.pt"),key=lambda p:int(p.stem.split("_")[-1]))
            final=files[-1];saved=spec.checkpoint(final)
            steps=sorted({int(s["step"]) for s in saved["optimizer_state_dict"]["state"].values()})
            with torch.inference_mode():
                obs,_=env.get_observations();export_actor(runner,obs,run)
            write_json(run/"training_result.json",{"checkpoint":str(final),"sha256":sha256(final),
                "updates_this_run":args["max_iterations"],"completed_updates":saved["iter"]+1,
                "adam_steps":steps,"single_actor":True,"behavior_acceptance":"not_evaluated"})
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
