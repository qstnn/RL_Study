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
    if actual!=(290,242,16):
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
            "rolling_coefficient":base.cfg.rolling_coefficient,"wheel_damping":3.,"passive_cart_drive_stiffness":0.,
            "payload_examples":base.r1_payload_audit,"physical_com_inertia_verified":base.r1_inertia_verified,"unloaded_inertia_urdf_verified":base.r1_unloaded_inertia_urdf_verified,"nonbase_inertia_unchanged":base.r1_nonbase_inertia_unchanged,"behavior_acceptance":"not_evaluated"}
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
    write_json(directory/"verification.json",{"actor_input":290,"actor_output":16,"samples":len(obs),
               "max_abs_error":float((expected-actual).abs().max()),"sha256":sha256(path),"history_required":True})


def evaluate(env,runner,args,run,app):
    import torch
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified import core,mdp
    base=env.unwrapped
    runner.load(args["checkpoint"],load_optimizer=False)
    policy=runner.get_inference_policy(device=env.device)
    cases=protocol.cases() if args["mode"]=="evaluate" else [{"slope_deg":args["slope_deg"],"payload_kg":args["payload_kg"],"towed_kg":args["towed_kg"],"speed":args["speed"]}]
    seeds=args["eval_seeds"] if args["mode"]=="evaluate" else [args["seed"]]
    reports=[]
    for seed in seeds:
        for index,case in enumerate(cases):
            base.r1_slope_rad.fill_(case["slope_deg"]*3.141592653589793/180)
            base.r1_tangent[:],base.r1_normal[:]=core.plane_basis(base.r1_slope_rad)
            base.r1_payload_kg.fill_(case["payload_kg"]);base.r1_towed_kg.fill_(case["towed_kg"])
            base.cfg.fixed_speed_mps=case["speed"]
            lane=SPEC.ANGLES.index(case["slope_deg"])
            base.scene.terrain.terrain_types.fill_(lane)
            base.scene.terrain.env_origins[:]=base.scene.terrain.terrain_origins[0,lane]
            mdp.set_payload_properties(base)
            env.seed(seed)
            rows=[];reason=None
            with torch.inference_mode():
                obs,_=env.reset()
                initial=base.scene["robot"].data.root_state_w[0].cpu().tolist()
                for step in range(protocol.STEPS):
                    start=time.monotonic()
                    if not app.is_running():reason="application_closed";break
                    action=policy(obs)
                    if not torch.isfinite(action).all():reason="non_finite_action";break
                    obs,_,done,_=env.step(action)
                    row={key:float(value[0]) for key,value in base.r1_last_metrics.items()}
                    row["t"]=step*.02
                    rows.append(row)
                    if done.any():reason="terminated";break
                    if not args["headless"]:
                        p=base.scene["robot"].data.root_pos_w[0].cpu().numpy()
                        base.sim.set_camera_view(p+[5.,-6.,3.5],p+[-.7,0.,0.])
                        time.sleep(max(0.,.02-(time.monotonic()-start)))
            prefix=f"seed{seed}_case{index:02d}"
            with (run/(prefix+".csv")).open("w",newline="") as stream:
                if rows:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            report=protocol.summarize(rows,case,reason)
            report.update(seed=seed,initial_root_state=initial)
            write_json(run/(prefix+".json"),report);reports.append(report)
            if reason=="application_closed":break
    write_json(run/"evaluation.json",{"checkpoint":args["checkpoint"],"sha256":sha256(args["checkpoint"]),
        "ppo_updates":0,"cases":reports,"all_cases_complete":all(r["complete"] for r in reports),
        "success_rate":sum(r["success"] for r in reports)/len(reports),"final_hardware_acceptance":"not_evaluated",
        "note":"Same initial states may produce identical seed results; no randomized robustness claim"})


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
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified import spec
        spec.register_task()
        cfg=load_cfg_from_registry(spec.TASK,"env_cfg_entry_point")
        agent=load_cfg_from_registry(spec.TASK,"rsl_rl_cfg_entry_point")
        cfg.scene.num_envs=args["num_envs"]
        cfg.seed=agent.seed=args["seed"]
        cfg.sim.device=agent.device=args["device"]
        cfg.ground_friction=args["ground_friction"];cfg.rolling_coefficient=args["rolling_coefficient"]
        cfg.scene.robot.spawn.usd_dir=str(run/"asset_cache/robot")
        cfg.scene.robot.spawn.usd_file_name="lighthw.usd"
        cfg.scene.trailer.spawn.asset_path=str(run/"assets/cart.urdf")
        cfg.scene.trailer.spawn.usd_dir=str(run/"asset_cache/trailer")
        cfg.scene.trailer.spawn.usd_file_name="trailer.usd"
        agent.max_iterations=args["max_iterations"]
        if args["mode"] in ("play","evaluate"):
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

        if args["mode"]=="smoke":
            terminations=0
            with torch.inference_mode():
                for _ in range(100):
                    obs,reward,done,extra=env.step(torch.zeros(env.num_envs,16,device=env.device))
                    if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():raise RuntimeError("Non-finite smoke")
                    terminations+=int(done.sum())
            write_json(run/"smoke.json",{"steps":100,"ppo_updates":0,"terminations":terminations,
                "condition_metrics":env.unwrapped.drain_metrics(),"behavior_acceptance":"not_evaluated",
                "note":"Zero-action control is not a trained loaded-slope policy; finite/reset-chain check only"})
            return
        runner=UnifiedRunner(env,agent.to_dict(),log_dir=str(run/"training") if args["mode"]=="train" else None,device=args["device"])
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
            elif args["warm_start"]:
                saved=validate_checkpoint(args["warm_start"])
                runner.alg.policy.load_state_dict(spec.transfer_stage1(saved,runner.alg.policy.state_dict()),strict=True)
                initialization.update(mode="stage1_network_transfer",source_iteration=saved["iter"],
                                      history_new_columns_zero=True,critic_extra_columns_zero=True)
                source_actor=deepcopy(runner.alg.policy.actor)
                source_actor[0]=torch.nn.Linear(57,512)
                source_actor.load_state_dict({key.removeprefix("actor."):value for key,value in saved["model_state_dict"].items() if key.startswith("actor.")})
                source_actor=source_actor.to(env.device)
                with torch.inference_mode():
                    probe,_=env.get_observations()
                    difference=(source_actor(probe[:,spec.latest_source_columns()])-runner.alg.policy.act_inference(probe)).abs().max()
                    if difference>1e-5:raise RuntimeError("GPU warm-start actor function differs")
                initialization["gpu_actor_function_max_abs_error"]=float(difference)
            write_json(run/"initialization.json",initialization)
            env.unwrapped.drain_metrics()
            runner.learn(num_learning_iterations=args["max_iterations"],init_at_random_ep_len=True)
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
