"""V5冻结Actor完整任务噪声检查；48环境，可加16环境原Actor同批对照，PPO/梯度更新为0。

调用需在lab环境中；本脚本复用V5 worker.evaluate原协议、50Hz CSV和1kHz力矩记录。
四个流是单seed矢量环境内的独立随机抽样，不冒充四个独立仿真种子或实机鲁棒验收。
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODES=(("clean",False,False),("observation_only",True,False),("action_only",False,True),("both",True,True))
NOISY_TERMS=("base_ang_vel","projected_gravity","joint_pos","joint_vel")
RAW_ACTION_STD=[.005]*12+[.01]*4


def build_trials(include_baseline=False):
    """每个工况内16环境使用相同物理初态；模式只改变对应env的输入噪声/动作采样。"""
    conditions=[("candidate","tow40_up25_static",0.,40.,0.),
                ("candidate","payload15_up25_distance",15.,0.,.3),
                ("candidate","unloaded_up25_fast",0.,0.,3.1)]
    if include_baseline:conditions.append(("baseline","tow40_up25_static",0.,40.,0.))
    trials=[];cases=[]
    for model,condition,payload,towed,speed in conditions:
        for mode_id,(mode,observation_noise,action_noise) in enumerate(MODES):
            for stream in range(4):
                index=len(trials)
                trial=dict(case_id=index,model=model,condition=condition,noise_mode=mode,noise_mode_id=mode_id,
                           stream=stream,observation_noise=observation_noise,action_noise=action_noise)
                case=dict(slope_deg=25.,payload_kg=payload,towed_kg=towed,speed=speed,goal_distance_m=1.,
                          source_model=model,condition=condition,noise_mode=mode,noise_stream=stream,diagnostic_case_id=index)
                trials.append(trial);cases.append(case)
    return trials,cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--baseline_checkpoint",type=Path,
                        help="Optional V4/V5 original Actor: append16 tow40/up25/static cases in the same V5 batch")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=43)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve(strict=True)
    sys.path.insert(0,str(ROOT/"scripts"))
    import launch
    if Path(launch.__file__).resolve()!=ROOT/"scripts/launch.py":
        raise RuntimeError("Frozen noise probe imported a foreign launch module")
    SPEC,CART=launch.SPEC,launch.CART
    sys.path.insert(0,str(launch.REPO/"source/rl_training"))
    write_json,now,sha256=launch.write_json,launch.now,launch.sha256
    saved=SPEC.checkpoint(checkpoint)
    checkpoint_sha=sha256(checkpoint)
    model_states={"candidate":saved}
    sources={"candidate":dict(path=str(checkpoint),sha256=checkpoint_sha,
                              version=saved["infos"]["r1_unified_contract"]["version"])}
    if args.baseline_checkpoint is not None:
        baseline=args.baseline_checkpoint.resolve(strict=True)
        original=SPEC.initialization_checkpoint(baseline)
        model_states["baseline"]=original
        sources["baseline"]=dict(path=str(baseline),sha256=sha256(baseline),
                                 version=original["infos"]["r1_unified_contract"]["version"])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    trials,cases=build_trials(include_baseline="baseline" in sources)
    for trial,case in zip(trials,cases):
        source=sources[trial["model"]]
        trial.update(source_checkpoint=source["path"],source_sha256=source["sha256"])
        case.update(source_checkpoint=source["path"],source_sha256=source["sha256"])
    request=dict(created_at=now(),checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,sources=sources,
                 seed=args.seed,device=args.device,num_envs=len(cases),trials=trials,cases=cases,
                 observation_noise_scale=.1,raw_action_sampling_std=RAW_ACTION_STD,ppo_updates=0,
                 full_original_protocol=True,static_duration_s=20.,moving_hold_settle_s=2.,moving_hold_s=5.,
                 loaded_goal_distance_m=1.,loaded_max_budget_s=60.,unloaded_max_budget_s=30.,
                 independence="Independent vectorized draws within one simulator seed; not four simulator seeds",
                 scope="Frozen stochastic screening; protocol completion is separate from all acceptance checks")
    write_json(out/"request.json",request)
    launch.snapshot(out)
    CART.write_cart_urdf(out/"assets/cart.urdf")
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=args.device).app
    env=None
    try:
        import torch
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.noise import uniform_noise
        from isaaclab.utils.io import dump_yaml
        worker=launch.module_from_file("_v5_frozen_noise_worker",ROOT/"scripts/worker.py")
        SPEC.register_task()
        cfg=load_cfg_from_registry(SPEC.TASK,"env_cfg_entry_point")
        agent=load_cfg_from_registry(SPEC.TASK,"rsl_rl_cfg_entry_point")
        cfg.scene.num_envs=len(cases);cfg.seed=agent.seed=args.seed
        cfg.sim.device=agent.device=args.device
        cfg.episode_length_s=65.;cfg.fixed_speed_mps=0.
        cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,0.,40.
        cfg.viewer.origin_type="world"
        if abs(cfg.observation_noise_scale-.1)>1e-12:
            raise RuntimeError("V5 native observation noise scale differs from the requested 0.1")
        cfg.observations.policy.enable_corruption=True
        obs_mask=torch.tensor([trial["observation_noise"] for trial in trials],device=args.device)[:,None]
        action_mask=torch.tensor([trial["action_noise"] for trial in trials],device=args.device)[:,None]
        noise_stats={};noise_config={}

        def plain(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
            if isinstance(value,(list,tuple)):return [plain(item) for item in value]
            return value

        def conditional_noise(name):
            def apply(data,noise_cfg):
                # 使用真实Isaac uniform_noise及原幅值；native ObservationManager继续负责clip/scale/history。
                noisy=uniform_noise(data,noise_cfg)
                result=torch.where(obs_mask,noisy,data)
                delta=(result-data).abs().amax(dim=-1)
                if name not in noise_stats:
                    noise_stats[name]=dict(calls=0,max_abs=torch.zeros_like(delta))
                noise_stats[name]["calls"]+=1
                noise_stats[name]["max_abs"]=torch.maximum(noise_stats[name]["max_abs"],delta)
                return result
            apply.__name__="conditional_native_uniform_"+name
            return apply

        for name in NOISY_TERMS:
            term=getattr(cfg.observations.policy,name)
            if term.noise.operation!="add":raise RuntimeError("Expected native additive proprioception noise")
            noise_config[name]=dict(operation=term.noise.operation,n_min=plain(term.noise.n_min),
                                   n_max=plain(term.noise.n_max),observation_scale=plain(term.scale),
                                   observation_clip=plain(term.clip),history_frames=cfg.observations.policy.history_length)
            term.noise.func=conditional_noise(name)
        cfg.scene.robot.spawn.usd_dir=str(out/"cache/robot");cfg.scene.robot.spawn.usd_file_name="robot.usd"
        cfg.scene.trailer.spawn.asset_path=str(out/"assets/cart.urdf")
        cfg.scene.trailer.spawn.usd_dir=str(out/"cache/cart");cfg.scene.trailer.spawn.usd_file_name="cart.usd"
        dump_yaml(str(out/"params/env.yaml"),cfg)
        dump_yaml(str(out/"params/agent.yaml"),agent)
        write_json(out/"noise_configuration.json",dict(native_observation_noise_scale=cfg.observation_noise_scale,
                   policy_enable_corruption=True,terms=noise_config,raw_action_std=RAW_ACTION_STD,
                   native_pipeline="func -> modifier -> noise -> clip -> scale -> history; conditional mask at noise only"))
        env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions)
        original_verify=worker.verify_runtime

        def verify_with_initial_state(actual_env,directory):
            report=original_verify(actual_env,directory)
            base=actual_env.unwrapped
            states={"robot_root":base.scene["robot"].data.root_state_w,
                    "robot_joints":torch.cat((base.scene["robot"].data.joint_pos,base.scene["robot"].data.joint_vel),dim=-1),
                    "trailer_root":base.scene["trailer"].data.root_state_w,
                    "trailer_joints":torch.cat((base.scene["trailer"].data.joint_pos,base.scene["trailer"].data.joint_vel),dim=-1)}
            comparison={}
            condition_indices={}
            for condition in dict.fromkeys(trial["condition"] for trial in trials):
                indices=[i for i,trial in enumerate(trials) if trial["condition"]==condition]
                ids=torch.tensor(indices,device=actual_env.device)
                # 可选原Actor的16环境与新Actor的16个同工况环境共同比较，不仅各自组内比较。
                group={name:float((value[ids]-value[ids[0]]).abs().max()) for name,value in states.items()}
                if max(group.values())>1e-7:
                    raise RuntimeError("Noise/model comparison has mismatched initial physical states: "+repr(group))
                comparison[condition]=group;condition_indices[condition]=indices
            write_json(out/"initial_state_comparison.json",dict(same_within_each_condition=True,
                candidate_baseline_checked_together="baseline" in sources,condition_case_indices=condition_indices,
                max_absolute_difference=comparison,initial_robot_root=states["robot_root"].cpu().tolist()))
            return report

        worker.verify_runtime=verify_with_initial_state

        class FrozenRunner:
            def __init__(self):
                self.actors={};self.before={};self.model_indices={};self.checkpoint_stds={}
                for model,model_saved in model_states.items():
                    state=model_saved["model_state_dict"]
                    actor=torch.nn.Sequential(torch.nn.Linear(300,512),torch.nn.ELU(),torch.nn.Linear(512,256),
                        torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16)).to(env.device).eval()
                    actor.load_state_dict({key.removeprefix("actor."):value for key,value in state.items() if key.startswith("actor.")})
                    actor.requires_grad_(False)
                    self.actors[model]=actor
                    self.before[model]={key:value.detach().clone() for key,value in actor.state_dict().items()}
                    self.model_indices[model]=torch.tensor([i for i,trial in enumerate(trials) if trial["model"]==model],device=env.device)
                    self.checkpoint_stds[model]=plain(state["log_std"].exp() if "log_std" in state else state["std"])
                # 原Actor也只加载300→16均值网络，不创建/恢复Critic或Adam；两模型用相同本轮扰动幅值。
                self.std=torch.tensor(RAW_ACTION_STD,device=env.device)
                self.checkpoint_std=self.checkpoint_stds["candidate"]
                self.steps=0;self.sampled_noise_max=torch.zeros(len(cases),device=env.device)
            def load(self,path,load_optimizer=False):
                if load_optimizer or Path(path).resolve()!=checkpoint or sha256(path)!=checkpoint_sha:
                    raise RuntimeError("Frozen diagnostic attempted foreign checkpoint or optimizer loading")
            def get_inference_policy(self,device=None):
                def infer(obs):
                    action=torch.empty((len(cases),16),dtype=obs.dtype,device=obs.device)
                    for model,actor in self.actors.items():
                        indices=self.model_indices[model]
                        action[indices]=actor(obs[indices])
                    noise=action_mask*torch.randn_like(action)*self.std
                    self.sampled_noise_max=torch.maximum(self.sampled_noise_max,noise.abs().amax(-1))
                    self.steps+=1
                    if self.steps%250==0:print("V5_NOISE_PROBE_STEPS",self.steps,flush=True)
                    return action+noise
                return infer

        runner=FrozenRunner()
        # 只替换case列表；finished/summarize/max_steps全部保留完整生产协议。
        worker.protocol.cases=lambda distance:cases
        evaluation_args=dict(mode="evaluate",eval_distance=1.,eval_seeds=[args.seed],seed=args.seed,
                             checkpoint=str(checkpoint),command_profile="slew",headless=True)
        with torch.inference_mode():worker.evaluate(env,runner,evaluation_args,out,app)
        model_integrity={}
        for model,actor in runner.actors.items():
            for name,value in actor.state_dict().items():
                if not torch.equal(value,runner.before[model][name]):raise RuntimeError("Frozen Actor weights changed: "+model)
            after=sha256(sources[model]["path"])
            if after!=sources[model]["sha256"]:raise RuntimeError("Input checkpoint changed during diagnostic: "+model)
            model_integrity[model]=dict(weights_changed=False,checkpoint_sha256_before=sources[model]["sha256"],
                                        checkpoint_sha256_after=after,critic_loaded=False,optimizer_loaded=False)
        actual_noise={name:dict(calls=stat["calls"],max_abs_by_case=stat["max_abs"].cpu().tolist())
                      for name,stat in noise_stats.items()}
        clean_ids=(~obs_mask[:,0]).nonzero().flatten()
        for name,stat in noise_stats.items():
            if torch.count_nonzero(stat["max_abs"][clean_ids]):raise RuntimeError("Observation noise leaked into a clean mode")
            if name not in noise_config:raise RuntimeError("Unexpected native noise term")
        if set(actual_noise)!=set(NOISY_TERMS):raise RuntimeError("Some native noise functions were not called")
        if torch.count_nonzero(runner.sampled_noise_max[(~action_mask[:,0]).nonzero().flatten()]):
            raise RuntimeError("Action sampling leaked into a mean-action mode")
        result=json.loads((out/"evaluation.json").read_text())
        summaries=[]
        for trial,report in zip(trials,result["cases"]):
            prefix=f"seed{args.seed}_case{trial['case_id']:02d}"
            report["noise_diagnostic"]=dict(**trial,observation_noise_scale=.1,
                native_noise_parameters=noise_config,raw_action_std=RAW_ACTION_STD,
                effective_action_std=RAW_ACTION_STD if trial["action_noise"] else [0.]*16,
                csv=prefix+".csv",motor_trace=prefix+"_motor_trace.npz")
            write_json(out/(prefix+".json"),report)
            summaries.append(dict(**trial,duration_s=report["observed"]["duration_s"],
                complete=report["complete"],success=report["success"],failed_checks=report.get("failed_checks",[]),
                termination=report["observed"]["termination_causes"],metrics=report.get("metrics",{}),
                report=prefix+".json",csv=prefix+".csv",motor_trace=prefix+"_motor_trace.npz"))
        result.update(noise_diagnostic=True,full_original_protocol=True,observation_noise_scale=.1,
                      raw_action_sampling_std=RAW_ACTION_STD,checkpoint_noise_std=runner.checkpoint_std,
                      checkpoint_noise_std_by_model=runner.checkpoint_stds,sources=sources,model_integrity=model_integrity,
                      independence=request["independence"],weights_changed=False)
        write_json(out/"evaluation.json",result)
        groups=[]
        for model,condition in dict.fromkeys((trial["model"],trial["condition"]) for trial in trials):
            for mode,_,_ in MODES:
                group=[row for row in summaries if row["model"]==model and row["condition"]==condition and row["noise_mode"]==mode]
                groups.append(dict(model=model,condition=condition,noise_mode=mode,trials=len(group),
                    source_checkpoint=sources[model]["path"],source_sha256=sources[model]["sha256"],
                    complete_count=sum(row["complete"] for row in group),pass_count=sum(row["success"] for row in group)))
        write_json(out/"diagnostic_summary.json",dict(finished_at=now(),checkpoint=str(checkpoint),
            checkpoint_sha256_before=checkpoint_sha,checkpoint_sha256_after=sha256(checkpoint),
            ppo_updates=0,weights_changed=False,sources=sources,model_integrity=model_integrity,
            full_original_protocol=True,groups=groups,trials=summaries,
            observation_noise=noise_config,observed_native_noise=actual_noise,
            raw_action_std=RAW_ACTION_STD,checkpoint_noise_std=runner.checkpoint_std,
            checkpoint_noise_std_by_model=runner.checkpoint_stds,
            observed_raw_action_noise_max_by_case=runner.sampled_noise_max.cpu().tolist(),
            noise_stat_scope="All native calls including setup/reset/inactive slots; task metrics use active protocol rows only",
            independence=request["independence"],final_hardware_acceptance="not_evaluated"))
        for group in groups:print("V5_NOISE_PROBE_RESULT",json.dumps(group),flush=True)
    except BaseException as exc:
        write_json(out/"worker_error.json",dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        traceback.print_exc()
        if env is None:
            import os
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=="__main__":
    main()
