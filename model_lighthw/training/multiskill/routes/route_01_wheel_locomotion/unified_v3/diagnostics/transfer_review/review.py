"""同一R1物理条件下冻结比较旧Actor；PPO0，真实速度仅用于教师诊断。"""
import argparse,json,sys,traceback
from pathlib import Path
from contract import REPO,V3,SOURCES,GATE,gate
sys.path.insert(0,str(V3/'scripts'));sys.path.insert(0,str(REPO/'source/rl_training'))
from launch import CART,SPEC,now,write_json,snapshot,asset_manifest,sha256,module_from_file

def load_actor(name,device='cpu'):
    import torch
    r=SOURCES[name]
    if sha256(r['checkpoint'])!=r['sha256']:raise ValueError('Pinned source changed: '+name)
    saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);state=saved['model_state_dict']
    if not all(torch.isfinite(t).all() for t in state.values()):raise ValueError('Non-finite source')
    net=torch.nn.Sequential(torch.nn.Linear(r['input_dim'],512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,16))
    net.load_state_dict({k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')},strict=True)
    return net.to(device).eval()

def compatibility():
    """逐项核对共有观测和实际动作语义，不能只比较维度。"""
    import yaml
    current=yaml.load((V3/'runs/2026-10-16_16-29-49.804831/params/env.yaml').read_text(),Loader=yaml.BaseLoader)
    result={}
    for name in ('route53','route38'):
        folder=SOURCES[name]['checkpoint'].parent
        old=yaml.load((folder/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
        agent=yaml.load((folder/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
        assert agent['policy']['activation']=='elu' and agent['empirical_normalization']=='false'
        checks={}
        for term in SPEC.BASE_TERMS:
            a,b=old['observations']['policy'][term],current['observations']['policy'][term]
            for k in ('func','params','scale','clip'):checks[f'observation/{term}/{k}']=a.get(k)==b.get(k)
        for term in ('joint_pos','joint_vel'):
            for k in ('joint_names','scale','clip','preserve_order'):checks[f'action/{term}/{k}']=old['actions'][term].get(k)==current['actions'][term].get(k)
        checks['default_pose']=old['scene']['robot']['init_state']['joint_pos']==current['scene']['robot']['init_state']['joint_pos']
        checks['default_velocity_zero']=set(old['scene']['robot']['init_state']['joint_vel'].values())=={'0.0'}
        checks['linear_velocity_scale']=old['observations']['policy']['base_lin_vel']['scale']=='2.0'
        if not all(checks.values()):raise ValueError(f'Incompatible semantic mapping {name}: {[k for k,v in checks.items() if not v]}')
        load_actor(name)
        result[name]=dict(checks=checks,source_actor_dim=60,source_critic_dim=236,
          extra_teacher_input='true body linear velocity * 2; diagnostic only',mirror_branch='inactive: R1 command vy=0',
          physical_differences=dict(wheel_damping=[1.,3.],leg_cap_Nm=[48.,96.],wheel_cap_Nm=[24.,48.]))
    return result

def cpu_check():
    import numpy as np
    import torch
    torch.set_num_threads(1);result=compatibility();actor=load_actor('route53')
    trace=REPO/'model_lighthw/training/routes/route_53_ppo_mixed_recovery/evaluations/2026-10-15_14-11-17_474350_retention/candidate/forward3/synchronized.npz'
    z=np.load(trace);obs=torch.from_numpy(z['control_obs'])
    with torch.inference_mode():action=actor(obs)
    error=float((action-torch.from_numpy(z['control_raw_action'])).abs().max());assert error<2e-4
    hist=torch.randn(len(obs),290);hist[:,SPEC.latest_source_columns()]=obs[:,3:]
    assert torch.equal(torch.cat((obs[:,:3],hist[:,SPEC.latest_source_columns()]),dim=-1),obs)
    baseline=json.loads((V3/'runs/2026-10-16_16-29-49.804831/periodic_evaluations/update_0500/evaluation.json').read_text())
    rejected=gate(baseline);assert not rejected['teacher_pre_screen_passed'] and not rejected['checks']['tow40_up25_survives_startup']
    assert baseline['cases'][10]['observed']['duration_s']==.6
    return dict(status='passed',compatibility=result,source_action_replay_max_error=error,latest_frame_mapping_exact=True,current_v3_gate=rejected,ppo_updates=0)

class FrozenRunner:
    """无优化器或训练接口；教师60维Actor的真线速度不用于部署。"""
    def __init__(self,name,env):
        self.name,self.env=name,env;self.actor=load_actor(name,env.device)
        self.inputs=[];self.source_inputs=[];self.outputs=[];self.max_observation_error=0.
    def load(self,path,load_optimizer=False):
        if Path(path)!=SOURCES[self.name]['checkpoint'] or load_optimizer:raise ValueError('Cannot change pinned source or restore optimizer')
    def get_inference_policy(self,device=None):
        import torch
        def policy(obs):
            base=self.env.unwrapped
            if SOURCES[self.name]['input_dim']==60:
                latest=obs[:,SPEC.latest_source_columns()]
                source=torch.cat((base.scene['robot'].data.root_lin_vel_b.clamp(-100.,100.)*2.,latest),dim=-1)
                # Critic首60列独立交叉核对同次具名观测，不调用compute推进history。
                error=float((source-base.obs_buf['critic'][:,:60]).abs().max())
                self.max_observation_error=max(self.max_observation_error,error)
                if error>2e-5:raise RuntimeError(f'Named source observation mismatch: {error}')
                if torch.count_nonzero(source[:,10]):raise RuntimeError('R1 vy must remain zero; mirror branch not applicable')
            else:source=obs
            action=self.actor(source)
            self.inputs.append(obs.detach().cpu().numpy().copy());self.source_inputs.append(source.detach().cpu().numpy().copy());self.outputs.append(action.detach().cpu().numpy().copy())
            return action
        return policy
    def save_inputs(self,directory,reports):
        import numpy as np
        np.savez_compressed(directory/'policy_inputs.npz',observation_history=np.stack(self.inputs),source_observation=np.stack(self.source_inputs),raw_action=np.stack(self.outputs),valid_case_lengths=np.asarray([c['samples'] for c in reports]))
        write_json(directory/'adapter_check.json',dict(source=self.name,max_named_observation_error=self.max_observation_error,source_checkpoint_sha256=SOURCES[self.name]['sha256'],deployable=self.name=='control_v3',privileged_velocity_used=SOURCES[self.name]['input_dim']==60,ppo_updates=0,note='Exclude post-termination padded samples using valid_case_lengths.'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--output',type=Path);parser.add_argument('--candidate',choices=tuple(SOURCES),action='append');args=parser.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(),ensure_ascii=False,indent=2));return
    if args.output is None:parser.error('--output required for a GPU screen')
    names=args.candidate or ['control_v3','route53','route38'];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checks=cpu_check();sources={k:dict(v,checkpoint=str(v['checkpoint'])) for k,v in SOURCES.items()}
    write_json(out/'plan.json',dict(created_at=now(),gate=GATE,candidates=names,sources=sources,seed=43,physics='unmodified V3',ppo_updates=0,maximum_cases=16*len(names),true_velocity='teacher diagnostic only; never deployable'))
    write_json(out/'cpu_checks.json',checks);write_json(out/'asset_manifest.json',asset_manifest());snapshot(out);CART.write_cart_urdf(out/'assets/cart.urdf')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app;env=None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        worker=module_from_file("_r1_transfer_v3_worker",V3/"scripts/worker.py")
        if worker.protocol is not __import__("launch").PROTOCOL:raise RuntimeError("Wrong evaluation protocol loaded")
        SPEC.register_task();cfg=load_cfg_from_registry(SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(SPEC.TASK,'rsl_rl_cfg_entry_point')
        cfg.scene.num_envs=16;cfg.seed=agent.seed=43;cfg.episode_length_s=65.;cfg.fixed_speed_mps=.3
        cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,15.,0.;cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
        cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
        dump_yaml(str(out/'params/env.yaml'),cfg)
        env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions);summary=[]
        for name in names:
            directory=out/name;directory.mkdir();runner=FrozenRunner(name,env)
            request=dict(mode='evaluate',eval_distance=1.,eval_seeds=[43],seed=43,checkpoint=str(SOURCES[name]['checkpoint']),command_profile='slew',headless=True)
            write_json(directory/'request.json',dict(request,diagnostic=True,teacher_true_velocity=name!='control_v3'))
            with torch.inference_mode():worker.evaluate(env,runner,request,directory,app)
            evaluation=json.loads((directory/'evaluation.json').read_text());runner.save_inputs(directory,evaluation['cases']);decision=gate(evaluation)
            write_json(directory/'gate.json',decision)
            row=dict(candidate=name,full_R1_pass_count=evaluation['pass_count'],complete_count=evaluation['complete_count'],**decision);summary.append(row)
            write_json(out/'summary.json',dict(status='running',candidates=summary,ppo_updates=0));print('TRANSFER_REVIEW',json.dumps(row,ensure_ascii=False),flush=True);del runner
        for record in SOURCES.values():
            if sha256(record['checkpoint'])!=record['sha256']:raise RuntimeError('Source changed during screen')
        write_json(out/'summary.json',dict(status='completed',finished_at=now(),candidates=summary,ppo_updates=0,note='Teacher pass permits adapter development, not training. Failure does not prove physical impossibility.'))
    except BaseException as exc:
        write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
        # Kit关闭可能覆盖异常退出码；错误已保存，结束本次独立诊断进程。
        import os
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
    finally:
        if env is not None:env.close()
        app.close()
if __name__=='__main__':main()
