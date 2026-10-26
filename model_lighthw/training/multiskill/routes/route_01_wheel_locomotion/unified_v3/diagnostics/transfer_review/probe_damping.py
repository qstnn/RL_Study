"""单变量诊断：旧Actor在R1中仅把轮阻尼3恢复1，保持原准入条件。"""
import argparse,json,sys,traceback
from pathlib import Path
from review import (REPO,V3,SOURCES,GATE,gate,cpu_check,FrozenRunner,
                    CART,SPEC,now,write_json,snapshot,asset_manifest,sha256,module_from_file)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--output',type=Path);parser.add_argument('--candidate',choices=('route53','route38'),action='append');args=parser.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(),ensure_ascii=False,indent=2));return
    if args.output is None:parser.error('--output required for a GPU screen')
    names=args.candidate or ['route53'];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checks=cpu_check();sources={k:dict(v,checkpoint=str(v['checkpoint'])) for k,v in SOURCES.items()}
    write_json(out/'plan.json',dict(created_at=now(),gate=GATE,candidates=names,sources=sources,seed=43,physics='V3 with wheel damping 1 instead of 3; all other parameters unchanged',ppo_updates=0,maximum_cases=16*len(names),true_velocity='teacher diagnostic only; never deployable'))
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
        cfg.scene.robot.actuators['wheel'].damping=1.
        cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,15.,0.;cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd'
        cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
        dump_yaml(str(out/'params/env.yaml'),cfg)
        env=RslRlVecEnvWrapper(gym.make(SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions);summary=[]
        actual=env.unwrapped.scene['robot'].actuators['wheel'].damping
        if not torch.allclose(actual,torch.ones_like(actual)):raise RuntimeError('Damping treatment not applied')
        write_json(out/'treatment.json',dict(wheel_damping=actual.detach().cpu().tolist(),leg_cap_Nm=96.,wheel_cap_Nm=48.,source='transfer_review_02',ppo_updates=0))
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
