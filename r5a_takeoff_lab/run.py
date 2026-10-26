"""USB 独立起跳探针，保留完整输入、源码和物理轨迹；不启动 PPO。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
CORE=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw'
ROUTE=ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/highjump'
sys.path.insert(0,str(ROUTE/'scripts'))
sys.path.insert(0,str(CORE))

def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(run):
    request=json.loads((run/'request.json').read_text());app=None;env=None;video=None
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=request.get('video',False),device='cuda:0').app
        import numpy as np
        import torch
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.env import JumpLandingEnv,JumpLandingEnvCfg
        from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.protocol import reference_pose
        from control import Reference
        class Probe(JumpLandingEnv):
            def _apply_action(self):
                self._advance_protocol(self._sim_step_counter-1)
                self._capture(self._sim_step_counter-1)
                t=self.episode_physics_steps*self.physics_dt
                pose=self.reference.pose(t)
                legacy=reference_pose(t,self.state,self.reference_profiles)
                pose=torch.where((self.reference.params['legacy']>0)[:,None],legacy,pose)
                limits=self.robot.data.joint_pos_limits[:,self.leg_ids]
                self.targets.copy_(pose.clamp(limits[:,:,0],limits[:,:,1]))
                self.robot.set_joint_position_target(self.targets,joint_ids=self.leg_ids)
                self.robot.set_joint_velocity_target(torch.zeros_like(self.actions[:,12:]),joint_ids=self.wheel_ids)
            def _get_dones(self):
                _,timeout=super()._get_dones()
                # 只在诊断子类允许倾斜和失败后继续记录；不能据此算 J1/后空翻通过。
                return torch.zeros_like(timeout),timeout
            def _get_rewards(self):return torch.zeros(self.num_envs,device=self.device)
        profiles=request['profiles'];n=len(profiles)
        cfg=JumpLandingEnvCfg();cfg.scene.num_envs=n;cfg.seed=request['seed']
        cfg.episode_length_s=request['seconds'];cfg.sim.device='cuda:0'
        cfg.scene.robot.spawn.asset_path=str(run/'input/robot.urdf')
        cfg.scene.robot.spawn.usd_dir=str(run/'asset_cache');cfg.scene.robot.spawn.usd_file_name='probe.usd'
        if request.get('video'):
            from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.appearance import sky_path
            cfg.scene.sky_light.spawn.texture_file=sky_path()
        env=Probe(cfg,render_mode='rgb_array' if request.get('video') else None)
        env.reset(seed=request['seed']);env.reference=Reference(env,profiles)
        if request.get('video'):
            import imageio.v2 as imageio
            import imageio_ffmpeg
            os.environ['IMAGEIO_FFMPEG_EXE']=imageio_ffmpeg.get_ffmpeg_exe()
            video=imageio.get_writer(str(run/'replay.mp4'),fps=50,codec='libx264',quality=7,macro_block_size=1)
        d=env.robot.data;phys=env.robot.root_physx_view
        metadata=dict(body_names=env.robot.body_names,joint_names=env.robot.joint_names,
            contact_body_names=env.contacts.body_names,wheel_body_ids=env.wheel_body_ids,
            masses=phys.get_masses()[0].cpu().tolist(),inertias_link_axes=phys.get_inertias()[0].cpu().tolist(),
            com_poses_in_link=phys.get_coms()[0].cpu().tolist(),
            effort_limits=d.joint_effort_limits[0].cpu().tolist(),velocity_limits=d.joint_vel_limits[0].cpu().tolist(),
            physics_dt=env.physics_dt,policy_dt=env.step_dt,seed=request['seed'],
            self_collisions=cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
            gravity=list(cfg.sim.gravity),friction=1.,restitution=0.,ppo_updates=0,
            production_termination_changed=False,probe_termination='fixed horizon, failures retained',
            inertia_frame='full tensor at COM expressed in rigid-body-prim axes')
        assert abs(sum(metadata['masses'])-11.66811148685705)<.002
        assert metadata['self_collisions'] and metadata['physics_dt']==.001
        for j,name in enumerate(env.robot.joint_names):
            assert abs(metadata['effort_limits'][j]-(24. if 'wheel' in name else 48.))<1e-4
            assert abs(metadata['velocity_limits'][j]-(37.7 if 'wheel' in name else 12.57))<1e-4
        write(run/'physics.json',metadata)
        fields={};buffers={};start=env._sim_step_counter;capacity=env.max_episode_length*20+1
        def sample(e,tick):
            index=tick-start
            if index<0:return
            data=e.robot.data;m=e.metrics();t=e.episode_physics_steps*e.physics_dt
            if index and e.reference.last_tick!=tick:
                e.reference.update(t,m);e.reference.last_tick=tick
            values=dict(com=m['com'],com_velocity=m['com_vel'],root_state=data.root_state_w,
                body_com_position=data.body_com_pos_w,body_com_velocity=data.body_com_lin_vel_w,
                body_link_quaternion=data.body_link_quat_w,body_angular_velocity=data.body_com_ang_vel_w,
                contact_force=e.contacts.data.net_forces_w,joint_position=data.joint_pos,
                joint_velocity=data.joint_vel,applied_torque=data.applied_torque,leg_targets=e.targets,
                wheel_clearance=m['clearance'],tilt=m['tilt'],base_height=m['base_z'],
                legacy_failed=e.state.failed,legacy_success=e.state.success,legacy_stable=e.state.stable_time,
                release=e.reference.released,landed=e.reference.landed,backflip_angle=e.reference.rotation)
            for key,value in values.items():
                if key not in buffers:buffers[key]=torch.empty((capacity,*value.shape),device=e.device,dtype=value.dtype)
                buffers[key][index].copy_(value)
            if video is not None and index%20==0:video.append_data(e.render())
            if index and index%500==0:print(f'[PROBE] physics {index}/{capacity-1}',flush=True)
        env.physics_observer=sample;env._last_capture_tick=-1;env._capture(start)
        with torch.inference_mode():
            for _ in range(env.max_episode_length):env.step(torch.zeros(n,16,device=env.device))
        env.physics_observer=None
        arrays={k:v.cpu().numpy() for k,v in buffers.items()};arrays['t']=np.arange(capacity)*env.physics_dt
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(run/'trace.npz',**arrays)
        if video is not None:video.close();video=None
        write(run/'result.json',dict(status='completed',profiles=n,physics_samples=capacity,ppo_updates=0))
        print('[PROBE] saved',run,flush=True)
    except BaseException:
        (run/'error.txt').write_text(traceback.format_exc());raise
    finally:
        if video is not None:video.close()
        if env is not None:env.close()
        if app is not None:app.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--profiles',type=Path)
    p.add_argument('--run-dir',type=Path);p.add_argument('--seconds',type=float,default=2.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--video',action='store_true')
    a=p.parse_args()
    if a.worker:return worker(a.worker.resolve())
    if not a.profiles:p.error('--profiles required')
    profiles=json.loads(a.profiles.read_text());assert 1<=len(profiles)<=128
    if a.video and len(profiles)!=1:p.error('--video requires exactly one profile')
    assert 1.4<=a.seconds<=4. and abs(a.seconds/.02-round(a.seconds/.02))<1e-8
    run=(a.run_dir or ROOT/'usb_review/route_05a'/('takeoff_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    run.mkdir(parents=True,exist_ok=False)
    from launch import asset_manifest, module_from_file, URDF, snapshot
    from control import baseline
    profiles=[{**baseline(),**x} for x in profiles]
    # Validate reference envelopes including front/rear asymmetric crouch before simulation.
    for x in profiles:
        assert x['push_start']+min(0.,x['rear_delay'])+min(0.,x['knee_delay'])-x['hold']>x['crouch_start']+.02
        for offset in (0.,x['rear_crouch_offset']):
            assert -1.57<=x['crouch_thigh']+offset<=1.57 and -2.5<=x['crouch_calf']-offset<=-.087
        assert -1.57<=x['tuck_thigh']<=1.57 and -2.5<=x['tuck_calf']<=-.087
    asset=module_from_file('_takeoff_payload_asset',CORE/'payload_asset.py')
    model=asset.write_payload_urdf(URDF,run/'input/robot.urdf',0.)
    write(run/'model.json',model);write(run/'assets.json',asset_manifest())
    snapshot(run)
    manifest=json.loads((run/'source_manifest.json').read_text())
    for source in HERE.glob('*.py'):
        rel=str(source.relative_to(ROOT));dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);manifest[rel]=sha(source)
    write(run/'source_manifest.json',manifest)
    write(run/'request.json',dict(profiles=profiles,seed=a.seed,seconds=a.seconds,video=a.video))
    with (run/'console.log').open('w') as f:
        code=subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(run)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    changed=[s for s,h in manifest.items() if sha(ROOT/s)!=h]
    write(run/'source_check.json',dict(changed=changed,passed=not changed))
    if code or changed:raise RuntimeError(f'Probe failed: code={code}, changed={changed}; see {run}/console.log')
    from analyze import analyze
    analyze(run)
    print('[USB REVIEW]',run,flush=True)


if __name__=='__main__':main()
