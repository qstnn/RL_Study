"""R5A 回放证据：1 kHz 原始状态、独立复判、曲线与本地 MP4。"""
import json
import math
from pathlib import Path
import numpy as np
from launch import write_json,spec


class Capture:
    def __init__(self,env,run,label,video=False):
        self.env=env; self.run=Path(run); self.label=label; self.start_tick=env._sim_step_counter
        self.rows={}; self.video=None; self.video_frames=0
        if video:
            import imageio.v2 as imageio
            import imageio_ffmpeg
            import os
            os.environ['IMAGEIO_FFMPEG_EXE']=imageio_ffmpeg.get_ffmpeg_exe()
            self.video=imageio.get_writer(str(self.run/(label+'.mp4')),fps=50,codec='libx264',quality=7,macro_block_size=1)
        env._last_capture_tick=-1; env.physics_observer=self.sample
        env._capture(env._sim_step_counter)

    def sample(self,env,tick):
        d=env.robot.data; m=env.metrics()
        values=dict(t=(tick-self.start_tick)*env.physics_dt,completed_physics_step=tick>self.start_tick,com=m['com'][0],com_velocity=m['com_vel'][0],
            root_state=d.root_state_w[0],joint_position=d.joint_pos[0],joint_velocity=d.joint_vel[0],
            applied_torque=d.applied_torque[0],computed_torque=d.computed_torque[0],
            wheel_force_z=m['force_z'][0],wheel_clearance=m['clearance'][0],body_force=m['body_force'][0],
            contact_force_world=env.contacts.data.net_forces_w[0],
            wheel_link_position=d.body_link_pos_w[0,env.wheel_body_ids],
            body_com_position=d.body_com_pos_w[0],body_com_velocity=d.body_com_lin_vel_w[0],
            body_link_quaternion=d.body_link_quat_w[0],body_angular_velocity=d.body_com_ang_vel_w[0],
            tilt=m['tilt'][0],base_height=m['base_z'][0],actions=env.actions[0],leg_targets=env.targets[0],
            online_launched=env.state.launched[0],online_landed=env.state.landed[0],
            online_failed=env.state.failed[0],online_stable_time=env.state.stable_time[0])
        for key,value in values.items():
            if hasattr(value,'detach'): value=value.detach().cpu().numpy().copy()
            self.rows.setdefault(key,[]).append(value)
        # Capture at policy boundaries before automatic reset, including the terminal frame.
        if self.video is not None and (tick-self.start_tick)%20==0:
            self.frame()

    def frame(self):
        if self.video is not None:
            pixels=self.env.render()
            if pixels is None: raise RuntimeError('Camera returned no RGB frame')
            self.video.append_data(pixels); self.video_frames+=1

    def finish(self,reason,policy_steps):
        self.env.physics_observer=None
        if self.video is not None: self.video.close(); self.video=None
        arrays={key:np.asarray(rows) for key,rows in self.rows.items()}
        path=self.run/(self.label+'_trace.npz'); np.savez_compressed(path,**arrays)
        complete=(len(arrays['t'])==policy_steps*20+1 and
                  np.allclose(np.diff(arrays['t']),.001,rtol=0,atol=1e-9))
        metadata=dict(schema_version=1,protocol=spec.VERSION,reason=reason,policy_samples=policy_steps,
            physics_samples=len(arrays['t']),physics_dt=.001,includes_initial_and_terminal_frames=True,
            uniform_physics_trace=bool(complete),video_frames=self.video_frames,
            initial_contact_and_torque_valid=False,initial_frame_reason="Reset pose; no completed physics step yet",
            joint_names=self.env.robot.joint_names,body_names=self.env.robot.body_names,
            contact_body_names=self.env.contacts.body_names,wheel_body_names=list(spec.WHEEL_BODIES),
            body_masses_kg=self.env.masses[0].cpu().tolist(),
            body_inertias_link_axes_at_com_kg_m2=self.env.robot.root_physx_view.get_inertias()[0].cpu().tolist(),
            physical_self_collisions=True,full_backflip=False)
        write_json(self.run/(self.label+'_trace.json'),metadata)
        report=judge(arrays,reason)
        report.update(trace=path.name,metadata=self.label+'_trace.json',video=self.label+'.mp4' if self.video_frames else None,
                      uniform_physics_trace=bool(complete),policy_samples=policy_steps,physics_samples=len(arrays['t']))
        if not complete: report.update(success=False,trace_error='Non-uniform or incomplete physics sampling')
        if self.video_frames and self.video_frames!=policy_steps+1:
            raise RuntimeError('Video does not include every policy boundary and terminal frame')
        write_json(self.run/(self.label+'.json'),report)
        plot(arrays,self.run/(self.label+'.png'),self.env.robot.joint_names)
        return report


def judge(a,reason):
    """从真实物理采样重新计算；训练时 50 Hz 判据不替代本次 1 kHz 筛查。"""
    import torch
    from r5a.protocol import JumpState
    if not all(np.isfinite(v).all() for v in a.values()):
        return dict(success=False,reason='non_finite_trace',final_acceptance='not_evaluated')
    s=JumpState(1); s.reset([0],torch.tensor([float(a['com'][0,2])]))
    origin=a['com'][0,:2].copy(); last=0.
    f=lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32).reshape(1,-1)
    # At t=0 PhysX contact/actuator buffers may describe the pre-reset scene.
    # Preserve that raw sample but only judge completed physical steps.
    for i in range(1,len(a['t'])):
        t=a['t'][i]
        if t<=.5: origin=a['com'][i,:2].copy()
        scalar=lambda value:f(value).flatten()
        s.update(float(t-last),scalar(t),scalar(a['com'][i,2]),scalar(a['com_velocity'][i,2]),
            f(a['wheel_force_z'][i]),f(a['wheel_clearance'][i]),scalar(a['tilt'][i]),scalar(a['base_height'][i]),
            scalar(np.linalg.norm(a['com_velocity'][i,:2])),scalar(np.linalg.norm(a['com'][i,:2]-origin)),
            scalar(a['body_force'][i]),scalar(np.abs(a['applied_torque'][i]).max()))
        last=t
    report=s.report(0)
    report['success']=bool(report['success'] and not report['failed'] and reason=='timeout' and last>=3.999)
    derivative=np.diff(a['com'],axis=0)/np.diff(a['t'])[:,None] if len(a['t'])>1 else np.empty((0,3))
    discrepancy=float(np.max(np.abs(derivative-(a['com_velocity'][1:]+a['com_velocity'][:-1])*.5))) if len(derivative) else None
    report.update(reason=reason,duration_s=float(last),criterion='1kHz_J1_v2_terminal_hold',screen=spec.SCREEN,
                  terminal_continuous_hold=True,
                  com_velocity_vs_position_derivative_max_error_m_s=discrepancy,
                  full_backflip_acceptance='not_evaluated',hardware_acceptance='not_evaluated')
    return report


def plot(a,path,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    t=a['t']
    axes[0].plot(t,a['com'][:,2],label='Whole-robot COM z'); axes[0].plot(t,a['base_height'],label='Base z')
    axes[0].set_ylabel('Height (m)'); axes[0].legend()
    axes[1].plot(t,a['tilt']*180/math.pi,label='Body tilt'); axes[1].set_ylabel('Tilt (deg)')
    for j,leg in enumerate(spec.LEGS): axes[2].plot(t,a['wheel_force_z'][:,j],label=leg)
    axes[2].set_ylabel('Wheel normal force (N)'); axes[2].legend(ncol=4)
    for name in (f'FL_{part}_joint' for part in ('hip','thigh','calf','wheel')):
        axes[3].plot(t,a['applied_torque'][:,names.index(name)],label=name)
    axes[3].set_ylabel('Applied torque (Nm)'); axes[3].set_xlabel('Time (s)'); axes[3].legend(ncol=2)
    for ax in axes: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
