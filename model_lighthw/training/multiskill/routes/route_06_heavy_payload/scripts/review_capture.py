"""R6离线审查采样：策略50Hz、可选物理1kHz、重置前终止帧和本地MP4。"""
import csv
import json
from pathlib import Path
import numpy as np
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg, DatasetExportMode
from isaaclab.utils import configclass


def json_safe(value):
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


class ReviewRecorder(RecorderTerm):
    """只读采样，不改动作/奖励/物理；post_step在自动reset前由Isaac Lab调用。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.r6_review = self
        self.active = False
        self.frames, self.physics = [], []
        self.metrics_func = None
        self.writer = None
        self.run = None
        self.current = None
        self.video_index = []
        self.want_physics = False
        self.in_physics = False
        self.policy_before = None

    @staticmethod
    def array(value):
        return value[0].detach().cpu().numpy().copy()

    def begin(self, run, label, record_video=False, physics_trace=False):
        self.run, self.current = Path(run), label
        self.frames, self.physics, self.video_index = [], [], []
        self.want_physics, self.active = physics_trace, True
        if record_video:
            import imageio.v2 as imageio
            import imageio_ffmpeg
            # get_writer使用wheel内置FFmpeg；该接口不下载二进制文件。
            import os
            binary = next((Path(imageio_ffmpeg.__file__).parent/'binaries').glob('ffmpeg-*'))
            os.environ['IMAGEIO_FFMPEG_EXE'] = str(binary)
            self.writer = imageio.get_writer(str(self.run/(label+'.mp4')), fps=25, codec='libx264',
                                            macro_block_size=1, ffmpeg_log_level='error')
            for _ in range(12):
                self._env.sim.render(); self._env.render()
            self.capture_video(0.0, False)

    def record_pre_step(self):
        if self.active:
            self.policy_before = self.array(self._env.obs_buf['policy'])
            self.in_physics = True
        return None, None

    def physical_sample(self):
        if not (self.active and self.in_physics and self.want_physics):
            return
        import torch
        data = self._env.scene['robot'].data
        values = {
            'root_state_w':data.root_state_w[0],
            'joint_pos':data.joint_pos[0], 'joint_vel':data.joint_vel[0],
            'applied_torque':data.applied_torque[0], 'computed_torque':data.computed_torque[0],
            'contact_forces_w':self._env.scene['contact_forces'].data.net_forces_w[0]}
        runtime = getattr(self._env, 'r6_payload_runtime', None)
        if runtime is not None:
            # 物理属性由每次变载回读校验；每个物理采样记录当时的载荷标签。
            values['payload_kg'] = data.root_state_w.new_tensor(runtime.mass_kg)
            values['loaded_mass_kg'] = data.root_state_w.new_tensor(runtime.total_mass_kg)
        self.physics_shapes = {k:tuple(v.shape) for k,v in values.items()}
        # 每个物理步只做一次GPU->CPU传输，避免各字段分别同步拖慢评估。
        self.physics.append(torch.cat([v.reshape(-1) for v in values.values()]).detach().cpu().numpy().copy())

    def record_post_step(self):
        if not self.active:
            return None, None
        env, data = self._env, self._env.scene['robot'].data
        self.in_physics = False
        from review_limits import actuator_effort_limits
        frame = {
            't':np.asarray((len(self.frames)+1)*env.step_dt),
            'root_state_w':self.array(data.root_state_w),
            'root_lin_vel_b':self.array(data.root_lin_vel_b), 'root_ang_vel_b':self.array(data.root_ang_vel_b),
            'joint_pos':self.array(data.joint_pos), 'joint_vel':self.array(data.joint_vel),
            'joint_pos_target':self.array(data.joint_pos_target), 'joint_vel_target':self.array(data.joint_vel_target),
            'applied_torque':self.array(data.applied_torque), 'computed_torque':self.array(data.computed_torque),
            'actuator_effort_limits':self.array(actuator_effort_limits(env.scene['robot'])),
            'sim_effort_limits':self.array(data.joint_effort_limits),
            'contact_forces_w':self.array(env.scene['contact_forces'].data.net_forces_w),
            'policy_observation_before':self.policy_before,
            'policy_observation_after':self.array(env.obs_buf['policy']),
            'critic_observation_after':self.array(env.obs_buf['critic']),
            'raw_action':self.array(env.action_manager.action),
            'command':self.array(env.command_manager.get_command('base_velocity')),
            'reward':self.array(env.reward_buf), 'reward_term_rates':self.array(env.reward_manager._step_reward),
            'terminated':self.array(env.reset_terminated), 'timeout':self.array(env.reset_time_outs),
            'termination_terms':np.asarray([bool(env.termination_manager.get_term(n)[0]) for n in env.termination_manager.active_terms]),
        }
        runtime = getattr(env, 'r6_payload_runtime', None)
        if runtime is not None:
            frame['payload_kg'] = np.asarray(runtime.mass_kg)
            frame['loaded_mass_kg'] = np.asarray(runtime.total_mass_kg)
        frame['metrics'] = {k:float(v[0]) for k,v in self.metrics_func(env).items()}
        self.frames.append(frame)
        terminal = bool(frame['terminated'] or frame['timeout'])
        if self.writer is not None and (len(self.frames)%2 == 0 or terminal):
            self.capture_video(float(frame['t']), terminal)
        return None, None

    def capture_video(self, t, terminal):
        self._env.sim.render()
        rgb = self._env.render()
        if rgb is None or rgb.size == 0:
            raise RuntimeError('R6 video renderer returned no image')
        self.writer.append_data(rgb)
        self.video_index.append({'frame':len(self.video_index), 'simulation_time_s':t, 'terminal':int(terminal)})
        self.last_rgb = rgb.copy()

    def finish(self, reason):
        self.active = self.in_physics = False
        if self.writer is not None:
            self.writer.close(); self.writer = None
        if self.current is None:
            return
        if self.frames:
            arrays={key:np.stack([f[key] for f in self.frames]) for key in self.frames[0] if key!='metrics'}
            np.savez_compressed(self.run/(self.current+'_trace.npz'), **arrays)
            with (self.run/(self.current+'_trace.csv')).open('w',newline='') as stream:
                rows=[dict(t=float(f['t']),terminated=int(f['terminated']),timeout=int(f['timeout']),**f['metrics']) for f in self.frames]
                if 'payload_kg' in self.frames[0]:
                    for row, frame in zip(rows, self.frames):
                        row.update(payload_kg=float(frame['payload_kg']), loaded_mass_kg=float(frame['loaded_mass_kg']))
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if self.physics:
            packed=np.stack(self.physics); values={}; start=0
            for key, shape in self.physics_shapes.items():
                width=int(np.prod(shape))
                values[key]=packed[:,start:start+width].reshape((len(packed),*shape))
                start += width
            values['t']=(np.arange(len(self.physics))+1)*self._env.physics_dt
            np.savez_compressed(self.run/(self.current+'_physics.npz'),**values)
        if self.video_index:
            from PIL import Image
            Image.fromarray(self.last_rgb).save(self.run/(self.current+'_preview.png'))
            with (self.run/(self.current+'_video_frames.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(self.video_index[0])); writer.writeheader(); writer.writerows(self.video_index)
        terminal = self.frames[-1] if self.frames and (self.frames[-1]['terminated'] or self.frames[-1]['timeout']) else None
        metadata={'schema_version':2, 'effort_limit_semantics':'nominal actuator/PhysX minimum; speed-dependent clipping also visible in computed vs applied torque','reason':reason,'policy_hz':1/self._env.step_dt,
            'physics_hz':1/self._env.physics_dt,'physics_samples':len(self.physics),'policy_samples':len(self.frames),
            'joint_order':self._env.scene['robot'].joint_names,'contact_body_order':self._env.scene['contact_forces'].body_names,
            'reward_term_order':self._env.reward_manager.active_terms,'reward_terms_unit':'weighted rate; multiply by policy dt for step contributions',
            'termination_term_order':self._env.termination_manager.active_terms,
            'terminal_before_reset':None if terminal is None else {'t':float(terminal['t']), 'metrics':terminal['metrics'],
                'reasons':[n for n,on in zip(self._env.termination_manager.active_terms,terminal['termination_terms']) if on]},
            'video_fps':25,'video_frames':len(self.video_index),'video_timestamps':'simulation time in *_video_frames.csv',
            'sampling':'post-physics, pre-auto-reset; initial state is in episode summary',
            'finite':all(np.isfinite(v).all() for f in self.frames for k,v in f.items() if k!='metrics')}
        if getattr(self._env, 'r6_payload_runtime', None) is not None:
            metadata.update(schema_version=3, protocol='interactive_payload_change_v1',
                payload_events=self.current+'_payload_events.jsonl',
                payload_units='kg; includes per-sample payload and total robot mass',
                final_acceptance='manual_variable_payload_not_standard_screen')
        (self.run/(self.current+'_trace.json')).write_text(json.dumps(json_safe(metadata),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        self.current = None


@configclass
class ReviewRecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    review = RecorderTermCfg(class_type=ReviewRecorder)


def attach_physics_capture(env):
    original = env.scene.update
    def update(dt):
        original(dt)
        env.r6_review.physical_sample()
    env.scene.update = update
