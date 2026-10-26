"""自主后空翻 DirectRLEnv：策略控制全部目标，事件/验收在1kHz执行。"""
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz,matrix_from_quat
from .scene import JumpSceneCfg
from .protocol import FlipState,targets
from .recovery import recovery_terms
from .shaping import flight_shaping,landing_progress,flight_quality,reward_potentials,preparation_potential
from .spec import CONTRACT as C,LEG_JOINTS,WHEELS,WHEEL_BODIES

@configclass
class AutoFlipCfg(DirectRLEnvCfg):
 scene:JumpSceneCfg=JumpSceneCfg(num_envs=512,env_spacing=4.,replicate_physics=True)
 sim:sim_utils.SimulationCfg=sim_utils.SimulationCfg(dt=.001,render_interval=20,
  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
 decimation=20
 episode_length_s=6.
 is_finite_horizon=True
 action_space=16
 observation_space=420
 state_space=104
 reset_jitter=0.
 def __post_init__(self):
  super().__post_init__();self.viewer.origin_type='world';self.viewer.eye=(2.,-2.,1.2);self.viewer.lookat=(0.,0.,.5)

class AutoFlipEnv(DirectRLEnv):
 num_states=104
 def __init__(self,cfg,render_mode=None,**kwargs):
  self.physics_observer=None;self.episode_sink=None;self._last_capture_tick=-1
  super().__init__(cfg,render_mode,**kwargs)
  self.robot=self.scene['robot'];self.contacts=self.scene['contact_forces']
  self.leg_ids=self.robot.find_joints(list(LEG_JOINTS),preserve_order=True)[0]
  self.wheel_ids=self.robot.find_joints(list(WHEELS),preserve_order=True)[0]
  self.joint_ids=self.leg_ids+self.wheel_ids
  self.wheel_body_ids=self.robot.find_bodies(list(WHEEL_BODIES),preserve_order=True)[0]
  self.contact_ids=self.contacts.find_bodies(list(WHEEL_BODIES),preserve_order=True)[0]
  self.other_contacts=[i for i,n in enumerate(self.contacts.body_names) if n not in WHEEL_BODIES]
  self.masses=self.robot.root_physx_view.get_masses().to(self.device);self.mass=self.masses.sum(-1)
  self.inertias=self.robot.root_physx_view.get_inertias().to(self.device).reshape(self.num_envs,-1,3,3)
  self.actions=torch.zeros(self.num_envs,16,device=self.device);self.previous_actions=torch.zeros_like(self.actions)
  self.targets=targets(self.actions);self.state=FlipState(self.num_envs,self.device)
  self.history=torch.zeros(self.num_envs,5,84,device=self.device);self.history_reset=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
  self._history_step=-1;self._protocol_tick=self._sim_step_counter
  self.episode_physics_steps=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  self.episode_count=torch.zeros_like(self.episode_physics_steps);self.last_reports=[]
  self.old_potential=torch.zeros(self.num_envs,5,device=self.device)
  self.old_preparation=torch.zeros(self.num_envs,device=self.device)
  self.old_extension=torch.zeros(self.num_envs,device=self.device)
  self.old_launch_score=torch.zeros(self.num_envs,device=self.device)
  self.old_air_rotation=torch.zeros(self.num_envs,device=self.device)
  self.old_compact=torch.zeros(self.num_envs,device=self.device)
  self.air_control_seen=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
  self.phase_switch=torch.zeros_like(self.air_control_seen)
  self.peak_support=torch.zeros(self.num_envs,device=self.device)
  self.finite=torch.tensor(True,device=self.device)
  self.reward_terms={};self.episode_rewards={k:torch.zeros(self.num_envs,device=self.device) for k in C['reward_weights']}
 def _setup_scene(self):pass
 def metrics(self):
  d=self.robot.data;force=self.contacts.data.net_forces_w
  return dict(com=(d.body_com_pos_w*self.masses[:,:,None]).sum(1)/self.mass[:,None],
   com_vel=(d.body_com_lin_vel_w*self.masses[:,:,None]).sum(1)/self.mass[:,None],force=force,
   force_z=force[:,self.contact_ids,2],clearance=d.body_link_pos_w[:,self.wheel_body_ids,2]-.0875,
   tilt=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1.,1.)),base_z=d.root_link_pos_w[:,2],
   body_force=force[:,self.other_contacts].norm(dim=-1).amax(-1),torque_peak=d.applied_torque.abs().amax(-1))
 def momentum(self,m):
  d=self.robot.data;r=d.body_com_pos_w-m['com'][:,None,:];v=d.body_com_lin_vel_w-m['com_vel'][:,None,:]
  rot=matrix_from_quat(d.body_link_quat_w);iw=rot@self.inertias@rot.transpose(-1,-2)
  h=(iw@d.body_com_ang_vel_w[...,None]).squeeze(-1)+torch.cross(r,self.masses[:,:,None]*v,dim=-1)
  h=h.sum(1)
  axis=self.state.axis
  locked=(torch.einsum('ni,nbij,nj->nb',axis,iw,axis)+self.masses*((r*r).sum(-1)-(r*axis[:,None,:]).sum(-1).square())).sum(-1)
  return h,locked
 def _capture(self,tick):
  if self.physics_observer is not None and tick!=self._last_capture_tick:
   self.physics_observer(self,tick);self._last_capture_tick=tick
 def _pre_physics_step(self,actions):
  if not torch.isfinite(actions).all():raise RuntimeError('Non-finite action')
  self.extras['log']={};self.finite.fill_(True);self.peak_support.zero_();self.state.landing_event.zero_()
  self.state.valid_air_event.zero_()
  self.old_potential.copy_(reward_potentials(self.state,self.mass))
  self.old_preparation.copy_(preparation_potential(self.state))
  self.old_extension.copy_(self.state.extension_peak)
  self.old_launch_score.copy_(self.state.launch_score_peak)
  self.old_air_rotation.copy_(self.state.air_rotation_peak)
  self.old_compact.copy_(self.state.compact_potential)
  self.previous_actions.copy_(self.actions)
  self.phase_switch.copy_(self.state.launched&~self.air_control_seen)
  self.air_control_seen.copy_(self.state.launched)
  self.actions.copy_(actions.clamp(-1,1)*(self.episode_physics_steps>=round(C['settle_seconds']/self.physics_dt))[:,None])
 def _apply_action(self):
  self._advance(self._sim_step_counter-1);self._capture(self._sim_step_counter-1)
  self.targets.copy_(targets(self.actions))
  self.robot.set_joint_position_target(self.targets,joint_ids=self.leg_ids)
  self.robot.set_joint_velocity_target(self.actions[:,12:]*C['wheel_speed_rad_s'],joint_ids=self.wheel_ids)
 def _advance(self,tick):
  if tick==self._protocol_tick:return
  if tick!=self._protocol_tick+1:raise RuntimeError('Physics event gap')
  m=self.metrics();self.finite.logical_and_(torch.isfinite(torch.cat([v.reshape(-1) for v in m.values()])).all())
  self.episode_physics_steps.add_(1);t=self.episode_physics_steps*self.physics_dt
  h,inertia=self.momentum(m)
  m.update(H_backward=-(h*self.state.axis).sum(-1),omega_backward=-(self.robot.data.root_ang_vel_w*self.state.axis).sum(-1),locked_inertia=inertia)
  self.state.update(self.physics_dt,t,m,self.robot.data.root_quat_w)
  self.peak_support.copy_(torch.maximum(self.peak_support,m['force_z'].clamp_min(0).sum(-1)))
  self.current_metrics=m;self._protocol_tick=tick
 def _get_dones(self):
  self._advance(self._sim_step_counter);self._capture(self._sim_step_counter)
  if not self.finite:raise RuntimeError('Non-finite physical state')
  if not torch.equal(self.episode_physics_steps,self.episode_length_buf*self.cfg.decimation):raise RuntimeError('Episode physics clock mismatch')
  return self.state.failed.clone(),self.episode_length_buf>=self.max_episode_length
 def _get_rewards(self):
  """每步返回加权奖励。辅助进度每回合有界，真实腾空后的恢复按有界单圈进度计分；成功门不变。"""
  s=self.state;m=self.current_metrics;w=C['reward_weights'];dt=self.step_dt
  p=reward_potentials(s,self.mass)
  delta=p-self.old_potential
  valid=s.flight_valid&~s.failed
  touch=s.landing_event&valid
  quality=flight_quality(s.flight_rise,s.flight_time)
  rotation=landing_progress(s.angle)*quality
  terminal=s.failed|(self.episode_length_buf>=self.max_episode_length)
  next_preparation=torch.where(terminal,0.,preparation_potential(s))
  upright=torch.exp(-(m['tilt']/.26).square())
  hold=s.qualified_landing&~s.failed
  quiet=torch.exp(-(m['com_vel'].norm(dim=-1)/.15).square())*(m['force_z']>5.).all(-1)*upright
  terms=dict(push_potential=delta[:,0],height_potential=delta[:,1],rotation_potential=delta[:,2],airtime_potential=delta[:,3],
   preparation_shaping=C['gamma']*next_preparation-self.old_preparation,
   extension_shaping=C['gamma']*torch.where(terminal,0.,s.extension_peak)-self.old_extension,
   launch_shaping=C['gamma']*torch.where(terminal,0.,s.launch_score_peak)-self.old_launch_score,
   air_rotation_potential=s.air_rotation_peak-self.old_air_rotation,
   flight_bonus=touch.float(),landing_rotation=touch*rotation,
   landing_pose=touch*rotation*((1+torch.cos(s.landing_tilt))*.5).square(),
   touchdown_deficit=touch*(1-(s.angle/(2*math.pi)).clamp(0,1)),
   stable_rate=dt*hold*quiet,terminal_success=(self.episode_length_buf>=self.max_episode_length)*s.success,
   failure=s.failed.float(),alive_rate=dt*(self.episode_physics_steps>round(C['settle_seconds']/self.physics_dt)).float(),
   action_rate=dt*(self.actions-self.previous_actions).square().sum(-1),
   handoff_action_change=self.phase_switch*(self.actions-self.previous_actions).square().mean(-1),
   torque_rate=dt*(self.robot.data.applied_torque[:,self.leg_ids]/48.).square().mean(-1),
   side_axis_rate=dt*(s.launched&~s.landed)*(s.axis_error/.35).square().clamp_max(9.),
   excessive_spin_rate=dt*(self.robot.data.root_ang_vel_b.norm(dim=-1)-25.).clamp_min(0.).square(),
   impact_rate=delta[:,4])
  terms.update(recovery_terms(s,m,self.robot.data.joint_vel[:,self.wheel_ids],dt))
  terms.update(flight_shaping(s,m,s.compact_potential-self.old_compact,dt))
  self.reward_terms={k:v*w[k] for k,v in terms.items()}
  reward=sum(self.reward_terms.values())
  for k,v in self.reward_terms.items():
   self.episode_rewards[k].add_(v);self.extras['log']['Reward/'+k]=v.mean()
  self.extras['log'].update({'AutoFlip/angle_deg':s.angle.mean()*180/math.pi,'AutoFlip/flight_rise':s.flight_rise.mean(),
   'AutoFlip/launch_fraction':s.launched.float().mean(),'AutoFlip/qualified_landing':s.qualified_landing.float().mean(),
   'AutoFlip/preparation_cm':s.compression_peak.mean()*100,'AutoFlip/compact_credit':s.compact_potential.mean()})
  if not torch.isfinite(reward).all():raise RuntimeError('Non-finite reward')
  return reward
 def _get_observations(self):
  d=self.robot.data;m=self.metrics();s=self.state;t=self.episode_length_buf*self.step_dt
  pos=(d.joint_pos-d.default_joint_pos)[:,self.joint_ids].clone();pos[:,12:]=0.
  frame=torch.cat((d.root_ang_vel_b*.2,d.projected_gravity_b,pos,d.joint_vel[:,self.joint_ids]*.05,self.actions,
   m['com_vel'].clamp(-8,8)*.25,(m['com'][:,:2]-s.origin).clamp(-2,2),(m['com'][:,2]-s.baseline_z)[:,None],
   (m['force_z']/(self.mass[:,None]*9.81)).clamp(-2,20),m['clearance'].clamp(-1,2),
   torch.nn.functional.one_hot(s.phase(t),4).float(),(t/6.)[:,None],(s.angle/(2*math.pi)).clamp(-2,2)[:,None],
   s.flight_time[:,None],(1-s.angle/(2*math.pi)).clamp(-2,2)[:,None],s.launched.float()[:,None],s.landed.float()[:,None],
   (m['com'][:,:2]-s.landing_xy).clamp(-2,2),m['base_z'][:,None],s.stable_time[:,None],s.axis_error[:,None],s.flight_rise[:,None]),-1)
  if frame.shape[1]!=84:raise RuntimeError(f'Frame {frame.shape}')
  if self._history_step!=self.common_step_counter:
   self.history[:,:-1].copy_(self.history[:,1:].clone());self.history[:,-1].copy_(frame);self._history_step=self.common_step_counter
  self.history[self.history_reset]=frame[self.history_reset,None,:].expand(-1,5,-1);self.history_reset.zero_()
  h,inertia=self.momentum(m)
  extras=torch.stack((s.launch_vz,s.flight_rise,s.flight_time,s.qualified_landing.float(),s.failed.float(),s.release_t/6.,
   s.landing_tilt,s.push_peak/4.,s.height_peak/.8,s.impact_peak/2000.,1-t/6.,s.rotation_potential,
   m['body_force']/100.,m['torque_peak']/48.,s.landing_t/6.,s.count.clamp_max(7)/7.),-1)
  critic=torch.cat((frame,h*.1,inertia[:,None],extras),-1)
  return {'policy':self.history.flatten(1).clone(),'critic':critic}
 def _reset_idx(self,env_ids):
  if env_ids is None:env_ids=self.robot._ALL_INDICES
  report_ids=env_ids[self.episode_length_buf[env_ids]>0]
  if report_ids.numel():
   reports=[]
   # Only end-of-episode values cross the CPU boundary; physics events stay on device.
   ids=report_ids.cpu().tolist()
   fields={k:getattr(self.state,k)[report_ids].detach().cpu().tolist() for k in self.state.FLOATS+self.state.BOOLS if not k.startswith('candidate')}
   failures=self.state.reason[report_ids].cpu().tolist();counts=self.episode_count[report_ids].cpu().tolist()
   durations=(self.episode_length_buf[report_ids]*self.step_dt).cpu().tolist()
   components={k:v[report_ids].detach().cpu().tolist() for k,v in self.episode_rewards.items()}
   for row,i in enumerate(ids):
    reports.append(dict(env_id=i,episode=counts[row],duration_s=durations[row],failure_code=failures[row],
      **{k:v[row] for k,v in fields.items()},reward_components={k:v[row] for k,v in components.items()}))
   self.last_reports=reports
   if self.episode_sink:self.episode_sink(reports)
   self.episode_count[report_ids]+=1
   self.extras['log'].update({f'Ended/{k}':getattr(self.state,k)[report_ids].float().mean() for k in ('success','launched','flight_valid','qualified_landing','flight_time','flight_rise','angle')})
   self.extras['log'].update({'Ended/angle_deg':self.state.angle[report_ids].mean()*180/math.pi,
    'Ended/preparation_cm':self.state.compression_peak[report_ids].mean()*100,
    'Ended/compact_credit':self.state.compact_potential[report_ids].mean(),
    'Ended/release_H':self.state.release_H[report_ids].mean()})
  super()._reset_idx(env_ids)
  root=self.robot.data.default_root_state[env_ids].clone();root[:,:3]+=self.scene.env_origins[env_ids]
  # 当前先学习固定工况：保留URDF默认姿态，不消耗环境随机采样。
  if self.cfg.reset_jitter:
   raise ValueError('v6 fixed-condition contract forbids initial-state randomization')
  root[:,7:]=0.
  q=self.robot.data.default_joint_pos[env_ids].clone()
  self.robot.write_root_pose_to_sim(root[:,:7],env_ids);self.robot.write_root_velocity_to_sim(root[:,7:],env_ids)
  self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),None,env_ids)
  self.actions[env_ids]=0;self.previous_actions[env_ids]=0;self.targets[env_ids]=q[:,self.leg_ids]
  self.state.reset(env_ids,root[:,2]-.085652,root[:,:2],root[:,3:7]);self.episode_physics_steps[env_ids]=0;self.history_reset[env_ids]=True
  self.air_control_seen[env_ids]=False;self.phase_switch[env_ids]=False
  for v in self.episode_rewards.values():v[env_ids]=0
