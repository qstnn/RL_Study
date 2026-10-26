"""Actor-free fixed-pose traction test; no policy network is evaluated or loaded.

Robot joint targets and explicitly selected wheel-gain diagnostics are changed. Cart physics, forces, limits, observations,
commands and the production acceptance protocol remain active and recorded.
"""
import argparse,json,math,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1].parent/'unified_v5';sys.path.insert(0,str(ROOT/'scripts'))
import launch
VARIANTS=('coordinated_ff12','coordinated_ff','attitude','traction','coordinated','baseline','slow','frozen_legs','preload_wheels','frozen_both','frozen_both_slow','shift_forward','shift_backward','pose_00','pose_15','pose_30','pose_45','crawl03','crawl06','crawl10','soft_05','soft_10','ik_03','ik_06','ik_09','fb_03','fb_06','fb_09','fb_03_w10','fb_03_w20','fb_03_w40','ik_03_w20','ik_03_w40','pose_15_w20')
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=launch.BASELINE,help='Only used for provenance; actor weights are not loaded.');p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0');p.add_argument('--output',type=Path,required=True);p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['coordinated_ff12']);p.add_argument('--repeats',type=int,default=2);p.add_argument('--seeds',type=int,nargs='+',default=[43]);p.add_argument('--layout',choices=['towing','all','hybrid'],default='towing');p.add_argument('--noise',action='store_true');p.add_argument('--action-noise',action='store_true');p.add_argument('--parallel-noise-seeds',type=int,nargs='+');p.add_argument('--position-iterations',type=int,default=4);p.add_argument('--velocity-iterations',type=int,default=1);p.add_argument('--hitch-height',type=float,default=.07);p.add_argument('--friction-offset',type=float);p.add_argument('--friction-correlation',type=float);p.add_argument('--attitude-kp',type=float,default=.8);p.add_argument('--attitude-kd',type=float,default=.15);p.add_argument('--velocity-gain',type=float,default=2.);p.add_argument('--slip-damping',type=float,default=.2);p.add_argument('--filter-tau',type=float,default=.04);p.add_argument('--move-speed',type=float,default=.3);p.add_argument('--wheel-mode',choices=['torque','velocity'],default='torque');p.add_argument('--traction-sign',type=float,choices=(-1.,1.),default=1.);p.add_argument('--pose-id',choices=('h0.38_p0_x0.00','h0.38_p-3_x0.00','h0.38_p0_x0.04'),default='h0.38_p0_x0.00');p.add_argument('--leg-kp',type=float,default=80.);p.add_argument('--leg-kd',type=float,default=3.);p.add_argument('--leg-error-limit',type=float,default=.5);p.add_argument('--ramp-start',type=float,default=0.0);p.add_argument('--ramp-duration',type=float,default=.30);p.add_argument('--replay-static-source',type=Path);p.add_argument('--replay-wheel-gain',type=float,default=0.);p.add_argument('--loaded-acceleration',type=float,default=.3);a=p.parse_args()
 source=a.checkpoint.resolve(strict=True);digest=launch.sha256(source);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 replay_source=a.replay_static_source.resolve(strict=True) if a.replay_static_source else None
 replay_digest=launch.sha256(replay_source) if replay_source else None
 if a.parallel_noise_seeds and (not a.noise or a.repeats!=1 or len(a.seeds)!=1):p.error('Parallel noise streams require --noise, --repeats 1 and one global evaluation seed')
 if a.action_noise and not a.parallel_noise_seeds:p.error('Action-noise validation requires explicit parallel noise seeds')
 if a.parallel_noise_seeds and len(set(a.parallel_noise_seeds))!=len(a.parallel_noise_seeds):p.error('Noise seeds must be distinct')
 cases=[]
 for variant in a.variants:
  templates=launch.PROTOCOL.cases() if (a.layout=='all' or a.layout=='hybrid' and variant=='baseline') else [dict(slope_deg=25.,payload_kg=0.,towed_kg=40.,speed=speed,goal_distance_m=1.) for speed in (0.,a.move_speed)]
  for repeat in range(len(a.parallel_noise_seeds) if a.parallel_noise_seeds else a.repeats):
   for case in templates:
    entry=dict(case,variant=variant,replicate=repeat)
    if a.parallel_noise_seeds:entry['noise_seed']=a.parallel_noise_seeds[repeat]
    cases.append(entry)
 launch.write_json(out/'request.json',dict(created_at=launch.now(),checkpoint=str(source),checkpoint_sha256=digest,cases=cases,seeds=a.seeds,noise=a.noise,action_noise=a.action_noise,parallel_noise_seeds=a.parallel_noise_seeds,coordinated_controls=dict(pose_id=a.pose_id,wheel_mode=a.wheel_mode,traction_sign=a.traction_sign,attitude_kp=a.attitude_kp,attitude_kd=a.attitude_kd,leg_error_limit_rad=a.leg_error_limit,velocity_gain=a.velocity_gain,slip_damping=a.slip_damping,filter_tau=a.filter_tau,ramp_start_s=a.ramp_start,ramp_duration_s=a.ramp_duration,move_speed=a.move_speed,paired_noise=True,leg_feedforward_reference='factorial_velocity_01 baseline, t=2.5..3.0s measured friction',teacher_privileged_inputs=['contact_normal','true_lane_speed','drawbar_force','nominal_slope','episode_elapsed_for_initial_blend']),device=a.device,physics_override=dict(hitch_height_m=a.hitch_height,leg_kp=a.leg_kp,leg_kd=a.leg_kd,position_iterations=a.position_iterations,velocity_iterations=a.velocity_iterations,scene_position_limit=a.position_iterations,scene_velocity_limit=a.velocity_iterations,friction_offset=a.friction_offset,friction_correlation=a.friction_correlation),probe_sha256=launch.sha256(Path(__file__)),ppo_updates=0,teacher_only=True,source_contract=launch.SPEC.CONTRACT,execution_contract=launch.SPEC.CONTRACT,replay_static_source=str(replay_source) if replay_source else None,replay_static_sha256=replay_digest,replay_wheel_gain=a.replay_wheel_gain,loaded_acceleration=a.loaded_acceleration,scope='Actor-free traction diagnostic; static action replay is identified when used; no policy inference, no root forces and no cart propulsion; same-seed slots are not independent robustness seeds.'))
 launch.snapshot(out);launch.write_json(out/'actor_free_contract.json',dict(actor_loaded=False,actor_inference=False,action_source='recorded_frozen_actor_action_replay' if replay_source else 'coordinator_fixed_pose_and_measured_force',replay_source=str(replay_source) if replay_source else None,replay_sha256=replay_digest,ppo_updates=0,ground_friction=3.0));(out/'probe_source.py').write_bytes(Path(__file__).read_bytes());(out/'actor_free_control_source.py').write_bytes((Path(__file__).parent/'actor_free_control.py').read_bytes());launch.CART.write_cart_urdf(out/'assets/cart.urdf')
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True,device=a.device).app;env=None
 try:
  import gymnasium as gym,numpy as np,torch
  from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
  from isaaclab.utils.io import dump_yaml
  launch.SPEC.register_task();worker=launch.module_from_file('_tow_control_probe',ROOT/'scripts/worker.py')
  from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.wheel_unified_v5 import mdp
  def serializable(value):
   if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
   if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
   if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
   return value
  worker.write_json=lambda path,value:launch.write_json(path,serializable(value))
  cfg=load_cfg_from_registry(launch.SPEC.TASK,'env_cfg_entry_point');agent=load_cfg_from_registry(launch.SPEC.TASK,'rsl_rl_cfg_entry_point')
  pose_path=ROOT.parent/'research_v7/results/2026-10-20_validation/poses.json'
  pose_record=next(record for record in json.loads(pose_path.read_text())['selected'] if record['id']==a.pose_id)
  fixed_q_reset=np.asarray(pose_record['q'],dtype=np.float32)
  action_joint_names=list(launch.SPEC.CONTRACT['action_order'])
  # Isaac's imported articulation stores joints grouped by joint type and then
  # leg; action and observation contracts use explicit leg-major ordering.
  sim_joint_names=[f'{leg}_{joint}_joint' for joint in ('hip','thigh','calf','wheel') for leg in ('FL','FR','RL','RR')]
  action_index={name:i for i,name in enumerate(action_joint_names)}
  fixed_q_reset_sim=np.asarray([fixed_q_reset[action_index[name]] for name in sim_joint_names],dtype=np.float32)
  pose_height=float(pose_record['height_m']);pose_shift=float(pose_record['forward_shift_m']);pose_pitch=math.radians(float(pose_record['pitch_deg']))
  def reset_fixed_base(env,env_ids):
   ids=torch.as_tensor(env_ids,device=env.device)
   robot,cart=env.scene['robot'],env.scene['trailer']
   angle=env.r1_slope_rad[ids]+pose_pitch
   pose=robot.data.default_root_state[ids].clone()
   pose[:,:3]=env.scene.env_origins[ids]+env.r1_normal[ids]*pose_height+env.r1_tangent[ids]*pose_shift
   pose[:,3:7]=0.;pose[:,3]=torch.cos(angle/2.);pose[:,5]=-torch.sin(angle/2.);pose[:,7:]=0.
   robot.write_root_pose_to_sim(pose[:,:7],env_ids=ids);robot.write_root_velocity_to_sim(pose[:,7:],env_ids=ids)
   trailer_pose=cart.data.default_root_state[ids].clone()
   trailer_pose[:,:3]=env.scene.env_origins[ids]+env.r1_normal[ids]*.30-env.r1_tangent[ids]*1.73
   slope_angle=env.r1_slope_rad[ids]
   trailer_pose[:,3:7]=0.;trailer_pose[:,3]=torch.cos(slope_angle/2.);trailer_pose[:,5]=-torch.sin(slope_angle/2.);trailer_pose[:,7:]=0.
   inactive=env.r1_towed_kg[ids]==0
   trailer_pose[inactive,1]=500.;trailer_pose[inactive,2]=1.
   cart.write_root_pose_to_sim(trailer_pose[:,:7],env_ids=ids);cart.write_root_velocity_to_sim(trailer_pose[:,7:],env_ids=ids)
   cart.write_joint_state_to_sim(cart.data.default_joint_pos[ids],torch.zeros_like(cart.data.default_joint_vel[ids]),env_ids=ids)
   env.r1_drawbar_force[ids]=0.;env.r1_drawbar_length[ids]=1.;env.r1_drawbar_guard[ids]=False;env.r1_drawbar_peak[ids]=0.
   if hasattr(env,'r1_eval_phase'):
    env.r1_eval_phase[ids]=0;env.r1_eval_stop_time[ids]=-1.;env.r1_eval_origin[ids]=pose[:,:3];env.r1_eval_progress[ids]=0.
  if replay_source is None:
   cfg.events.reset_base.func=reset_fixed_base
   cfg.events.reset_base.params={}
  def reset_fixed_joints(env,env_ids):
   ids=torch.as_tensor(env_ids,device=env.device)
   q=torch.as_tensor(fixed_q_reset_sim,device=env.device,dtype=torch.float32).expand(len(ids),-1).clone()
   env.scene['robot'].write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
  if replay_source is None:
   cfg.events.reset_joints.func=reset_fixed_joints
   cfg.events.reset_joints.params={}
  cfg.scene.num_envs=len(cases);cfg.seed=agent.seed=a.seeds[0];cfg.sim.device=a.device;cfg.episode_length_s=65.;cfg.fixed_speed_mps=0.;cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg=25.,0.,40.;cfg.ground_friction=3.;cfg.rolling_coefficient=.02;cfg.scene.terrain.physics_material.static_friction=3.;cfg.scene.terrain.physics_material.dynamic_friction=3.;cfg.sim.physics_material.static_friction=3.;cfg.sim.physics_material.dynamic_friction=3.
  cfg.scene.robot.actuators['joint'].stiffness=a.leg_kp;cfg.scene.robot.actuators['joint'].damping=a.leg_kd
  cfg.observations.policy.enable_corruption=a.noise;cfg.viewer.origin_type='world';cfg.hitch_height_m=a.hitch_height
  for asset in (cfg.scene.robot,cfg.scene.trailer):
   asset.spawn.articulation_props.solver_position_iteration_count=a.position_iterations
   asset.spawn.articulation_props.solver_velocity_iteration_count=a.velocity_iterations
  cfg.sim.physx.max_position_iteration_count=a.position_iterations
  cfg.sim.physx.max_velocity_iteration_count=a.velocity_iterations
  if a.friction_offset is not None:cfg.sim.physx.friction_offset_threshold=a.friction_offset
  if a.friction_correlation is not None:cfg.sim.physx.friction_correlation_distance=a.friction_correlation
  noise_audit={};noise_generators={}
  if a.parallel_noise_seeds:
   noise_indices={(seed,variant):torch.tensor([i for i,c in enumerate(cases) if c['noise_seed']==seed and c['variant']==variant],device=a.device) for seed in a.parallel_noise_seeds for variant in a.variants}
   noise_generators={key:torch.Generator(device=a.device).manual_seed(key[0]) for key in noise_indices}
   def seeded_noise(data,cfg):
    if cfg.operation!='add':raise ValueError('Expected native additive observation noise')
    result=data.clone()
    for seed,indices in noise_indices.items():
     shape=(len(indices),*data.shape[1:])
     noise=torch.rand(shape,device=data.device,dtype=data.dtype,generator=noise_generators[seed])*(cfg.n_max-cfg.n_min)+cfg.n_min
     result[indices]+=noise
    key=str((tuple(data.shape[1:]),cfg.n_min,cfg.n_max))
    noise_audit[key]=noise_audit.get(key,0)+1
    return result
   for name in ('base_ang_vel','projected_gravity','joint_pos','joint_vel'):
    getattr(cfg.observations.policy,name).noise.func=seeded_noise
  cfg.scene.robot.spawn.usd_dir=str(out/'cache/robot');cfg.scene.robot.spawn.usd_file_name='robot.usd';cfg.scene.trailer.spawn.asset_path=str(out/'assets/cart.urdf');cfg.scene.trailer.spawn.usd_dir=str(out/'cache/cart');cfg.scene.trailer.spawn.usd_file_name='cart.usd'
  dump_yaml(str(out/'params/env.yaml'),cfg);dump_yaml(str(out/'params/agent.yaml'),agent)
  env=RslRlVecEnvWrapper(gym.make(launch.SPEC.TASK,cfg=cfg),clip_actions=agent.clip_actions);base=env.unwrapped
  launch.write_json(out/'joint_mapping.json',dict(data_joint_names=list(base.scene['robot'].joint_names),r1_joint_ids=list(base.r1_joint_ids),config_joint_names=sim_joint_names,action_joint_names=action_joint_names))
  from pxr import Usd,PhysxSchema,UsdPhysics
  import isaaclab.sim as sim_utils
  stage=sim_utils.get_current_stage();solver_audit={}
  for prim in stage.Traverse():
   path=str(prim.GetPath())
   if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or (path.startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
    attrs={str(attr.GetName()):attr.Get() for attr in prim.GetAttributes() if 'Iteration' in str(attr.GetName())}
    solver_audit[path]=attrs
  launch.write_json(out/'solver_runtime_usd.json',solver_audit)
  pos_names=list(cfg.observations.policy.joint_pos.params['asset_cfg'].joint_names)
  leg_observation_indices=[pos_names.index(name) for name in launch.SPEC.CONTRACT['action_order'][:12]]
  pos_start=0
  for name,width in launch.SPEC.POLICY_TERMS.items():
   if name=='joint_pos':break
   pos_start+=width*launch.SPEC.HISTORY
  pos_start+=(launch.SPEC.HISTORY-1)*launch.SPEC.POLICY_TERMS['joint_pos']
  leg_observation_columns=torch.tensor([pos_start+i for i in leg_observation_indices],device=env.device)
  pos_scale=cfg.observations.policy.joint_pos.scale
  leg_observation_scale=torch.as_tensor(1. if pos_scale is None else pos_scale,device=env.device).expand(16)[leg_observation_indices]
  leg_default=torch.tensor([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5],device=env.device)
  launch.write_json(out/'feedback_observation_mapping.json',dict(position_names=pos_names,leg_names=launch.SPEC.CONTRACT['action_order'][:12],latest_columns=leg_observation_columns.cpu().tolist(),relative_position=True,source='resolved observation joint names; action ordering differs from interleaved observation ordering'))
  base.cfg.loaded_acceleration=torch.tensor([.1 if c['variant'] in ('slow','frozen_both_slow') else a.loaded_acceleration for c in cases],device=env.device)
  replay_actions=None
  if replay_source:
   with np.load(replay_source,allow_pickle=False) as reference:
    replay_actions=torch.as_tensor(reference['executed_action'],device=env.device,dtype=torch.float32)
   if replay_actions.ndim!=2 or replay_actions.shape[1]!=16 or len(replay_actions)<1000:
    raise ValueError('Static replay must have at least 1000 rows and 16 actions')
  import numpy as np
  fixed_q=np.asarray(pose_record['q'][:12],dtype=np.float32)
  fixed_default=np.asarray([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5],dtype=np.float32)
  fixed_scale=np.tile([.08,.14,.14],4).astype(np.float32)
  fixed_action=np.zeros(16,dtype=np.float32);fixed_action[:12]=(fixed_q-fixed_default)/fixed_scale
  offline_pose=torch.tensor(fixed_action,device=env.device)
  evidence={'fb_03':dict(pitch_relative_slope_deg=pose_record['pitch_deg'],target_height_m=pose_record['height_m'],forward_shift_m=pose_record['forward_shift_m'],source_actual_joint_pos=fixed_q.tolist(),target_actual_joint_pos=fixed_q.tolist(),kinematics='static_pose_search_selected')}
  ik_poses={'fb_03':offline_pose}
  launch.write_json(out/'pose_source.json',dict(path=str(pose_path),sha256=launch.sha256(pose_path),pose_id=pose_record['id'],height_m=pose_record['height_m'],pitch_deg=pose_record['pitch_deg'],forward_shift_m=pose_record['forward_shift_m'],actual_joint_pos=fixed_q.tolist(),raw_action=offline_pose.cpu().tolist(),source='V7 static pose search; no Actor action'))
  launch.write_json(out/'ik_poses.json',evidence)
  coordinated_control=launch.module_from_file('_actor_free_coordinated_control',Path(__file__).parent/'actor_free_control.py')
  physical=coordinated_control.PhysicsRecorder(base,worker)
  coordinator=coordinated_control.Coordinator(base,cfg,cases,evidence['fb_03'],offline_pose,launch.REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf',physical,kp=a.attitude_kp,kd=a.attitude_kd,velocity_gain=a.velocity_gain,slip_damping=a.slip_damping,filter_tau=a.filter_tau,wheel_mode=a.wheel_mode)
  coordinator.ramp_start=a.ramp_start;coordinator.ramp_duration=a.ramp_duration;coordinator.traction_sign=a.traction_sign;coordinator.position_error_limit=a.leg_error_limit
  launch.write_json(out/'controller_geometry.json',coordinator.geometry)
  launch.write_json(out/'physics_recorder.json',dict(sample_hz=1000,contact_force_w_is_normal_only=True,native_friction_view=physical.contact_meta,phase='post-integration before terminal reset',robot_body_names=base.scene['robot'].body_names,body_masses=physical.masses.cpu().tolist(),wheel_names=['FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link'],note='Whole-robot COM is mass-weighted over all robot links; excludes the cart. Drawbar tension is the force applied during the preceding 1ms interval.'))
  step_original=env.step
  def recorded_step(action):
   result=step_original(action);physical.capture();return result
  env.step=recorded_step
  class Runner:
   def __init__(self):
    self.actor=None
    self.trace={k:[] for k in ('observation','action','executed_action','actor_action','joint_pos','joint_vel','phase','command_vx','target_vx','wheel_positions','body_com_pos','body_gravity','wheel_damping','body_omega','wheel_contact_force','wheel_center_vel','wheel_angular_vel','decoded_leg_pos')}
    self.action_std=torch.tensor(agent.algorithm.initial_std_by_action,device=env.device)
    self.action_generators={key:torch.Generator(device=env.device).manual_seed(key[0]+100000) for key in (noise_indices if a.parallel_noise_seeds else {})}
    self.n=0;self.preload=torch.zeros(len(cases),16,device=env.device);self.count=torch.zeros(len(cases),device=env.device)
    self.ids={v:torch.tensor([i for i,c in enumerate(cases) if c['variant']==v and c['slope_deg']==25 and c['towed_kg']==40],device=env.device,dtype=torch.long) for v in a.variants}
   def load(self,path,load_optimizer=False):assert not load_optimizer and Path(path).resolve()==source
   def get_inference_policy(self,device=None):
    def infer(obs):
     altered=obs.clone()
     for variant,ids in self.ids.items():
      if variant.startswith('crawl'):
       gain=float(variant[5:])/30.
       for col in (30,33,36,39,42):altered[ids,col]*=gain
     decoded_leg_pos=obs[:,leg_observation_columns]/leg_observation_scale+leg_default
     action0=torch.zeros((obs.shape[0],16),device=obs.device,dtype=obs.dtype);action=action0.clone()
     if self.actor is not None:
      raise RuntimeError('Actor-free test unexpectedly has an actor')
     term=base.command_manager.get_term('base_velocity')
     t=base.episode_length_buf*.02
     # A run-wide step counter handles seed boundaries without mistaking a failed env reset for a new trial.
     if self.n and bool(torch.all(base.episode_length_buf==0)):
      self.preload.zero_();self.count.zero_();base.scene['robot'].actuators['wheel'].damping.fill_(2.)
     gather=(t>=2.)&(t<2.8)
     self.preload[gather]+=action0[gather];self.count[gather]+=1
     held=self.preload/self.count.clamp_min(1.)[:,None]
     cmd=base.command_manager.get_command('base_velocity')[:,0]
     heading=obs[:,289] # newest declared heading input (scale 1)
     omega_scale=cfg.observations.policy.base_ang_vel.scale
     yaw_scale=1. if omega_scale is None else float(omega_scale[-1] if isinstance(omega_scale,(tuple,list)) else omega_scale)
     yaw=obs[:,14]/yaw_scale
     correction=(-8.*heading-1.*yaw).clamp(-2.,2.)
     for variant,ids0 in self.ids.items():
      ids=ids0[(t[ids0]>=2.8)&(self.count[ids0]>0)]
      if variant in ('frozen_legs','frozen_both','frozen_both_slow','shift_forward','shift_backward'):
       action[ids,:12]=held[ids,:12]
      if variant in ('preload_wheels','frozen_both','frozen_both_slow','shift_forward','shift_backward'):
       action[ids,12:]=held[ids,12:]+cmd[ids,None]/.0875/5.
       action[ids,12:]+=correction[ids,None]*torch.tensor([1.,-1.,1.,-1.],device=env.device)/5.
      if variant in ('shift_forward','shift_backward'):
       amount=.06 if variant=='shift_forward' else -.06
       ramp=((t[ids]-2.8)/.5).clamp(0.,1.)
       for j in (1,4,7,10):action[ids,j]+=amount*ramp/.14
     for variant,ids0 in self.ids.items():
      if variant.startswith('soft_'):
       ids=ids0[(t[ids0]>=2.8)&(self.count[ids0]>0)]
       reset_ids=ids0[t[ids0]<2.8]
       gain=float(variant.split('_')[1])/10.
       base.scene['robot'].actuators['wheel'].damping[reset_ids]=2.
       base.scene['robot'].actuators['wheel'].damping[ids]=gain
       # Preserve the observed static preload torque while lowering velocity feedback gain.
       # This is an explicitly recorded teacher-only actuator intervention.
       action[ids,12:]=held[ids,12:]*(2./gain)+cmd[ids,None]/.0875/5.
       action[ids,12:]+=correction[ids,None]*torch.tensor([1.,-1.,1.,-1.],device=env.device)/5.
     for variant,ids in self.ids.items():
      if variant.startswith('pose_'):
       # Diagnostic fixed joint targets, reached before the 3-second movement origin.
       amount=float(variant.split('_')[1])/100.
       ramp=((t[ids]-1.5)/1.).clamp(0.,1.)
       target=offline_pose[None,:].expand(len(ids),-1).clone()
       for j in (1,4,7,10):target[:,j]+=amount/.14
       motion_gain=float(variant.rsplit('_w',1)[1])/100. if '_w' in variant else 1.
       target[:,12:]+=motion_gain*cmd[ids,None]/.0875/5.
       target[:,12:]+=correction[ids,None]*torch.tensor([1.,-1.,1.,-1.],device=env.device)/5.
       action[ids]=torch.lerp(action[ids],target,ramp[:,None])
     for variant,ids in self.ids.items():
      if variant.startswith('ik_'):
       ramp=((t[ids]-1.5)/1.).clamp(0.,1.)
       target=ik_poses[variant][None,:].expand(len(ids),-1).clone()
       motion_gain=float(variant.rsplit('_w',1)[1])/100. if '_w' in variant else 1.
       target[:,12:]+=motion_gain*cmd[ids,None]/.0875/5.
       target[:,12:]+=correction[ids,None]*torch.tensor([1.,-1.,1.,-1.],device=env.device)/5.
       action[ids]=torch.lerp(action[ids],target,ramp[:,None])
     for variant,ids in self.ids.items():
      if variant.startswith('fb_'):
       scale=torch.tensor([.08,.14,.14]*4,device=env.device)
       default=torch.tensor([0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5],device=env.device)
       actual=decoded_leg_pos[ids]
       q0=torch.tensor(evidence[variant]['source_actual_joint_pos'],device=env.device)
       desired=torch.tensor(evidence[variant]['target_actual_joint_pos'],device=env.device)
       load_offset=default+offline_pose[:12]*scale-q0
       # Proprioceptive feedback, no time/phase/contact/linear-velocity oracle.
       target_q=actual+(desired-actual).clamp(-.05,.05)+load_offset
       action[ids,:12]=(target_q-default)/scale
       motion_gain=float(variant.rsplit('_w',1)[1])/100. if '_w' in variant else 1.
       action[ids,12:]=offline_pose[None,12:]+motion_gain*cmd[ids,None]/.0875/5.
       action[ids,12:]+=correction[ids,None]*torch.tensor([1.,-1.,1.,-1.],device=env.device)/5.
     if replay_actions is None:
      action,control_metrics=coordinator.apply(obs,action,cmd,decoded_leg_pos,t)
     else:
      frame=base.episode_length_buf.long().clamp(max=len(replay_actions)-1)
      action=replay_actions[frame].clone()
      moving=torch.tensor([c['speed']>0 for c in cases],device=env.device)&(t>=3.)
      action[moving,12:]+=a.replay_wheel_gain*cmd[moving,None]/.0875/5.
      control_metrics={}
     robot=base.scene['robot'];ids=base.r1_joint_ids
     wheel_ids=[robot.body_names.index(name) for name in ('FR_wheel_link','FL_wheel_link','RR_wheel_link','RL_wheel_link')]
     executed=action.clone()
     if a.action_noise:
      for seed,indices in noise_indices.items():
       executed[indices]+=torch.randn((len(indices),16),device=env.device,generator=self.action_generators[seed])*self.action_std
     values=dict(observation=obs,action=action,executed_action=executed,actor_action=action0,joint_pos=robot.data.joint_pos[:,ids],joint_vel=robot.data.joint_vel[:,ids],phase=base.r1_eval_phase,command_vx=cmd,target_vx=term.target_vx,wheel_positions=robot.data.body_pos_w[:,wheel_ids],body_com_pos=robot.data.body_com_pos_w[:,base.r1_base_id],body_gravity=robot.data.projected_gravity_b,wheel_damping=robot.actuators['wheel'].damping,body_omega=robot.data.root_ang_vel_b,wheel_contact_force=base.scene['contact_forces'].data.net_forces_w[:,base.r1_wheel_sensor_ids],wheel_center_vel=robot.data.body_lin_vel_w[:,wheel_ids],wheel_angular_vel=robot.data.body_ang_vel_w[:,wheel_ids],decoded_leg_pos=decoded_leg_pos)
     values.update(control_metrics)
     for k,v in worker.cpu_measurements(values).items():self.trace.setdefault(k,[]).append(v.copy())
     self.n+=1
     if self.n%250==0:print('TOW_CONTROL_STEPS',self.n,'TOW_PROGRESS',[(c['variant'],c.get('noise_seed'),round(float(base.r1_eval_progress[i]),4),round(float(t[i]),2)) for i,c in enumerate(cases) if c['slope_deg']==25 and c['towed_kg']==40 and c['speed']>0],flush=True)
     return executed
    return infer
  runner=Runner();worker.protocol.cases=lambda distance:cases
  original_readback=worker.cpu_measurements;readback_audit={}
  def audited_readback(values):
   result=original_readback(values);key='|'.join(values)
   if key not in readback_audit:
    for name,value in values.items():
     old=value.detach().cpu().numpy()
     if old.dtype!=result[name].dtype or not np.array_equal(old,result[name],equal_nan=True):raise ValueError('Batched readback differs: '+name)
    readback_audit[key]=dict(fields=len(values),device=str(next(iter(values.values())).device),exact_values=True,exact_dtype=True,exact_order=list(result)==list(values))
    launch.write_json(out/'readback_equivalence.json',list(readback_audit.values()))
   return result
  worker.cpu_measurements=audited_readback
  args=dict(mode='evaluate',eval_distance=1.,eval_seeds=a.seeds,seed=a.seeds[0],checkpoint=str(source),command_profile='slew',headless=True)
  with torch.inference_mode():worker.evaluate(env,runner,args,out,app)
  launch.write_json(out/'noise_streams.json',dict(seeds=a.parallel_noise_seeds,calls=noise_audit,native_additive_bounds=True,applied_before_history=True,world_initial_state_randomized=False,action_noise=a.action_noise,action_seed_offset=100000,paired_across_variants=True,raw_action_std=runner.action_std.cpu().tolist()))
  result=json.loads((out/'evaluation.json').read_text());traces={k:np.stack(v) for k,v in runner.trace.items()}
  physics_traces={k:np.stack(value) for k,value in physical.history.items()}
  offset=0
  for seed in a.seeds:
   seedcases=[r for r in result['cases'] if r['seed']==seed]
   for i,r in enumerate(seedcases):
    n=r['samples'];np.savez_compressed(out/f'seed{seed}_case{i:02d}_policy_trace.npz',**{k:v[offset:offset+n,i] for k,v in traces.items()})
    np.savez_compressed(out/f'seed{seed}_case{i:02d}_physics_trace.npz',**{k:v[offset:offset+n,i].reshape(n*base.cfg.decimation,*v.shape[3:]) for k,v in physics_traces.items()},physics_dt=.001,post_integration=True)
   offset+=max(r['samples'] for r in seedcases)
  assert offset==len(traces['action'])
  summary=[]
  for variant in a.variants:
   for speed in (0.,a.move_speed):
    rr=[r for r in result['cases'] if r['case']['variant']==variant and r['case']['slope_deg']==25 and r['case']['towed_kg']==40 and r['case']['speed']==speed]
    summary.append(dict(variant=variant,speed=speed,pass_count=sum(r['success'] for r in rr),total=len(rr),support=[r.get('metrics',{}).get('hold_four_wheel_fraction') for r in rr],drift=[r.get('metrics',{}).get('hold_drift_m') for r in rr],progress=[r['observed']['progress_m'] for r in rr],duration=[r['observed']['duration_s'] for r in rr],failed=[r['failed_checks'] for r in rr]))
  assert launch.sha256(source)==digest
  launch.write_json(out/'summary.json',dict(teacher_only=True,ppo_updates=0,source_sha_unchanged=True,summary=summary));print('TOW_CONTROL_COMPLETE',json.dumps(summary),flush=True)
 except BaseException as exc:
  launch.write_json(out/'worker_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
  import os
  sys.stdout.flush();sys.stderr.flush();os._exit(1)
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
