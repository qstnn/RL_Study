"""自主后空翻 v6：动作/观测/奖励/验收契约。旧 J1 模型不兼容。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION='lighthw_r5a_teacher_guided_v2'
LEGS=('FR','FL','RR','RL')
LEG_JOINTS=tuple(f'{l}_{j}_joint' for l in LEGS for j in ('hip','thigh','calf'))
WHEELS=tuple(f'{l}_wheel_joint' for l in LEGS)
WHEEL_BODIES=tuple(n.replace('_joint','_link') for n in WHEELS)
STAND=(0.,.6,-1.5,0.,.6,-1.5,0.,-.6,1.5,0.,-.6,1.5)
# 前后关节坐标镜像；留出几何极限余量，但不施加时间参考。
LOW=(-.38,-1.45,-2.40)*2+(-.38,-1.45,.12)*2
HIGH=(.38,1.45,-.12)*2+(.38,1.45,2.40)*2
CONTRACT=dict(robot_mass_kg=22.0, urdf_sha256="781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9", robot_model_version="lighthw_uniform_22kg_v1",
 version=VERSION,stage='autonomous_backflip',physics_dt=.001,policy_dt=.02,
 episode_seconds=6.,settle_seconds=.4,launch_deadline_s=3.,action_dim=16,frame_dim=84,policy_dim=420,critic_dim=104,
 initial_state_randomization=False,observation_noise=False,physics_domain_randomization=False,
 training_action_sampling='PPO tanh-Normal exploration retained',evaluation_action_sampling=False,
 history=5,action_order=list(LEG_JOINTS+WHEELS),action_lower=list(LOW),action_upper=list(HIGH),stand=list(STAND),
 action_transform='absolute bounded joint positions around neutral; wheels +/-37.7rad/s; no reference trajectory',
 leg_effort_Nm=48.,wheel_effort_Nm=24.,leg_speed_rad_s=12.57,wheel_speed_rad_s=37.7,
 rigid_max_angular_velocity_deg_s=100000.,gyroscopic_forces=True,
 solver_position_iterations=8,solver_velocity_iterations=16,wheel_armature=.00243216,
 policy_distribution='tanh_normal_bounded_latent_mean_3',rollout_kl_limit=.03,
 momentum_absolute_tolerance=.02,momentum_relative_tolerance=.05,
 self_collisions=True,payload_kg=0.,friction=1.,restitution=0.,leg_pd=[80.,3.],wheel_damping=1.,
 airborne_force_N=2.,illegal_contact_N=5.,release_samples=7,min_takeoff_vz=.25,
 min_flight_s=.10,min_flight_rise_m=.02,rotation_range_deg=[350.,385.],max_axis_error_deg=20.,
 max_landing_tilt_deg=20.,stable_tilt_deg=15.,stable_speed_m_s=.1,stable_vertical_speed_m_s=.15,
 stable_landing_drift_m=.10,stable_hold_s=1.,standing_base_z=.416,standing_height_tolerance_m=.07,
 no_height_curriculum=True,no_root_velocity_injection=True,reference_controller=False,ground_actor_initial_output_scale=1.,
 fixed_flight_rise_target_m=.30,fixed_flight_time_target_s=.40,preparation_depth_scale_m=.06,
 preparation_max_tilt_deg=30.,preparation_support_N=5.,preparation_min_depth_m=.01,preparation_credit_window_s=.8,
 takeoff_H_scale=2.6,takeoff_omega_scale=8.,compact_inertia_reduction_scale=.20,
 compact_credit_rule='first-flight monotone-angle integral; early credit escrowed until valid flight; no repeat payment',
 ground_exploration_leg_floor=.03,ground_exploration_wheel_floor=.02,
 ground_exploration_multiplier=1.,air_exploration_multiplier=.8,landing_exploration_multiplier=.5,exploration_cap=.15,
 selection_joint_progress_tolerance=.02,policy_architecture='independent ground and air actors initialized by verified teacher imitation',
 actor_privileged_inputs='simulated whole COM velocity/height and wheel contact estimates; not hardware validated',
 initialization='phase-balanced teacher behavior cloning; fresh Critic and PPO optimizer',gamma=.995,
 critic_warmup_updates=20,demonstration_weight=.2,demonstration_decay_updates=1000,
 reward_weights=dict(push_potential=.5,height_potential=1.5,airtime_potential=.5,rotation_potential=12.,air_rotation_potential=16.,
  takeoff_coordination=3.,preparation_shaping=4.,extension_shaping=4.,launch_shaping=2.,preparation_credit=4.,compact_rotation=8.,
  flight_bonus=.5,landing_rotation=4.,landing_pose=6.,touchdown_deficit=-5.,stable_rate=4.,
  recovery_upright=1.,recovery_height=1.,recovery_support=3.,recovery_quiet=3.,recovery_drift=3.,recovery_combined=6.,landing_wheel_spin=-.3,
  terminal_success=80.,failure=-6.,alive_rate=-.03,action_rate=-.02,handoff_action_change=-.5,torque_rate=-.02,side_axis_rate=-1.,excessive_spin_rate=-.02,impact_rate=-1.))

def runner_config():
 return dict(seed=42,device='cuda:0',num_steps_per_env=32,max_iterations=3000,save_interval=200,
  empirical_normalization=False,clip_actions=1.,logger='tensorboard',
  policy=dict(class_name='SquashedActorCritic',init_noise_std=.25,noise_std_type='log',actor_hidden_dims=[256,256,128],
              critic_hidden_dims=[256,256,128],activation='elu'),
  algorithm=dict(class_name='PPO',num_learning_epochs=5,num_mini_batches=4,clip_param=.2,gamma=.995,lam=.95,
   value_loss_coef=1.,entropy_coef=.002,learning_rate=1e-5,max_grad_norm=1.,use_clipped_value_loss=True,
   schedule='fixed',desired_kl=.01))
