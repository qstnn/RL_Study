"""自主后空翻 v2：动作/观测/奖励/验收契约。旧 J1 模型不兼容。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7]
VERSION='lighthw_r5a_autonomous_backflip_v2'
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
 history=5,action_order=list(LEG_JOINTS+WHEELS),action_lower=list(LOW),action_upper=list(HIGH),stand=list(STAND),
 action_transform='absolute bounded joint positions around neutral; wheels +/-37.7rad/s; no reference trajectory',
 leg_effort_Nm=48.,wheel_effort_Nm=24.,leg_speed_rad_s=12.57,wheel_speed_rad_s=37.7,
 rigid_max_angular_velocity_deg_s=100000.,gyroscopic_forces=True,
 policy_distribution='tanh_normal_bounded_latent_mean_3',rollout_kl_limit=.03,
 momentum_absolute_tolerance=.1,momentum_relative_tolerance=.1,
 self_collisions=True,payload_kg=0.,friction=1.,restitution=0.,leg_pd=[80.,3.],wheel_damping=1.,
 airborne_force_N=2.,illegal_contact_N=5.,release_samples=7,min_takeoff_vz=.25,
 min_flight_s=.10,min_flight_rise_m=.02,rotation_range_deg=[350.,385.],max_axis_error_deg=20.,
 max_landing_tilt_deg=20.,stable_tilt_deg=15.,stable_speed_m_s=.1,stable_vertical_speed_m_s=.15,
 stable_landing_drift_m=.10,stable_hold_s=1.,standing_base_z=.416,standing_height_tolerance_m=.07,
 no_height_curriculum=True,no_root_velocity_injection=True,reference_controller=False,
 actor_privileged_inputs='simulated whole COM velocity/height and wheel contact estimates; not hardware validated',
 initialization='new actor, no legacy checkpoint migration',gamma=.99,
 reward_weights=dict(push_potential=2.,height_potential=4.,rotation_potential=8.,
  flight_bonus=2.,landing_rotation=12.,landing_pose=6.,stable_rate=4.,recovery_upright=2.,recovery_height=2.,recovery_support=2.,recovery_quiet=1.,recovery_drift=1.,
  landing_wheel_spin=-.1,terminal_success=40.,
  failure=-6.,alive_rate=-.1,action_rate=-.02,torque_rate=-.02,side_axis_rate=-1.,
  excessive_spin_rate=-.02,impact_rate=-.01))

def runner_config():
 return dict(seed=42,device='cuda:0',num_steps_per_env=32,max_iterations=2000,save_interval=100,
  empirical_normalization=False,clip_actions=1.,logger='tensorboard',
  policy=dict(class_name='SquashedActorCritic',init_noise_std=.25,noise_std_type='log',actor_hidden_dims=[256,256,128],
              critic_hidden_dims=[256,256,128],activation='elu'),
  algorithm=dict(class_name='PPO',num_learning_epochs=5,num_mini_batches=4,clip_param=.2,gamma=.99,lam=.95,
   value_loss_coef=1.,entropy_coef=.002,learning_rate=1e-4,max_grad_norm=1.,use_clipped_value_loss=True,
   schedule='fixed',desired_kl=.01))
