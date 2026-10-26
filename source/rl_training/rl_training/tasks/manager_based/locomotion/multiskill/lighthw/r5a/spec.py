"""R5A J1 的纯数据契约；不依赖 Isaac Sim。"""
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p/'offline/routes.json').is_file())
MULTISKILL = REPO/'model_lighthw/training/multiskill'
ROUTE = MULTISKILL/'routes/route_05_acrobatics/backflip'
URDF = REPO/'model_lighthw/lighthw_urdf/urdf/lighthw.urdf'
VERSION = 'lighthw_r5a_jump_landing_v2'
TASK = 'MultiSkill-LightHW-R5A-JumpLanding-v1'
LEGS = ('FR', 'FL', 'RR', 'RL')
JOINTS = tuple(f'{leg}_{part}_joint' for leg in LEGS for part in ('hip','thigh','calf','wheel'))
LEG_JOINTS = tuple(n for n in JOINTS if 'wheel' not in n)
WHEELS = tuple(f'{leg}_wheel_joint' for leg in LEGS)
WHEEL_BODIES = tuple(n.replace('_joint','_link') for n in WHEELS)
STAND = (0., .6, -1.5, 0., .6, -1.5, 0., -.6, 1.5, 0., -.6, 1.5)
CROUCH = (0., .95, -1.95, 0., .95, -1.95, 0., -.95, 1.95, 0., -.95, 1.95)
EXTEND = (0., .18, -.45, 0., .18, -.45, 0., -.18, .45, 0., -.18, .45)
RESIDUAL = (.10, .25, .35)*4
SCREEN = dict(min_com_rise_m=.05, min_air_seconds=.10, min_takeoff_vz=.25,
              min_wheel_clearance_m=.015, support_force_N=5., airborne_force_N=2.,
              min_stable_seconds=1., max_stable_tilt_deg=15., max_stable_speed_m_s=.05,
              max_stable_vertical_speed_m_s=.15, max_landing_drift_m=.05,
              base_height_m=.416, base_height_tolerance_m=.05,
              illegal_contact_N=5., fail_tilt_deg=60., fail_base_height_m=.16,
              fail_drift_m=.6, no_takeoff_deadline_s=1.8)
CONTRACT = dict(robot_mass_kg=22.0, urdf_sha256="781d9998885ed5962eacaf6363271480a3b98f346879ce8101a311e0314242a9", robot_model_version="lighthw_uniform_22kg_v1",
 version=VERSION, task=TASK, stage='J1_recovery_v2', payload_kg=0.,
                policy_dim=285, policy_frame_dim=57, policy_history=5, critic_dim=86, action_dim=16,
                joint_order=list(JOINTS), action_order=list(LEG_JOINTS+WHEELS),
                physics_dt=.001, policy_dt=.02, episode_seconds=4., finite_horizon=True,
                leg_effort_Nm=48., wheel_effort_Nm=24., leg_speed_rad_s=12.57, wheel_speed_rad_s=37.7,
                leg_pd=[80.,3.], wheel_damping=1., actuator_delay_steps=0,
                self_collision=True, friction=1., restitution=0., wheel_radius_m=.0875,
                reference_times_s=[0.,.5,.9,1.04,1.06,1.20,4.],
                stand=list(STAND), crouch=list(CROUCH), extend=list(EXTEND), residual_scale=list(RESIDUAL),
                action_transform='reference + scale * clamp(raw,-1,1); absolute URDF leg limits; wheel=5*clamp(raw,-1,1)',
                action_disabled_before_s=.5, empirical_normalization=False,
                target_com_rise_m=.08, screen=SCREEN,
                landing_force_cost_threshold_bw=3., landing_force_cost_rate=.3,
                event_sampling_hz=1000, terminal_continuous_hold=True,
                recovery_speed_scale_m_s=.03, recovery_drift_scale_m=.05,
                braking_start_s=1.5, braking_speed_cost_rate=1., braking_wheel_cost_rate=.02,
                recovery_drift_cost_rate=.2, terminal_success_bonus=10.,
                refinement_updates=300, refinement_noise_std=.05, refinement_learning_rate=1e-4,
                external_push=False, injected_root_velocity=False,
                full_backflip_implemented=False, hardware_validated=False)


def register():
    import gymnasium as gym
    entry = 'rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a.env:JumpLandingEnv'
    if TASK in gym.registry:
        if gym.spec(TASK).entry_point != entry:
            raise ValueError('Conflicting R5A registration')
        return
    gym.register(id=TASK, entry_point=entry, disable_env_checker=True)
