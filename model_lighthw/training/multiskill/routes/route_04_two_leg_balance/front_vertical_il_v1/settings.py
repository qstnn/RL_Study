"""R4b-only contract; the rear-standing checkpoint is never an initializer."""
from two_leg_balance_sustain_v7.spec import WEIGHTS as BASE_WEIGHTS

PROFILE = 'front_vertical_v1'
WEIGHTS = dict(BASE_WEIGHTS)
VERSION = 'r4b_front_vertical_il_v1'
OBSERVATION = 'r4b_raw305_positive_capture2_previous_action16_front_fk2_v1'
PPO_CONFIG = dict(actor_obs_dim=325, critic_obs_dim=88, num_actions=16,
    num_steps_per_env=32, initial_std=.0005, learning_rate=3e-6,
    num_learning_epochs=2, num_mini_batches=1, gamma=.995, lam=.95)


def contract():
    return dict(version=VERSION, skill='R4b', support='FR/FL', airborne='RR/RL',
        target_pitch_deg=90., max_target_error_deg=5., capture_seconds=2.,
        capture_deadline_after_rise_start_s=8., observation_horizons_s=[10, 60, 300],
        automatic_return=False, raw_actor_dim=305, actor_dim=325, critic_dim=88, action_dim=16,
        actor_truth_inputs=False, episode_clock_input=False, initial_policy='fresh R4b IL',
        observation_noise=False, reset_randomization=False, domain_randomization=False,
        physics_dt=.001, control_dt=.02, leg_pd=[80., 3.], wheel_damping=1.,
        torque_limits_nm=[48., 24.], max_com_speed_m_s=.12, max_angular_speed_rad_s=.25,
        max_support_line_error_m=.06, min_airborne_clearance_m=.03,
        support_force_n=5., airborne_force_n=2., cumulative_drift_limit=None,
        failure_pitch_range_deg=[-35., 110.], failure_lateral_gravity_angle_deg=35.,
        min_base_height_m=.18, illegal_body_contact_n=5.,
        high_frequency=dict(cutoff_hz=5., rms_tau_s=.1,
            limits=[.02, .05, .003490658503988659, .5]),
        weights=WEIGHTS, reward_profile=PROFILE, capture_pose_width_rad=.12,
        ppo=PPO_CONFIG, ppo_time_limit_s=360., automatic_frozen_evaluation=False,
        per_100_updates='save only', formal_training='explicit user command')
