"""持续任务合同：旧Actor只能通过显式旧观测适配器用于接管诊断。"""
from two_leg_balance_smooth_v6.spec import WEIGHTS as OLD_WEIGHTS, contract as old_contract
from two_leg_balance_split_v4.spec import STAND, LEG_JOINTS, WHEELS

PROFILE = 'continuous_stand_v7'
WEIGHTS = dict(OLD_WEIGHTS, hold_once=0., success_once=0., missed_hold=0., drift=0.)


def contract():
    c = old_contract()
    # 物理/稳定门沿用；删除未实现的旧PPO探索/优化器和固定回合语义，不能混作当前合同。
    for key in ('hold_seconds', 'episode_seconds', 'task_seconds_after_entry',
                'entry_noise_scale', 'quiet_noise_phases', 'quiet_noise_scale', 'hold_noise_scale',
                'hold_exploration', 'hold_noise_target_tilt_deg', 'hold_noise_ang_speed_rad_s', 'optimizer'):
        c.pop(key, None)
    c.update(version='lighthw_r4a_sustain_v7',
        reward_profile=PROFILE, weights=WEIGHTS, finite_horizon=False,
        task='MultiSkill-LightHW-R4a-Sustain-v7',
        task_behavior='continuous rear standing until explicit return command',
        capture_milestone_s=2., capture_deadline_s=8.,
        phase_clock='real entry then local 3s rise reference; indefinite hold; explicit return event then local 3s return reference',
        actor='IMU/gravity, encoders, previous action, mode, target gravity, command/local-reference phase, stand_request; 61 x 5; no episode clock',
        privileged='COM relative to body, velocities, contacts, clearances, base height, support error, actual RMS, entered/held/failed/stand_request; no elapsed timers or origin drift',
        default_technical_time_limit_s=40., diagnostic_time_limit='request hold_seconds + 15s; recorded in sustain_runtime.json',
        recovery_reward_requires_held=False, recovery_reward_requires_return_command=True, missed_hold_penalty_once=0.,
        time_cost_phase='rise only',
        old_checkpoint_compatible=False, automatic_return=False,
        observation_duration_is_success=False, first_post_capture_break_is_latched=True,
        balance_reward='12 * integral of 1kHz qualified support time per control step',
        hold_drift_cost='none; physical ground boundary is separate truncation',
        max_episode_displacement_m=None, world_ground_margin_m=2.,
        timeout_bootstrap='gamma * V(pre-reset terminal critic observation), exactly once',
        initialization='explicitly versioned new observations; old Actor only with separate adapter for diagnostic prefix; no automatic resume',
        ppo_admission=False, automatic_frozen_evaluation=False)
    return c
