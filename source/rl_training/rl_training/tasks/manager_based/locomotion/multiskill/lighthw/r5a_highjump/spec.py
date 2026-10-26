"""R5A J1 v3：固定高目标，不使用逐级增高课程。"""
from copy import deepcopy
try:
    from ..r5a import spec as legacy
except ImportError:
    from r5a import spec as legacy

REPO, MULTISKILL, URDF = legacy.REPO, legacy.MULTISKILL, legacy.URDF
ROUTE = legacy.ROUTE/'highjump'
VERSION = 'lighthw_r5a_highjump_v3'
TASK = 'MultiSkill-LightHW-R5A-HighJump-v3'
LEGS, JOINTS, LEG_JOINTS = legacy.LEGS, legacy.JOINTS, legacy.LEG_JOINTS
WHEELS, WHEEL_BODIES = legacy.WHEELS, legacy.WHEEL_BODIES
STAND, CROUCH, EXTEND, RESIDUAL = legacy.STAND, legacy.CROUCH, legacy.EXTEND, legacy.RESIDUAL
# 探针只改变关节参考，不改变质量、重力、驱动上限或根速度。
REFERENCE_PROFILES = [dict(crouch_thigh=.95,crouch_calf=-1.95,push_duration_s=.16,
    thigh_duration_s=.16,calf_duration_s=.05,calf_delay_s=-.075,
    extend_thigh=.05,extend_calf=-.10,crouch_end_s=.85,
    tuck_fraction=1.8,landing_delay_s=0.,prepare_duration_s=.08)]
EXTEND=(0.,.05,-.10,0.,.05,-.10,0.,-.05,.10,0.,-.05,.10)

SCREEN = deepcopy(legacy.SCREEN)
SCREEN.update(min_flight_rise_m=.10, min_free_flight_s=.30,
    min_first_flight_peak_rise_m=.16, min_peak_wheel_clearance_m=.22, max_landing_impact_N=2300.)
CONTRACT = deepcopy(legacy.CONTRACT)
CONTRACT.update(version=VERSION, task=TASK, stage='J1_highjump_v3',
    policy_dim=400, policy_frame_dim=80, critic_dim=109,
    target_flight_rise_m=.12, target_takeoff_vz_m_s=(2*9.81*.12)**.5,
    target_wheel_clearance_m=.25, target_free_flight_s=.34, target_standing_rise_m=.20,
    target_range_m=[.22,.28], initialization_min_flight_rise_m=.08,
    height_curriculum=False, target_com_rise_m=None,
    screen=SCREEN, reference_profiles=REFERENCE_PROFILES, reference_profile=0,
    push_start_s=.85, tuck_transition_s=.06, 
    cushion_seconds=.25, critic_warmup_updates=20, actor_learning_rate=1e-5,
    critic_learning_rate=1e-4, exploration_std=.05, quality_interval=50,
    training_actor_updates=300, quality_consecutive_failures=2,
    reference_control_hz=1000, stable_wheel_speed_cost_rate=.02,
    target_height_reward_sigma_m=.04, terminal_success_bonus=20.,
    recovery_drift_cost_rate=.10, landing_force_cost_rate=.15,
    action_transform='event reference + residual; unchanged 48/24 Nm and 12.57/37.7 rad/s',
    actor_feedback='legacy 57 + feedback 23 (COM velocity, relative XY, normal forces, event phase, flight/clearance/airtime/standing-rise goals, current clearance and COM height); history 5',
    full_backflip_implemented=False)

CONTRACT['extend']=list(EXTEND)
CONTRACT['jump_goal']='fixed clearance/flight/COM goals from first update; no height curriculum'
for obsolete in ('reference_times_s','target_com_rise_m','refinement_updates','refinement_noise_std','refinement_learning_rate','braking_start_s'):
    CONTRACT.pop(obsolete,None)


def register():
    import gymnasium as gym
    entry='rl_training.tasks.manager_based.locomotion.multiskill.lighthw.r5a_highjump.env:JumpLandingEnv'
    if TASK in gym.registry:
        if gym.spec(TASK).entry_point != entry: raise ValueError('Conflicting highjump registration')
        return
    gym.register(id=TASK,entry_point=entry,disable_env_checker=True)
