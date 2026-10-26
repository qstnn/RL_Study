"""用户确认的低速漂移标准及独立v6奖励；抖动阈值为本轮工程筛查值。"""
from two_leg_balance_split_v4.spec import (
    STAND, LOW, HIGH, LEG_JOINTS, WHEELS, WHEEL_BODIES, runner_config,
    SOFT_TRANSFER_WEIGHTS, TRANSFER_GATE_FLOOR, contract as original_contract)
from .jitter import CUTOFF_HZ, RMS_TAU_S, LIMITS, NAMES

VERSION = 'lighthw_r4a_slow_drift_smooth_v6'
PROFILE = 'slow_drift_smooth_v6'
REWARD_PROFILES = (PROFILE,)
MAX_SPEED = .12
WEIGHTS = dict(SOFT_TRANSFER_WEIGHTS, action_rate=-.05, jitter=-.5,
               action_curvature=-.25, slow_speed=-2.)


def weights_for(profile=PROFILE):
    if profile != PROFILE:
        raise ValueError('Smooth v6 requires its own reward profile')
    return dict(WEIGHTS)


def contract():
    c = original_contract(2., 'rear', 'soft_transfer_v2')
    c.update(version=VERSION, task='MultiSkill-LightHW-R4a-Smooth-v6', reward_profile=PROFILE,
        max_com_speed_m_s=MAX_SPEED, max_hold_drift_m=None, max_recovery_drift_m=None,
        hold_drift_cost='no hold/recovery position cost; episode displacement margin only above 1.2m',
        capture_stability='quiet with speed deadband <=0.12m/s; actual-state high-pass RMS quality in hold',
        jitter_filter=dict(kind='causal second-order Butterworth high-pass', cutoff_hz=CUTOFF_HZ,
            rms_time_constant_s=RMS_TAU_S, sample_dt=.001,
            limits=dict(zip(NAMES, LIMITS)), phase_gate=['hold', 'recovery'],
            classification='engineering screening thresholds, not a claim of zero vibration'),
        weights=WEIGHTS, automatic_frozen_evaluation=False, automatic_promotion=False,
        initialization='explicit legacy actor transfer only; new protocol/reward/source lineage')
    return c
