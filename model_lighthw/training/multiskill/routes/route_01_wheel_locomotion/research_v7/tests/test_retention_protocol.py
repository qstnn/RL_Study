"""原真实12项CSV重算与非拖载协议接线回归；不启动Isaac。"""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from migration_v1 import retention_protocol as p
from migration_v1.dataset import EvidenceError


def test_original_twelve_real_success_episodes_keep_the_same_acceptance():
    folder = p.V7.parent/'unified_v5/validation/frozen_candidates_03'
    saved = json.loads((folder/'model1_evaluation.json').read_text())['cases']
    original_cases = [(i,c) for i,c in enumerate(saved) if c['case']['towed_kg']==0.]
    assert p.cases() == [c['case'] for _,c in original_cases]
    assert len(p.cases()) == 12
    for i,expected in original_cases:
        with (folder/f'seed43_case{i+16:02d}.csv').open() as stream:
            rows = [{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
        actual = p.original.summarize(rows,expected['case'],None)
        assert actual['success'] is True
        assert actual['checks'] == expected['checks']
        assert actual['window_samples'] == expected['window_samples']
        assert actual['metrics'] == expected['metrics']


def test_original_case_budgets_and_accelerations_do_not_follow_heavy_tow():
    profile = dict(ground_friction=3.,rolling_coefficient=.02,hitch_height_m=.04,
        loaded_acceleration_mps2=.05)
    for case in p.cases():
        cfg = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)))
        p.configure_case(cfg,case,profile)
        assert cfg.fixed_towed_kg == 0. and cfg.fixed_payload_kg == case['payload_kg']
        assert cfg.unloaded_acceleration == 1. and cfg.loaded_acceleration == .3
        assert cfg.eval_distance_m == 1. and cfg.command_profile == 'slew'
        assert not cfg.observations.policy.enable_corruption
        expected = 20. if case['speed']==0. else 60. if case['payload_kg'] else 30.
        assert p.original.max_steps(case)*.02 == expected
        assert cfg.episode_length_s > expected


def test_static_and_moving_commands_use_original_clock_and_nominal_context():
    for case in p.cases():
        obs = np.zeros((1,300))
        obs[:,290:300] = np.tile([case['payload_kg']/15.,0.],5)
        p.check_pre_action(obs,2.98,0,0.,case)
        if case['speed']:
            p.check_pre_action(obs,3.,1,.02 if not case['payload_kg'] else .006,case)
            with pytest.raises(EvidenceError,match='3-second'):
                p.check_pre_action(obs,3.,0,0.,case)
        else:
            p.check_pre_action(obs,19.98,0,0.,case)
        bad = obs.copy(); bad[0,299]=1.
        with pytest.raises(EvidenceError,match='nominal'):
            p.check_pre_action(bad,0.,0,0.,case)


def test_no_towing_requires_actual_zero_forces_not_only_nominal_metadata():
    keys = ('real_cart_robot_force_w_N','real_cart_robot_torque_w_Nm',
        'real_cart_cart_forces_w_N','real_cart_cart_torques_w_Nm')
    physics = {key:np.zeros((20,3)) for key in keys}
    rows = [dict(towed_kg=0.,drawbar_force_N=0.)]
    assert p.check_no_towing(physics,rows)['no_towing_force_verified']
    physics[keys[0]][2,0]=.1
    with pytest.raises(EvidenceError,match='external force'):
        p.check_no_towing(physics,rows)


def test_worker_import_is_cpu_only_and_invalid_case_creates_no_output(tmp_path):
    from migration_v1 import retention_worker as worker
    out = tmp_path/'absent'
    with pytest.raises(EvidenceError,match='index'):
        worker.prepare(tmp_path/'missing.pt',12,out)
    assert not out.exists()
