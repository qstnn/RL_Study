"""Static posture comparisons and bounded physical-trial contract (no Isaac app)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model import Model
from robot_pose_sweep import evaluate_candidate, reference_wheels, trial_command


@pytest.fixture(scope='module')
def screened_poses():
    model = Model()
    q, centers = reference_wheels(model)
    return {pitch: evaluate_candidate(model, q, centers, .40, pitch, 0.)
            for pitch in (-3., 0., 3.)}


def test_flat_static_pose_reproduces_prior_independent_screen(screened_poses):
    reference = Path(__file__).resolve().parents[1] / \
        'results/2026-10-22_no_ppo_1m_search/pose_h040.json'
    earlier = next(c for c in json.loads(reference.read_text())['candidates']
                   if c['id'] == 'h0.40_p0_x0.00')
    old_static = next(s for s in earlier['scenarios']
                      if s['slope_deg'] == 25. and s['phase'] == 'static')
    new_static = screened_poses[0.]['scenarios'][0]
    np.testing.assert_allclose(new_static['normal_N'], old_static['normal_N'], atol=1.e-6)
    np.testing.assert_allclose(new_static['com_lane_m'], old_static['com_lane_m'], atol=1.e-8)
    assert abs(new_static['hitch_clearance_m'] - old_static['hitch_clearance_m']) < 1.e-8


def test_front_low_tradeoff_is_not_mistaken_for_better_traction(screened_poses):
    low, flat, high = (screened_poses[p] for p in (-3., 0., 3.))
    assert low['hitch_clearance_m'] > flat['hitch_clearance_m'] > high['hitch_clearance_m']
    assert low['scenarios'][1]['front_normal_N'] < flat['scenarios'][1]['front_normal_N'] \
        < high['scenarios'][1]['front_normal_N']
    assert all(p['joint_limit_margin_rad'] > 0 for p in (low, flat, high))


def test_trial_requires_feasible_pose_and_short_diagnostic(tmp_path):
    pose_file = tmp_path / 'poses.json'
    pose_file.write_text(json.dumps({'selected': [{'id': 'known'}]}))
    with pytest.raises(ValueError, match='<=15'):
        trial_command(pose_file, 'known', tmp_path/'run', 20., 'cpu')
    with pytest.raises(ValueError, match='exactly once'):
        trial_command(pose_file, 'missing', tmp_path/'run', 15., 'cpu')
    command = trial_command(pose_file, 'known', tmp_path/'run', 15., 'cpu')
    assert '--seconds' in command and '--pose-file' in command
    assert '--loaded-acceleration' in command and command[command.index('--loaded-acceleration')+1] == '.05'
