"""真实源工况列表的CPU配置/覆盖测试；合成元数据不算实际首批运行证据。"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from migration_v1 import schedule


@pytest.mark.parametrize('num_envs', [64, 128, 256])
def test_full_catalog_before_weighting_and_deterministic(num_envs):
    result = schedule.make_schedule(num_envs)
    cases = result['conditions']
    assert len(cases) == num_envs
    assert len(set(map(tuple, cases[:64]))) == 64
    assert set(map(tuple, cases)) == set(schedule.permitted_catalog()['cases'])
    assert result == schedule.make_schedule(num_envs)
    assert result['configured_full_coverage']
    assert result['runtime_pre_action_verified'] is False
    assert result['formal_ppo_authorized'] is False
    assert not any(s == -25 and t > 0 or p > 0 and t > 0 for s, p, t in cases)
    metadata = np.c_[cases, np.zeros(num_envs)]  # Synthetic only; no environment execution claim.
    review = schedule.validate_pre_action_conditions(metadata)
    assert review['supplied_metadata_covers_all_64']
    assert review['execution_provenance_verified'] is False
    assert review['formal_ppo_authorized'] is False


def test_extra_quotas_use_declared_weights_after_64_required_rows():
    result = schedule.make_schedule(128)
    assert result['extra_class_quotas'] == {'unloaded': 16, 'payload_only': 16, 'towing_only': 32}
    assert result['class_counts'] == {'unloaded': 27, 'payload_only': 49, 'towing_only': 52}
    assert [25., 0., 40.] in result['conditions']
    assert [-20., 0., 40.] in result['conditions']


@pytest.mark.parametrize('num_envs', [1, 16, 63])
def test_small_smoke_must_be_explicit_diagnostic(num_envs):
    with pytest.raises(ValueError, match='at least 64'):
        schedule.make_schedule(num_envs)
    result = schedule.make_schedule(num_envs, mode='diagnostic')
    assert result['diagnostic_only'] and not result['configured_full_coverage']
    metadata = np.c_[result['conditions'], np.zeros(num_envs)]
    with pytest.raises(ValueError, match='missing permitted'):
        schedule.validate_pre_action_conditions(metadata)
    review = schedule.validate_pre_action_conditions(metadata, mode='diagnostic')
    assert review['diagnostic_only'] and not review['supplied_metadata_covers_all_64']


def test_configured_64_does_not_hide_runtime_missing_or_forbidden_cases():
    metadata = np.c_[schedule.make_schedule(64)['conditions'], np.zeros(64)]
    metadata[-1] = metadata[0]
    with pytest.raises(ValueError, match='missing permitted'):
        schedule.validate_pre_action_conditions(metadata)
    metadata[0] = [-25., 0., 40., 0.]
    with pytest.raises(ValueError, match='forbidden'):
        schedule.validate_pre_action_conditions(metadata, mode='diagnostic')
    metadata[0] = [0., 15., 40., 0.]
    with pytest.raises(ValueError, match='forbidden'):
        schedule.validate_pre_action_conditions(metadata, mode='diagnostic')


@pytest.mark.parametrize('kind', ['nan', 'radians', 'phase', 'shape'])
def test_runtime_metadata_units_and_phase_validation(kind):
    metadata = np.c_[schedule.make_schedule(64)['conditions'], np.zeros(64)]
    if kind == 'nan':
        metadata[0, 0] = np.nan
    elif kind == 'radians':
        metadata[:, 0] *= np.pi/180.
    elif kind == 'phase':
        metadata[0, 3] = 5.
    else:
        metadata = metadata[:, :3]
    with pytest.raises(ValueError):
        schedule.validate_pre_action_conditions(metadata)
