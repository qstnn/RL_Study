import pytest

from migration_v1.command_adapter import command_parameters
from migration_v1.dataset import EvidenceError


def declared_sources():
    # Only protocol metadata for CPU validation, never physical success evidence.
    return {'teacher_bundle': {'episodes': [
        {'case': {'slope_deg': slope, 'goal_distance_m': 1.2},
         'command_timing': {'actual_move_start_s': start},
         'physics_profile': {'loaded_acceleration_mps2': .05}}
        for slope, start in [(25., 3.), (-20., 7.48)]]}}


def test_fixed_task_clocks_keep_non_towing_protocol_and_mark_extrapolation():
    result = command_parameters([(25., 0., 40.), (-20., 0., 40.), (0., 0., 0.),
                                 (-25., 15., 0.), (-10., 0., 20.)], declared_sources())
    assert result['preparation_seconds'] == [3., 7.48, 3., 3., 7.48]
    assert result['loaded_acceleration_mps2'] == [.05, .05, .3, .3, .05]
    assert result['goal_distance_m'] == [1.2, 1.2, 1., 1., 1.2]
    assert not result['live_teacher_or_contact_ready']
    assert not result['non_anchor_behavior_verified']


def test_same_direction_disagreeing_source_times_are_rejected():
    sources = declared_sources()
    sources['teacher_bundle']['episodes'].append(dict(sources['teacher_bundle']['episodes'][0],
        command_timing={'actual_move_start_s': 8.}))
    with pytest.raises(EvidenceError, match='disagree'):
        command_parameters([(25., 0., 40.)], sources)


def test_forbidden_towing_or_mixed_payload_cannot_enter_command_table():
    for case in [(-25., 0., 40.), (0., 15., 40.)]:
        with pytest.raises(ValueError, match='forbidden'):
            command_parameters([case], declared_sources())
