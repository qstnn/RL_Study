"""Actual CPU/GPU loss-of-contact states must not retain fictitious support.

Fixture provenance/SHA values are adjacent to the small NPZ. These are failed
physical states; offline controller queries do not create successful labels.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from coordinated_capture import CoordinatedCapture


def sample(index):
    with np.load(Path(__file__).parent/'fixtures/native_contact_failure_states.npz') as data:
        return ({k:data[k][index] for k in data.files if k not in ('nominal','reference_json')},
                data['nominal'][index],json.loads(data['reference_json'].item()))


@pytest.mark.parametrize('index,unsupported',[(0,[2]),(1,[1,2])])
def test_actual_lost_contact_is_not_reintroduced_by_ideal_wheel_geometry(index,unsupported):
    z,nominal,reference=sample(index)
    controller=CoordinatedCapture(reference,25.,servo_discrete=True)
    _,result,diagnostic=controller.compute(z,nominal)
    assert not np.any(diagnostic['mask'][unsupported]),diagnostic
    if result['success']:
        np.testing.assert_allclose(result['contact_forces'][unsupported],0.,atol=1.e-8)


def test_first_control_cycle_keeps_original_startup_contact_assumption():
    z,nominal,reference=sample(2)
    controller=CoordinatedCapture(reference,25.,servo_discrete=True)
    _,result,diagnostic=controller.compute(z,nominal)
    assert z['time']==0. and not z['counts'].any()
    assert diagnostic['mask'].all()
    assert result['success']
