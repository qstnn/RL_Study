import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rolling_load_audit import rolling_metrics


def data():
    # Only rear wheels support; the airborne wheels can spin without spoiling valid rolling.
    load=np.tile([0.,0.,50.,50.],(1000,1))
    contact=load>0
    center=np.full_like(load,.2)
    spin=np.tile([3.,3.,.2,.2],(1000,1))
    slip=abs(spin-center)
    return [load,contact,center,spin,slip]


def test_two_wheel_rolling_passes_without_four_wheel_gate():
    result=rolling_metrics(*data())
    assert result['rolling_dominates']
    assert result['weighted_slip_ratio']==pytest.approx(0.)
    assert result['spin_travel_ratio']==pytest.approx(1.)


@pytest.mark.parametrize('spin_speed',[0.,.6])
def test_locked_skid_and_spinning_are_rejected(spin_speed):
    d=data();d[3][:]=spin_speed;d[4][:]=abs(spin_speed-.2)
    assert not rolling_metrics(*d)['rolling_dominates']


def test_long_airborne_gap_and_no_support_are_rejected():
    d=data();d[0][300:400]=0
    assert not rolling_metrics(*d)['rolling_dominates']
    d[0][:]=0
    assert not rolling_metrics(*d)['rolling_dominates']


def test_bad_observations_are_rejected():
    d=data();d[2][0,0]=np.nan
    with pytest.raises(ValueError):rolling_metrics(*d)
