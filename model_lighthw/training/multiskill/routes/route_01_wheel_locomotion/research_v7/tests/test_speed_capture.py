import numpy as np
from speed_capture import SpeedCapture


def test_bounded_command_can_recover_a_falling_speed_reference_without_following_it():
    c=SpeedCapture(0.,.2)
    assert np.isclose(c.command(0.,-.5),-.3)
    assert np.isclose(c.command(1.,-1.),-.2)
    assert c.command(3.,-2.)==0.
    assert not c.finished


def test_preparation_cannot_end_at_zero_crossing_or_with_fast_wheels():
    c=SpeedCapture(0.,.2)
    def observe(v,w):
        return c.observe(0.,v,0.,np.zeros(2),np.zeros(3),0.,np.array([20.,20.,w,w]),np.array([0.,0.,60.,60.]))
    for _ in range(15):observe(0.,0.)
    observe(-.3,-3.)
    for _ in range(30):observe(0.,10.)
    assert not c.finished
    for _ in range(21):observe(0.,0.)
    assert c.finished
