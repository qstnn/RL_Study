"""Front/rear-neutral geometry with a stricter five-degree vertical target gate."""
import math
from two_leg_balance_smooth_v6.core import targets, desired_gravity, support_geometry
from two_leg_balance_smooth_v6.core import masks as original_masks


def masks(m, mode, target_g):
    value = original_masks(m, mode, target_g)
    value['balance'] = (value['pair'] & value['quiet'] & (m['line_error'] <= .06)
                        & (value['attitude'] <= math.radians(5.)))
    return value
