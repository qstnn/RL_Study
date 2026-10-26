"""Front-support teacher outputs only original PD positions and wheel speeds."""
import itertools
import math
import torch
from .model import pitch_rate


def candidates():
    # Bounded first screen: 16 rows x two independent nominal slots, no endless search.
    return [dict(id=i, rise_scale=scale, ramp_s=ramp, trigger_deg=5., blend_s=blend,
                 rise_wheel=-.20, line_gain=8., velocity_gain=2.,
                 pitch_gain=.05, pitch_rate_gain=.20, leg_line_gain=0.)
            for i, (scale, ramp, blend) in enumerate(itertools.product((.6, .8), (1., 2.), (1., 1.5, 2., 3.)))]


class Teacher:
    def __init__(self, rows, geometry, device):
        self.rows = rows; self.n = len(rows); self.device = device
        self.params = {k: torch.tensor([r[k] for r in rows], device=device) for k in rows[0]}
        self.target = torch.tensor(geometry['selected']['pd_action'], device=device)

    def __call__(self, env, obs):
        p = self.params; raw = obs[:, :305]
        t = env.state.clock(env.physics_steps*env.physics_dt)
        u = ((t-1.)/p['ramp_s']).clamp(0, 1); smooth = u*u*(3-2*u)
        initial = torch.zeros(self.n, 16, device=self.device)
        initial[:, [1, 4]] = (-p['rise_scale']*smooth)[:, None]
        initial[:, [7, 10]] = (p['rise_scale']*smooth)[:, None]
        initial[:, 12:14] = (p['rise_wheel']*smooth)[:, None]
        m = env.metrics()
        lateral = m['wheel_pos'][:, 1, :2]-m['wheel_pos'][:, 0, :2]
        forward = torch.nn.functional.normalize(torch.stack((lateral[:, 1], -lateral[:, 0]), -1), dim=-1)
        gap = ((m['com'][:, :2]-m['wheel_pos'][:, :2, :2].mean(1))*forward).sum(-1)
        velocity = (m['com_vel'][:, :2]*forward).sum(-1)
        pitch, rate = pitch_rate(raw)
        final = initial.clone(); final[:, :12] = self.target
        final[:, 12:14] = (p['line_gain']*gap + p['velocity_gain']*velocity
            + p['pitch_gain']*(pitch-math.pi/2)+p['pitch_rate_gain']*rate)[:, None]
        final[:, 14:16] = 0
        final[:, [1, 4]] += (p['leg_line_gain']*gap.clamp(-.12, .12))[:, None]
        blend = obs[:, 306:307]
        action = torch.where(obs[:, 305:306] > .5, obs[:, 307:323]*(1-blend)+final*blend, initial)
        return action.clamp(-1, 1)
