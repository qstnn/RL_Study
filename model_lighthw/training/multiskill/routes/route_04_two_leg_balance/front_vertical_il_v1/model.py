"""Fresh IL rise network plus learned front-wheel capture feedback; no R4a weights."""
import math
import torch
from torch import nn
from .common import URDF
from .estimator import TorchBodyEstimator


def pitch_rate(raw):
    f = raw[..., -61:]; g = f[..., 3:6]; w = f[..., :3] / .2
    dg = -torch.linalg.cross(w, g, dim=-1)
    p = torch.atan2(g[..., 0], -g[..., 2])
    r = (-g[..., 2]*dg[..., 0]+g[..., 0]*dg[..., 2])/(g[..., 0]**2+g[..., 2]**2).clamp_min(1e-6)
    return p, r


def basis(x):
    p, r = pitch_rate(x[..., :305]); gap, velocity = x[..., 323:325].unbind(-1)
    return torch.stack((torch.ones_like(p), gap, velocity, p, r, gap.clamp(-.12, .12)), -1)


class Features:
    def __init__(self, n, device, trigger_deg=5., blend_s=1.5):
        self.trigger = torch.as_tensor(trigger_deg, device=device)*math.pi/180.
        self.steps = torch.as_tensor(blend_s, device=device)/.02
        self.latched = torch.zeros(n, dtype=torch.bool, device=device)
        self.age = torch.zeros(n, device=device)
        self.start = torch.zeros(n, 16, device=device)
        self.estimator = TorchBodyEstimator(URDF).to(device=device, dtype=torch.float32).eval()

    @torch.no_grad()
    def __call__(self, raw, done):
        f = raw[:, -61:]; clear = done.bool() | (f[:, 55] >= .5) | (f[:, 60] <= .5)
        self.latched[clear] = False; self.age[clear] = 0; self.start[clear] = 0
        self.age[self.latched] += 1
        pitch, _ = pitch_rate(raw)
        fire = ~self.latched & ~clear & (pitch >= self.trigger)
        self.start[fire] = f[fire, 34:50]; self.latched |= fire
        u = (self.age/self.steps).clamp(0, 1); blend = u*u*(3-2*u)
        estimate = self.estimator(raw)
        return torch.cat((raw, self.latched.float()[:, None], blend[:, None], self.start.clone(),
                          torch.stack((estimate['gap_m'], estimate['velocity_m_s']), -1)), -1)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Sequential(nn.Linear(307, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 16))
        self.coefficients = nn.Parameter(torch.zeros(16, 6))
        mask = torch.zeros(16, 6); mask[:, 0] = 1; mask[[1, 4], 5] = 1; mask[12:14, 1:5] = 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        if x.shape[-1] != 325: raise ValueError('R4b actor requires its own 325-dim features')
        prefix = self.prefix(x[..., :307])
        steady = basis(x) @ (self.coefficients*self.mask).T
        blend = x[..., 306:307]
        return torch.where(x[..., 305:306] > .5,
                           x[..., 307:323]*(1-blend)+steady*blend, prefix).clamp(-1, 1)
