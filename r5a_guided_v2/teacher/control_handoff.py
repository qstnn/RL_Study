"""Continuous touchdown handoff, with the unchanged bounded teacher underneath."""
import torch
from .control_split_bound import Teacher as BaseTeacher, ACTION_LIMIT

DIM = 40

class Overlay:
    def __init__(self, original, **values):
        self._original = original
        self.__dict__.update(values)
    def __getattr__(self, name):
        return getattr(self._original, name)

class Teacher:
    def __init__(self, actor, parameters):
        self.actor = actor
        self.parameters = torch.as_tensor(parameters, device=next(actor.parameters()).device, dtype=torch.float32)
        self.base = BaseTeacher(actor, self.parameters[:, :39])

    def __call__(self, obs, env):
        # 保留空中展开规律，在首次触地角度处得到连续的交接起点。
        # 只覆盖教师所读的阶段标志；不修改环境状态、接触或终止判定。
        capture = self.base(obs, env)
        state = Overlay(env.state, landed=torch.zeros_like(env.state.landed))
        opening = self.base(obs, Overlay(env, state=state))
        elapsed = env.episode_physics_steps * env.physics_dt - env.state.landing_t
        duration = self.parameters[:, 39].clamp(.02, .16)
        u = (elapsed / duration).clamp(0., 1.)
        weight = u * u * (3. - 2. * u)
        blended = opening * (1. - weight[:, None]) + capture * weight[:, None]
        return torch.where(env.state.landed[:, None], blended, capture).clamp(-ACTION_LIMIT, ACTION_LIMIT)
