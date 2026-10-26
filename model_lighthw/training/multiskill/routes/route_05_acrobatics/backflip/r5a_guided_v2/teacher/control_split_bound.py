"""Teacher outputs confined to the current student's reachable mean-action range."""
import math
import torch
from .control_split import Teacher as SplitTeacher,DIM,score
ACTION_LIMIT=math.tanh(3.)
class Teacher(SplitTeacher):
 def __call__(self,obs,env):
  action=super().__call__(obs,env).clamp(-ACTION_LIMIT,ACTION_LIMIT)
  active=env.episode_physics_steps>=round(.4/env.physics_dt)
  return action*active[:,None]
