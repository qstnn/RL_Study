"""三套自主Actor按观测中的离地/首次接触切换，隔离起跳、飞行与恢复梯度。
没有教师在线接管或参考轨迹；PPO采样/log_prob/KL共用同一tanh-Normal分布。
"""
import copy,math
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from .spec import CONTRACT as C

class SquashedActorCritic(ActorCritic):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.ground_actor=copy.deepcopy(self.actor)
  self.landing_actor=copy.deepcopy(self.actor)
 def grounded(self,obs):
  # 84维最新帧第76项为首次离地标记；复用小维度解析概率测试时只使用原Actor。
  return obs[:,-84+76]<.5 if obs.shape[-1]==420 else torch.zeros(len(obs),device=obs.device,dtype=torch.bool)
 def raw_mean(self,observations):
  raw=self.actor(observations)
  if observations.shape[-1]==420:
   raw=torch.where(self.grounded(observations)[:,None],self.ground_actor(observations),raw)
   raw=torch.where((observations[:,-84+77]>.5)[:,None],self.landing_actor(observations),raw)
  return raw
 def latent_mean(self,observations):
  # 教师会精确达到tanh(3)。直接限幅让这些目标可达；模仿损失使用raw_mean防止饱和零梯度。
  return self.raw_mean(observations).clamp(-3.,3.)
 def update_distribution(self,observations):
  mean=self.latent_mean(observations);std=self.log_std.exp().expand_as(mean)
  if observations.shape[-1]==420:
   ground=self.grounded(observations);landed=observations[:,-84+77]>.5
   ground_std=std*C['ground_exploration_multiplier']
   floor=std.new_tensor([C['ground_exploration_leg_floor']]*12+[C['ground_exploration_wheel_floor']]*4)
   ground_std=torch.maximum(ground_std,floor).clamp_max(C['exploration_cap'])
   from .landing import observation_prelanding
   multiplier=torch.where(observation_prelanding(observations),C['prelanding_exploration_multiplier'],C['air_exploration_multiplier'])
   multiplier=torch.where(landed,C['landing_exploration_multiplier'],multiplier)
   moving_std=std*multiplier[:,None]
   std=torch.where(ground[:,None],ground_std,moving_std.clamp_min(C['moving_exploration_floor']))
  self.distribution=Normal(mean,std)
 def act(self,observations,**kwargs):
  self.update_distribution(observations);return torch.tanh(self.distribution.sample())
 def act_inference(self,observations):return torch.tanh(self.latent_mean(observations))
 def get_actions_log_prob(self,actions):
  a=actions.clamp(-1.+1e-6,1.-1e-6);z=torch.atanh(a)
  return (self.distribution.log_prob(z)-torch.log1p(-a.square())).sum(-1)
 @property
 def entropy(self):
  z=self.distribution.rsample();jac=2.*(math.log(2.)-z-torch.nn.functional.softplus(-2.*z))
  return (self.distribution.entropy()+jac).sum(-1)
