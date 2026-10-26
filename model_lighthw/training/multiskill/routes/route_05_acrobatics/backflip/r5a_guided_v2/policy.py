"""两套自主Actor：v5地面特征加温和输出初始化，原样继承v5空中/恢复权重。

切换只读取Actor观测中的launched事件；没有参考动作、规定起跳时刻或外力。
PPO采样、log_prob、熵和KL都使用同一个按相位确定的tanh-Normal分布。
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
 def grounded(self,obs):
  # 84维最新帧第76项为首次离地标记；复用小维度解析概率测试时只使用原Actor。
  return obs[:,-84+76]<.5 if obs.shape[-1]==420 else torch.zeros(len(obs),device=obs.device,dtype=torch.bool)
 def latent_mean(self,observations):
  raw=self.actor(observations)
  if observations.shape[-1]==420:raw=torch.where(self.grounded(observations)[:,None],self.ground_actor(observations),raw)
  return 3.*torch.tanh(raw/3.)
 def update_distribution(self,observations):
  mean=self.latent_mean(observations);std=self.log_std.exp().expand_as(mean)
  if observations.shape[-1]==420:
   ground=self.grounded(observations);landed=observations[:,-84+77]>.5
   ground_std=std*C['ground_exploration_multiplier']
   floor=std.new_tensor([C['ground_exploration_leg_floor']]*12+[C['ground_exploration_wheel_floor']]*4)
   ground_std=torch.maximum(ground_std,floor).clamp_max(C['exploration_cap'])
   moving_std=std*torch.where(landed,C['landing_exploration_multiplier'],C['air_exploration_multiplier'])[:,None]
   std=torch.where(ground[:,None],ground_std,moving_std.clamp_min(.01))
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
