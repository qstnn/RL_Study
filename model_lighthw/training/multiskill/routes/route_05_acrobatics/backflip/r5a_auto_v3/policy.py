"""有界策略：PPO在同一tanh变换前后的概率密度上计算比值，KL用潜在正态精确值。"""
import math
import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

class SquashedActorCritic(ActorCritic):
 def latent_mean(self,observations):
  # 连续限幅避免原始均值无限增长到动作裁剪区；确定性输出可达±tanh(3)。
  return 3.*torch.tanh(self.actor(observations)/3.)
 def update_distribution(self,observations):
  mean=self.latent_mean(observations)
  self.distribution=Normal(mean,self.log_std.exp().expand_as(mean))
 def act(self,observations,**kwargs):
  self.update_distribution(observations)
  return torch.tanh(self.distribution.sample())
 def act_inference(self,observations):
  return torch.tanh(self.latent_mean(observations))
 def get_actions_log_prob(self,actions):
  # float32极端尾样本可四舍五入为±1；仅在逆变换处作数值保护。
  a=actions.clamp(-1.+1e-6,1.-1e-6)
  z=torch.atanh(a)
  return (self.distribution.log_prob(z)-torch.log1p(-a.square())).sum(-1)
 @property
 def entropy(self):
  z=self.distribution.rsample()
  # H(tanh Z)=H(Z)+E log|d tanh(Z)/dZ|；重参数样本保持均值梯度。
  log_jac=2.*(math.log(2.)-z-torch.nn.functional.softplus(-2.*z))
  return (self.distribution.entropy()+log_jac).sum(-1)
