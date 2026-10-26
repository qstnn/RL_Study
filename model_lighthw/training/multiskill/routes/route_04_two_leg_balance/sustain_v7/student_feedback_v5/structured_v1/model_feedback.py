"""从已执行成功标签学习低维捕获反馈；原起身网络冻结。"""
import torch
from torch import nn


def feedback_basis(raw,gap_velocity):
    """返回[1,gap,rolling速度,pitch,pitch_rate,clip(gap)]，全为部署本体量。"""
    f=raw[...,-61:];g=f[...,3:6];omega=f[...,:3]/.2
    derivative=-torch.linalg.cross(omega,g,dim=-1)
    pitch=torch.atan2(g[...,0],-g[...,2])
    rate=(-g[...,2]*derivative[...,0]+g[...,0]*derivative[...,2])/(g[...,0].square()+g[...,2].square()).clamp_min(1e-6)
    gap,velocity=gap_velocity.unbind(-1)
    return torch.stack((torch.ones_like(gap),gap,velocity,pitch,rate,gap.clamp(-.12,.12)),-1)


class FeedbackActor(nn.Module):
    def __init__(self,prefix,variant='full'):
        super().__init__()
        if variant not in ('full','no_velocity'):raise ValueError('Unknown feedback variant')
        self.prefix=prefix;self.prefix.requires_grad_(False);self.variant=variant
        mask=torch.zeros(16,6);mask[:,0]=1;mask[[7,10],5]=1
        mask[14:16,1:5]=1
        if variant=='no_velocity':mask[:,2]=0
        self.register_buffer('mask',mask)
        self.coefficients=nn.Parameter(torch.zeros(16,6))

    def forward(self,x):
        if x.shape[-1]!=325:raise ValueError('Structured feedback requires 325-dim explicit policy input')
        raw=x[...,:305];memory=x[...,305:307];start=x[...,307:323];estimated=x[...,323:325]
        prefix=self.prefix(x[...,:307])
        final=feedback_basis(raw,estimated)@(self.coefficients*self.mask).T
        blend=memory[...,1:2]
        mixed=start*(1.-blend)+final*blend
        return torch.where(memory[...,0:1]>.5,mixed,prefix).clamp(-1.,1.)
