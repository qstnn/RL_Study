"""单因素对照：相同共享主干/双输出头，只改变输出头路由。"""
from copy import deepcopy
import math
import torch
from torch import nn


def observed_capture(observation):
    """原本体观测路由：站立有效且不在入场、pitch<=-5度；无教师或COM输入。"""
    if observation.shape[-1]!=305:
        raise ValueError('Expected unchanged v7 observation 5x61')
    frame=observation.reshape(*observation.shape[:-1],5,61)[...,-1,:]
    pitch=torch.atan2(frame[...,3],-frame[...,5])
    return (pitch<=math.radians(-5.)) & (frame[...,55]<.5) & (frame[...,60]>.5)


class PhaseActor(nn.Module):
    def __init__(self, original, routing):
        super().__init__()
        if routing not in ('uniform','observed'):
            raise ValueError('Unknown routing')
        modules=list(original.children())
        assert isinstance(modules[-1],nn.Linear) and modules[-1].in_features==128 and modules[-1].out_features==16
        self.trunk=nn.Sequential(*deepcopy(modules[:-1]))
        self.rise=deepcopy(modules[-1])
        self.capture=deepcopy(modules[-1])
        self.routing=routing

    def forward(self, observation):
        hidden=self.trunk(observation)
        a,b=self.rise(hidden),self.capture(hidden)
        if self.routing=='uniform':return .5*(a+b)
        return torch.where(observed_capture(observation)[...,None],b,a)
