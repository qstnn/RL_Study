"""保留已选实际俯仰路由，显式迁移305到307输入；新列零初始化。"""
from copy import deepcopy
import torch
from torch import nn


class MemoryActor(nn.Module):
    def __init__(self,original,mode):
        super().__init__()
        if mode not in ('zero','exposed'):raise ValueError('Unknown memory mode')
        assert original.routing=='observed'
        self.mode=mode
        self.trunk=deepcopy(original.trunk)
        source=self.trunk[0]
        first=nn.Linear(307,source.out_features)
        with torch.no_grad():
            first.weight.zero_();first.weight[:,:305].copy_(source.weight);first.bias.copy_(source.bias)
        self.trunk[0]=first
        self.rise=deepcopy(original.rise);self.capture=deepcopy(original.capture)

    def forward(self,x):
        if x.shape[-1]!=307:raise ValueError('MemoryActor requires explicit 307-dim contract')
        from model_phase import observed_capture
        h=self.trunk(x)
        return torch.where(observed_capture(x[...,:305])[...,None],self.capture(h),self.rise(h))
