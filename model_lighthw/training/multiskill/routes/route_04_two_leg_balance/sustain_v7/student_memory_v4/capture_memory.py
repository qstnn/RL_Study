"""部署侧捕获记忆：只用原本体观测，每个50Hz控制样本推进一次。"""
import math
import torch


class CaptureMemory:
    def __init__(self,n,device):
        self.latched=torch.zeros(n,dtype=torch.bool,device=device)
        self.age=torch.zeros(n,dtype=torch.long,device=device)

    def update(self,raw,reset=None):
        """返回[触发锁存, smoothstep(触发后步数/40)]；返回/入场/reset清空，40步后恒定。"""
        if raw.ndim!=2 or raw.shape[-1]!=305:raise ValueError('Expected batch x 305 raw v7 input')
        f=raw.reshape(-1,5,61)[:,-1]
        allowed=(f[:,55]<.5)&(f[:,60]>.5)
        clear=~allowed
        if reset is not None:clear=clear | reset.bool()
        self.latched[clear]=False;self.age[clear]=0
        self.age=torch.where(self.latched,(self.age+1).clamp(max=40),self.age)
        pitch=torch.atan2(f[:,3],-f[:,5])
        fire=~self.latched & allowed & (pitch<=math.radians(-5.))
        self.latched |= fire
        self.age[fire]=0
        u=self.age.to(raw.dtype)/40.
        return torch.stack((self.latched.to(raw.dtype),u*u*(3.-2.*u)),dim=-1)


def augment(raw,memory,mode):
    if mode not in ('zero','exposed'):raise ValueError('Unknown memory mode')
    if raw.shape[-1]!=305 or memory.shape!=(*raw.shape[:-1],2):raise ValueError('Invalid memory observation shape')
    return torch.cat((raw,memory if mode=='exposed' else torch.zeros_like(memory)),dim=-1)
