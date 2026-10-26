"""状态构造每控制步仅推进一次；仅用raw305与冻结起身Actor。"""
import torch
from common_feedback import CaptureMemory,TorchBodyEstimator,URDF


class FeedbackFeatures:
    def __init__(self,prefix,n,device):
        self.prefix=prefix.to(device).eval()
        self.memory=CaptureMemory(n,device)
        self.start=torch.zeros(n,16,device=device)
        self.estimator=TorchBodyEstimator(URDF).to(device=device,dtype=torch.float32).eval()

    @torch.no_grad()
    def __call__(self,raw,done):
        frame=raw[:,-61:]
        clear=done.bool() | (frame[:,55]>=.5) | (frame[:,60]<=.5)
        previous=self.memory.latched.clone()
        memory=self.memory.update(raw,done)
        self.start[clear]=0
        fire=(memory[:,0]>.5)&(~previous|clear)
        # u=0重复上一实际执行动作，避免旧capture头切换突变。
        if fire.any():self.start[fire]=frame[fire,34:50]
        estimate=self.estimator(raw)
        values=torch.stack((estimate['gap_m'],estimate['velocity_m_s']),-1)
        return torch.cat((raw,memory,self.start.clone(),values),-1)
