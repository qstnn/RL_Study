"""纯张量地形课程状态；不含奖励、动作、终止或PPO更新。"""
import torch


class TraversalState:
    """50Hz完整地块证据。出生半块、横向换列、回退和重置均不能伪造完整跨越。

    课程通过仅为升级依据，不等于独立1kHz行为验收。整段episode发生非轮碰撞、
    倾角越界或非有限会取消已完成的升级资格；运动低载从不参与判定。
    """
    def __init__(self,n,device):
        # 也允许在inference_mode内构造，避免后续reset/优化器日志发生inference tensor原位写入错误。
        with torch.inference_mode(False):
            self.row=torch.zeros(n,device=device,dtype=torch.long)
            self.col=torch.zeros_like(self.row)
            self.initialized=torch.zeros(n,device=device,dtype=torch.bool)
            self.entered=torch.zeros_like(self.initialized)
            self.clean=torch.ones_like(self.initialized)
            self.unsafe=torch.zeros_like(self.initialized)
            self.qualified=torch.zeros_like(self.initialized)
            self.command_distance=torch.zeros(n,device=device)
            self.first_x=torch.zeros_like(self.command_distance)
            self.forward_distance=torch.zeros_like(self.command_distance)
            self.steps=torch.zeros_like(self.row)

    @torch.no_grad()
    def observe(self,row,col,inside,moving,bad,x,command,dt,assigned_col):
        pair=self.initialized
        # 未跨越完整边界的出生地块不计；必须同列+worldX进入并离开下一个地块。
        eligible=inside&(col==assigned_col)
        forward=pair&eligible&(col==self.col)&(row==self.row+1)&moving
        complete=forward&self.entered&self.clean&~bad
        self.unsafe.logical_or_(bad | (inside&(col!=assigned_col)))
        self.qualified.logical_or_(complete)
        same=pair&eligible&(col==self.col)&(row==self.row)
        self.entered.copy_(torch.where(forward,True,self.entered&same))
        self.clean.copy_(torch.where(forward,~bad,self.clean&~bad))
        self.first_x.copy_(torch.where(self.steps==0,x,self.first_x))
        self.forward_distance.copy_(torch.nan_to_num(x-self.first_x))
        self.command_distance.add_(torch.where(moving,command.clamp_min(0)*dt,0.))
        self.steps.add_(1)
        self.row.copy_(row);self.col.copy_(col);self.initialized.copy_(eligible)

    @torch.no_grad()
    def decisions(self,ids,levels,flat,terminated,max_level=9):
        sampled=self.steps[ids]>0
        unsafe=self.unsafe[ids]|terminated
        success=self.qualified[ids]&~unsafe
        # 高层地图最后一行之后没有完整下一块；最高级不升级，只有受阻(<4m)或不安全才降级。
        incomplete=~self.qualified[ids]
        incomplete=torch.where(levels==max_level,self.forward_distance[ids]<4.,incomplete)
        enough=self.command_distance[ids]>=12.
        up=sampled&~flat&(levels<max_level)&success
        down=sampled&~flat&~up&(unsafe|(enough&incomplete))
        next_level=(levels+up.long()-down.long()).clamp(0,max_level)
        next_level=torch.where(flat,torch.zeros_like(next_level),next_level)
        return next_level,up,down,unsafe,sampled&~enough&~unsafe&~success

    @torch.no_grad()
    def reset(self,ids):
        self.initialized[ids]=False;self.entered[ids]=False;self.clean[ids]=True
        self.unsafe[ids]=False;self.qualified[ids]=False;self.command_distance[ids]=0
        self.first_x[ids]=0;self.forward_distance[ids]=0;self.steps[ids]=0
