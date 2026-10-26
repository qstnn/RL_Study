"""连续上楼课程的纯Tensor状态；不参与奖励、动作或PPO更新。"""
import torch
from .terrain_traversal_state_v2 import TraversalStateV2

VERSION = 'lighthw_continuous_stair_flights_v1'
WHEEL_RADIUS = .0875
SPAWN_OFFSET_X = -3.6
FIRST_EDGE_X = -3.
STEP_WIDTH = .3
RISERS = 7
LANDING_EDGE_X = -1.2


def bottom_origins(grid, levels, columns, stair_mask):
    """只移动正向金字塔楼梯的出生点；网格中心/地形几何不变。"""
    result=grid[levels,columns].clone()
    result[stair_mask,0]+=SPAWN_OFFSET_X
    result[stair_mask,2]=0.
    return result


class StairFlightState(TraversalStateV2):
    """必须从首级前方开始，依次使四轮跨过全部7级并在平台停留0.2s。

    保留整episode安全判据。单步/前轮登顶/换列/绕侧/传送跳级均不能晋级；
    移动低载不参与判定。其他地形保持原完整8m地块课程。
    """
    def __init__(self,n,device):
        super().__init__(n,device)
        with torch.inference_mode(False):
            self.stair=torch.zeros(n,dtype=torch.bool,device=device)
            self.from_bottom=torch.zeros_like(self.stair)
            self.flight_invalid=torch.zeros_like(self.stair)
            self.risers=torch.zeros(n,dtype=torch.long,device=device)
            self.landing_ticks=torch.zeros_like(self.risers)
            self.flight_done=torch.zeros_like(self.stair)

    @torch.no_grad()
    def observe_flight(self,wheel_xyz,center,stair_mask,dt=.02):
        self.stair.copy_(stair_mask)
        rel=wheel_xyz-center[:,None,:]
        # x/y以地块中心为原点；高度地面在worldZ=0，平台为center.z。
        first=self.steps==1
        started=(rel[:,:,0].max(1).values<FIRST_EDGE_X-.02)
        self.from_bottom.copy_(torch.where(first,started,self.from_bottom))
        corridor=(rel[:,:,1].abs()<=-LANDING_EDGE_X-.05).all(1)
        self.flight_invalid.logical_or_(stair_mask & ~corridor)
        crossed=torch.floor((rel[:,:,0].min(1).values-FIRST_EDGE_X)/STEP_WIDTH).long()+1
        crossed=crossed.clamp(0,RISERS)
        # 跳过多个台阶的传送/错误reset不能当连续跨越。回退允许重新尝试。
        self.flight_invalid.logical_or_(stair_mask & (crossed>self.risers+1))
        h=center[:,2]/RISERS
        enough_height=(wheel_xyz[:,:,2].min(1).values-WHEEL_RADIUS)>=crossed*h-.035
        self.risers.copy_(torch.where(stair_mask & enough_height,torch.maximum(self.risers,crossed),self.risers))
        landing=((rel[:,:,0]>=LANDING_EDGE_X+.05)&(rel[:,:,0]<=-LANDING_EDGE_X-.05)
            &(rel[:,:,2]-WHEEL_RADIUS).abs().le(.035)).all(1)&corridor
        eligible=stair_mask&self.from_bottom&~self.flight_invalid&~self.unsafe&(self.risers==RISERS)
        self.landing_ticks.copy_(torch.where(landing&eligible,self.landing_ticks+1,0))
        self.flight_done.logical_or_(eligible&(self.landing_ticks*dt>=.2-1e-6))
        self.qualified.copy_(torch.where(stair_mask,self.flight_done,self.qualified))

    @torch.no_grad()
    def decisions(self,ids,levels,flat,terminated,max_level=9):
        old=super().decisions(ids,levels,flat,terminated,max_level)
        stair=self.stair[ids];sampled=self.steps[ids]>0
        unsafe=self.unsafe[ids]|terminated|self.flight_invalid[ids]
        success=self.flight_done[ids]&~unsafe
        # 约3.3m即可使后轮到顶；命令积分>=4m才将未完成视为受阻。
        enough=self.command_distance[ids]>=4.
        up=sampled&(levels<max_level)&success
        down=sampled&~up&(unsafe|(enough&~success))
        level=(levels+up.long()-down.long()).clamp(0,max_level)
        insufficient=sampled&~enough&~unsafe&~success
        return tuple(torch.where(stair,a,b) for a,b in zip((level,up,down,unsafe,insufficient),old))

    @torch.no_grad()
    def reset(self,ids):
        super().reset(ids)
        self.from_bottom[ids]=False;self.flight_invalid[ids]=False
        self.risers[ids]=0;self.landing_ticks[ids]=0;self.flight_done[ids]=False
