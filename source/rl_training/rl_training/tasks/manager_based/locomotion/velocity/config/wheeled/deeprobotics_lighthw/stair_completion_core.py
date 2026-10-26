"""全四轮共同过阶的纯状态机：只生成奖励证据，不控制动作/课程/reset。"""
import torch

VERSION = 'lighthw_all_wheel_stair_completion_v1'
PARAMETERS = dict(wheel_radius=.0875, first_edge=-3., step_width=.3, risers=7,
                  corridor_half_width=1.15, height_tolerance=.035,
                  landing_half_width=1.15, landing_seconds=.2,
                  command_threshold=.05, max_displacement=.25,
                  progress_bonus=1., completion_bonus=3.)


class StairCompletionState:
    """每个episode每级最多1分；四轮登顶留台0.2秒额外3分，最多10分。

    激活：正向M20楼梯、底部出生，过阶时前后帧命令vx/世界X均>0.05且持续安全。第7级已合法计分后可零命令留台完成。
    以最慢一轮的有序进展决定共同级数；无需人为抬高、无需运动四轮承载。
    轮底允许既有3.5cm几何容差，平台使用既有内缩5cm和高度容差。
    回退/重跨不重复付分；关闭门控时已到达的里程碑也消耗，不积攒奖金。
    终止碰撞之前已经取得的进展分不追溯扣回；终止/姿态代价保持原实现。
    """
    def __init__(self, n, device):
        with torch.inference_mode(False):
            self.previous = torch.zeros(n, 4, 3, device=device)
            self.center = torch.zeros(n, 3, device=device)
            self.initialized = torch.zeros(n, device=device, dtype=torch.bool)
            self.eligible = torch.zeros_like(self.initialized)
            self.invalid = torch.zeros_like(self.initialized)
            self.previous_gate = torch.zeros_like(self.initialized)
            self.wheel_riser = torch.zeros(n, 4, device=device, dtype=torch.long)
            self.common_riser = torch.zeros(n, device=device, dtype=torch.long)
            self.landing_ticks = torch.zeros_like(self.common_riser)
            self.completed = torch.zeros_like(self.initialized)
            self.top_earned = torch.zeros_like(self.initialized)
            self.last_progress = torch.zeros_like(self.common_riser)
            self.last_completion = torch.zeros_like(self.initialized)
            self.episode_bonus = torch.zeros(n, device=device)

    @torch.no_grad()
    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in vars(self).values(): value[ids] = 0

    @torch.no_grad()
    def observe(self, wheel_xyz, center, positive_stairs, vx, world_vx, safe, dt=.02):
        p = PARAMETERS
        if abs(dt-.02)>1e-9: raise ValueError('Pinned 50Hz control sampling required')
        xyz = wheel_xyz - center[:, None, :]
        fresh = ~self.initialized
        finite = torch.isfinite(wheel_xyz).all((1,2)) & torch.isfinite(center).all(1)
        finite &= torch.isfinite(vx) & torch.isfinite(world_vx)
        gate = (vx > p['command_threshold']) & (world_vx > p['command_threshold'])
        corridor = (xyz[:,:,1].abs() <= p['corridor_half_width']).all(1)
        self.eligible[fresh] = (positive_stairs & (xyz[:,:,0] < p['first_edge']-.02).all(1) & (center[:,2] > 0))[fresh]
        self.center[fresh] = center[fresh]
        index = (torch.floor((xyz[:,:,0]-p['first_edge'])/p['step_width'])+1).long().clamp(0,p['risers'])
        displaced = (wheel_xyz-self.previous).norm(dim=2).amax(1)>p['max_displacement']
        skipped = (index > self.wheel_riser+1).any(1)
        self.invalid |= ~finite | ~safe | ~corridor | (center!=self.center).any(1) | ((displaced|skipped)&~fresh)
        top = center[:,None,2]*index/p['risers']
        high = wheel_xyz[:,:,2]-p['wheel_radius'] >= top-p['height_tolerance']
        attained = torch.where(high, index, self.wheel_riser)
        self.wheel_riser.copy_(torch.maximum(self.wheel_riser,attained))
        common = self.wheel_riser.amin(1)
        increment = (common-self.common_riser).clamp(0,1)
        allowed = self.eligible & positive_stairs & ~self.invalid & gate & self.previous_gate & ~fresh
        self.last_progress.copy_(torch.where(allowed,increment,0))
        self.common_riser.copy_(torch.maximum(common,self.common_riser))
        self.top_earned |= (self.last_progress>0)&(self.common_riser==7)
        landing = (common==7) & (xyz[:,:,0].abs()<=p['landing_half_width']).all(1)
        landing &= (xyz[:,:,1].abs()<=p['landing_half_width']).all(1)
        landing &= ((wheel_xyz[:,:,2]-p['wheel_radius']-center[:,None,2]).abs()<p['height_tolerance']).all(1)
        self.landing_ticks.copy_(torch.where(landing & ~self.invalid,self.landing_ticks+1,0))
        done = self.landing_ticks >= round(p['landing_seconds']/dt)
        self.last_completion.copy_(done & ~self.completed & self.top_earned & self.eligible & positive_stairs & ~self.invalid)
        self.completed |= done
        bonus = self.last_progress*p['progress_bonus'] + self.last_completion*p['completion_bonus']
        self.episode_bonus += bonus
        self.previous.copy_(torch.nan_to_num(wheel_xyz)); self.previous_gate.copy_(gate & finite)
        self.initialized.fill_(True)
        return bonus
