"""台阶跨越事件的纯 Tensor 状态机；不输出动作、不改变课程或 reset。"""
import torch

VERSION = 'lighthw_stair_clearance_event_v1'
PARAMETERS = dict(wheel_radius=0.0875, first_edge=-3.0, step_width=0.3, risers=7,
                  corridor_half_width=1.15, clearance_deadband=0.002,
                  clearance_full=0.015, bonus_per_wheel=0.5,
                  command_threshold=0.05, max_step_displacement=0.25)


class StairClearanceState:
    """从楼梯脚下出发，逐轮逐级最多付一次净空奖金。

    两个相邻控制采样之间轮心向 +X 穿越阶沿时，线性插值轮心高度，
    减轮半径和该级踏面高度得到几何净空。2mm 死区、15mm 满分，
    每轮每阶最多 0.5（积分后的 reward）。额外抬高不多给分。
    前后采样均须有朝上楼方向的前进命令；悬停、原地抬腿、回退重跨
    均不付分。已经跨过但未获奖的阶也消耗机会，避免来回刷分。
    这里只是 50Hz 轮端几何证据，不证明连续碰撞净空或协调步态。
    """
    def __init__(self, n, device):
        with torch.inference_mode(False):
            self.previous = torch.zeros(n, 4, 3, device=device)
            self.center = torch.zeros(n, 3, device=device)
            self.initialized = torch.zeros(n, device=device, dtype=torch.bool)
            self.eligible = torch.zeros_like(self.initialized)
            self.invalid = torch.zeros_like(self.initialized)
            self.previous_gate = torch.zeros_like(self.initialized)
            self.next_riser = torch.zeros(n, 4, device=device, dtype=torch.long)
            self.paid_events = torch.zeros_like(self.next_riser)
            self.crossings = torch.zeros_like(self.next_riser)
            self.episode_bonus = torch.zeros(n, device=device)
            self.last_clearance = torch.zeros(n, 4, device=device)
            self.last_paid = torch.zeros(n, 4, device=device, dtype=torch.bool)
            self.last_crossed = torch.zeros_like(self.last_paid)

    @torch.no_grad()
    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in vars(self).values():
            value[ids] = 0

    @torch.no_grad()
    def observe(self, wheel_xyz, center, positive_stairs, forward_command,
                world_x_command, safe):
        p = PARAMETERS
        xyz = wheel_xyz - center[:, None, :]
        finite = torch.isfinite(wheel_xyz).all((1, 2)) & torch.isfinite(center).all(1)
        finite &= torch.isfinite(forward_command) & torch.isfinite(world_x_command)
        gate = (forward_command > p['command_threshold']) & (world_x_command > p['command_threshold'])
        corridor = (xyz[:, :, 1].abs() <= p['corridor_half_width']).all(1)
        fresh = ~self.initialized
        # 只有从第一阶前方开始的正向楼梯 episode 可以得分。平台出生不补发。
        started_before = (xyz[:, :, 0] < p['first_edge'] - .02).all(1)
        self.eligible[fresh] = (positive_stairs & started_before & (center[:, 2] > 0))[fresh]
        self.center[fresh] = center[fresh]
        changed = (center != self.center).any(1)
        displaced = (wheel_xyz - self.previous).norm(dim=2).amax(1) > p['max_step_displacement']
        current_index = (torch.floor((xyz[:, :, 0] - p['first_edge']) / p['step_width']) + 1).long().clamp(0, p['risers'])
        skipped = (current_index > self.next_riser + 1).any(1)
        self.invalid |= ~finite | ~safe | ~corridor | changed | ((displaced | skipped) & ~fresh)
        old_x = self.previous[:, :, 0] - center[:, None, 0]
        edge = p['first_edge'] + self.next_riser * p['step_width']
        dx = xyz[:, :, 0] - old_x
        crossed = (old_x < edge) & (xyz[:, :, 0] >= edge) & (dx > 0)
        crossed &= (self.next_riser < p['risers']) & ~fresh[:, None]
        alpha = ((edge - old_x) / dx.clamp_min(1e-9)).clamp(0, 1)
        z_at_edge = self.previous[:, :, 2] + alpha * (wheel_xyz[:, :, 2] - self.previous[:, :, 2])
        top_z = center[:, None, 2] * (self.next_riser + 1) / p['risers']
        clearance = z_at_edge - p['wheel_radius'] - top_z
        score = ((clearance - p['clearance_deadband']) /
                 (p['clearance_full'] - p['clearance_deadband'])).clamp(0, 1)
        allowed = self.eligible & positive_stairs & ~self.invalid & gate & self.previous_gate
        event = crossed & allowed[:, None]
        wheel_bonus = torch.where(event, score * p['bonus_per_wheel'], 0.0)
        wheel_bonus = torch.nan_to_num(wheel_bonus, nan=0., posinf=0., neginf=0.)
        self.last_paid.copy_(wheel_bonus > 0)
        self.last_crossed.copy_(crossed & self.eligible[:, None])
        self.last_clearance.copy_(torch.where(crossed & finite[:, None], clearance, 0.0))
        self.paid_events += self.last_paid.long()
        self.crossings += self.last_crossed.long()
        # 永不回退最高已越过的阶数；未开门控/净空不够的第一次跨越也不重复付。
        self.next_riser.copy_(torch.maximum(self.next_riser, current_index))
        bonus = wheel_bonus.sum(1)
        self.episode_bonus += bonus
        self.previous.copy_(torch.nan_to_num(wheel_xyz))
        self.previous_gate.copy_(gate & finite)
        self.initialized.fill_(True)
        return bonus
