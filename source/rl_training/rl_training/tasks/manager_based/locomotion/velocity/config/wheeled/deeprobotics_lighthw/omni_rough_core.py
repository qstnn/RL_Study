"""完整rough迁移的纯张量规则；不修改旧路线，也不直接生成关节控制。"""
from copy import deepcopy
import torch

VERSION = 'lighthw_omni_rough_retention_v1'
COURSE_VERSION = 'lighthw_omni_and_retained_flights_v1'
COMMAND_CONTRACT = dict(
    version=VERSION, retention_rule='even_environment_ids', retention_fraction=.5,
    zero_command_probability=.25, resampling_seconds=8.,
    retention_flat_vx=[0.,3.], retention_rough_vx=[0.,1.], retention_anchor_probability=.5,
    general_vx=[-2.,2.], general_vy=[-1.,1.], general_wz=[-1.,1.],
    general_moving_modes=['longitudinal','lateral','yaw','mixed','heading'],
    mode_probabilities=[.2]*5, heading_range=[-3.141592653589793,3.141592653589793],
    heading_control_stiffness=.5, obstacle_planar_speed_cap=1.,
    general_initial_level=0, retained_initial_levels='exact source per-env levels',
    all_16_actions_from_actor=True, behavior_cloning=False, wheel_feedforward=False,
)
NOISE_CONTRACT = dict(
    version='lighthw_commanded_motion_wheel_sigma2_v1', command_index=9,
    command_threshold=.05, forward_multiplier=2., otherwise_multiplier=1.,
    rule='norm(physical_command_vx_vy_wz) > 0.05',
    base_log_std_reset=False, base_bounds_unchanged=True, leg_distribution_unchanged=True,
    stop_distribution_unchanged=True, environment_side_noise=False,
)


def retention_mask(ids):
    return ids.remainder(2) == 0


def moving_command(command, threshold=.05):
    """所有方向共用命令范数门控；不以实际车速判断停车。"""
    return command.norm(dim=-1) > threshold


def gated_contact(original, command):
    return torch.where(moving_command(command), torch.zeros_like(original), original)


def acceleration_factor(command, threshold=.05, multiplier=.05):
    return torch.where(moving_command(command, threshold), multiplier, 1.)


def limit_obstacle_command(command, obstacle):
    """台阶/方块仅缩放平移速度，保持方向与yaw命令；不更改Actor输出。"""
    result = command.clone()
    speed = result[:, :2].norm(dim=1).clamp_min(1e-8)
    factor = torch.where(obstacle, (1. / speed).clamp_max(1.), 1.)
    result[:, :2] *= factor[:, None]
    return result


def migrated_mapping(source, target):
    """仅显式迁移命令分布签名，腿轮物理映射与正确观测契约逐字段保持。"""
    expected = deepcopy(source)
    expected['command_distribution'] = deepcopy(COMMAND_CONTRACT)
    if expected != target:
        raise ValueError('Unexpected action/observation migration outside command distribution')
    return expected


class OmniTraversalState:
    """M20式任意方向离开出生区域课程，不能把原地转向算作越障成功。

    在出生地块内或离开该地块的首帧，实际平移距离达到4m才记录通行资格；
    之后整个episode保持非轮无碰撞/倾角安全/有限才晋级。转向、零命令暴露
    不作为行驶受阻；运动低载不参与判定。课程资格不等于完整地形独立验收。
    """
    def __init__(self, n, device):
        with torch.inference_mode(False):
            self.steps=torch.zeros(n,device=device,dtype=torch.long)
            self.unsafe=torch.zeros(n,device=device,dtype=torch.bool)
            self.qualified=torch.zeros_like(self.unsafe)
            self.left_spawn=torch.zeros_like(self.unsafe)
            self.exposure=torch.zeros(n,device=device)
            self.max_distance=torch.zeros(n,device=device)
            self.start=torch.zeros(n,2,device=device)

    @torch.no_grad()
    def observe(self, xy, command, bad, spawn_cell, current_cell, inside, dt, spawn_xy=None):
        origin=xy if spawn_xy is None else spawn_xy
        self.start.copy_(torch.where((self.steps==0)[:,None],origin,self.start))
        distance=torch.nan_to_num((xy-self.start).norm(dim=1),nan=0.,posinf=0.)
        at_spawn=(spawn_cell==current_cell).all(1)&inside
        just_leaving=~at_spawn&~self.left_spawn
        # 拒绝坐标跳变/错误reset带来的虚假首帧通行。
        local_jump=(current_cell-spawn_cell).abs().amax(1)>1
        self.exposure.add_(command[:,:2].norm(dim=1)*dt)
        qualified=(at_spawn|just_leaving)&~local_jump&(distance>=4.)&(self.steps>0)&(self.exposure>=1.)
        self.qualified.logical_or_(qualified)
        self.left_spawn.logical_or_(~at_spawn)
        self.unsafe.logical_or_(bad)
        self.max_distance.copy_(torch.maximum(self.max_distance,distance))
        self.steps.add_(1)

    def decisions(self, ids, levels, flat, terminated):
        sampled=self.steps[ids]>0
        unsafe=self.unsafe[ids]|terminated
        success=self.qualified[ids]&~unsafe
        enough=self.exposure[ids]>=8.
        up=sampled&~flat&success&(levels<9)
        down=sampled&~flat&~up&(unsafe|(enough&~success))
        next_level=torch.where(flat,0,(levels+up.long()-down.long()).clamp(0,9))
        return next_level,up,down,unsafe,sampled&~enough&~unsafe&~success

    def reset(self, ids):
        for value in vars(self).values():
            value[ids]=0
